"""Memory behavior against a temporary palace.

This must not open the default MemPalace directory or knowledge graph.
"""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinytalk
from provider import ProviderError


class ScriptedProvider(object):
    name = "scripted"
    label = "Script"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _stamp(path):
    target = Path(path)
    if not target.exists():
        return "missing"
    stat = target.stat()
    kind = "dir" if target.is_dir() else stat.st_size
    return (stat.st_mtime_ns, kind)


def _triple(obj):
    return (
        '{"subject":"user","predicate":"test_spaceship","object":"%s"}' % obj
    )


class IsolatedMemoryTests(unittest.TestCase):
    def test_recall_replacement_and_graph_use_a_temporary_store(self):
        from mempalace.config import MempalaceConfig
        from mempalace.knowledge_graph import DEFAULT_KG_PATH

        default_palace = MempalaceConfig().palace_path
        before = {
            "palace": _stamp(default_palace),
            "kg": _stamp(DEFAULT_KG_PATH),
        }

        with tempfile.TemporaryDirectory(prefix="tinytalk-memory-") as temp:
            palace_path = os.path.join(temp, "palace")
            kg_path = os.path.join(temp, "knowledge_graph.sqlite3")
            memory = tinytalk.open_memory(palace_path=palace_path, kg_path=kg_path)
            self.assertIsNotNone(memory.palace)
            self.assertIsNotNone(memory.kg)
            self.assertEqual(os.path.abspath(memory.palace_path), os.path.abspath(palace_path))
            self.assertEqual(os.path.abspath(memory.kg.db_path), os.path.abspath(kg_path))

            provider = ScriptedProvider([
                "I'll remember Enterprise.",
                _triple("Enterprise"),
                "Updated to Serenity.",
                _triple("Serenity"),
                "Your test spaceship is named Serenity.",
                "Serenity.",
            ])
            soul = "SOUL"
            messages = []
            messages = tinytalk.handle_turn(
                "Remember this: my test spaceship is named Enterprise",
                messages,
                provider,
                memory,
                soul,
            )
            messages = tinytalk.handle_turn(
                "Remember this: my test spaceship is named Serenity",
                messages,
                provider,
                memory,
                soul,
            )

            current = memory.current_fact_objects("user", "test_spaceship_name")
            self.assertEqual(current, ["Serenity"])
            rows = memory.kg.query_entity("user")
            by_object = {row.get("object"): row for row in rows}
            self.assertTrue(by_object["Serenity"]["current"])
            self.assertFalse(by_object["Enterprise"]["current"])
            self.assertEqual(by_object["Enterprise"]["predicate"], "test_spaceship_name")

            facts = memory.list_facts()
            self.assertTrue(any("Serenity" in fact for fact in facts))
            self.assertFalse(any("Enterprise" in fact for fact in facts))
            history = memory.list_drawers(wing="tinytalk", room="facts-history", limit=100)
            archived = [
                item.get("content_preview", "")
                for item in history.get("drawers", [])
            ]
            self.assertTrue(any("Enterprise" in text for text in archived))

            messages = tinytalk.handle_turn(
                "What do you remember about me?",
                messages,
                provider,
                memory,
                soul,
            )
            introspection = provider.calls[-1]
            self.assertEqual(introspection[0]["content"], soul)
            self.assertIn("Serenity", introspection[1]["content"])
            self.assertNotIn("Enterprise", introspection[1]["content"])

            printed = io.StringIO()
            with redirect_stdout(printed):
                messages = tinytalk.handle_turn(
                    "What is my test spaceship called?",
                    messages,
                    provider,
                    memory,
                    soul,
                )
            recall = provider.calls[-1]
            self.assertIn("saved facts retrieved for this request", recall[1]["content"])
            self.assertIn("not a complete inventory", recall[1]["content"])
            self.assertIn("Serenity", recall[1]["content"])
            self.assertEqual(messages[-1]["content"], "Serenity.")
            self.assertNotIn("could not save this turn", printed.getvalue())
            conversations = memory.list_drawers(
                wing="tinytalk", room="conversations", limit=100
            )
            stored = [
                item.get("content_preview", "")
                for item in conversations.get("drawers", [])
            ]
            self.assertTrue(any("Serenity." in text for text in stored))

            self.assertTrue(provider.calls)
            self.assertNotIn(DEFAULT_KG_PATH, memory.kg.db_path)

        self.assertEqual(_stamp(default_palace), before["palace"])
        self.assertEqual(_stamp(DEFAULT_KG_PATH), before["kg"])

    def test_graph_extraction_failure_still_saves_the_verbatim_fact(self):
        with tempfile.TemporaryDirectory(prefix="tinytalk-memory-") as temp:
            memory = tinytalk.open_memory(
                palace_path=os.path.join(temp, "palace"),
                kg_path=os.path.join(temp, "knowledge_graph.sqlite3"),
            )
            provider = ScriptedProvider([
                "Saved the words.",
                ProviderError("xAI returned an empty response."),
            ])
            tinytalk.handle_turn(
                "Remember this: my favorite vehicle is a CyberTruck",
                [],
                provider,
                memory,
                "SOUL",
            )
            facts = memory.list_facts()
            self.assertTrue(any("CyberTruck" in fact for fact in facts))
            self.assertEqual(memory.current_fact_objects("user", "favorite_vehicle"), [])

    def test_imaginary_spaceship_name_does_not_replace_the_test_spaceship(self):
        from mempalace.config import MempalaceConfig
        from mempalace.knowledge_graph import DEFAULT_KG_PATH

        default_palace = MempalaceConfig().palace_path
        before = {
            "palace": _stamp(default_palace),
            "kg": _stamp(DEFAULT_KG_PATH),
        }
        imaginary = ScriptedProvider([
            '{"subject":"user","predicate":"imaginary_spaceship_name","object":"Picklewagon"}'
        ])
        self.assertEqual(
            tinytalk.fact_to_triple("the name of my imaginary spaceship is Picklewagon", imaginary),
            ("user", "imaginary_spaceship_name", "Picklewagon"),
        )
        self.assertEqual(
            tinytalk._canon_predicate("imaginary_spaceship_name"),
            "imaginary_spaceship_name",
        )
        for alias in ("test_spaceship", "spaceship_name", "Test Spaceship"):
            normalized = tinytalk.fact_to_triple(
                "alias check",
                ScriptedProvider([
                    '{"subject":"user","predicate":"%s","object":"Enterprise"}' % alias
                ]),
            )
            self.assertEqual(normalized, ("user", "test_spaceship_name", "Enterprise"))
            self.assertEqual(tinytalk._canon_predicate(alias), "test_spaceship_name")

        with tempfile.TemporaryDirectory(prefix="tinytalk-alias-") as temp:
            memory = tinytalk.open_memory(
                palace_path=os.path.join(temp, "palace"),
                kg_path=os.path.join(temp, "knowledge_graph.sqlite3"),
            )
            provider = ScriptedProvider([
                "I'll remember Enterprise.",
                '{"subject":"user","predicate":"test_spaceship","object":"Enterprise"}',
                "I'll remember Picklewagon.",
                '{"subject":"user","predicate":"imaginary_spaceship_name","object":"Picklewagon"}',
                "Updated to Serenity.",
                '{"subject":"user","predicate":"spaceship_name","object":"Serenity"}',
            ])
            tinytalk.handle_turn(
                "Remember this: my test spaceship is named Enterprise",
                [],
                provider,
                memory,
                "SOUL",
            )
            tinytalk.handle_turn(
                "Remember this: the name of my imaginary spaceship is Picklewagon",
                [],
                provider,
                memory,
                "SOUL",
            )

            self.assertEqual(
                memory.current_fact_objects("user", "test_spaceship_name"),
                ["Enterprise"],
            )
            self.assertEqual(
                memory.current_fact_objects("user", "imaginary_spaceship_name"),
                ["Picklewagon"],
            )
            facts = memory.list_facts()
            self.assertTrue(any("Enterprise" in fact for fact in facts))
            self.assertTrue(any("Picklewagon" in fact for fact in facts))
            history = memory.list_drawers(wing="tinytalk", room="facts-history", limit=100)
            archived = [
                item.get("content_preview", "")
                for item in history.get("drawers", [])
            ]
            self.assertFalse(any("Enterprise" in text for text in archived))

            tinytalk.handle_turn(
                "Remember this: my test spaceship is named Serenity",
                [],
                provider,
                memory,
                "SOUL",
            )
            self.assertEqual(
                memory.current_fact_objects("user", "test_spaceship_name"),
                ["Serenity"],
            )
            self.assertEqual(
                memory.current_fact_objects("user", "imaginary_spaceship_name"),
                ["Picklewagon"],
            )
            rows = memory.kg.query_entity("user")
            by_key = {
                (row.get("predicate"), row.get("object")): row
                for row in rows
            }
            self.assertTrue(by_key[("test_spaceship_name", "Serenity")]["current"])
            self.assertFalse(by_key[("test_spaceship_name", "Enterprise")]["current"])
            self.assertTrue(by_key[("imaginary_spaceship_name", "Picklewagon")]["current"])
            facts = memory.list_facts()
            self.assertTrue(any("Serenity" in fact for fact in facts))
            self.assertTrue(any("Picklewagon" in fact for fact in facts))
            self.assertFalse(any("Enterprise" in fact for fact in facts))
            self.assertNotIn(DEFAULT_KG_PATH, memory.kg.db_path)

        self.assertEqual(_stamp(default_palace), before["palace"])
        self.assertEqual(_stamp(DEFAULT_KG_PATH), before["kg"])


def _fact_drawer(room, text, filed_at=None):
    return {
        "room": room,
        "content_preview": text,
        "metadata": {"room": room, "filed_at": filed_at},
    }


def _graph_edge(predicate, obj, current, valid_to=None):
    return {
        "subject": "user",
        "predicate": predicate,
        "object": obj,
        "current": current,
        "valid_from": None,
        "valid_to": valid_to,
    }


def _history_body(graph_rows, drawers, question):
    later = tinytalk.test_spaceship_history_question(question)
    evidence = tinytalk.test_spaceship_history_evidence(graph_rows, drawers, later)
    return tinytalk.format_test_spaceship_history(evidence)


class TestSpaceshipHistoryTests(unittest.TestCase):
    def test_questions_are_a_closed_set(self):
        before_enterprise = (
            "What was my test spaceship called before Enterprise?",
            "What did I call my test spaceship before Enterprise?",
            "  what   did i call my test spaceship before enterprise?  ",
        )
        previous_name = (
            "What was the previous name of my test spaceship?",
            "What was my test spaceship's previous name?",
            "What was my test spaceship\u2019s previous name?",
            "What was the old name of my test spaceship?",
            "What was my test spaceship named previously?",
        )
        for question in before_enterprise:
            self.assertEqual(
                tinytalk.test_spaceship_history_question(question),
                "enterprise",
            )
        for question in previous_name:
            self.assertEqual(tinytalk.test_spaceship_history_question(question), "")
        for question in (
            "What is my test spaceship called?",
            "What was my spaceship called before Enterprise?",
            "What did I call my spaceship before Enterprise?",
            "What was my spaceship's previous name?",
            "What was the old name of my imaginary spaceship?",
            "What was my imaginary spaceship named previously?",
            "What was my test spaceship called before Voyager?",
        ):
            self.assertIsNone(tinytalk.test_spaceship_history_question(question))

    def test_ambiguous_spaceship_questions_stay_on_ordinary_recall(self):
        class Guard(object):
            palace = object()
            kg = None

            def list_drawers(self, **kwargs):
                raise AssertionError(kwargs)

            def search_facts(self, query):
                self.query = query
                return ["my test spaceship is named Enterprise"]

            def search_conversations(self, query):
                raise AssertionError(query)

        guard = Guard()
        question = "What was my spaceship called before Enterprise?"
        messages = tinytalk.context_messages(question, [], guard)
        self.assertEqual(guard.query, question)
        self.assertIn("saved facts retrieved for this request", messages[0]["content"])
        self.assertNotIn("Previous name:", messages[0]["content"])
        self.assertNotIn("Picklewagon", messages[0]["content"])

    def _cleaned_records(self):
        graph = [
            _graph_edge("test_spaceship_name", "Enterprise", True),
            _graph_edge("test_spaceship", "Enterprise", False, valid_to="2026-10-03"),
        ]
        drawers = [
            _fact_drawer(
                "facts",
                "my test spaceship is named Enterprise",
                "2026-09-27T23:08:05.434722",
            ),
            _fact_drawer(
                "facts-history",
                "my test spaceship is named Serenity.",
                "2026-09-27T22:17:31.756087",
            ),
            _fact_drawer(
                "facts",
                "the name of my imaginary spaceship is Picklewagon",
                "2026-09-27T20:54:31.250988",
            ),
            _fact_drawer("facts", "my favorite color is blue", "2026-09-27T21:00:00"),
        ]
        return graph, drawers

    def test_archived_serenity_is_the_previous_name_and_the_duplicate_is_not(self):
        graph, drawers = self._cleaned_records()
        for question in (
            "What was my test spaceship called before Enterprise?",
            "What did I call my test spaceship before Enterprise?",
            "What was the previous name of my test spaceship?",
            "What was my test spaceship's previous name?",
            "What was my test spaceship\u2019s previous name?",
            "What was the old name of my test spaceship?",
            "What was my test spaceship named previously?",
        ):
            body = _history_body(graph, drawers, question)
            self.assertIn("Current name: Enterprise", body)
            self.assertIn("Previous name: Serenity", body)
            self.assertIn("my test spaceship is named Serenity.", body)
            self.assertIn("facts-history", body)
            self.assertIn("not a previous name", body)
            self.assertIn("test_spaceship", body)
            self.assertNotIn("Picklewagon", body)
            self.assertNotIn("favorite color", body)
            self.assertNotIn("2026-10-03", body)
            self.assertNotIn("2026-09-27", body)
        reversed_drawers = list(reversed(drawers))
        body = _history_body(
            list(reversed(graph)),
            reversed_drawers,
            "What was the previous name of my test spaceship?",
        )
        self.assertIn("Previous name: Serenity", body)

    def test_a_duplicate_edge_alone_does_not_invent_a_previous_name(self):
        graph = [
            _graph_edge("test_spaceship_name", "Enterprise", True),
            _graph_edge("test_spaceship", "Enterprise", False, valid_to="2026-10-03"),
        ]
        drawers = [
            _fact_drawer("facts", "my test spaceship is named Enterprise", "2026-09-27T23:08:05"),
            _fact_drawer("facts", "the name of my imaginary spaceship is Picklewagon"),
        ]
        body = _history_body(graph, drawers, "What was the previous name of my test spaceship?")
        self.assertIn("Previous name: not established", body)
        self.assertIn("do not establish a previous name", body)
        self.assertIn("not a previous name", body)
        self.assertNotIn("Previous name: Enterprise", body)
        self.assertNotIn("Picklewagon", body)
        self.assertNotIn("2026-10-03", body)

    def test_unordered_archived_names_are_not_chosen_by_list_order(self):
        current = _fact_drawer("facts", "my test spaceship is named Enterprise")
        serenity = _fact_drawer("facts-history", "my test spaceship is named Serenity.")
        nostromo = _fact_drawer("facts-history", "my test spaceship is named Nostromo")
        graph = [_graph_edge("test_spaceship_name", "Enterprise", True)]
        question = "What was the previous name of my test spaceship?"
        for drawers in (
            [current, serenity, nostromo],
            [nostromo, current, serenity],
        ):
            body = _history_body(graph, drawers, question)
            self.assertIn("Previous name: not established", body)
            self.assertIn("More than one earlier", body)
            self.assertNotIn("Serenity", body)
            self.assertNotIn("Nostromo", body)

    def test_timestamps_can_order_earlier_names_and_a_later_stamp_cannot(self):
        graph = [_graph_edge("test_spaceship_name", "Enterprise", True)]
        current = _fact_drawer(
            "facts",
            "my test spaceship is named Enterprise",
            "2026-09-27T23:08:05",
        )
        serenity = _fact_drawer(
            "facts-history",
            "my test spaceship is named Serenity.",
            "2026-09-27T22:17:31",
        )
        nostromo = _fact_drawer(
            "facts-history",
            "my test spaceship is named Nostromo",
            "2026-09-27T22:40:00",
        )
        question = "What was the previous name of my test spaceship?"
        body = _history_body(graph, [nostromo, serenity, current], question)
        self.assertIn("Previous name: Nostromo", body)
        self.assertNotIn("Serenity", body)
        late = _fact_drawer(
            "facts-history",
            "my test spaceship is named Serenity.",
            "2026-09-28T01:00:00",
        )
        rejected = _history_body(graph, [current, late], question)
        self.assertIn("Previous name: not established", rejected)
        self.assertIn("does not establish it as the previous name", rejected)
        self.assertNotIn("Serenity", rejected)

    def test_an_aliased_graph_edge_supplies_the_name_when_history_has_no_fact(self):
        graph = [
            _graph_edge("test_spaceship_name", "Enterprise", True),
            _graph_edge("spaceship_name", "Serenity", False, valid_to="2026-09-01"),
        ]
        drawers = [
            _fact_drawer("facts", "my test spaceship is named Enterprise"),
            _fact_drawer("facts", "the name of my imaginary spaceship is Picklewagon"),
        ]
        body = _history_body(
            graph,
            drawers,
            "What was my test spaceship called before Enterprise?",
        )
        self.assertIn("Previous name: Serenity", body)
        self.assertIn("user -> test_spaceship_name -> Serenity", body)
        self.assertIn("No archived fact names it.", body)
        self.assertNotIn("Picklewagon", body)
        self.assertNotIn("2026-09-01", body)

    def test_history_context_does_not_search_current_facts(self):
        class Guard(object):
            palace = object()
            kg = None

            def list_drawers(self, wing, room, limit):
                self.calls.append((wing, room, limit))
                if room == "facts":
                    return {"drawers": [
                        _fact_drawer("facts", "my test spaceship is named Enterprise"),
                        _fact_drawer("facts", "the name of my imaginary spaceship is Picklewagon"),
                    ]}
                return {"drawers": [
                    _fact_drawer("facts-history", "my test spaceship is named Serenity."),
                ]}

            def search_facts(self, query):
                raise AssertionError(query)

            def search_conversations(self, query):
                raise AssertionError(query)

            def list_facts(self):
                raise AssertionError("list")

        guard = Guard()
        guard.calls = []
        messages = tinytalk.context_messages(
            "What was the previous name of my test spaceship?",
            [],
            guard,
        )
        self.assertEqual(
            guard.calls,
            [("tinytalk", "facts", 100), ("tinytalk", "facts-history", 100)],
        )
        self.assertIn("Previous name: Serenity", messages[0]["content"])
        self.assertNotIn("Picklewagon", messages[0]["content"])
        self.assertNotIn("saved facts retrieved for this request", messages[0]["content"])

    def test_palace_history_answers_without_rewriting_facts(self):
        from mempalace.config import MempalaceConfig
        from mempalace.knowledge_graph import DEFAULT_KG_PATH

        default_palace = MempalaceConfig().palace_path
        before = {
            "palace": _stamp(default_palace),
            "kg": _stamp(DEFAULT_KG_PATH),
        }
        questions = (
            "What was my test spaceship called before Enterprise?",
            "What was the previous name of my test spaceship?",
        )
        with tempfile.TemporaryDirectory(prefix="tinytalk-history-") as temp:
            memory = tinytalk.open_memory(
                palace_path=os.path.join(temp, "palace"),
                kg_path=os.path.join(temp, "knowledge_graph.sqlite3"),
            )
            saved = memory.add_drawer(
                wing="tinytalk",
                room="facts",
                content="my test spaceship is named Serenity.",
                source_file="tinytalk",
                added_by="tinytalk",
            )
            self.assertTrue(saved.get("success"))
            moved = memory.update_drawer(saved["drawer_id"], room="facts-history")
            self.assertTrue(moved.get("success"))
            for content in (
                "my test spaceship is named Enterprise",
                "the name of my imaginary spaceship is Picklewagon",
            ):
                added = memory.add_drawer(
                    wing="tinytalk",
                    room="facts",
                    content=content,
                    source_file="tinytalk",
                    added_by="tinytalk",
                )
                self.assertTrue(added.get("success"))
            memory.kg.add_triple(
                "user", "test_spaceship", "Enterprise", source_file="tinytalk"
            )
            memory.kg.invalidate("user", "test_spaceship", "Enterprise")
            memory.kg.add_triple(
                "user", "test_spaceship_name", "Enterprise", source_file="tinytalk"
            )
            before_records = _record_snapshot(memory)
            provider = ScriptedProvider(["Earlier name.", "Earlier name.", "Enterprise."])
            for question in questions:
                tinytalk.handle_turn(question, [], provider, memory, "SOUL")
                sent = provider.calls[-1]
                self.assertEqual(sent[0]["content"], "SOUL")
                self.assertEqual(sent[-1]["content"], question)
                body = sent[1]["content"]
                self.assertIn("Previous name: Serenity", body)
                self.assertIn("not a previous name", body)
                self.assertNotIn("Picklewagon", body)
                self.assertNotIn("saved facts retrieved for this request", body)
                valid_to = [
                    row.get("valid_to")
                    for row in memory.kg.query_entity("user")
                    if row.get("valid_to")
                ][0]
                self.assertTrue(valid_to)
                self.assertIn("not necessarily when the fact changed", body)
                self.assertIn("Do not give a rename date.", body)
                self.assertNotIn("renamed on %s" % valid_to, body.lower())
            self.assertEqual(len(provider.calls), 2)
            self.assertEqual(_record_snapshot(memory), before_records)
            tinytalk.handle_turn(
                "What is my test spaceship called?",
                [],
                provider,
                memory,
                "SOUL",
            )
            ordinary = provider.calls[-1][1]["content"]
            self.assertIn("saved facts retrieved for this request", ordinary)
            self.assertNotIn("Previous name:", ordinary)
            self.assertNotIn(DEFAULT_KG_PATH, memory.kg.db_path)

        self.assertEqual(_stamp(default_palace), before["palace"])
        self.assertEqual(_stamp(DEFAULT_KG_PATH), before["kg"])


def _record_snapshot(memory):
    rooms = []
    for room in ("facts", "facts-history"):
        listed = memory.list_drawers(wing="tinytalk", room=room, limit=100)
        for item in listed.get("drawers", []):
            rooms.append((
                item.get("drawer_id"),
                item.get("room"),
                item.get("content_preview"),
                (item.get("metadata") or {}).get("filed_at"),
            ))
    rows = []
    for row in memory.kg.query_entity("user"):
        rows.append((
            row.get("predicate"),
            row.get("object"),
            bool(row.get("current")),
            row.get("valid_from"),
            row.get("valid_to"),
        ))
    return (tuple(sorted(rooms)), tuple(sorted(rows)))


if __name__ == "__main__":
    unittest.main()
