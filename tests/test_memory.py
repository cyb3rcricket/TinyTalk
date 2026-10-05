"""Memory behavior against a temporary palace.

memory_isolation keeps every MemPalace path inside a temporary root and
refuses to open a store anywhere else.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_isolation  # noqa: E402  (must come before tinytalk and mempalace)
import tinytalk
from provider import ProviderError


class GraphScanTests(unittest.TestCase):
    def _memory(self, triples):
        calls = []

        class KG(object):
            def dump_rows(self, table, after_rowid=0, limit=1000):
                calls.append(table)
                if table == "entities":
                    rows = [{"id": "user", "name": "user", "_rowid": 1}]
                    for index, row in enumerate(triples, 2):
                        rows.append({
                            "id": row["object_id"],
                            "name": row["object"],
                            "_rowid": index,
                        })
                    return rows
                return [
                    {
                        "id": "t-%s" % index,
                        "subject": "user",
                        "predicate": row["predicate"],
                        "object": row["object_id"],
                        "valid_to": None,
                        "source_drawer_id": row.get("source_drawer_id"),
                        "_rowid": index,
                    }
                    for index, row in enumerate(triples, 1)
                ]

        def list_drawers(wing, room, limit, offset=0):
            if offset or room != "facts-pending":
                return {"drawers": [], "total": 0}
            drawers = []
            for text, predicate, obj in (
                ("call me Al", "preferred_name", "Al"),
                ("call me Sam", "preferred_name", "Sam"),
            ):
                drawers.append({
                    "drawer_id": text,
                    "content_preview": text,
                    "metadata": {
                        "source_file": "tinytalk?predicate=%s&object=%s" % (predicate, obj),
                    },
                })
            return {"drawers": drawers, "total": len(drawers)}

        memory = tinytalk.MemoryStore(object(), "/tmp/unused", KG(), None, list_drawers, None)
        return memory, calls

    def test_unfinished_updates_read_the_graph_once(self):
        memory, calls = self._memory([{
            "predicate": "preferred_name",
            "object": "Al",
            "object_id": "al",
        }])
        updates = memory.unfinished_updates()
        self.assertEqual(calls, ["entities", "triples"])
        self.assertEqual([update["last_value"] for update in updates], ["Al", "Al"])

    def test_current_links_match_an_alias_and_skip_the_imaginary_ship(self):
        memory, _calls = self._memory([
            {"predicate": "test_spaceship", "object": "Serenity", "object_id": "serenity"},
            {"predicate": "spaceship_name", "object": "Serenity", "object_id": "serenity"},
            {
                "predicate": "imaginary_spaceship_name",
                "object": "Nostromo",
                "object_id": "nostromo",
            },
        ])
        ships = memory.current_links("user", "test_spaceship_name")
        self.assertEqual([link["object"] for link in ships], ["Serenity", "Serenity"])
        imaginary = memory.current_links("user", "imaginary_spaceship_name")
        self.assertEqual([link["object"] for link in imaginary], ["Nostromo"])


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


def _triple(obj):
    return (
        '{"subject":"user","predicate":"test_spaceship","object":"%s"}' % obj
    )


class IsolatedMemoryTests(unittest.TestCase):
    def test_recall_replacement_and_graph_use_a_temporary_store(self):
        with memory_isolation.temp_dir("tinytalk-memory-") as temp:
            palace_path = os.path.join(temp, "palace")
            kg_path = os.path.join(temp, "knowledge_graph.sqlite3")
            memory = memory_isolation.open_isolated_memory(temp)
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

    def test_graph_extraction_failure_still_saves_the_verbatim_fact(self):
        with memory_isolation.temp_dir("tinytalk-memory-") as temp:
            memory = memory_isolation.open_isolated_memory(temp)
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

        with memory_isolation.temp_dir("tinytalk-alias-") as temp:
            memory = memory_isolation.open_isolated_memory(temp)
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


def _fact_drawer(room, text, filed_at=None):
    return {
        "room": room,
        "content_preview": text,
        "metadata": {"room": room, "filed_at": filed_at},
    }


def _graph_edge(predicate, obj, current, valid_to=None, valid_from=None):
    return {
        "subject": "user",
        "predicate": predicate,
        "object": obj,
        "current": current,
        "valid_from": valid_from,
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

    def test_filing_times_do_not_order_earlier_names(self):
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
        self.assertIn("Previous name: not established", body)
        self.assertIn("More than one earlier", body)
        self.assertNotIn("Previous name: Nostromo", body)
        self.assertNotIn("Previous name: Serenity", body)
        late = _fact_drawer(
            "facts-history",
            "my test spaceship is named Serenity.",
            "2026-09-28T01:00:00",
        )
        accepted = _history_body(graph, [current, late], question)
        self.assertIn("Previous name: Serenity", accepted)
        self.assertNotIn("2026-09-28", accepted)

    def test_graph_boundaries_order_names_when_filing_times_do_not(self):
        graph = [
            _graph_edge(
                "test_spaceship_name", "Serenity", False,
                valid_from="2026-08-01T00:00:00Z",
                valid_to="2026-09-01T00:00:00Z",
            ),
            _graph_edge(
                "test_spaceship_name", "Enterprise", False,
                valid_from="2026-09-01T00:00:00Z",
                valid_to="2026-10-01T00:00:00Z",
            ),
            _graph_edge(
                "test_spaceship_name", "Voyager", True,
                valid_from="2026-10-01T00:00:00Z",
            ),
        ]
        # Filing times disagree with the graph: Serenity is filed last.
        drawers = [
            _fact_drawer("facts", "my test spaceship is named Voyager", "2026-09-01T00:00:00"),
            _fact_drawer(
                "facts-history", "my test spaceship is named Enterprise", "2026-08-01T00:00:00"
            ),
            _fact_drawer(
                "facts-history", "my test spaceship is named Serenity.", "2026-10-02T00:00:00"
            ),
        ]
        body = _history_body(graph, drawers, "What was the previous name of my test spaceship?")
        self.assertIn("Current name: Voyager", body)
        self.assertIn("Previous name: Enterprise", body)
        self.assertNotIn("Previous name: Serenity", body)
        self.assertNotIn("2026-10-01", body)
        self.assertIn("Do not give a rename date.", body)

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

            def list_drawers(self, wing, room, limit, offset=0):
                self.calls.append((wing, room, limit, offset))
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
            [("tinytalk", "facts", 100, 0), ("tinytalk", "facts-history", 100, 0)],
        )
        self.assertIn("Previous name: Serenity", messages[0]["content"])
        self.assertNotIn("Picklewagon", messages[0]["content"])
        self.assertNotIn("saved facts retrieved for this request", messages[0]["content"])

    def test_palace_history_answers_without_rewriting_facts(self):
        questions = (
            "What was my test spaceship called before Enterprise?",
            "What was the previous name of my test spaceship?",
        )
        with memory_isolation.temp_dir("tinytalk-history-") as temp:
            memory = memory_isolation.open_isolated_memory(temp)
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


class FullFactReadTests(unittest.TestCase):
    def test_introspection_reads_past_the_first_hundred_facts(self):
        facts = ["fact number %s" % index for index in range(102)]

        class Memory(object):
            palace = object()

            def list_drawers(self, wing, room, limit=100, offset=0):
                page = facts[offset:offset + limit]
                return {
                    "drawers": [
                        {
                            "drawer_id": "fact-%s" % (offset + index),
                            "content_preview": text,
                            "metadata": {"room": room},
                        }
                        for index, text in enumerate(page)
                    ],
                    "total": len(facts),
                }

            def unfinished_updates(self):
                return []

        _messages, sources = tinytalk.build_context(
            "What do you remember about me?",
            [{"role": "user", "content": "What do you remember about me?"}],
            Memory(),
        )
        texts = [source["text"] for source in sources if source["kind"] == "saved fact"]
        self.assertEqual(texts, facts)

    def test_history_reads_a_name_past_the_preview(self):
        prefix = "a" * 190
        full = prefix + " my test spaceship is named Serenity."
        preview = full[:200] + "..."

        class Memory(object):
            palace = object()
            kg = None

            def list_drawers(self, wing, room, limit=100, offset=0):
                if offset:
                    return {"drawers": [], "total": 1}
                if room == "facts":
                    item = _fact_drawer("facts", "my test spaceship is named Enterprise")
                else:
                    item = {
                        "drawer_id": "long-serenity",
                        "room": room,
                        "content_preview": preview,
                        "metadata": {"room": room},
                    }
                return {"drawers": [item], "total": 1}

            def get_drawer(self, drawer_id):
                if drawer_id != "long-serenity":
                    return {"error": "Drawer not found: %s" % drawer_id}
                return {"content": full, "room": "facts-history", "metadata": {}}

            def search_facts(self, query):
                raise AssertionError(query)

        messages = tinytalk.context_messages(
            "What was the previous name of my test spaceship?",
            [],
            Memory(),
        )
        self.assertIn("Previous name: Serenity", messages[0]["content"])
        self.assertNotIn(preview, messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
