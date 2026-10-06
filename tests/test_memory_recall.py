"""Current-profile recall. No live model and no real MemPalace.

memory_isolation keeps every MemPalace path inside a temporary root.
A semantic miss must not promote an old assistant answer to the current name.
"""

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_isolation  # noqa: E402  (must come before tinytalk and mempalace)
import tinytalk


class StubProvider(object):
    name = "stub"
    label = "Stub"

    def __init__(self, answers=None, reply="Okay."):
        self.answers = dict(answers or {})
        self.reply = reply
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        if messages and messages[0].get("content") == tinytalk.TRIPLE_INSTRUCTIONS:
            return self.answers.get(messages[-1]["content"], "not json")
        return self.reply


class GraphMemory(object):
    """In-memory graph and drawers. search_facts stays empty unless told otherwise."""

    palace = object()

    def __init__(self):
        self.kg = self
        self.entities = [{"id": "e-user", "name": "user", "_rowid": 1}]
        self.triples = []
        self.drawers = {
            "facts": [],
            "facts-history": [],
            "facts-pending": [],
            "facts-abandoned": [],
        }
        self.fact_hits = []
        self.conversations = []
        self.fail_graph = False
        self.fail_drawers = False
        self.fail_one_drawer = None
        self._next_row = 1
        self.conversation_calls = []

    def add_edge(self, predicate, obj, drawer_id=None, valid_to=None):
        self._next_row += 1
        entity_id = "e-%s" % self._next_row
        self.entities.append({"id": entity_id, "name": obj, "_rowid": self._next_row})
        self._next_row += 1
        self.triples.append({
            "id": "t-%s" % self._next_row,
            "subject": "e-user",
            "predicate": predicate,
            "object": entity_id,
            "valid_from": None,
            "valid_to": valid_to,
            "source_drawer_id": drawer_id,
            "_rowid": self._next_row,
        })
        return self.triples[-1]["id"]

    def add_drawer(self, room, drawer_id, text, predicate=None, obj=None, unclassified=False):
        if unclassified:
            source = tinytalk._fact_source_file(unclassified=True)
        elif predicate:
            source = tinytalk._fact_source_file(predicate, obj)
        else:
            source = "tinytalk"
        self.drawers[room].append({
            "drawer_id": drawer_id,
            "content_preview": text,
            "metadata": {
                "source_file": source,
                "filed_at": "2026-10-05T12:00:00",
            },
        })

    def dump_rows(self, table, after_rowid=0, limit=1000):
        if self.fail_graph:
            raise RuntimeError("graph locked")
        rows = self.entities if table == "entities" else self.triples
        return [row for row in rows if row["_rowid"] > after_rowid][:limit]

    def list_drawers(self, wing, room, limit=100, offset=0):
        if self.fail_drawers:
            raise RuntimeError("drawers locked")
        found = self.drawers.get(room, [])
        return {"drawers": found[offset:offset + limit], "total": len(found)}

    def get_drawer(self, drawer_id):
        if self.fail_one_drawer and drawer_id == self.fail_one_drawer:
            raise RuntimeError("drawer unreadable")
        for room, drawers in self.drawers.items():
            for drawer in drawers:
                if drawer["drawer_id"] == drawer_id:
                    return {
                        "content": drawer["content_preview"],
                        "room": room,
                        "metadata": drawer["metadata"],
                    }
        return {"error": "Drawer not found: %s" % drawer_id}

    def search_facts(self, query):
        return list(self.fact_hits)

    def search_conversations(self, query):
        self.conversation_calls.append(query)
        return list(self.conversations)

    def unfinished_updates(self):
        return []

    def save_exchange(self, user_prompt, response, memory_note=None):
        self.saved = getattr(self, "saved", [])
        self.saved.append((user_prompt, response))


def _willow():
    memory = GraphMemory()
    memory.add_edge("preferred_name", "Willow", "drawer-willow")
    memory.add_drawer("facts", "drawer-willow", "call me Willow", "preferred_name", "Willow")
    memory.conversations = [{
        "text": "User: what is my name?\n\nAssistant: Your name is Tommi.",
        "drawer_id": "drawer-tommi",
        "similarity": 0.91,
    }]
    return memory


def _entry(memory, predicate):
    return next(
        entry for entry in tinytalk.read_current_profile(memory)
        if entry["predicate"] == predicate
    )


class StaleNameRegressionTests(unittest.TestCase):
    def test_a_semantic_miss_does_not_promote_an_old_assistant_name(self):
        memory = _willow()
        messages, sources = tinytalk.build_context("What is my name?", [], memory)
        body = "\n".join(message.get("content") or "" for message in messages)
        self.assertIn("Willow", body)
        self.assertNotIn("Tommi", body)
        self.assertNotIn("conversation excerpt", [source["kind"] for source in sources])
        self.assertEqual(memory.conversation_calls, [])
        current = next(
            source for source in sources
            if source.get("predicate") == "preferred_name"
        )
        self.assertEqual(current["status"], "current")
        self.assertEqual(current["selection"], "profile lookup")
        self.assertEqual(current["text"], "call me Willow")


class ProfileStateTests(unittest.TestCase):
    def test_a_supported_fact_is_current(self):
        entry = _entry(_willow(), "preferred_name")
        self.assertEqual(entry["state"], "current")
        self.assertEqual(entry["value"], "Willow")
        self.assertEqual(entry["source_drawer_id"], "drawer-willow")
        self.assertEqual(entry["source_text"], "call me Willow")
        self.assertIsNone(entry["pending_value"])

    def test_other_profile_fields_are_missing(self):
        entry = _entry(_willow(), "home_city")
        self.assertEqual(entry["state"], "missing")
        self.assertIsNone(entry["value"])

    def test_a_graph_only_edge_is_unresolved(self):
        memory = GraphMemory()
        memory.add_edge("preferred_name", "Willow", None)
        entry = _entry(memory, "preferred_name")
        self.assertEqual(entry["state"], "conflict")
        self.assertIsNone(entry["value"])
        self.assertIn("no fact drawer", entry["reason"])

    def test_a_missing_drawer_is_a_conflict(self):
        memory = GraphMemory()
        memory.add_edge("preferred_name", "Willow", "missing-drawer")
        entry = _entry(memory, "preferred_name")
        self.assertEqual(entry["state"], "conflict")
        self.assertIn("missing", entry["reason"])
        self.assertIsNone(entry["value"])

    def test_metadata_that_disagrees_is_a_conflict(self):
        memory = GraphMemory()
        memory.add_edge("preferred_name", "Willow", "drawer-willow")
        memory.add_drawer("facts", "drawer-willow", "call me Willow", "preferred_name", "Tommi")
        entry = _entry(memory, "preferred_name")
        self.assertEqual(entry["state"], "conflict")
        self.assertIn("metadata", entry["reason"])

    def test_unclassified_text_is_not_current(self):
        memory = GraphMemory()
        memory.add_edge("preferred_name", "Willow", "drawer-willow")
        memory.add_drawer("facts", "drawer-willow", "call me Willow", unclassified=True)
        entry = _entry(memory, "preferred_name")
        self.assertNotEqual(entry["state"], "current")

    def test_two_active_values_conflict(self):
        memory = _willow()
        memory.add_edge("preferred_name", "Tommi", "drawer-tommi")
        memory.add_drawer("facts", "drawer-tommi", "call me Tommi", "preferred_name", "Tommi")
        entry = _entry(memory, "preferred_name")
        self.assertEqual(entry["state"], "conflict")
        self.assertIn("Several active values", entry["reason"])
        self.assertIsNone(entry["value"])

    def test_a_pending_update_keeps_the_last_confirmed_value(self):
        memory = _willow()
        memory.drawers["facts-history"] = memory.drawers["facts"]
        memory.drawers["facts"] = []
        memory.add_drawer(
            "facts-pending", "drawer-pending", "call me River", "preferred_name", "River"
        )
        entry = _entry(memory, "preferred_name")
        self.assertEqual(entry["state"], "pending")
        self.assertEqual(entry["value"], "Willow")
        self.assertEqual(entry["pending_value"], "River")
        self.assertNotEqual(entry["pending_value"], entry["value"])

    def test_pending_without_a_last_value_names_no_current_value(self):
        memory = GraphMemory()
        memory.add_drawer(
            "facts-pending", "drawer-pending", "call me River", "preferred_name", "River"
        )
        entry = _entry(memory, "preferred_name")
        self.assertEqual(entry["state"], "pending")
        self.assertIsNone(entry["value"])
        self.assertEqual(entry["pending_value"], "River")

    def test_an_unreadable_graph_is_unavailable(self):
        memory = _willow()
        memory.fail_graph = True
        entry = _entry(memory, "preferred_name")
        self.assertEqual(entry["state"], "unavailable")
        self.assertIsNone(entry["value"])

    def test_a_memory_without_a_graph_is_unavailable(self):
        class Bare(object):
            palace = object()
            kg = None

        states = [entry["state"] for entry in tinytalk.read_current_profile(Bare())]
        self.assertEqual(states, ["unavailable"] * len(tinytalk.PROFILE_PREDICATES))

    def test_an_unreadable_linked_drawer_is_unavailable(self):
        memory = _willow()
        memory.fail_one_drawer = "drawer-willow"
        entry = _entry(memory, "preferred_name")
        self.assertEqual(entry["state"], "unavailable")


class DirectAnswerTests(unittest.TestCase):
    def _ask(self, memory, prompt, messages=None, provider=None):
        provider = provider or StubProvider()
        sources = tinytalk.SessionSources()
        shown = io.StringIO()
        with redirect_stdout(shown):
            updated = tinytalk.handle_turn(
                prompt, list(messages or []), provider, memory, "SOUL", sources
            )
        return provider, sources, shown.getvalue(), updated

    def test_three_name_questions_answer_without_the_model_or_embeddings(self):
        memory = _willow()
        memory.search_facts = lambda query: (_ for _ in ()).throw(RuntimeError("embeddings down"))
        for prompt in (
            "What is my name?",
            "  What   is my preferred name?  ",
            "What should you call me",
        ):
            provider, sources, shown, _updated = self._ask(memory, prompt)
            self.assertEqual(provider.calls, [])
            self.assertIn("Your saved preferred name is Willow.", shown)
            self.assertNotIn("Tommi", shown)
            self.assertEqual(sources.records[0]["selection"], "profile lookup")
            self.assertEqual(sources.records[0]["status"], "current")
            self.assertEqual(sources.records[0]["predicate"], "preferred_name")

    def test_other_wording_uses_the_model_and_still_includes_the_profile(self):
        memory = _willow()
        provider, _sources, _shown, _updated = self._ask(
            memory, "What was my spaceship called before Enterprise?"
        )
        # That wording is not a test-spaceship history question and not a
        # direct name question. The model sees the profile.
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("call me Willow", provider.calls[0][1]["content"])
        self.assertNotIn("Tommi", provider.calls[0][1]["content"])

    def test_a_missing_name_abstains(self):
        _provider, _sources, shown, _updated = self._ask(GraphMemory(), "What is my name?")
        self.assertIn("No confirmed saved preferred name.", shown)
        self.assertNotIn("Tommi", shown)

    def test_an_unsaved_session_name_is_not_stored(self):
        memory = _willow()
        provider, _sources, shown, updated = self._ask(memory, "call me River")
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("not a saved fact", provider.calls[0][1]["content"])
        self.assertEqual(_entry(memory, "preferred_name")["value"], "Willow")
        _provider, sources, shown, _updated = self._ask(
            memory, "What should you call me?", updated
        )
        self.assertIn("Your saved preferred name is Willow.", shown)
        self.assertIn("asked to be called River", shown)
        self.assertIn("not a saved fact", shown)
        self.assertEqual(_entry(memory, "preferred_name")["value"], "Willow")
        self.assertTrue(all(source.get("status") != "current" or "River" not in source["text"]
                            for source in sources.records))

    def test_a_coworker_line_is_not_a_session_name(self):
        memory = _willow()
        provider, _sources, _shown, updated = self._ask(
            memory, "my coworker asked you to call me Tommi"
        )
        self.assertNotIn("not a saved fact", provider.calls[0][1]["content"])
        _provider, _sources, shown, _updated = self._ask(memory, "What is my name?", updated)
        self.assertNotIn("Tommi", shown)

    def test_both_providers_see_the_same_profile(self):
        memory = _willow()
        question = "How should we start the day?"
        first = StubProvider(reply="One.")
        second = StubProvider(reply="Two.")
        self._ask(memory, question, provider=first)
        self._ask(memory, question, provider=second)
        self.assertEqual(first.calls[0][1]["content"], second.calls[0][1]["content"])
        self.assertIn("Willow", first.calls[0][1]["content"])


class ProfileBudgetTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("TINYTALK_PROFILE_BUDGET_CHARS", None)
        os.environ.pop("TINYTALK_MAX_REQUEST_CHARS", None)

    def test_an_entry_that_does_not_fit_is_omitted_whole(self):
        memory = _willow()
        memory.drawers["facts"][0]["content_preview"] = "call me " + ("Willow " * 80)
        os.environ["TINYTALK_PROFILE_BUDGET_CHARS"] = "120"
        messages, sources = tinytalk.build_context("How are you?", [], memory)
        body = messages[0]["content"]
        self.assertNotIn("Willow Willow", body)
        self.assertIn("did not fit", body)
        self.assertTrue(any(source.get("status") == "unavailable" for source in sources))

    def test_required_profile_survives_optional_trimming(self):
        memory = _willow()
        memory.fact_hits = [
            {"text": "optional-%s %s" % (index, "z" * 400), "similarity": 0.9}
            for index in range(6)
        ]
        os.environ["TINYTALK_MAX_REQUEST_CHARS"] = "4000"
        provider = StubProvider(reply="partial")
        shown = io.StringIO()
        with redirect_stdout(shown):
            tinytalk.handle_turn("How are you?", [], provider, memory, "SOUL")
        self.assertEqual(len(provider.calls), 1)
        body = provider.calls[0][1]["content"]
        self.assertIn("call me Willow", body)
        self.assertIn("retrieved records were omitted", body)

    def test_a_required_profile_that_cannot_fit_is_not_sent(self):
        memory = _willow()
        memory.drawers["facts"][0]["content_preview"] = "W" * 9000
        os.environ["TINYTALK_PROFILE_BUDGET_CHARS"] = "20000"
        os.environ["TINYTALK_MAX_REQUEST_CHARS"] = "8000"
        provider = StubProvider(reply="nope")
        sources = tinytalk.SessionSources()
        sources.replace([{"kind": "saved fact", "text": "keep me"}])
        shown = io.StringIO()
        with redirect_stdout(shown):
            updated = tinytalk.handle_turn(
                "How are you?", [], provider, memory, "SOUL", sources
            )
        self.assertEqual(provider.calls, [])
        self.assertEqual(updated, [])
        self.assertEqual(sources.records, [{"kind": "saved fact", "text": "keep me"}])
        self.assertIn("Nothing was saved.", shown.getvalue())


class RecallCommandTests(unittest.TestCase):
    def test_memory_and_bare_recall_do_not_call_the_model_or_save(self):
        memory = _willow()
        provider = StubProvider()
        sources = tinytalk.SessionSources()
        sources.replace([{"kind": "saved fact", "text": "keep me", "id": "keep"}])
        shown = io.StringIO()
        with redirect_stdout(shown):
            updated = tinytalk.handle_user_line(
                "/memory", [], provider, memory, "SOUL", None, sources
            )
            updated = tinytalk.handle_user_line(
                "/recall", updated, provider, memory, "SOUL", None, sources
            )
        self.assertEqual(provider.calls, [])
        self.assertEqual(updated, [])
        self.assertFalse(hasattr(memory, "saved"))
        self.assertEqual(sources.records[0]["id"], "keep")
        text = shown.getvalue()
        self.assertIn("preferred_name: current", text)
        self.assertIn("value: Willow", text)
        self.assertIn("drawer: drawer-willow", text)
        self.assertIn("Use /recall followed by", text)

    def test_recall_labels_an_old_assistant_line_as_historical(self):
        memory = _willow()
        provider = StubProvider(reply="From the chat.")
        sources = tinytalk.SessionSources()
        shown = io.StringIO()
        with redirect_stdout(shown):
            tinytalk.handle_user_line(
                "/recall what did we say about my name",
                [], provider, memory, "SOUL", None, sources,
            )
        body = provider.calls[0][1]["content"]
        self.assertIn("historical", body)
        self.assertIn("Tommi", body)
        self.assertIn("explicit history recall", body)
        excerpt = next(
            source for source in sources.records
            if source["kind"] == "conversation excerpt"
        )
        self.assertEqual(excerpt["status"], "historical")
        self.assertIn("Assistant: Your name is Tommi.", excerpt["text"])

    def test_opaque_history_cannot_become_the_current_name(self):
        memory = _willow()
        memory.conversations = [{
            "text": "the assistant once guessed Tommi",
            "similarity": 0.95,
            "drawer_id": "legacy",
        }]
        _messages, sources = tinytalk.recall_context("name", [], memory)
        opaque = next(source for source in sources if source["kind"] == "conversation excerpt")
        self.assertIn("Opaque historical text", opaque["text"])
        self.assertEqual(_entry(memory, "preferred_name")["value"], "Willow")
        self.assertNotEqual(opaque["status"], "current")


class RestartProfileTests(unittest.TestCase):
    def setUp(self):
        self._temp = memory_isolation.temp_dir("tinytalk-profile-")
        self.memory = memory_isolation.open_isolated_memory(self._temp.name)

    def tearDown(self):
        self.memory.kg.close()
        self._temp.cleanup()

    def test_a_saved_name_survives_restart_when_embeddings_miss(self):
        provider = StubProvider({
            "call me Willow": '{"subject":"user","predicate":"preferred_name","object":"Willow"}',
        })
        shown = io.StringIO()
        with redirect_stdout(shown):
            saved = tinytalk.remember_fact("Remember this: call me Willow", self.memory, provider)
        self.assertEqual(saved["status"], "saved")
        palace_path = self.memory.palace_path
        kg_path = self.memory.kg.db_path
        self.memory.kg.close()
        reopened = tinytalk.open_memory(palace_path=palace_path, kg_path=kg_path)
        self.memory = reopened

        def embeddings_down(query):
            raise RuntimeError("embeddings down")

        reopened.search_facts = embeddings_down
        asker = StubProvider(reply="should not be used")
        out = io.StringIO()
        with redirect_stdout(out):
            for prompt in (
                "What is my name?",
                "What is my preferred name?",
                "What should you call me?",
            ):
                tinytalk.handle_turn(prompt, [], asker, reopened, "SOUL")
        self.assertEqual(asker.calls, [])
        self.assertEqual(out.getvalue().count("Your saved preferred name is Willow."), 3)
        self.assertNotIn("Tommi", out.getvalue())


if __name__ == "__main__":
    unittest.main()
