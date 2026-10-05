"""Fact replacement against real, temporary MemPalace stores.

Failures are injected by replacing one store method on a real store.
The model is a stub that answers by the text it is given. Each failure test
checks the stored rooms and graph, and the context the next answer would get.
"""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_isolation  # noqa: E402  (must come before tinytalk and mempalace)
import tinytalk
from provider import ProviderError

SHIP = "test_spaceship_name"


def triple(predicate, obj, subject="user"):
    return json.dumps({"subject": subject, "predicate": predicate, "object": obj})


class TextProvider(object):
    """Extraction stub keyed by the fact text. Chat replies use `reply`."""

    name = "stub"
    label = "Stub"

    def __init__(self, answers=None, reply="Okay."):
        self.answers = dict(answers or {})
        self.reply = reply
        self.extractions = []

    def complete(self, messages):
        text = messages[-1]["content"]
        if messages[0]["content"] != tinytalk.TRIPLE_INSTRUCTIONS:
            return self.reply
        self.extractions.append(text)
        answer = self.answers.get(text, "not json")
        if isinstance(answer, Exception):
            raise answer
        return answer


def remember(memory, provider, fact):
    out = io.StringIO()
    with redirect_stdout(out):
        result = tinytalk.remember_fact("Remember this: " + fact, memory, provider)
    return result


def room_texts(memory, room):
    return sorted(d["text"] for d in tinytalk._room_drawers(memory, room))


def ship_rows(memory, predicate=SHIP):
    return [row for row in tinytalk._graph_rows(memory.kg, "triples") if row["predicate"] == predicate]


def context(memory, question):
    """The memory message and sources the next answer would get."""
    messages, sources = tinytalk.build_context(
        question, [{"role": "user", "content": question}], memory
    )
    first = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    return first, sources


def kinds(sources, kind):
    return [source["text"] for source in sources if source["kind"] == kind]


UNFINISHED = tinytalk.UNFINISHED_KIND
SHIP_QUESTION = "What is my test spaceship called?"
PREVIOUS_QUESTION = "What was the previous name of my test spaceship?"


class FactReplacementTests(unittest.TestCase):
    def setUp(self):
        self._temp = memory_isolation.temp_dir("tinytalk-replace-")
        self.memory = memory_isolation.open_isolated_memory(self._temp.name)

    def tearDown(self):
        self.memory.kg.close()
        self._temp.cleanup()

    def provider(self, **extra):
        answers = {
            "call me Al": triple("preferred_name", "Al"),
            "call me Sam": triple("preferred_name", "Sam"),
            "my favorite meal is ramen": triple("favorite_meal", "ramen"),
            "I grew up in Dallas": triple("childhood_city", "Dallas"),
            "I live in Nashville.": triple("home_city", "Nashville"),
            "my test spaceship is named Serenity": triple(SHIP, "Serenity"),
            "my test spaceship is named Enterprise": triple(SHIP, "Enterprise"),
            "my test spaceship is named Voyager": triple(SHIP, "Voyager"),
            "my imaginary spaceship is named Serenity Two": triple(
                "imaginary_spaceship_name", "Serenity Two"
            ),
        }
        answers.update(extra)
        return TextProvider(answers)

    def seed_serenity(self):
        provider = self.provider()
        self.assertEqual(remember(self.memory, provider, "my test spaceship is named Serenity")["status"], "saved")
        return provider

    def legacy_fact(self, text):
        self.assertTrue(self.memory.add_drawer(
            wing="tinytalk", room="facts", content=text, source_file="tinytalk", added_by="tinytalk",
        )["success"])

    def turn(self, provider, prompt):
        out = io.StringIO()
        with redirect_stdout(out):
            tinytalk.handle_turn(prompt, [], provider, self.memory, "SOUL")
        return out.getvalue().splitlines()

    def rooms(self):
        return {
            room: room_texts(self.memory, room)
            for room in ("facts", "facts-history", "facts-pending", "facts-abandoned")
        }

    def assert_serenity_is_still_current(self, enterprise_note):
        """Recall, introspection, and the history matcher after an unfinished Enterprise update."""
        body, sources = context(self.memory, SHIP_QUESTION)
        self.assertNotIn("my test spaceship is named Enterprise", kinds(sources, "saved fact"))
        unfinished = kinds(sources, UNFINISHED)
        self.assertEqual(len(unfinished), 1)
        self.assertIn('Not a current fact: "my test spaceship is named Enterprise"', unfinished[0])
        self.assertIn(enterprise_note, unfinished[0])
        self.assertIn("is not a current fact", body)
        for excerpt in kinds(sources, "conversation excerpt"):
            self.assertNotIn("Enterprise", excerpt)

        body, sources = context(self.memory, "What do you remember about me?")
        self.assertNotIn("my test spaceship is named Enterprise", kinds(sources, "saved fact"))
        self.assertEqual(len(kinds(sources, UNFINISHED)), 1)

        body, sources = context(self.memory, PREVIOUS_QUESTION)
        current = "Current name: Serenity" if "Serenity" in enterprise_note else "Current name: not established"
        self.assertIn(current, body)
        self.assertNotIn("Current name: Enterprise", body)
        self.assertNotIn("Previous name: Enterprise", body)
        self.assertIn("Unfinished update, not the current name", body)
        self.assertEqual(len(kinds(sources, UNFINISHED)), 1)

    def assert_finished(self, current):
        self.assertEqual(self.memory.current_fact_objects("user", SHIP), [current])
        self.assertEqual(room_texts(self.memory, "facts-pending"), [])
        body, sources = context(self.memory, SHIP_QUESTION)
        self.assertEqual(kinds(sources, UNFINISHED), [])
        self.assertNotIn("unfinished", body)

    # -- exact-record replacement ---------------------------------------

    def test_replacement_archives_only_the_replaced_fact(self):
        provider = self.provider()
        for fact in (
            "call me Al",
            "my favorite meal is ramen",
            "I grew up in Dallas",
            "my test spaceship is named Serenity",
            "my imaginary spaceship is named Serenity Two",
        ):
            self.assertEqual(remember(self.memory, provider, fact)["status"], "saved")

        sam = remember(self.memory, provider, "call me Sam")
        ship = remember(self.memory, provider, "my test spaceship is named Enterprise")

        self.assertEqual(sam, {"status": "replaced", "message": "Memory: saved. Replaced Al."})
        self.assertEqual(ship, {"status": "replaced", "message": "Memory: saved. Replaced Serenity."})
        self.assertEqual(self.rooms(), {
            "facts": sorted([
                "I grew up in Dallas",
                "call me Sam",
                "my favorite meal is ramen",
                "my imaginary spaceship is named Serenity Two",
                "my test spaceship is named Enterprise",
            ]),
            "facts-history": sorted(["call me Al", "my test spaceship is named Serenity"]),
            "facts-pending": [],
            "facts-abandoned": [],
        })
        self.assertEqual(
            self.memory.current_fact_objects("user", "imaginary_spaceship_name"), ["Serenity Two"]
        )
        self.assert_finished("Enterprise")
        body, sources = context(self.memory, PREVIOUS_QUESTION)
        self.assertIn("Current name: Enterprise", body)
        self.assertIn("Previous name: Serenity", body)

    # -- records written before drawer links ----------------------------

    def test_legacy_test_spaceship_edge_uses_the_parser_not_the_model(self):
        for fact in (
            "my test spaceship is named Serenity",
            "my imaginary spaceship is named Serenity Two",
            "call me Al",
        ):
            self.legacy_fact(fact)
        self.memory.kg.add_triple("user", SHIP, "Serenity", source_file="tinytalk")
        provider = self.provider()

        result = remember(self.memory, provider, "my test spaceship is named Enterprise")

        self.assertEqual(result["status"], "replaced")
        self.assertEqual(provider.extractions, ["my test spaceship is named Enterprise"])
        self.assertEqual(room_texts(self.memory, "facts-history"), ["my test spaceship is named Serenity"])
        self.assertIn("my imaginary spaceship is named Serenity Two", room_texts(self.memory, "facts"))
        self.assert_finished("Enterprise")

    def test_unlinked_legacy_value_is_not_replaced(self):
        # The graph spells the value differently from the only saved fact.
        self.legacy_fact("I live in NYC.")
        self.legacy_fact("my favorite meal is ramen")
        self.memory.kg.add_triple("user", "home_city", "New York City", source_file="tinytalk")
        provider = self.provider()

        for _ in range(2):
            result = remember(self.memory, provider, "I live in Nashville.")
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["message"].startswith("Memory: saved as written, but not as a current fact."))
            self.assertIn("New York City is not linked to one saved fact", result["message"])
        self.assertEqual(provider.extractions, ["I live in Nashville.", "I live in Nashville."])
        self.assertEqual(self.rooms(), {
            "facts": ["I live in NYC.", "my favorite meal is ramen"],
            "facts-history": [],
            "facts-pending": ["I live in Nashville."],
            "facts-abandoned": [],
        })
        self.assertEqual(self.memory.current_fact_objects("user", "home_city"), ["New York City"])

        body, sources = context(self.memory, "What do you remember about me?")
        self.assertEqual(sorted(kinds(sources, "saved fact")), ["I live in NYC.", "my favorite meal is ramen"])
        unfinished = kinds(sources, UNFINISHED)
        self.assertEqual(len(unfinished), 1)
        self.assertIn('Not a current fact: "I live in Nashville."', unfinished[0])
        self.assertIn("The last completed home_city is New York City.", unfinished[0])

    def test_legacy_parser_disagreement_or_ambiguity_is_not_guessed(self):
        cases = (
            # The graph value and the parsed fact disagree. The model is not asked.
            (["my test spaceship is named Serenity"], "USS Serenity"),
            # Two current test-spaceship facts: no single drawer holds the value.
            (["my test spaceship is named Serenity", "my test spaceship is named Serenity, I think"],
             "Serenity"),
            # The value only appears in a fact the parser does not read.
            (["Serenity is what I call my test spaceship"], "Serenity"),
        )
        for facts, value in cases:
            with memory_isolation.temp_dir("tinytalk-legacy-") as temp:
                memory = memory_isolation.open_isolated_memory(temp)
                for fact in facts:
                    memory.add_drawer(wing="tinytalk", room="facts", content=fact,
                                      source_file="tinytalk", added_by="tinytalk")
                memory.kg.add_triple("user", SHIP, value, source_file="tinytalk")
                provider = self.provider()

                result = remember(memory, provider, "my test spaceship is named Enterprise")

                self.assertEqual(result["status"], "incomplete", facts)
                self.assertIn("is not linked to one saved fact", result["message"])
                self.assertEqual(provider.extractions, ["my test spaceship is named Enterprise"])
                self.assertEqual(room_texts(memory, "facts"), sorted(facts))
                self.assertEqual(room_texts(memory, "facts-history"), [])
                self.assertEqual(room_texts(memory, "facts-pending"), ["my test spaceship is named Enterprise"])
                self.assertEqual(memory.current_fact_objects("user", SHIP), [value])
                body, sources = context(memory, PREVIOUS_QUESTION)
                self.assertNotIn("Current name: Enterprise", body)
                memory.kg.close()

    # -- failures, recall while unfinished, and retries -----------------

    def test_failed_drawer_save_changes_nothing_and_retry_finishes(self):
        provider = self.seed_serenity()
        real_add = self.memory.add_drawer
        self.memory.add_drawer = lambda **kwargs: {"success": False, "error": "disk full"}

        failed = remember(self.memory, provider, "my test spaceship is named Enterprise")

        self.assertEqual(failed, {
            "status": "not_saved",
            "message": "Memory: not saved. MemPalace could not store this fact.",
        })

        def boom(**kwargs):
            raise RuntimeError("palace locked")

        self.memory.add_drawer = boom
        self.assertEqual(
            remember(self.memory, provider, "my test spaceship is named Enterprise")["status"],
            "not_saved",
        )
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Serenity"])
        self.assertEqual(self.rooms()["facts-pending"], [])
        self.assertEqual(self.memory.current_fact_objects("user", SHIP), ["Serenity"])

        self.memory.add_drawer = real_add
        retry = remember(self.memory, provider, "my test spaceship is named Enterprise")
        self.assertEqual(retry["status"], "replaced")
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Enterprise"])
        self.assertEqual(self.rooms()["facts-history"], ["my test spaceship is named Serenity"])
        self.assert_finished("Enterprise")

    def test_graph_read_failure_keeps_the_last_value_current(self):
        provider = self.seed_serenity()
        real_lookup = self.memory.current_links
        added = []

        def unreadable(subject, predicate):
            raise RuntimeError("database is locked")

        self.memory.current_links = unreadable
        real_add_triple = self.memory.kg.add_triple
        self.memory.kg.add_triple = lambda *a, **k: added.append(a) or real_add_triple(*a, **k)

        result = remember(self.memory, provider, "my test spaceship is named Enterprise")

        self.assertEqual(result["status"], "incomplete")
        self.assertIn("could not be read, so nothing was replaced", result["message"])
        self.assertEqual(added, [])
        self.memory.current_links = real_lookup
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Serenity"])
        self.assertEqual(self.rooms()["facts-pending"], ["my test spaceship is named Enterprise"])
        self.assertEqual(self.memory.current_fact_objects("user", SHIP), ["Serenity"])
        self.assert_serenity_is_still_current("The last completed test_spaceship_name is Serenity.")

        retry = remember(self.memory, provider, "my test spaceship is named Enterprise")
        self.assertEqual(retry["status"], "replaced")
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Enterprise"])
        self.assert_finished("Enterprise")

    def test_archive_failure_keeps_the_last_value_current(self):
        provider = self.seed_serenity()
        self.memory.save_exchange("hello", "Hi there.")
        real_update = self.memory.update_drawer
        self.memory.update_drawer = lambda drawer_id, **kwargs: {"success": False, "error": "locked"}

        lines = self.turn(provider, "Remember this: my test spaceship is named Enterprise")

        self.assertIn("Memory: saved as written, but not as a current fact. "
                      "The earlier fact for Serenity could not be archived. "
                      "Repeat the command to finish.", lines)
        self.memory.update_drawer = real_update
        self.assertEqual(self.rooms(), {
            "facts": ["my test spaceship is named Serenity"],
            "facts-history": [],
            "facts-pending": ["my test spaceship is named Enterprise"],
            "facts-abandoned": [],
        })
        self.assertEqual(self.memory.current_fact_objects("user", SHIP), ["Serenity"])
        self.assert_serenity_is_still_current("The last completed test_spaceship_name is Serenity.")

        retry = remember(self.memory, provider, "my test spaceship is named Enterprise")
        self.assertEqual(retry["status"], "replaced")
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Enterprise"])
        self.assertEqual(self.rooms()["facts-history"], ["my test spaceship is named Serenity"])
        self.assert_finished("Enterprise")

    def test_supersede_failure_does_not_present_the_new_value(self):
        provider = self.seed_serenity()
        self.memory.save_exchange("hello", "Hi there.")
        kg = self.memory.kg
        real_supersede = kg.supersede
        added = []

        def broken(*args, **kwargs):
            raise RuntimeError("database is locked")

        kg.supersede = broken
        real_add_triple = kg.add_triple
        kg.add_triple = lambda *a, **k: added.append(a) or real_add_triple(*a, **k)

        lines = self.turn(provider, "Remember this: my test spaceship is named Enterprise")

        self.assertIn("Memory: saved as written, but not as a current fact. "
                      "The knowledge graph still lists Serenity. Repeat the command to finish.", lines)
        self.assertEqual(added, [])
        self.assertEqual(self.rooms(), {
            "facts": [],
            "facts-history": ["my test spaceship is named Serenity"],
            "facts-pending": ["my test spaceship is named Enterprise"],
            "facts-abandoned": [],
        })
        self.assertEqual(self.memory.current_fact_objects("user", SHIP), ["Serenity"])
        # No current fact matches now, so recall falls back to conversations.
        # The stored "Remember this:" turn for Enterprise is left out.
        body, sources = context(self.memory, SHIP_QUESTION)
        self.assertTrue(any("hello" in text for text in kinds(sources, "conversation excerpt")))
        self.assert_serenity_is_still_current("The last completed test_spaceship_name is Serenity.")

        kg.supersede = real_supersede
        retry = remember(self.memory, provider, "my test spaceship is named Enterprise")
        self.assertEqual(retry["status"], "replaced")
        self.assertEqual(added, [])
        self.assertEqual(len([row for row in ship_rows(self.memory) if row["valid_to"] is None]), 1)
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Enterprise"])
        self.assertEqual(self.rooms()["facts-history"], ["my test spaceship is named Serenity"])
        self.assert_finished("Enterprise")

    def test_failed_move_into_facts_leaves_the_name_unsettled(self):
        provider = self.seed_serenity()
        real_update = self.memory.update_drawer

        def no_promotion(drawer_id, **kwargs):
            if kwargs.get("room") == "facts":
                return {"success": False, "error": "locked"}
            return real_update(drawer_id, **kwargs)

        self.memory.update_drawer = no_promotion
        result = remember(self.memory, provider, "my test spaceship is named Enterprise")
        self.memory.update_drawer = real_update

        self.assertEqual(result["status"], "incomplete")
        self.assertIn("could not be moved into current facts", result["message"])
        # The graph moved on, but its fact did not: neither name is settled.
        self.assertEqual(self.memory.current_fact_objects("user", SHIP), ["Enterprise"])
        self.assertEqual(self.rooms()["facts"], [])
        self.assertEqual(self.rooms()["facts-pending"], ["my test spaceship is named Enterprise"])
        self.assert_serenity_is_still_current(
            "The saved records do not establish the current test_spaceship_name."
        )

        retry = remember(self.memory, provider, "my test spaceship is named Enterprise")
        self.assertEqual(retry, {"status": "saved", "message": "Memory: saved."})
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Enterprise"])
        self.assert_finished("Enterprise")

    # -- a different command after an unfinished one --------------------

    def test_failed_enterprise_then_voyager_leaves_only_voyager_current(self):
        provider = self.seed_serenity()
        real_update = self.memory.update_drawer
        self.memory.update_drawer = lambda drawer_id, **kwargs: {"success": False, "error": "locked"}
        self.assertEqual(
            remember(self.memory, provider, "my test spaceship is named Enterprise")["status"],
            "incomplete",
        )
        self.memory.update_drawer = real_update

        voyager = remember(self.memory, provider, "my test spaceship is named Voyager")

        self.assertEqual(voyager, {"status": "replaced", "message": "Memory: saved. Replaced Serenity."})
        self.assertEqual(self.rooms(), {
            "facts": ["my test spaceship is named Voyager"],
            "facts-history": ["my test spaceship is named Serenity"],
            "facts-pending": [],
            "facts-abandoned": ["my test spaceship is named Enterprise"],
        })
        self.assert_finished("Voyager")
        body, sources = context(self.memory, PREVIOUS_QUESTION)
        self.assertIn("Current name: Voyager", body)
        self.assertIn("Previous name: Serenity", body)
        self.assertNotIn("Enterprise", body)
        body, sources = context(self.memory, "What do you remember about me?")
        self.assertEqual(kinds(sources, "saved fact"), ["my test spaceship is named Voyager"])

        # Repeating Voyager does not bring the abandoned Enterprise back.
        self.assertEqual(
            remember(self.memory, provider, "my test spaceship is named Voyager"),
            {"status": "already_saved", "message": "Memory: already saved."},
        )
        self.assertEqual(self.rooms()["facts-abandoned"], ["my test spaceship is named Enterprise"])

        # Repeating the abandoned Enterprise command is a new replacement of Voyager.
        again = remember(self.memory, provider, "my test spaceship is named Enterprise")
        self.assertEqual(again, {"status": "replaced", "message": "Memory: saved. Replaced Voyager."})
        self.assertEqual(self.rooms(), {
            "facts": ["my test spaceship is named Enterprise"],
            "facts-history": sorted([
                "my test spaceship is named Serenity",
                "my test spaceship is named Voyager",
            ]),
            "facts-pending": [],
            "facts-abandoned": [],
        })
        self.assert_finished("Enterprise")

    def test_an_unfinished_update_that_cannot_be_cleared_is_reported(self):
        provider = self.seed_serenity()
        real_update = self.memory.update_drawer
        self.memory.update_drawer = lambda drawer_id, **kwargs: {"success": False, "error": "locked"}
        remember(self.memory, provider, "my test spaceship is named Enterprise")

        def no_abandon(drawer_id, **kwargs):
            if kwargs.get("room") == "facts-abandoned":
                return {"success": False, "error": "locked"}
            return real_update(drawer_id, **kwargs)

        self.memory.update_drawer = no_abandon
        result = remember(self.memory, provider, "my test spaceship is named Voyager")
        self.assertEqual(result, {
            "status": "incomplete",
            "message": "Memory: saved, but the unfinished update to Enterprise could not be cleared. "
                       "Repeat the command to finish.",
        })
        self.assertEqual(self.rooms()["facts-pending"], ["my test spaceship is named Enterprise"])
        body, sources = context(self.memory, SHIP_QUESTION)
        self.assertEqual(len(kinds(sources, UNFINISHED)), 1)

        self.memory.update_drawer = real_update
        retry = remember(self.memory, provider, "my test spaceship is named Voyager")
        self.assertEqual(retry, {"status": "saved", "message": "Memory: saved."})
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Voyager"])
        self.assertEqual(self.rooms()["facts-abandoned"], ["my test spaceship is named Enterprise"])
        self.assert_finished("Voyager")

    # -- repeats --------------------------------------------------------

    def test_retry_repairs_a_graph_only_value(self):
        # The old code could leave a current edge with no saved fact behind it.
        self.memory.kg.add_triple("user", SHIP, "Enterprise", source_file="tinytalk")
        provider = self.provider()

        repaired = remember(self.memory, provider, "my test spaceship is named Enterprise")

        self.assertEqual(repaired, {"status": "saved", "message": "Memory: saved."})
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Enterprise"])
        self.assertEqual(len(ship_rows(self.memory)), 1)

        again = remember(self.memory, provider, "my test spaceship is named Enterprise")
        self.assertEqual(again["status"], "already_saved")
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Enterprise"])
        self.assertEqual(len(ship_rows(self.memory)), 1)
        self.assert_finished("Enterprise")

    def test_repeated_success_does_not_duplicate(self):
        provider = self.provider()
        for _ in range(3):
            remember(self.memory, provider, "my test spaceship is named Enterprise")
            remember(self.memory, provider, "my favorite meal is ramen")

        self.assertEqual(
            remember(self.memory, provider, "my test spaceship is named Enterprise"),
            {"status": "already_saved", "message": "Memory: already saved."},
        )
        self.assertEqual(
            remember(self.memory, provider, "my favorite meal is ramen"),
            {"status": "already_saved", "message": "Memory: already saved."},
        )
        self.assertEqual(self.rooms()["facts"], sorted([
            "my favorite meal is ramen",
            "my test spaceship is named Enterprise",
        ]))
        self.assertEqual(self.rooms()["facts-pending"], [])
        self.assertEqual(len(ship_rows(self.memory)), 1)
        self.assertEqual(len(ship_rows(self.memory, "favorite_meal")), 1)

    def test_restating_an_archived_value_makes_it_current_again(self):
        provider = self.seed_serenity()
        remember(self.memory, provider, "my test spaceship is named Enterprise")

        back = remember(self.memory, provider, "my test spaceship is named Serenity")

        self.assertEqual(back["status"], "replaced")
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Serenity"])
        self.assertEqual(self.rooms()["facts-history"], ["my test spaceship is named Enterprise"])
        self.assert_finished("Serenity")

    # -- extraction -----------------------------------------------------

    def test_unclear_extraction_saves_the_words_and_replaces_nothing(self):
        provider = self.seed_serenity()
        for fact, answer in (
            ("my test spaceship is named Defiant, maybe", "not json"),
            ("my test spaceship is named Orville", ProviderError("xAI returned an empty response.")),
        ):
            provider.answers[fact] = answer
            result = remember(self.memory, provider, fact)
            self.assertEqual(result["status"], "saved_as_written", fact)
            self.assertIn("No clear fact could be extracted", result["message"])
            self.assertIn("nothing was replaced", result["message"])
            self.assertIn(fact, self.rooms()["facts"])
        self.assertEqual(self.memory.current_fact_objects("user", SHIP), ["Serenity"])
        self.assertEqual(self.rooms()["facts-history"], [])

    def test_mismatched_extraction_is_kept_out_of_current_facts(self):
        provider = self.seed_serenity()
        for fact, answer in (
            ("my test spaceship is named Voyager", triple(SHIP, "Nostromo")),
            ("my test spaceship is named Rocinante", triple(SHIP, "Rocinante", subject="my ship")),
        ):
            provider.answers[fact] = answer
            result = remember(self.memory, provider, fact)
            self.assertEqual(result["status"], "incomplete", fact)
            self.assertIn("did not clearly match the text, so nothing was replaced", result["message"])
        self.assertEqual(self.rooms()["facts"], ["my test spaceship is named Serenity"])
        self.assertEqual(self.rooms()["facts-pending"], sorted([
            "my test spaceship is named Rocinante",
            "my test spaceship is named Voyager",
        ]))
        self.assertEqual(self.memory.current_fact_objects("user", SHIP), ["Serenity"])
        body, sources = context(self.memory, PREVIOUS_QUESTION)
        self.assertIn("Current name: Serenity", body)
        self.assertEqual(len(kinds(sources, UNFINISHED)), 2)

    # -- chat status line -----------------------------------------------

    def test_chat_prints_the_status_after_the_reply(self):
        provider = self.provider()
        provider.reply = "Got it, I'll remember that."

        lines = self.turn(provider, "Remember this: my test spaceship is named Serenity")
        self.assertEqual(lines[:2], ["Stub: Got it, I'll remember that.", "Memory: saved."])

        class Unavailable(object):
            palace = None
            kg = None

        out = io.StringIO()
        with redirect_stdout(out):
            tinytalk.handle_turn(
                "Remember this: my test spaceship is named Enterprise", [], provider, Unavailable(), "SOUL"
            )
        self.assertEqual(out.getvalue().splitlines()[:2], [
            "Stub: Got it, I'll remember that.",
            "Memory: not saved. Saved memory is unavailable in this session.",
        ])

        real_add = self.memory.add_drawer
        self.memory.add_drawer = lambda **kwargs: {"success": False}
        lines = self.turn(provider, "Remember this: my test spaceship is named Enterprise")
        self.assertEqual(lines[1], "Memory: not saved. MemPalace could not store this fact.")
        self.memory.add_drawer = real_add

        lines = self.turn(provider, "What is my test spaceship called?")
        self.assertFalse(any(line.startswith("Memory:") for line in lines))
        lines = self.turn(provider, "Remember this:   ")
        self.assertFalse(any(line.startswith("Memory:") for line in lines))


class IsolationTests(unittest.TestCase):
    def test_paths_resolve_inside_the_test_root(self):
        memory_isolation.check_default_paths()
        with memory_isolation.temp_dir("tinytalk-iso-") as temp:
            memory = memory_isolation.open_isolated_memory(temp)
            self.assertTrue(memory_isolation.inside_root(memory.palace_path))
            self.assertTrue(memory_isolation.inside_root(memory.kg.db_path))
            memory.kg.close()

    def test_a_store_outside_the_root_is_refused(self):
        with self.assertRaises(RuntimeError):
            memory_isolation.open_isolated_memory(str(Path.home().parent.parent / "elsewhere"))


if __name__ == "__main__":
    unittest.main()
