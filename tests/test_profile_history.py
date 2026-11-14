"""Profile history through real temporary palaces and the normal chat path."""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_isolation  # Must precede MemPalace imports.
import memory_core as core
import tinytalk


CASES = {
    "preferred_name": ("River", "Willow", "my preferred name is %s", "preferred name"),
    "home_city": ("Austin", "Nashville", "my home city is %s", "home city"),
    "current_job": ("teacher", "engineer", "my current job is %s", "job"),
    "favorite_vehicle": ("bicycle", "CyberTruck", "my favorite vehicle is %s", "favorite vehicle"),
}


class HistoryProvider:
    """Extract known test facts and echo the history context supplied to chat."""

    name = "test"
    label = "Test"

    def complete(self, messages):
        if messages[0]["content"] == core.TRIPLE_INSTRUCTIONS:
            fact = messages[-1]["content"]
            for predicate, (old, new, wording, _label) in CASES.items():
                for value in (old, new):
                    if fact == wording % value:
                        return json.dumps({"subject": "user", "predicate": predicate, "object": value})
            raise AssertionError("Unexpected extraction input")
        return next(
            message["content"] for message in messages
            if message["content"].startswith("This context is the saved ")
        )


def edge(value, current, end=None, start=None, predicate="home_city"):
    return {
        "subject": "user", "predicate": predicate, "object": value,
        "current": current, "valid_from": start, "valid_to": end,
    }


class ProfileHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = memory_isolation.temp_dir("tinytalk-profile-history-")
        self.memory = memory_isolation.open_isolated_memory(self.temp.name)
        self.provider = HistoryProvider()

    def tearDown(self):
        self.memory.kg.close()
        self.temp.cleanup()

    def save(self, fact):
        with redirect_stdout(io.StringIO()):
            return core.remember_fact("Remember this: " + fact, self.memory, self.provider)

    def ask(self, question):
        sources = core.SessionSources()
        out = io.StringIO()
        with redirect_stdout(out):
            messages = tinytalk.handle_turn(
                question, [], self.provider, self.memory, "SOUL", sources,
            )
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertIn(messages[-1]["content"], out.getvalue())
        return messages[-1]["content"], sources.records

    def check_replacement(self, predicate):
        old, new, wording, label = CASES[predicate]
        self.assertEqual(self.save(wording % old)["status"], "saved")
        self.assertEqual(self.save(wording % new)["status"], "replaced")
        for question in core.PROFILE_HISTORY_QUESTIONS[predicate]:
            with self.subTest(question=question):
                answer, sources = self.ask(question)
                self.assertIn("Previous %s: %s" % (label, old), answer)
                self.assertIn("This previous value is historical, not current.", answer)
                self.assertIn("Current %s: %s" % (label, new), answer)
                self.assertNotIn("Current %s: %s" % (label, old), answer)
                self.assertIn(wording % old, [s["text"] for s in sources if s["kind"] == "archived fact"])
                self.assertNotIn(wording % old, [s["text"] for s in sources if s["kind"] == "saved fact"])
                profile = next(p for p in core.read_current_profile(self.memory) if p["predicate"] == predicate)
                self.assertEqual((profile["state"], profile["value"]), ("current", new))
                self.assertEqual(self.memory.current_fact_objects("user", predicate), [new])

    def test_preferred_name_history(self):
        self.check_replacement("preferred_name")

    def test_home_city_history(self):
        self.check_replacement("home_city")

    def test_current_job_history(self):
        self.check_replacement("current_job")

    def test_favorite_vehicle_history(self):
        self.check_replacement("favorite_vehicle")

    def test_no_history_is_not_invented(self):
        for predicate, (old, new, wording, label) in CASES.items():
            with self.subTest(predicate=predicate):
                self.assertEqual(self.save(wording % new)["status"], "saved")
                question = next(iter(core.PROFILE_HISTORY_QUESTIONS[predicate]))
                answer, sources = self.ask(question)
                self.assertIn("Previous %s: not established" % label, answer)
                self.assertIn("Do not guess a value or a date.", answer)
                self.assertNotIn(old, answer)
                self.assertFalse(any(s["kind"] == "archived fact" for s in sources))

    def test_failed_replacement_is_not_current_or_history(self):
        self.assertEqual(self.save("my home city is Austin")["status"], "saved")
        real_update = self.memory.update_drawer

        def fail_archive(drawer_id, **kwargs):
            if kwargs.get("room") == core.HISTORY_ROOM:
                return {"success": False, "error": "archive unavailable"}
            return real_update(drawer_id, **kwargs)

        self.memory.update_drawer = fail_archive
        self.assertEqual(self.save("my home city is Nashville")["status"], "incomplete")
        answer, sources = self.ask("what was my previous home city")
        self.assertIn("Current home city: Austin", answer)
        self.assertIn("Previous home city: not established", answer)
        self.assertIn("Unfinished update, not the current value", answer)
        self.assertNotIn("Current home city: Nashville", answer)
        self.assertFalse(any(s["kind"] == "archived fact" for s in sources))


class ProfileHistoryEvidenceTests(unittest.TestCase):
    def test_wordings_are_closed_normalized_and_do_not_collide(self):
        existing = set(core.PREFERRED_NAME_QUESTIONS) | core.MEMORY_INTROSPECTION_QUESTIONS
        existing |= set(core.TEST_SPACESHIP_HISTORY_QUESTIONS)
        seen = set(existing)
        for predicate, questions in core.PROFILE_HISTORY_QUESTIONS.items():
            self.assertFalse(seen.intersection(questions))
            seen.update(questions)
            for question, later in questions.items():
                normalized_variant = "  " + question.upper().replace(" ", "   ") + "?  "
                self.assertEqual(core.profile_history_question(normalized_variant), (predicate, later))
        for question in core.PREFERRED_NAME_QUESTIONS:
            self.assertIsNone(core.profile_history_question(question))
            self.assertEqual(tinytalk.preferred_name_question(question), "preferred_name")
        for question in (
            "what was my name before sam", "what was my home city before memphis",
            "what was my job before teacher", "what was my favorite vehicle before bicycle",
            "what was it before", "what is my previous job", "what was my legal name before",
        ):
            self.assertIsNone(core.profile_history_question(question))
        for question, later in core.TEST_SPACESHIP_HISTORY_QUESTIONS.items():
            self.assertEqual(core.profile_history_question(question), ("test_spaceship_name", later))

    def test_metadata_scopes_archived_facts_without_parsing_text(self):
        drawers = [
            {"drawer_id": "old", "room": "facts-history", "text": "I used to live there",
             "metadata": {"source_file": core._fact_source_file("home_city", "Austin")}},
            {"room": "facts-history", "text": "my home city is Imaginary", "metadata": {}},
            {"room": "facts-history", "text": "I lived in Other",
             "metadata": {"source_file": core._fact_source_file("childhood_city", "Other")}},
            {"room": "facts-pending", "text": "I live in Pending",
             "metadata": {"source_file": core._fact_source_file("home_city", "Pending")}},
        ]
        evidence = core.profile_history_evidence([edge("Nashville", True)], drawers, "home_city", "")
        self.assertEqual(evidence["previous_name"], "Austin")
        self.assertEqual(evidence["candidate_count"], 1)
        self.assertEqual([s["text"] for s in core.history_sources(evidence) if s["kind"] == "archived fact"],
                         ["I used to live there"])

    def test_graph_boundaries_order_values_not_input_or_filing_order(self):
        rows = [
            edge("Nashville", True, start="2026-03-01T00:00:00Z"),
            edge("Austin", False, end="2026-02-01T00:00:00Z"),
            edge("Dallas", False, end="2026-01-01T00:00:00Z"),
        ]
        for ordered in (rows, list(reversed(rows))):
            evidence = core.profile_history_evidence(ordered, [], "home_city", "nashville")
            self.assertEqual(evidence["previous_name"], "Austin")
            self.assertNotIn("Dallas", core.format_profile_history(evidence, "home_city"))

    def test_ambiguous_or_late_boundaries_do_not_guess(self):
        for first, second in (
            (None, None),
            ("2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            ("2026-04-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ):
            rows = [edge("Nashville", True, start="2026-03-01T00:00:00Z"),
                    edge("Austin", False, end=first), edge("Dallas", False, end=second)]
            evidence = core.profile_history_evidence(rows, [], "home_city", "")
            self.assertIsNone(evidence["previous_name"])
            self.assertIn("not established", core.format_profile_history(evidence, "home_city"))
            self.assertFalse(any(s["kind"] == "archived fact" for s in core.history_sources(evidence)))

    def test_before_value_mismatch_suppresses_history(self):
        evidence = core.profile_history_evidence(
            [edge("Memphis", True), edge("Austin", False)], [], "home_city", "nashville",
        )
        self.assertTrue(evidence["requested_mismatch"])
        self.assertIsNone(evidence["previous_name"])
        self.assertNotIn("Austin", core.format_profile_history(evidence, "home_city"))

    def test_spaceship_history_prefers_metadata_over_the_legacy_parser(self):
        drawers = [
            {"drawer_id": "meta", "room": "facts-history",
             "text": "the old test spaceship was named Serenity",
             "metadata": {"source_file": core._fact_source_file("test_spaceship_name", "Nostromo")}},
            {"drawer_id": "legacy", "room": "facts-history",
             "text": "my test spaceship is named Serenity",
             "metadata": {"source_file": "tinytalk"}},
            {"drawer_id": "unclassified", "room": "facts-history",
             "text": "my test spaceship is named Defiant",
             "metadata": {"source_file": core._fact_source_file(unclassified=True)}},
        ]
        facts = core._history_facts(drawers, "test_spaceship_name")
        self.assertEqual(
            [(fact["id"], fact["name"]) for fact in facts],
            [("meta", "Nostromo"), ("legacy", "Serenity")],
        )

    def test_duplicate_current_value_is_not_previous(self):
        evidence = core.profile_history_evidence(
            [edge("Nashville", True), edge("Nashville", False)], [], "home_city", "",
        )
        self.assertIsNone(evidence["previous_name"])
        self.assertEqual(len(evidence["duplicates"]), 1)


if __name__ == "__main__":
    unittest.main()
