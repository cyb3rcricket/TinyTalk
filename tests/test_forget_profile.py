"""Forget facts through real isolated palaces; retain abandoned provenance."""

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
    "preferred_name": ("River", "Willow", "Hazel", "my preferred name is %s", "what was my previous name"),
    "home_city": ("Austin", "Nashville", "Chicago", "my home city is %s", "what was my previous home city"),
    "current_job": ("teacher", "engineer", "carpenter", "my current job is %s", "what was my previous job"),
    "favorite_vehicle": ("bicycle", "CyberTruck", "scooter", "my favorite vehicle is %s", "what was my previous favorite vehicle"),
    "test_spaceship_name": ("Alpha", "Beta", "Gamma", "my test spaceship is named %s", "what was the previous name of my test spaceship"),
}


class ExtractionProvider:
    def complete(self, messages):
        if messages[0]["content"] != core.TRIPLE_INSTRUCTIONS:
            raise AssertionError("Forgetting must not call the model")
        text = messages[-1]["content"]
        for predicate, (old, new, fresh, wording, _question) in CASES.items():
            for value in (old, new, fresh):
                if text == wording % value:
                    return json.dumps({"subject": "user", "predicate": predicate, "object": value})
        raise AssertionError("Unexpected fact")


class ForgetProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = memory_isolation.temp_dir("tinytalk-forget-")
        self.memory = memory_isolation.open_isolated_memory(self.temp.name)
        self.provider = ExtractionProvider()

    def tearDown(self):
        self.memory.kg.close()
        self.temp.cleanup()

    def save(self, predicate, value):
        wording = CASES[predicate][3]
        with redirect_stdout(io.StringIO()):
            return core.remember_fact("Remember this: " + wording % value, self.memory, self.provider)

    def profile(self):
        return {entry["predicate"]: entry for entry in core.read_current_profile(self.memory)}

    def snapshot(self):
        return {
            "drawers": core._drawers_by_id(self.memory),
            "entities": core._graph_rows(self.memory.kg, "entities"),
            "triples": core._graph_rows(self.memory.kg, "triples"),
            "conversations": core._room_drawers(self.memory, "conversations"),
        }

    def command(self, text):
        messages = [{"role": "assistant", "content": "The earlier chat stays."}]
        original = list(messages)
        sources = core.SessionSources()
        sources.replace([{"kind": "saved fact", "text": "earlier supplied source"}])
        original_sources = list(sources.records)
        output = io.StringIO()
        with redirect_stdout(output):
            returned = tinytalk.handle_user_line(
                text, messages, self.provider, self.memory, "SOUL", None, sources,
            )
        self.assertIs(returned, messages)
        self.assertEqual(messages, original)
        self.assertEqual(sources.records, original_sources)
        return output.getvalue().strip()

    def assert_no_history(self, predicate, old_values):
        question = CASES[predicate][4]
        messages, sources = tinytalk.build_context(question, [], self.memory)
        body = messages[0]["content"]
        self.assertIn("not established", body)
        for value in old_values:
            self.assertNotIn(value, body)
        self.assertFalse(any(source["kind"] == "archived fact" for source in sources))

    def check_field(self, predicate):
        old, new, fresh, wording, _question = CASES[predicate]
        for field, case in CASES.items():
            self.assertEqual(self.save(field, case[0])["status"], "saved")
        self.assertEqual(self.save(predicate, new)["status"], "replaced")
        self.memory.save_exchange("An earlier conversation.", "It stays in history.")
        before = self.snapshot()
        before_profile = self.profile()
        originals = {
            key: drawer for key, drawer in before["drawers"].items()
            if core._fact_relationship(drawer["metadata"])[0] == predicate
        }
        reply = self.command("/forget " + predicate)
        self.assertEqual(reply, (
            "Forgot facts for %s. Moved to facts-abandoned: facts=1, facts-history=1, "
            "facts-pending=0. /forget forgets facts, it does not rewrite conversation history."
        ) % predicate)
        profile = self.profile()
        self.assertEqual(profile[predicate]["state"], "missing")
        self.assertIsNone(profile[predicate]["value"])
        for other in CASES:
            if other != predicate:
                self.assertEqual(profile[other], before_profile[other])
        after = self.snapshot()
        self.assertEqual(after["conversations"], before["conversations"])
        self.assertEqual(set(after["drawers"]), set(before["drawers"]))
        self.assertEqual(len(after["triples"]), len(before["triples"]))
        for drawer_id, original in originals.items():
            abandoned = after["drawers"][drawer_id]
            self.assertEqual(abandoned["room"], core.ABANDONED_ROOM)
            self.assertEqual(abandoned["text"], original["text"])
            self.assertEqual(abandoned["metadata"]["source_file"], original["metadata"]["source_file"])
            self.assertFalse(self.memory.get_drawer(drawer_id).get("error"))
        for row in after["triples"]:
            if row["predicate"] == predicate:
                self.assertIsNotNone(row["valid_to"])
            else:
                self.assertIn(row, before["triples"])
        hits = self.memory.search_facts(wording % new)
        for hit in hits:
            self.assertNotIn(hit["text"], (wording % old, wording % new))
        self.assert_no_history(predicate, (old, new))
        self.assertEqual(self.save(predicate, fresh)["status"], "saved")
        self.assertEqual(self.profile()[predicate]["state"], "current")
        self.assertEqual(self.profile()[predicate]["value"], fresh)
        self.assert_no_history(predicate, (old, new))

    def test_forget_preferred_name(self):
        self.check_field("preferred_name")

    def test_forget_home_city(self):
        self.check_field("home_city")

    def test_forget_current_job(self):
        self.check_field("current_job")

    def test_forget_favorite_vehicle(self):
        self.check_field("favorite_vehicle")

    def test_forget_test_spaceship_name(self):
        self.check_field("test_spaceship_name")

    def test_current_only_forget_and_same_value_fresh_save(self):
        self.assertEqual(self.save("home_city", "Austin")["status"], "saved")
        original = core._room_drawers(self.memory, core.FACT_ROOM)[0]
        result = core.forget_profile_field(self.memory, "home_city")
        self.assertEqual(result["status"], "forgotten")
        self.assertEqual(result["moved"], {"facts": 1, "facts-history": 0, "facts-pending": 0})
        self.assertEqual(self.profile()["home_city"]["state"], "missing")
        self.assertEqual(self.memory.search_facts("my home city is Austin"), [])
        self.assertEqual(self.save("home_city", "Austin")["status"], "saved")
        self.assertEqual(self.profile()["home_city"]["value"], "Austin")
        self.assertEqual(self.profile()["home_city"]["source_drawer_id"], original["drawer_id"])

    def test_unknown_missing_and_extra_arguments_change_nothing(self):
        self.save("home_city", "Austin")
        before = self.snapshot()
        for command in ("/forget", "/forget unknown", "/forget home_city extra", "/forgetful home_city"):
            self.assertEqual(self.command(command), core.FORGET_USAGE)
            self.assertEqual(self.snapshot(), before)
        self.assertEqual(core.forget_profile_field(self.memory, "unknown")["status"], "invalid_field")
        self.assertEqual(self.snapshot(), before)

    def test_empty_field_changes_nothing(self):
        self.save("preferred_name", "River")
        before = self.snapshot()
        self.assertEqual(self.command("/forget home_city"), (
            "Nothing stored for home_city. "
            "/forget forgets facts, it does not rewrite conversation history."
        ))
        self.assertEqual(self.snapshot(), before)

    def test_repeated_forget_is_a_no_op(self):
        self.save("home_city", "Austin")
        core.forget_profile_field(self.memory, "home_city")
        before = self.snapshot()
        result = core.forget_profile_field(self.memory, "home_city")
        self.assertEqual(result["status"], "nothing_stored")
        self.assertEqual(result["moved"], {"facts": 0, "facts-history": 0, "facts-pending": 0})
        self.assertEqual(self.snapshot(), before)

    def test_pending_update_is_abandoned_so_profile_is_missing(self):
        self.save("home_city", "Austin")
        update = self.memory.update_drawer

        def fail_archive(drawer_id, **kwargs):
            if kwargs.get("room") == core.HISTORY_ROOM:
                return {"success": False}
            return update(drawer_id, **kwargs)

        self.memory.update_drawer = fail_archive
        self.assertEqual(self.save("home_city", "Nashville")["status"], "incomplete")
        self.memory.update_drawer = update
        result = core.forget_profile_field(self.memory, "home_city")
        self.assertEqual(result["status"], "forgotten")
        self.assertEqual(result["moved"], {"facts": 1, "facts-history": 0, "facts-pending": 1})
        self.assertEqual(self.profile()["home_city"]["state"], "missing")
        self.assertEqual(self.memory.pending_facts(), [])
        self.assertEqual(self.save("home_city", "Chicago")["status"], "saved")
        self.assert_no_history("home_city", ("Austin", "Nashville"))

    def test_shared_drawer_is_refused_before_any_write(self):
        self.save("home_city", "Austin")
        drawer_id = self.profile()["home_city"]["source_drawer_id"]
        self.memory.kg.add_triple("user", "current_job", "teacher", source_drawer_id=drawer_id)
        before = self.snapshot()
        result = core.forget_profile_field(self.memory, "home_city")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("Nothing was changed", result["message"])
        self.assertEqual(self.snapshot(), before)

    def test_failed_move_reports_only_completed_moves_and_retry_finishes(self):
        self.save("home_city", "Austin")
        self.save("home_city", "Nashville")
        update = self.memory.update_drawer
        calls = []

        def fail_second(drawer_id, **kwargs):
            calls.append(drawer_id)
            return {"success": False} if len(calls) == 2 else update(drawer_id, **kwargs)

        self.memory.update_drawer = fail_second
        result = core.forget_profile_field(self.memory, "home_city")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["moved"], {"facts": 1, "facts-history": 0, "facts-pending": 0})
        self.assertIn("facts=1, facts-history=0, facts-pending=0", result["message"])
        self.memory.update_drawer = update
        retry = core.forget_profile_field(self.memory, "home_city")
        self.assertEqual(retry["status"], "forgotten")
        self.assertEqual(retry["moved"], {"facts": 0, "facts-history": 1, "facts-pending": 0})
        self.assertEqual(self.profile()["home_city"]["state"], "missing")

    def test_failed_graph_retirement_is_reported_and_retry_finishes(self):
        self.save("home_city", "Austin")
        invalidate = self.memory.kg.invalidate

        def fail(*args, **kwargs):
            raise RuntimeError("graph unavailable")

        self.memory.kg.invalidate = fail
        result = core.forget_profile_field(self.memory, "home_city")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["moved"]["facts"], 1)
        self.assertNotIn("Forgot facts", result["message"])
        self.memory.kg.invalidate = invalidate
        self.assertEqual(core.forget_profile_field(self.memory, "home_city")["status"], "forgotten")
        self.assertEqual(self.profile()["home_city"]["state"], "missing")

    def test_legacy_spaceship_aliases_and_unlinked_history_are_forgotten(self):
        for room, name in ((core.FACT_ROOM, "Beta"), (core.HISTORY_ROOM, "Alpha")):
            self.memory.add_drawer(wing="tinytalk", room=room,
                                   content="my test spaceship is named " + name,
                                   source_file="tinytalk", added_by="tinytalk")
        self.memory.kg.add_triple("user", "test_spaceship", "Alpha")
        self.memory.kg.invalidate("user", "test_spaceship", "Alpha")
        self.memory.kg.add_triple("user", "spaceship_name", "Beta")
        result = core.forget_profile_field(self.memory, "test_spaceship_name")
        self.assertEqual(result["status"], "forgotten")
        self.assertEqual(result["moved"], {"facts": 1, "facts-history": 1, "facts-pending": 0})
        self.assertEqual(self.profile()["test_spaceship_name"]["state"], "missing")
        self.assert_no_history("test_spaceship_name", ("Alpha", "Beta"))
        self.assertEqual(self.save("test_spaceship_name", "Gamma")["status"], "saved")
        self.assert_no_history("test_spaceship_name", ("Alpha", "Beta"))


if __name__ == "__main__":
    unittest.main()
