"""Dream pass against isolated MemPalace stores.

memory_isolation keeps every store inside a temporary root. Import it before
dream_pass, which imports memory_core.
"""

import io
import json
import os
import sys
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import memory_isolation  # noqa: E402
import memory_core  # noqa: E402
import dream_pass  # noqa: E402

RAMEN_OLD = "my favorite meal is spicy ramen from the night market stall"
RAMEN_NEW = "my favorite meal is spicy ramen from the night market"
WALK_OLD = "I like quiet morning walks beside the river."
WALK_NEW = "I like quiet morning walks beside the river"
GARDEN = "I grow tomatoes on the south balcony"
DOCK_BELL = "the dock bell rings twice at dusk"
SPARE_KEY = "a spare key lives under the blue pot"
NASHVILLE = "I live in Nashville"
MEMPHIS = "I live in Memphis"
ROOMS = ("facts", "facts-history", "facts-pending", "facts-abandoned")


def triple(predicate, obj):
    return json.dumps({"subject": "user", "predicate": predicate, "object": obj})


class Extractor(object):
    """Returns one fixed extraction string."""

    def __init__(self, raw):
        self.raw = raw

    def complete(self, messages):
        return self.raw


def remember(memory, fact, raw):
    out = io.StringIO()
    with redirect_stdout(out):
        return memory_core.remember_fact("Remember this: " + fact, memory, Extractor(raw))


def add_fact(memory, text, predicate=None, obj=None, room="facts", unclassified=False):
    if unclassified:
        source = memory_core._fact_source_file(unclassified=True)
    elif predicate:
        source = memory_core._fact_source_file(predicate, obj or "")
    else:
        source = memory_core._fact_source_file()
    saved = memory.add_drawer(
        wing="tinytalk",
        room=room,
        content=text,
        source_file=source,
        added_by="tinytalk",
    )
    if not isinstance(saved, dict) or not saved.get("success"):
        raise AssertionError(saved)
    return saved["drawer_id"]


def snapshot(memory):
    found = {}
    for room in ROOMS:
        for drawer in memory_core._room_drawers(memory, room):
            found[drawer["drawer_id"]] = room
    return found


def section(report, title):
    marker = "## " + title
    start = report.index(marker)
    rest = report[start:]
    ends = []
    for mark in ("\n## ", "\nSummary:"):
        at = rest.find(mark, 1)
        if at != -1:
            ends.append(at)
    return rest[:min(ends)] if ends else rest


class DuplicateThresholdTests(unittest.TestCase):
    def test_threshold_is_stricter_than_the_recall_floor(self):
        self.assertEqual(dream_pass.DUPLICATE_SIMILARITY, 0.90)
        self.assertGreater(
            dream_pass.DUPLICATE_SIMILARITY, memory_core.MIN_FACT_SIMILARITY
        )


class DreamPassTests(unittest.TestCase):
    def setUp(self):
        self._temp = memory_isolation.temp_dir("tinytalk-dream-")
        self.memory = memory_isolation.open_isolated_memory(self._temp.name)

    def tearDown(self):
        self.memory.kg.close()
        self._temp.cleanup()

    def _drawer(self, text):
        matches = []
        for room in ROOMS:
            for drawer in memory_core._room_drawers(self.memory, room):
                if drawer["text"] == text:
                    matches.append(drawer)
        self.assertEqual(len(matches), 1, matches)
        return matches[0]

    def _seed_report_store(self):
        add_fact(self.memory, RAMEN_OLD, predicate="favorite_meal", obj="spicy ramen")
        time.sleep(0.05)
        add_fact(self.memory, RAMEN_NEW, predicate="favorite_meal", obj="spicy ramen")
        add_fact(self.memory, WALK_OLD, unclassified=True)
        time.sleep(0.05)
        add_fact(self.memory, WALK_NEW, unclassified=True)
        add_fact(self.memory, GARDEN, predicate="garden_crop", obj="tomatoes")
        add_fact(self.memory, SPARE_KEY)
        saved = remember(self.memory, DOCK_BELL, "not a fact")
        self.assertEqual(saved["status"], "saved_as_written")
        current = remember(self.memory, NASHVILLE, triple("home_city", "Nashville"))
        self.assertEqual(current["status"], "saved")
        add_fact(self.memory, MEMPHIS, predicate="home_city", obj="Memphis")

    def test_report_finds_cluster_stale_echo_and_orphan(self):
        self._seed_report_store()
        ramen_old = self._drawer(RAMEN_OLD)
        ramen_new = self._drawer(RAMEN_NEW)
        walk_old = self._drawer(WALK_OLD)
        walk_new = self._drawer(WALK_NEW)
        garden = self._drawer(GARDEN)
        dock = self._drawer(DOCK_BELL)
        spare = self._drawer(SPARE_KEY)
        nashville = self._drawer(NASHVILLE)
        memphis = self._drawer(MEMPHIS)

        self.assertLess(ramen_old["metadata"]["filed_at"], ramen_new["metadata"]["filed_at"])
        self.assertLess(walk_old["metadata"]["filed_at"], walk_new["metadata"]["filed_at"])
        self.assertGreater(len(RAMEN_OLD), len(RAMEN_NEW))
        self.assertGreater(len(WALK_OLD), len(WALK_NEW))

        profile = {
            entry["predicate"]: entry
            for entry in memory_core.read_current_profile(self.memory)
        }
        self.assertEqual(profile["home_city"]["state"], "current")
        self.assertEqual(profile["home_city"]["value"], "Nashville")
        self.assertEqual(profile["home_city"]["source_drawer_id"], nashville["drawer_id"])

        before = snapshot(self.memory)
        with patch(
            "dream_pass._move_drawer",
            side_effect=AssertionError("report mode called _move_drawer"),
        ):
            clusters = dream_pass.find_duplicate_clusters(self.memory)
            echoes = dream_pass.find_stale_echoes(self.memory)
            orphans = dream_pass.find_orphaned_as_written(self.memory)
            report = dream_pass.render_report(self.memory)
            out = io.StringIO()
            with redirect_stdout(out):
                code = dream_pass.main([], memory=self.memory)
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue(), report)
        self.assertEqual(snapshot(self.memory), before)

        self.assertEqual(
            sorted(tuple(sorted(member["text"] for member in cluster["members"]))
                   for cluster in clusters),
            sorted([
                tuple(sorted([RAMEN_OLD, RAMEN_NEW])),
                tuple(sorted([WALK_OLD, WALK_NEW])),
            ]),
        )
        ramen = next(cluster for cluster in clusters if RAMEN_NEW in {
            member["text"] for member in cluster["members"]
        })
        walks = next(cluster for cluster in clusters if WALK_NEW in {
            member["text"] for member in cluster["members"]
        })
        self.assertEqual(ramen["keeper_id"], ramen_new["drawer_id"])
        self.assertEqual(ramen["members"][0]["drawer_id"], ramen_new["drawer_id"])
        self.assertEqual(walks["keeper_id"], walk_new["drawer_id"])
        self.assertEqual(
            {member["drawer_id"] for member in ramen["members"]},
            {ramen_old["drawer_id"], ramen_new["drawer_id"]},
        )

        self.assertEqual([echo["drawer_id"] for echo in echoes], [memphis["drawer_id"]])
        self.assertEqual(echoes[0]["predicate"], "home_city")
        self.assertEqual(echoes[0]["drawer_value"], "Memphis")
        self.assertEqual(echoes[0]["current_value"], "Nashville")
        self.assertEqual(echoes[0]["text"], MEMPHIS)

        self.assertEqual(
            {orphan["drawer_id"] for orphan in orphans},
            {
                walk_old["drawer_id"],
                walk_new["drawer_id"],
                dock["drawer_id"],
                spare["drawer_id"],
            },
        )
        self.assertIn(
            {"drawer_id": dock["drawer_id"], "text": DOCK_BELL},
            orphans,
        )

        duplicates = section(report, "Duplicate clusters")
        stale = section(report, "Stale echoes")
        orphan_section = section(report, "Orphaned as-written facts")
        self.assertNotIn("none found", report)
        self.assertIn(
            "Summary: duplicate clusters: 2; stale echoes: 1; orphaned as-written facts: 4",
            report,
        )
        for drawer_id, body, keeper in (
            (ramen_new["drawer_id"], duplicates, True),
            (ramen_old["drawer_id"], duplicates, False),
            (walk_new["drawer_id"], duplicates, True),
            (walk_old["drawer_id"], duplicates, False),
        ):
            line = next(item for item in body.splitlines() if drawer_id in item)
            if keeper:
                self.assertIn("suggested keeper", line)
            else:
                self.assertNotIn("suggested keeper", line)
        self.assertIn(RAMEN_OLD, duplicates)
        self.assertIn(RAMEN_NEW, duplicates)
        self.assertIn(WALK_OLD, duplicates)
        self.assertIn(WALK_NEW, duplicates)
        self.assertIn(memphis["drawer_id"], stale)
        self.assertIn('"Memphis"', stale)
        self.assertIn('"Nashville"', stale)
        self.assertIn(MEMPHIS, stale)
        self.assertIn(dock["drawer_id"], orphan_section)
        self.assertIn(DOCK_BELL, orphan_section)
        self.assertIn(spare["drawer_id"], orphan_section)
        self.assertIn(SPARE_KEY, orphan_section)

        for drawer_id in (
            nashville["drawer_id"],
            garden["drawer_id"],
            memphis["drawer_id"],
        ):
            self.assertNotIn(drawer_id, duplicates)
            self.assertNotIn(drawer_id, orphan_section)
        self.assertNotIn(nashville["drawer_id"], stale)
        self.assertNotIn(nashville["drawer_id"], report)
        self.assertNotIn(NASHVILLE, report)
        self.assertNotIn(garden["drawer_id"], report)
        self.assertNotIn(GARDEN, report)

    def test_empty_store_names_every_section(self):
        before = snapshot(self.memory)
        report = dream_pass.render_report(self.memory)
        self.assertEqual(snapshot(self.memory), before)
        for title in (
            "Duplicate clusters",
            "Stale echoes",
            "Orphaned as-written facts",
        ):
            body = section(report, title)
            self.assertIn("none found", body)
            self.assertEqual(body.count("none found"), 1)
        self.assertEqual(report.count("none found"), 3)
        self.assertIn(
            "Summary: duplicate clusters: 0; stale echoes: 0; orphaned as-written facts: 0",
            report,
        )
        out_path = os.path.join(self._temp.name, "dream.md")
        out = io.StringIO()
        with redirect_stdout(out):
            code = dream_pass.main(["--out", out_path], memory=self.memory)
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(Path(out_path).read_text(encoding="utf-8"), report)
        self.assertEqual(snapshot(self.memory), before)

        err = io.StringIO()
        with redirect_stdout(io.StringIO()), patch("sys.stderr", err):
            code = dream_pass.main(
                ["--apply", os.path.join(self._temp.name, "missing.txt"), "--out", out_path],
                memory=self.memory,
            )
        self.assertEqual(code, 2)
        self.assertEqual(snapshot(self.memory), before)

    def test_apply_retires_only_approved_ids(self):
        retire_id = add_fact(
            self.memory, "the shed door sticks in winter",
            predicate="shed_condition", obj="sticks",
        )
        keep_id = add_fact(
            self.memory, "the north window faces the alley",
            predicate="window_view", obj="alley",
        )
        history_id = add_fact(
            self.memory, "a note already archived",
            predicate="archive_note", obj="archived", room="facts-history",
        )
        pending_id = add_fact(
            self.memory, "an unfinished color note",
            predicate="favorite_color", obj="blue", room="facts-pending",
        )
        abandoned_id = add_fact(
            self.memory, "a discarded scrap",
            predicate="scrap_status", obj="discarded", room="facts-abandoned",
        )
        saved = remember(self.memory, "call me River", triple("preferred_name", "River"))
        self.assertEqual(saved["status"], "saved")
        profile_id = self._drawer("call me River")["drawer_id"]

        approvals = os.path.join(self._temp.name, "approvals.txt")
        Path(approvals).write_text("\n".join([
            "# keep the newer wording",
            "",
            retire_id,
            "   ",
            "not-a-drawer",
            "# profile is out of scope",
            profile_id,
            abandoned_id,
            pending_id,
            history_id,
            "# end",
            "",
        ]), encoding="utf-8")

        before = snapshot(self.memory)
        out = io.StringIO()
        with redirect_stdout(out):
            code = dream_pass.main(["--apply", approvals], memory=self.memory)
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().splitlines(), [
            "%s facts moved to facts-abandoned" % retire_id,
            "REFUSED not-a-drawer: unknown drawer",
            "REFUSED %s: profile predicate preferred_name" % profile_id,
            "REFUSED %s: not in facts or facts-history (currently facts-abandoned)"
            % abandoned_id,
            "REFUSED %s: not in facts or facts-history (currently facts-pending)"
            % pending_id,
            "%s facts-history moved to facts-abandoned" % history_id,
        ])
        after = snapshot(self.memory)
        moved = sorted(drawer_id for drawer_id in before if before[drawer_id] != after[drawer_id])
        self.assertEqual(moved, sorted([retire_id, history_id]))
        self.assertEqual(after[retire_id], "facts-abandoned")
        self.assertEqual(after[history_id], "facts-abandoned")
        self.assertEqual(after[keep_id], "facts")
        self.assertEqual(after[profile_id], "facts")
        self.assertEqual(after[abandoned_id], "facts-abandoned")
        self.assertEqual(after[pending_id], "facts-pending")
        self.assertNotIn("not-a-drawer", after)
        profile = {
            entry["predicate"]: entry
            for entry in memory_core.read_current_profile(self.memory)
        }
        self.assertEqual(profile["preferred_name"]["state"], "current")
        self.assertEqual(profile["preferred_name"]["value"], "River")
        self.assertEqual(profile["preferred_name"]["source_drawer_id"], profile_id)

    def test_apply_continues_after_a_move_failure(self):
        first_id = add_fact(self.memory, "the first note stays", predicate="note_order", obj="first")
        second_id = add_fact(self.memory, "the second note can leave", predicate="note_order", obj="second")
        real = self.memory.update_drawer

        def update(drawer_id, **kwargs):
            if drawer_id == first_id:
                return {"success": False}
            return real(drawer_id, **kwargs)

        self.memory.update_drawer = update
        approvals = os.path.join(self._temp.name, "approvals.txt")
        Path(approvals).write_text(first_id + "\n" + second_id + "\n", encoding="utf-8")
        out = io.StringIO()
        with redirect_stdout(out):
            code = dream_pass.main(["--apply", approvals], memory=self.memory)
        self.assertEqual(code, 1)
        self.assertEqual(out.getvalue().splitlines(), [
            "FAILED %s: could not move from facts to facts-abandoned" % first_id,
            "%s facts moved to facts-abandoned" % second_id,
        ])
        rooms = snapshot(self.memory)
        self.assertEqual(rooms[first_id], "facts")
        self.assertEqual(rooms[second_id], "facts-abandoned")

    def test_graph_linked_additive_facts_are_not_orphans(self):
        dune = "my favorite book is Dune"
        dune_again = "my favorite book is Dune."
        written = "the porch light stays on until midnight"
        first = remember(self.memory, dune, triple("favorite_book", "Dune"))
        second = remember(self.memory, dune_again, triple("favorite_book", "Dune."))
        as_written = remember(self.memory, written, "not a fact")
        self.assertEqual(first["status"], "saved")
        self.assertEqual(second["status"], "saved")
        self.assertEqual(as_written["status"], "saved_as_written")

        dune_drawer = self._drawer(dune)
        again_drawer = self._drawer(dune_again)
        written_drawer = self._drawer(written)
        self.assertEqual(
            memory_core._fact_relationship(dune_drawer["metadata"]), (None, None)
        )
        self.assertEqual(
            memory_core._fact_relationship(again_drawer["metadata"]), (None, None)
        )
        linked = {
            row.get("source_drawer_id")
            for row in memory_core._graph_rows(self.memory.kg, "triples")
        }
        self.assertIn(dune_drawer["drawer_id"], linked)
        self.assertIn(again_drawer["drawer_id"], linked)
        self.assertNotIn(written_drawer["drawer_id"], linked)

        before = snapshot(self.memory)
        orphans = dream_pass.find_orphaned_as_written(self.memory)
        clusters = dream_pass.find_duplicate_clusters(self.memory)
        report = dream_pass.render_report(self.memory)
        self.assertEqual(snapshot(self.memory), before)

        self.assertEqual(
            {orphan["drawer_id"] for orphan in orphans},
            {written_drawer["drawer_id"]},
        )
        self.assertEqual(orphans[0]["text"], written)
        book_clusters = [
            cluster for cluster in clusters
            if {member["text"] for member in cluster["members"]} == {dune, dune_again}
        ]
        self.assertEqual(len(book_clusters), 1)
        orphan_section = section(report, "Orphaned as-written facts")
        duplicate_section = section(report, "Duplicate clusters")
        self.assertIn(written_drawer["drawer_id"], orphan_section)
        self.assertNotIn(dune_drawer["drawer_id"], orphan_section)
        self.assertNotIn(again_drawer["drawer_id"], orphan_section)
        self.assertIn(dune, duplicate_section)
        self.assertIn(dune_again, duplicate_section)

    def test_apply_approvals_reads_drawers_once(self):
        first_id = add_fact(self.memory, "first cached note", predicate="cache_note", obj="one")
        second_id = add_fact(self.memory, "second cached note", predicate="cache_note", obj="two")
        calls = []
        real_drawers_by_id = dream_pass._drawers_by_id

        def counting_drawers_by_id(memory):
            calls.append(True)
            return real_drawers_by_id(memory)

        with patch("dream_pass._drawers_by_id", side_effect=counting_drawers_by_id):
            lines, ok = dream_pass.apply_approvals(
                self.memory, [first_id, second_id, first_id, "missing-id"]
            )

        self.assertTrue(ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(lines, [
            "%s facts moved to facts-abandoned" % first_id,
            "%s facts moved to facts-abandoned" % second_id,
            "REFUSED %s: not in facts or facts-history (currently facts-abandoned)" % first_id,
            "REFUSED missing-id: unknown drawer",
        ])
