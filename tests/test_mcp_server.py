"""MCP tool logic against an isolated MemPalace, plus one in-process transport smoke.

memory_isolation keeps every store inside a temporary root. Import it before
mcp_server, which imports tinytalk.
"""

import asyncio
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_isolation  # noqa: E402  (must come before tinytalk and mempalace)
import mcp_server
import memory_core
from mcp import Client


def triple(predicate, obj, subject="user"):
    return json.dumps({"subject": subject, "predicate": predicate, "object": obj})


def _print_not_protocol():
    print("not-protocol")
    return {"ok": True}


class Extractor(object):
    """Fact-extraction stub. A string is returned as the model text. An exception is raised."""

    def __init__(self, raw):
        self.raw = raw

    def complete(self, messages):
        if isinstance(self.raw, Exception):
            raise self.raw
        return self.raw


class McpToolTests(unittest.TestCase):
    def setUp(self):
        self._temp = memory_isolation.temp_dir("tinytalk-mcp-")
        self.temp = self._temp.__enter__()
        self.memory = memory_isolation.open_isolated_memory(self.temp)

    def tearDown(self):
        self._temp.__exit__(None, None, None)

    def _profile(self, predicate, entries=None):
        if entries is None:
            entries = memory_core.read_current_profile(self.memory)
        for entry in entries:
            if entry["predicate"] == predicate:
                return entry
        self.fail("missing profile predicate %s" % predicate)

    def test_get_current_profile_returns_entries_unchanged(self):
        entries = mcp_server.get_current_profile(self.memory)
        self.assertEqual(entries, memory_core.read_current_profile(self.memory))
        self.assertEqual(
            [entry["predicate"] for entry in entries],
            list(memory_core.PROFILE_PREDICATES),
        )
        for entry in entries:
            self.assertEqual(entry["state"], "missing")
            self.assertIsNone(entry["value"])

    def test_save_memory_returns_remember_fact_status(self):
        provider = Extractor(triple("preferred_name", "Willow"))
        result = mcp_server.save_memory(self.memory, provider, "call me Willow")
        self.assertEqual(result, {"status": "saved", "message": "Memory: saved."})
        entry = self._profile("preferred_name")
        self.assertEqual(entry["state"], "current")
        self.assertEqual(entry["value"], "Willow")

    def test_save_memory_surfaces_saved_as_written(self):
        text = "the dock bell rings twice at dusk"
        result = mcp_server.save_memory(self.memory, Extractor("not a fact"), text)
        self.assertEqual(result["status"], "saved_as_written")
        self.assertTrue(result["message"].startswith("Memory: saved as written."))
        self.assertIn(text, self.memory.list_facts())
        entry = self._profile("preferred_name")
        self.assertEqual(entry["state"], "missing")

        with memory_isolation.temp_dir("tinytalk-mcp-raise-") as other_temp:
            other = memory_isolation.open_isolated_memory(other_temp)
            raised = mcp_server.save_memory(other, Extractor(RuntimeError("no model")), text)
        self.assertEqual(raised["status"], "saved_as_written")
        self.assertEqual(raised["message"], result["message"])

    def test_save_memory_blank_text_is_not_a_command(self):
        self.assertIsNone(mcp_server.save_memory(self.memory, Extractor("unused"), "   "))

    def test_save_and_read_back_through_both_directions(self):
        ship = mcp_server.save_memory(
            self.memory,
            Extractor(triple("test_spaceship_name", "Enterprise")),
            "my test spaceship is named Enterprise",
        )
        self.assertEqual(ship["status"], "saved")
        stored = self._profile("test_spaceship_name")
        self.assertEqual(stored["state"], "current")
        self.assertEqual(stored["value"], "Enterprise")
        hits = self.memory.search_facts("my test spaceship is named Enterprise")
        self.assertTrue(any("Enterprise" in (hit.get("text") or "") for hit in hits))

        direct = memory_core.remember_fact(
            "Remember this: call me River",
            self.memory,
            Extractor(triple("preferred_name", "River")),
        )
        self.assertEqual(direct["status"], "saved")
        found = mcp_server.search_memories(self.memory, "call me River")
        self.assertEqual(list(found)[:2], ["profile", "semantic_facts"])
        self.assertEqual(found["profile"], memory_core.read_current_profile(self.memory))
        self.assertEqual(self._profile("preferred_name", found["profile"])["value"], "River")
        self.assertEqual(self._profile("preferred_name", found["profile"])["state"], "current")
        self.assertEqual(
            self._profile("test_spaceship_name", found["profile"])["value"],
            "Enterprise",
        )
        self.assertTrue(found["semantic_facts"])
        for hit in found["semantic_facts"]:
            self.assertEqual(hit["label"], "semantic fact")
            self.assertIn("similarity", hit)
            self.assertIn("text", hit)
            self.assertNotIn("state", hit)
        self.assertEqual(
            [(hit["similarity"], hit["text"]) for hit in found["semantic_facts"]],
            [(hit.get("similarity"), hit.get("text") or "") for hit in self.memory.search_facts("call me River")],
        )

    def test_sync_thread_saves_pairs_and_skips_the_tail(self):
        result = mcp_server.sync_thread(
            self.memory,
            "Docking notes",
            [
                {"role": "user", "content": "The amber dock light is on"},
                {"role": "assistant", "content": "I noted the amber dock light."},
                {"role": "user", "content": "The shuttle is named Kestrel"},
                {"role": "assistant", "content": "I noted the shuttle Kestrel."},
                {"role": "user", "content": "trailing unpaired lighthouse"},
            ],
            "field-notes",
        )
        self.assertEqual(result, {"saved": 2, "skipped": 1})

        listed = self.memory.list_drawers(wing="tinytalk", room="conversations", limit=10)
        drawers = listed.get("drawers") or []
        self.assertEqual(len(drawers), 2)
        texts = []
        for item in drawers:
            full = self.memory.get_drawer(item["drawer_id"])
            texts.append(full.get("content") or "")
        noted = [text for text in texts if "Thread: Docking notes (source: field-notes)" in text]
        self.assertEqual(len(noted), 1)
        self.assertTrue(any("amber dock light" in text for text in texts))
        self.assertTrue(any("Kestrel" in text for text in texts))
        self.assertFalse(any("trailing unpaired lighthouse" in text for text in texts))

        recalled = mcp_server.recall_thread(self.memory, "The amber dock light is on")
        self.assertTrue(recalled["excerpts"])
        excerpt = recalled["excerpts"][0]
        self.assertEqual(excerpt["status"], "historical")
        self.assertEqual(excerpt["kind"], "conversation excerpt")
        self.assertIn("amber dock light", excerpt["text"])
        self.assertNotEqual(excerpt["kind"], "saved fact")

    def test_sync_thread_rejects_invalid_shapes(self):
        for messages in (
            "not a list",
            [{"role": "system", "content": "nope"}],
            [{"role": "user", "content": 5}],
            [None],
        ):
            result = mcp_server.sync_thread(self.memory, "Title", messages, "app")
            self.assertIn("error", result)
        self.assertIn("error", mcp_server.sync_thread(self.memory, None, [], "app"))
        listed = self.memory.list_drawers(wing="tinytalk", room="conversations", limit=10)
        self.assertEqual(listed.get("drawers") or [], [])

    def test_recall_thread_drops_a_retired_command(self):
        provider = Extractor(triple("preferred_name", "Al"))
        self.assertEqual(
            mcp_server.save_memory(self.memory, provider, "call me Al")["status"],
            "saved",
        )
        self.memory.save_exchange("Remember this: call me Al", "Memory: saved.")
        self.memory.save_exchange("The amber dock light is on", "I noted the amber dock light.")
        replaced = mcp_server.save_memory(
            self.memory,
            Extractor(triple("preferred_name", "Sam")),
            "call me Sam",
        )
        self.assertEqual(replaced["status"], "replaced")

        raw = self.memory.search_conversations("Remember this: call me Al")
        self.assertTrue(any("call me Al" in (hit.get("text") or "") for hit in raw))
        dropped = mcp_server.recall_thread(self.memory, "Remember this: call me Al")
        for excerpt in dropped["excerpts"]:
            self.assertNotIn("call me Al", excerpt.get("text") or "")

        kept = mcp_server.recall_thread(self.memory, "The amber dock light is on")
        self.assertTrue(any("amber dock light" in excerpt.get("text", "") for excerpt in kept["excerpts"]))
        self.assertEqual(kept["excerpts"][0]["status"], "historical")

    def test_search_labels_an_unclassified_hit(self):
        written = mcp_server.save_memory(self.memory, Extractor("NONE"), "call me Al")
        self.assertEqual(written["status"], "saved_as_written")
        found = mcp_server.search_memories(self.memory, "call me Al")
        self.assertEqual(self._profile("preferred_name", found["profile"])["state"], "missing")
        self.assertTrue(found["semantic_facts"])
        self.assertTrue(all(
            hit["label"] == "unclassified saved text" for hit in found["semantic_facts"]
        ))
        self.assertNotIn("semantic fact", [hit["label"] for hit in found["semantic_facts"]])

    def test_sync_thread_reports_unavailable_when_the_palace_is_closed(self):
        self.memory.palace = None
        result = mcp_server.sync_thread(
            self.memory,
            "Docking notes",
            [
                {"role": "user", "content": "The amber dock light is on"},
                {"role": "assistant", "content": "I noted the amber dock light."},
            ],
            "field-notes",
        )
        self.assertEqual(result["error"], "Saved memory is unavailable.")
        self.assertEqual(result["saved"], 0)

    def test_recall_thread_filters_a_retired_fact_after_hydration(self):
        provider = Extractor(triple("preferred_name", "Al"))
        self.assertEqual(
            mcp_server.save_memory(self.memory, provider, "call me Al\nstill Al")["status"],
            "saved",
        )
        mcp_server.save_memory(
            self.memory,
            Extractor(triple("preferred_name", "Sam")),
            "call me Sam",
        )
        self.memory.save_exchange("Remember this: call me Al", "short preview")
        listed = self.memory.list_drawers(wing="tinytalk", room="conversations", limit=10)
        drawer_id = listed["drawers"][0]["drawer_id"]
        self.memory.update_drawer(
            drawer_id,
            content="User: Remember this: call me Al\nstill Al\n\nAssistant: short preview",
        )
        dropped = mcp_server.recall_thread(self.memory, "Remember this: call me Al")
        for excerpt in dropped["excerpts"]:
            self.assertNotIn("still Al", excerpt.get("text") or "")

    def test_list_sources_returns_the_stored_record(self):
        mcp_server.save_memory(
            self.memory,
            Extractor(triple("preferred_name", "Al")),
            "call me Al",
        )
        current = memory_core._room_drawers(self.memory, "facts")
        self.assertEqual(len(current), 1)
        current_id = current[0]["drawer_id"]
        found = mcp_server.list_sources(self.memory, current_id)
        self.assertTrue(found["found"])
        self.assertEqual(found["kind"], "saved fact")
        self.assertEqual(found["room"], "facts")
        self.assertEqual(found["status"], "facts")
        self.assertEqual(found["id"], current_id)
        self.assertEqual(found["text_preview"], memory_core._preview(current[0]["text"]))
        self.assertEqual(found["filed_at"], current[0]["metadata"]["filed_at"])

        mcp_server.save_memory(
            self.memory,
            Extractor(triple("preferred_name", "Sam")),
            "call me Sam",
        )
        history = memory_core._room_drawers(self.memory, "facts-history")
        self.assertEqual(len(history), 1)
        archived = mcp_server.list_sources(self.memory, history[0]["drawer_id"])
        self.assertEqual(archived["kind"], "archived fact")
        self.assertEqual(archived["room"], "facts-history")
        self.assertEqual(archived["id"], history[0]["drawer_id"])
        self.assertIn("call me Al", archived["text_preview"])
        self.assertEqual(archived["filed_at"], history[0]["metadata"]["filed_at"])

        mcp_server.sync_thread(
            self.memory,
            "Docking notes",
            [
                {"role": "user", "content": "The amber dock light is on"},
                {"role": "assistant", "content": "I noted the amber dock light."},
            ],
            "field-notes",
        )
        listed = self.memory.list_drawers(wing="tinytalk", room="conversations", limit=10)
        convo = listed["drawers"][0]
        record = mcp_server.list_sources(self.memory, convo["drawer_id"])
        self.assertEqual(record["kind"], "conversation excerpt")
        self.assertEqual(record["room"], "conversations")
        self.assertEqual(record["id"], convo["drawer_id"])
        self.assertEqual(record["filed_at"], convo["metadata"]["filed_at"])
        self.assertIn("amber dock light", record["text_preview"])

    def test_list_sources_reports_a_missing_drawer(self):
        missing = mcp_server.list_sources(self.memory, "no-such-drawer")
        self.assertEqual(missing, {"found": False, "message": "No drawer with that id."})
        self.assertEqual(
            mcp_server.list_sources(self.memory, ""),
            {"found": False, "message": "No drawer with that id."},
        )

    def test_transport_initialize_lists_six_tools_and_reads_profile(self):
        server = mcp_server.create_server(self.memory, None)

        async def session():
            async with Client(server, mode="legacy") as client:
                self.assertTrue(client.protocol_version)
                listed = await client.list_tools()
                self.assertEqual([tool.name for tool in listed.tools], list(mcp_server.TOOL_NAMES))
                result = await client.call_tool("get_current_profile", {})
                self.assertFalse(result.is_error)
                return result.structured_content

        stdout = io.StringIO()
        previous = os.environ.get("TINYTALK_DEBUG")
        os.environ["TINYTALK_DEBUG"] = "1"
        try:
            with redirect_stdout(stdout):
                payload = asyncio.run(session())
        finally:
            if previous is None:
                os.environ.pop("TINYTALK_DEBUG", None)
            else:
                os.environ["TINYTALK_DEBUG"] = previous
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(payload["result"], memory_core.read_current_profile(self.memory))

    def test_search_tool_does_not_print_debug_to_stdout(self):
        mcp_server.save_memory(
            self.memory,
            Extractor(triple("preferred_name", "Willow")),
            "call me Willow",
        )
        server = mcp_server.create_server(self.memory, None)

        async def session():
            async with Client(server, mode="legacy") as client:
                result = await client.call_tool("search_memories", {"query": "call me Willow"})
                self.assertFalse(result.is_error)
                return result.structured_content

        stdout = io.StringIO()
        stderr = io.StringIO()
        previous = os.environ.get("TINYTALK_DEBUG")
        os.environ["TINYTALK_DEBUG"] = "1"
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                payload = asyncio.run(session())
                guarded = mcp_server._call(_print_not_protocol)
        finally:
            if previous is None:
                os.environ.pop("TINYTALK_DEBUG", None)
            else:
                os.environ["TINYTALK_DEBUG"] = previous
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("[debug]", stdout.getvalue())
        self.assertIn("not-protocol", stderr.getvalue())
        self.assertEqual(guarded, {"ok": True})
        self.assertEqual(payload["profile"], memory_core.read_current_profile(self.memory))
        self.assertTrue(payload["semantic_facts"])
        self.assertEqual(payload["semantic_facts"][0]["label"], "semantic fact")

    def test_list_sources_uses_get_drawer_without_scanning_rooms(self):
        mcp_server.save_memory(
            self.memory,
            Extractor(triple("preferred_name", "Al")),
            "call me Al",
        )
        current = memory_core._room_drawers(self.memory, "facts")
        drawer_id = current[0]["drawer_id"]

        def fail_list_drawers(**kwargs):
            raise AssertionError("list_drawers should not be called when get_drawer exists")

        original_list_drawers = self.memory.list_drawers
        self.memory.list_drawers = fail_list_drawers
        try:
            found = mcp_server.list_sources(self.memory, drawer_id)
        finally:
            self.memory.list_drawers = original_list_drawers

        self.assertTrue(found["found"])
        self.assertEqual(found["id"], drawer_id)
        self.assertEqual(found["room"], "facts")
        self.assertEqual(found["kind"], "saved fact")

    def test_main_closes_kg_on_exit(self):
        from unittest.mock import patch

        closed = []

        class FakeKG(object):
            def close(self):
                closed.append(True)

        class FakeMemory(object):
            kg = FakeKG()

        class FakeServer(object):
            def run(self, transport="stdio"):
                return None

        with (
            patch.object(mcp_server, "load_dotenv"),
            patch.object(mcp_server, "build_provider", return_value=None),
            patch.object(mcp_server, "open_memory", return_value=FakeMemory()),
            patch.object(mcp_server, "create_server", return_value=FakeServer()),
        ):
            code = mcp_server.main()

        self.assertEqual(code, 0)
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
