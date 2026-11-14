"""Local stdio MCP server for TinyTalk memory.

The tool functions take the memory and provider explicitly so tests can call
them without speaking MCP. main opens one memory and registers thin wrappers.
"""

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

import memory_core
from memory_core import (
    ABANDONED_ROOM,
    FACT_ROOM,
    HISTORY_ROOM,
    PENDING_ROOM,
    open_memory,
    read_current_profile,
    remember_fact,
)
from provider import build_provider, load_settings
from tinytalk import load_dotenv

# save_exchange and search_conversations use this room name. There is no
# memory_core constant for it.
CONVERSATIONS_ROOM = "conversations"
SOURCE_ROOMS = (FACT_ROOM, HISTORY_ROOM, PENDING_ROOM, ABANDONED_ROOM, CONVERSATIONS_ROOM)

TOOL_NAMES = (
    "search_memories",
    "get_current_profile",
    "save_memory",
    "sync_thread",
    "recall_thread",
    "list_sources",
)


def _write_stderr(message):
    """Write a diagnostic line to stderr so stdout stays clean for MCP."""
    text = str(message)
    sys.stderr.write(text)
    if not text.endswith("\n"):
        sys.stderr.write("\n")


def _call(fn, *args, **kwargs):
    """Run a tool with debug off and any print sent to stderr."""
    captured = io.StringIO()
    with memory_core.scoped_io(output=_write_stderr, debug=False), redirect_stdout(captured):
        result = fn(*args, **kwargs)
    leaked = captured.getvalue()
    if leaked:
        _write_stderr(leaked)
    return result


def search_memories(memory, query):
    """Profile entries first, then semantic fact hits. The two lists stay separate."""
    profile = read_current_profile(memory)
    if not isinstance(query, str):
        return {
            "profile": profile,
            "semantic_facts": [],
            "error": "query must be a string.",
        }
    try:
        hits = [] if memory is None else memory.search_facts(query)
    except Exception:
        return {
            "profile": profile,
            "semantic_facts": [],
            "error": "Fact search failed.",
        }
    semantic_facts = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        source = memory_core._coerce_saved_fact(hit)
        if source.get("kind") == memory_core.UNCLASSIFIED_KIND:
            label = "unclassified saved text"
        else:
            label = "semantic fact"
        semantic_facts.append({
            "label": label,
            "similarity": hit.get("similarity"),
            "text": hit.get("text") or "",
        })
    return {"profile": profile, "semantic_facts": semantic_facts}


def get_current_profile(memory):
    """The current-profile entries, as read_current_profile stored them."""
    return read_current_profile(memory)


def save_memory(memory, provider, text):
    """Wrap text as a Remember this command and return remember_fact's result."""
    if not isinstance(text, str):
        return {"error": "text must be a string."}
    return remember_fact("Remember this: " + text, memory, provider)


def _message_pairs(messages):
    """Validated (role, content) pairs, or an error result."""
    if not isinstance(messages, list):
        return None, {"error": "messages must be a list of {role, content} objects."}
    parsed = []
    for item in messages:
        if not isinstance(item, dict):
            return None, {"error": "messages must be a list of {role, content} objects."}
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            return None, {
                "error": "Each message needs role 'user' or 'assistant' and string content.",
            }
        parsed.append((role, content))
    return parsed, None


def sync_thread(memory, title, messages, source_app):
    """Save consecutive user→assistant pairs. The first pair carries the thread note.

    A message that is not part of one of those pairs is skipped, including an
    unpaired trailing message. Invalid shapes return an error and save nothing.
    """
    if not isinstance(title, str) or not isinstance(source_app, str):
        return {"error": "title and source_app must be strings."}
    parsed, error = _message_pairs(messages)
    if error is not None:
        return error
    if (
        memory is None
        or getattr(memory, "palace", None) is None
        or not callable(getattr(memory, "save_exchange", None))
    ):
        return {"error": "Saved memory is unavailable.", "saved": 0, "skipped": 0}

    note = "Thread: %s (source: %s)" % (title, source_app)
    saved = 0
    skipped = 0
    index = 0
    while index < len(parsed):
        role, content = parsed[index]
        nxt = parsed[index + 1] if index + 1 < len(parsed) else None
        if role == "user" and nxt is not None and nxt[0] == "assistant":
            memory_note = note if saved == 0 else None
            try:
                memory.save_exchange(content, nxt[1], memory_note=memory_note)
            except Exception:
                return {
                    "error": "Could not save an exchange.",
                    "saved": saved,
                    "skipped": skipped,
                }
            saved += 1
            index += 2
        else:
            skipped += 1
            index += 1
    return {"saved": saved, "skipped": skipped}


def recall_thread(memory, query):
    """Historical excerpts, filtered the same way as tinytalk.recall_context."""
    if not isinstance(query, str):
        return {"error": "query must be a string.", "excerpts": []}
    if memory is None or getattr(memory, "palace", None) is None:
        return {"excerpts": []}
    try:
        found = memory.search_conversations(query)
    except Exception:
        found = []
    _unused, updates = memory_core.unfinished_memory(memory)
    retired = memory_core._retired_fact_texts(memory)
    excerpts = []
    for item in found:
        if not item or not memory_core._keep_conversation(item):
            continue
        hydrated = memory_core._hydrate_conversation(memory, item)
        if (
            memory_core._from_unfinished_command(hydrated, updates)
            or memory_core._from_retired_command(hydrated, retired)
        ):
            continue
        excerpts.append(memory_core._historical_source(hydrated))
    return {"excerpts": excerpts}


def _source_result(source, room):
    """One drawer, from a memory_core source record plus the room it was found in."""
    record = {
        "found": True,
        "kind": source.get("kind"),
        "room": room,
        "status": room,
        "text_preview": memory_core._preview(source.get("text") or ""),
        "id": source.get("id"),
    }
    if source.get("filed_at"):
        record["filed_at"] = source["filed_at"]
    return record


def _source_from_drawer(drawer, drawer_id):
    if not isinstance(drawer, dict) or drawer.get("error"):
        return None
    metadata = drawer.get("metadata") or {}
    wing = drawer.get("wing") or metadata.get("wing")
    if wing and wing != "tinytalk":
        return None
    room = drawer.get("room") or metadata.get("room")
    if room not in SOURCE_ROOMS:
        return None
    text = drawer.get("text")
    if text is None:
        text = drawer.get("content") or ""
    found_id = drawer.get("drawer_id") or drawer_id
    if room == CONVERSATIONS_ROOM:
        source = memory_core._coerce_conversation({
            "text": text,
            "drawer_id": found_id,
            "metadata": metadata,
        })
        return _source_result(source, room)
    if not text.strip():
        return None
    source = memory_core._coerce_saved_fact({
        "text": text,
        "drawer_id": found_id,
        "room": room,
        "metadata": metadata,
        "filed_at": metadata.get("filed_at"),
    })
    return _source_result(source, room)


def list_sources(memory, drawer_id):
    """The stored source for one drawer id, looked up across the five rooms.

    This is not the chat /sources list and it does not use session state.
    No match returns a not-found result.
    """
    if not isinstance(drawer_id, str) or not drawer_id:
        return {"found": False, "message": "No drawer with that id."}
    if memory is None:
        return {"found": False, "message": "Saved memory is unavailable."}
    get_drawer = getattr(memory, "get_drawer", None)
    if callable(get_drawer):
        try:
            drawer = get_drawer(drawer_id)
        except Exception:
            return {"found": False, "message": "The drawer could not be read."}
        result = _source_from_drawer(drawer, drawer_id)
        if result is not None:
            return result
        return {"found": False, "message": "No drawer with that id."}
    if getattr(memory, "list_drawers", None) is None:
        return {"found": False, "message": "Saved memory is unavailable."}
    try:
        for room in SOURCE_ROOMS:
            if room == CONVERSATIONS_ROOM:
                for drawer in memory_core._room_drawers(memory, room):
                    if str(drawer.get("drawer_id")) != drawer_id:
                        continue
                    source = memory_core._coerce_conversation({
                        "text": drawer.get("text") or "",
                        "drawer_id": drawer.get("drawer_id"),
                        "metadata": drawer.get("metadata") or {},
                    })
                    return _source_result(source, room)
                continue
            sources = memory_core._drawer_sources(memory, room)
            if sources is None:
                return {"found": False, "message": "Saved memory is unavailable."}
            for source in sources:
                if source.get("id") == drawer_id:
                    return _source_result(source, room)
    except Exception:
        return {"found": False, "message": "The drawer could not be read."}
    return {"found": False, "message": "No drawer with that id."}


def create_server(memory, provider):
    """Register the six tools as thin wrappers. Does not open a memory."""
    mcp = MCPServer("TinyTalk")

    @mcp.tool(name="search_memories")
    def search_memories_tool(query: str) -> dict[str, Any]:
        """Current-profile entries first, then separate semantic fact hits."""
        return _call(search_memories, memory, query)

    @mcp.tool(name="get_current_profile")
    def get_current_profile_tool() -> list[dict]:
        """The five current-profile entries, unchanged."""
        return _call(get_current_profile, memory)

    @mcp.tool(name="save_memory")
    def save_memory_tool(text: str) -> dict[str, Any]:
        """Save text with Remember this. Returns status and message unchanged."""
        result = _call(save_memory, memory, provider, text)
        if result is None:
            return {"status": None, "message": None}
        return result

    @mcp.tool(name="sync_thread")
    def sync_thread_tool(title: str, messages: list, source_app: str) -> dict[str, Any]:
        """Save user→assistant pairs from another app. The trailing unpaired message is skipped."""
        return _call(sync_thread, memory, title, messages, source_app)

    @mcp.tool(name="recall_thread")
    def recall_thread_tool(query: str) -> dict[str, Any]:
        """Historical conversation excerpts. Not current facts."""
        return _call(recall_thread, memory, query)

    @mcp.tool(name="list_sources")
    def list_sources_tool(drawer_id: str) -> dict[str, Any]:
        """Look up one stored drawer by id across the five memory rooms."""
        return _call(list_sources, memory, drawer_id)

    return mcp


def main():
    """Open one memory and serve it on stdio. Debug stays off."""
    load_dotenv(Path(__file__).with_name(".env"))
    provider = None
    try:
        provider = build_provider(load_settings(os.environ))
    except Exception as exc:
        sys.stderr.write(
            "TinyTalk MCP: provider unavailable (%s). "
            "The server will stay up; structured fact extraction is off.\n" % exc
        )
        provider = None
    captured = io.StringIO()
    with memory_core.scoped_io(output=_write_stderr, debug=False), redirect_stdout(captured):
        memory = open_memory()
    leaked = captured.getvalue()
    if leaked:
        sys.stderr.write(leaked)
        if not leaked.endswith("\n"):
            sys.stderr.write("\n")
    try:
        create_server(memory, provider).run(transport="stdio")
    finally:
        kg = getattr(memory, "kg", None)
        if kg is not None and callable(getattr(kg, "close", None)):
            kg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
