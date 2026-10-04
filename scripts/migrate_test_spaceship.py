"""Move the legacy Serenity test-spaceship fact into history and canonicalize
the Enterprise graph edge.

This is a one-time data migration. It does not load .env, call a model, delete
history, or rewrite conversation drawers. Dry-run is the default. Apply writes
only the Serenity room change, the legacy triple's valid_to, and one new
current triple.

MemPalace's update_drawer stamps last_modified. After that call, the script
writes the previous last_modified back so the saved drawer metadata matches
the original record except for room.
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinytalk

SERENITY_ID = "drawer_tinytalk_facts_90939b23152b1c735929a70d"
SERENITY_TEXT = "my test spaceship is named Serenity."
ENTERPRISE_ID = "drawer_tinytalk_facts_72a37b3c35227d88fe604275"
ENTERPRISE_TEXT = "my test spaceship is named Enterprise"
PICKLE_ID = "drawer_tinytalk_facts_2f78771e8901a937ce50555d"
PICKLE_TEXT = "the name of my imaginary spaceship is Picklewagon"
LEGACY_ID = "t_user_test_spaceship_enterprise_842d9d8544ac"
LEGACY_PREDICATE = "test_spaceship"
CANONICAL_PREDICATE = "test_spaceship_name"
SPACESHIP_TOKENS = ("spaceship", "serenity", "enterprise", "picklewagon")


class MigrationError(Exception):
    """The live records do not match a known safe state. Nothing was written."""


class MigrationWriteError(Exception):
    """A write failed or did not land as expected."""

    def __init__(self, message, completed):
        super(MigrationWriteError, self).__init__(message)
        self.completed = completed


def _abspath(path):
    return os.path.abspath(os.path.expanduser(path))


def real_store_paths():
    from mempalace.config import MempalaceConfig
    from mempalace.knowledge_graph import DEFAULT_KG_PATH

    return _abspath(MempalaceConfig().palace_path), _abspath(DEFAULT_KG_PATH)


def _meta_value(string_value, int_value, float_value, bool_value):
    if string_value is not None:
        return string_value
    if int_value is not None:
        return int_value
    if float_value is not None:
        return float_value
    if bool_value is not None:
        return bool(bool_value)
    return None


def read_drawers(palace_path, read_only=True):
    """Read drawer text and metadata from Chroma's SQLite file.

    Opening the Chroma client rewrites index files even for a read, so the
    migration inspects and verifies through SQLite instead. ``read_only`` is
    accepted so callers can say which pass this is; both passes use a
    query-only connection.
    """
    del read_only
    db_path = Path(palace_path) / "chroma.sqlite3"
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA query_only = ON")
    try:
        rows = connection.execute(
            """
            SELECT e.embedding_id, em.key, em.string_value, em.int_value,
                   em.float_value, em.bool_value
            FROM embeddings e
            JOIN embedding_metadata em ON em.id = e.id
            ORDER BY e.embedding_id, em.key
            """
        ).fetchall()
    finally:
        connection.close()
    drawers = {}
    for embedding_id, key, string_value, int_value, float_value, bool_value in rows:
        drawer = drawers.setdefault(embedding_id, {"content": "", "metadata": {}})
        value = _meta_value(string_value, int_value, float_value, bool_value)
        if key == "chroma:document":
            drawer["content"] = value or ""
        else:
            drawer["metadata"][key] = value
    return drawers


def read_triples(kg_path):
    # A WAL database has no durable sidecar until a connection creates one.
    # query_only blocks writes. Opening it can create an empty -wal/-shm
    # pair; it does not rewrite triple rows.
    connection = sqlite3.connect(str(kg_path))
    connection.execute("PRAGMA query_only = ON")
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT t.id, t.subject, t.predicate, t.object, t.valid_from, t.valid_to,
                   t.confidence, t.source_closet, t.source_file, t.source_drawer_id,
                   t.adapter_name, t.extracted_at,
                   s.name AS subject_name, o.name AS object_name
            FROM triples t
            JOIN entities s ON s.id = t.subject
            JOIN entities o ON o.id = t.object
            ORDER BY t.id
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def _is_spaceship(row):
    blob = " ".join(
        str(row.get(key) or "")
        for key in ("subject", "subject_name", "predicate", "object", "object_name")
    ).lower()
    return any(token in blob for token in SPACESHIP_TOKENS)


def _same_legacy_identity(row):
    return (
        row["id"] == LEGACY_ID
        and row["subject_name"] == "user"
        and row["predicate"] == LEGACY_PREDICATE
        and row["object_name"] == "Enterprise"
        and row["valid_from"] is None
        and row["confidence"] == 1.0
        and row["source_closet"] is None
        and row["source_file"] == "tinytalk"
        and row["source_drawer_id"] is None
        and row["adapter_name"] is None
    )


def _is_open_canonical(row):
    return (
        row["subject_name"] == "user"
        and row["predicate"] == CANONICAL_PREDICATE
        and row["object_name"] == "Enterprise"
        and row["valid_to"] is None
        and row["valid_from"] is None
        and row["source_file"] == "tinytalk"
    )


def inspect(palace_path, kg_path, read_only=True):
    drawers = read_drawers(palace_path, read_only=read_only)
    triples = read_triples(kg_path)
    problems = []

    def require_drawer(drawer_id, text, allowed_rooms):
        drawer = drawers.get(drawer_id)
        if drawer is None:
            problems.append("%s is missing" % drawer_id)
            return None
        if drawer["content"] != text:
            problems.append("%s content does not match the expected fact" % drawer_id)
            return None
        room = (drawer["metadata"] or {}).get("room")
        if room not in allowed_rooms:
            problems.append("%s is in room %r" % (drawer_id, room))
            return None
        copies = [
            other_id
            for other_id, other in drawers.items()
            if other_id != drawer_id and other["content"] == text
        ]
        if copies:
            problems.append("duplicate content for %s: %s" % (drawer_id, ", ".join(copies)))
        return drawer

    serenity = require_drawer(SERENITY_ID, SERENITY_TEXT, ("facts", "facts-history"))
    enterprise = require_drawer(ENTERPRISE_ID, ENTERPRISE_TEXT, ("facts",))
    pickle = require_drawer(PICKLE_ID, PICKLE_TEXT, ("facts",))

    for drawer_id, drawer in drawers.items():
        if drawer_id in (SERENITY_ID, ENTERPRISE_ID, PICKLE_ID):
            continue
        room = (drawer["metadata"] or {}).get("room")
        if room in ("facts", "facts-history") and "test spaceship" in drawer["content"].lower():
            problems.append("unexpected test-spaceship fact %s in %s" % (drawer_id, room))

    spaceship = [row for row in triples if _is_spaceship(row)]
    legacy = [row for row in spaceship if row["id"] == LEGACY_ID]
    canonical = [
        row
        for row in spaceship
        if row["predicate"] == CANONICAL_PREDICATE and row["object_name"] == "Enterprise"
    ]
    other = [row for row in spaceship if row not in legacy and row not in canonical]
    if other:
        problems.append(
            "unexpected spaceship graph rows: %s"
            % ", ".join(row["id"] for row in other)
        )
    if len(legacy) != 1 or not _same_legacy_identity(legacy[0] if legacy else {}):
        problems.append("legacy graph row %s is missing or does not match" % LEGACY_ID)
        legacy_row = None
    else:
        legacy_row = legacy[0]
    open_canonical = [row for row in canonical if row["valid_to"] is None]
    closed_canonical = [row for row in canonical if row["valid_to"] is not None]
    if closed_canonical:
        problems.append("canonical graph row is already closed")
    if len(open_canonical) > 1:
        problems.append("more than one current canonical spaceship triple")
    for row in open_canonical:
        if not _is_open_canonical(row):
            problems.append("current canonical triple does not match Enterprise")

    if problems:
        raise MigrationError("; ".join(problems))

    actions = []
    if serenity["metadata"].get("room") == "facts":
        actions.append("move_serenity_to_facts_history")
    if legacy_row["valid_to"] is None:
        actions.append("invalidate_legacy_test_spaceship")
    if not open_canonical:
        actions.append("add_canonical_test_spaceship_name")

    return {
        "drawers": drawers,
        "triples": triples,
        "legacy": legacy_row,
        "canonical": open_canonical[0] if open_canonical else None,
        "actions": actions,
        "enterprise_room": enterprise["metadata"].get("room"),
        "pickle_room": pickle["metadata"].get("room"),
        "serenity_room": serenity["metadata"].get("room"),
    }


def _release_chroma(mcp, palace_path):
    """Drop the MCP Chroma client so a later SQLite read is not locked."""
    collection = getattr(mcp, "_collection_cache", None)
    backend = getattr(collection, "_backend", None) if collection is not None else None
    if backend is not None:
        try:
            backend.close_palace(palace_path)
        except Exception:
            pass
    for name in ("_collection_cache", "_client_cache", "_collection_cache_palace"):
        if hasattr(mcp, name):
            setattr(mcp, name, None)


def _restore_serenity_metadata(collection, original):
    """Put back every metadata field except the room update_drawer changed."""
    restored = dict(original)
    restored["room"] = "facts-history"
    collection.update(ids=[SERENITY_ID], metadatas=[restored])


def _legacy_row_after(kg_path):
    rows = [row for row in read_triples(kg_path) if row["id"] == LEGACY_ID]
    if len(rows) != 1:
        return None
    return rows[0]


def apply_actions(palace_path, kg_path, state):
    completed = []
    mcp = None
    kg = None
    try:
        if "move_serenity_to_facts_history" in state["actions"]:
            original = state["drawers"][SERENITY_ID]["metadata"]
            mcp = tinytalk._import_memory_tools()
            tinytalk._point_tools_at_palace(mcp, palace_path)
            saved = mcp.tool_update_drawer(SERENITY_ID, room="facts-history")
            if not isinstance(saved, dict) or not saved.get("success"):
                raise MigrationWriteError(
                    "update_drawer failed: %s" % (saved,),
                    completed,
                )
            try:
                _restore_serenity_metadata(mcp._get_collection(), original)
            except Exception as exc:
                raise MigrationWriteError(
                    "room changed, but original metadata could not be restored: %s" % exc,
                    ["move_serenity_room_only"],
                )
            completed.append("move_serenity_to_facts_history")
        if (
            "invalidate_legacy_test_spaceship" in state["actions"]
            or "add_canonical_test_spaceship_name" in state["actions"]
        ):
            from mempalace.knowledge_graph import KnowledgeGraph

            # Release Chroma before opening the graph so the two stores are
            # not held together if the graph write fails.
            if mcp is not None:
                _release_chroma(mcp, palace_path)
                mcp = None
            kg = KnowledgeGraph(db_path=kg_path)
        if "invalidate_legacy_test_spaceship" in state["actions"]:
            before = dict(state["legacy"])
            kg.invalidate("user", LEGACY_PREDICATE, "Enterprise")
            after = _legacy_row_after(kg_path)
            comparable = dict(before)
            comparable["valid_to"] = after["valid_to"] if after else None
            if after is None or after != comparable or not after["valid_to"]:
                raise MigrationWriteError(
                    "legacy triple was not closed cleanly",
                    completed,
                )
            completed.append({
                "action": "invalidate_legacy_test_spaceship",
                "id": LEGACY_ID,
                "valid_to": after["valid_to"],
            })
        if "add_canonical_test_spaceship_name" in state["actions"]:
            new_id = kg.add_triple(
                "user",
                CANONICAL_PREDICATE,
                "Enterprise",
                source_file="tinytalk",
            )
            rows = [
                row for row in read_triples(kg_path)
                if row["id"] == new_id or _is_open_canonical(row)
            ]
            open_rows = [row for row in rows if _is_open_canonical(row)]
            if len(open_rows) != 1:
                raise MigrationWriteError(
                    "canonical triple was not added as the single current edge",
                    completed,
                )
            completed.append({
                "action": "add_canonical_test_spaceship_name",
                "id": open_rows[0]["id"],
                "valid_from": open_rows[0]["valid_from"],
            })
    except MigrationWriteError:
        raise
    except Exception as exc:
        raise MigrationWriteError("%s: %s" % (type(exc).__name__, exc), completed)
    finally:
        if kg is not None:
            kg.close()
        if mcp is not None:
            _release_chroma(mcp, palace_path)
    return completed


def public_report(state, completed=None):
    return {
        "actions": state["actions"],
        "serenity": {
            "id": SERENITY_ID,
            "room": state["serenity_room"],
            "text": SERENITY_TEXT,
        },
        "enterprise": {
            "id": ENTERPRISE_ID,
            "room": state["enterprise_room"],
            "text": ENTERPRISE_TEXT,
        },
        "picklewagon": {
            "id": PICKLE_ID,
            "room": state["pickle_room"],
            "text": PICKLE_TEXT,
        },
        "legacy": {
            "id": state["legacy"]["id"],
            "predicate": state["legacy"]["predicate"],
            "object": state["legacy"]["object_name"],
            "valid_from": state["legacy"]["valid_from"],
            "valid_to": state["legacy"]["valid_to"],
            "current": state["legacy"]["valid_to"] is None,
        },
        "canonical": None
        if state["canonical"] is None
        else {
            "id": state["canonical"]["id"],
            "predicate": state["canonical"]["predicate"],
            "object": state["canonical"]["object_name"],
            "valid_from": state["canonical"]["valid_from"],
            "valid_to": state["canonical"]["valid_to"],
            "current": True,
        },
        "drawer_count": len(state["drawers"]),
        "triple_count": len(state["triples"]),
        "completed": completed or [],
    }


def confirm_goal(state):
    """Structural check used after apply. Raises MigrationError on mismatch."""
    problems = []
    if state["actions"]:
        problems.append("actions still pending: %s" % ", ".join(state["actions"]))
    if state["serenity_room"] != "facts-history":
        problems.append("Serenity room is %s" % state["serenity_room"])
    if state["enterprise_room"] != "facts":
        problems.append("Enterprise room is %s" % state["enterprise_room"])
    if state["pickle_room"] != "facts":
        problems.append("Picklewagon room is %s" % state["pickle_room"])
    active = [
        row for row in state["triples"]
        if row["valid_to"] is None and _is_spaceship(row)
    ]
    if len(active) != 1 or not _is_open_canonical(active[0] if active else {}):
        problems.append("active spaceship graph is not the single canonical Enterprise edge")
    if state["legacy"]["valid_to"] is None or state["legacy"]["predicate"] != LEGACY_PREDICATE:
        problems.append("legacy graph edge is not inactive history")
    if problems:
        raise MigrationError("; ".join(problems))


def _backup_is_present(backup_dir):
    root = Path(backup_dir)
    palace = root / "palace" / "chroma.sqlite3"
    graph = root / "knowledge_graph.sqlite3"
    if not palace.is_file() or not graph.is_file():
        return False
    for path in (palace, graph):
        connection = sqlite3.connect(str(path))
        try:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                return False
        finally:
            connection.close()
    return True


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palace", required=True)
    parser.add_argument("--kg", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir")
    args = parser.parse_args(argv)

    palace_path = _abspath(args.palace)
    kg_path = _abspath(args.kg)
    real_palace, real_kg = real_store_paths()
    targets_real = palace_path == real_palace or kg_path == real_kg
    if args.apply and targets_real:
        if not args.backup_dir or not _backup_is_present(args.backup_dir):
            print(json.dumps({
                "ok": False,
                "error": "Refusing to apply to the real store without a verified backup directory.",
            }))
            return 2

    mode = "apply" if args.apply else "dry-run"
    try:
        state = inspect(palace_path, kg_path, read_only=True)
    except MigrationError as exc:
        print(json.dumps({"ok": False, "mode": mode, "error": str(exc)}))
        return 2

    if not args.apply:
        print(json.dumps({"ok": True, "mode": "dry-run", "report": public_report(state)}, indent=2))
        return 0

    if not state["actions"]:
        try:
            confirm_goal(state)
        except MigrationError as exc:
            print(json.dumps({"ok": False, "mode": "apply", "error": str(exc)}))
            return 2
        print(json.dumps({
            "ok": True,
            "mode": "apply",
            "changed": False,
            "report": public_report(state),
        }, indent=2))
        return 0

    try:
        again = inspect(palace_path, kg_path, read_only=True)
    except MigrationError as exc:
        print(json.dumps({"ok": False, "mode": "apply", "error": str(exc), "completed": []}))
        return 2
    if again["actions"] != state["actions"]:
        print(json.dumps({
            "ok": False,
            "mode": "apply",
            "error": "records changed between planning and apply",
            "completed": [],
        }))
        return 2

    try:
        completed = apply_actions(palace_path, kg_path, again)
        after = inspect(palace_path, kg_path, read_only=True)
        confirm_goal(after)
    except MigrationWriteError as exc:
        print(json.dumps({
            "ok": False,
            "mode": "apply",
            "error": str(exc),
            "completed": exc.completed,
            "recovery": (
                "TinyTalk must stay stopped. Move the current palace directory "
                "and knowledge_graph.sqlite3 aside, then copy palace/ and "
                "knowledge_graph.sqlite3 back from the backup directory. "
                "Do not delete the backup."
            ),
            "backup_dir": args.backup_dir,
        }, indent=2))
        return 1
    except MigrationError as exc:
        print(json.dumps({
            "ok": False,
            "mode": "apply",
            "error": "verification failed after writes: %s" % exc,
            "completed": completed,
            "recovery": (
                "TinyTalk must stay stopped. Move the current palace directory "
                "and knowledge_graph.sqlite3 aside, then copy palace/ and "
                "knowledge_graph.sqlite3 back from the backup directory. "
                "Do not delete the backup."
            ),
            "backup_dir": args.backup_dir,
        }, indent=2))
        return 1

    print(json.dumps({
        "ok": True,
        "mode": "apply",
        "changed": True,
        "report": public_report(after, completed),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
