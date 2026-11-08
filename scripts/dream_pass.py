"""Offline consolidation check for TinyTalk facts.

Report mode is the default. It reads the facts room and writes nothing to the
palace. Apply mode (--apply FILE) is the only path that moves drawers, and
only ids listed in that file.

A duplicate cluster is facts-room drawers whose relationship is not a profile
predicate, including drawers saved with no predicate, joined by search_facts
hits at DUPLICATE_SIMILARITY or above. The suggested keeper is the latest
filed_at, then the longest text. Stale echoes are facts whose metadata names
a profile predicate and an object other than the current profile value. They
are reported and are not cleanup candidates. Orphaned as-written facts are
facts-room drawers with no relationship predicate in metadata.

search_facts returns at most three hits. A duplicate past that cutoff is missed.
"""

import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_core import (
    ABANDONED_ROOM,
    FACT_ROOM,
    HISTORY_ROOM,
    PROFILE_PREDICATES,
    _canon_predicate,
    _drawers_by_id,
    _fact_relationship,
    _move_drawer,
    _names_match,
    _room_drawers,
    open_memory,
    read_current_profile,
)

# Stricter than memory_core.MIN_FACT_SIMILARITY. search_facts already applies
# that 0.45 floor; clustering uses the score on each returned hit.
DUPLICATE_SIMILARITY = 0.90

_MOVABLE_ROOMS = (FACT_ROOM, HISTORY_ROOM)


def _call_quietly(function, *args):
    """Run function with debug off. Incidental prints go to stderr."""
    previous = os.environ.get("TINYTALK_DEBUG")
    os.environ["TINYTALK_DEBUG"] = ""
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            return function(*args)
    finally:
        if previous is None:
            os.environ.pop("TINYTALK_DEBUG", None)
        else:
            os.environ["TINYTALK_DEBUG"] = previous
        leaked = captured.getvalue()
        if leaked:
            sys.stderr.write(leaked)
            if not leaked.endswith("\n"):
                sys.stderr.write("\n")


def _filed_at(drawer):
    metadata = drawer.get("metadata") or {}
    filed = metadata.get("filed_at")
    if filed is None:
        filed = drawer.get("filed_at")
    if filed is None:
        return ""
    return str(filed)


def _relationship(drawer):
    return _fact_relationship(drawer.get("metadata"))


def _profile_predicate(drawer):
    """Canonical profile predicate on this drawer, or None."""
    predicate, _obj = _relationship(drawer)
    if not predicate:
        return None
    canonical = _canon_predicate(predicate)
    if canonical in PROFILE_PREDICATES:
        return canonical
    return None


def _duplicate_candidate(drawer):
    """Profile predicates are out of scope. Everything else in facts can cluster."""
    return _profile_predicate(drawer) is None


def _hit_similarity(hit):
    if not isinstance(hit, dict):
        return None
    try:
        return float(hit.get("similarity"))
    except (TypeError, ValueError):
        return None


def _keeper_rank(member):
    """Latest filed_at, then longest text, then drawer id for a stable tie."""
    return (member["filed_at"], len(member["text"]), member["drawer_id"])


def find_duplicate_clusters(memory):
    """Connected facts-room candidates at DUPLICATE_SIMILARITY or above.

    Each member is searched with search_facts. A hit joins the cluster when
    its similarity clears DUPLICATE_SIMILARITY and its drawer is a candidate.
    A single drawer is not a cluster. Members are ordered with the suggested
    keeper first.
    """
    by_id = {}
    for drawer in _room_drawers(memory, FACT_ROOM):
        drawer_id = drawer.get("drawer_id")
        if drawer_id and _duplicate_candidate(drawer):
            by_id[drawer_id] = drawer
    parent = {drawer_id: drawer_id for drawer_id in by_id}

    def find(drawer_id):
        root = drawer_id
        while parent[root] != root:
            root = parent[root]
        while parent[drawer_id] != root:
            parent[drawer_id], drawer_id = root, parent[drawer_id]
        return root

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for drawer_id, drawer in by_id.items():
        text = drawer.get("text") or ""
        if not text:
            continue
        for hit in memory.search_facts(text) or []:
            score = _hit_similarity(hit)
            if score is None or score < DUPLICATE_SIMILARITY:
                continue
            other = hit.get("drawer_id")
            if other in by_id:
                union(drawer_id, other)

    groups = {}
    for drawer_id in by_id:
        groups.setdefault(find(drawer_id), []).append(by_id[drawer_id])

    clusters = []
    for drawers in groups.values():
        if len(drawers) < 2:
            continue
        members = [
            {
                "drawer_id": drawer["drawer_id"],
                "filed_at": _filed_at(drawer),
                "text": drawer.get("text") or "",
            }
            for drawer in drawers
        ]
        members.sort(key=_keeper_rank, reverse=True)
        clusters.append({
            "keeper_id": members[0]["drawer_id"],
            "members": members,
        })
    clusters.sort(
        key=lambda cluster: (_keeper_rank(cluster["members"][0]), cluster["keeper_id"]),
        reverse=True,
    )
    return clusters


def find_stale_echoes(memory):
    """Facts whose profile predicate disagrees with the current profile value.

    Only a profile entry in state "current" supplies that value. These rows are
    anomalies to report. Apply mode still refuses them.
    """
    current = {}
    for entry in read_current_profile(memory):
        if entry.get("state") != "current":
            continue
        value = entry.get("value")
        if isinstance(value, str) and value.strip():
            current[entry["predicate"]] = value

    echoes = []
    for drawer in _room_drawers(memory, FACT_ROOM):
        predicate = _profile_predicate(drawer)
        if predicate is None or predicate not in current:
            continue
        _stored, obj = _relationship(drawer)
        if _names_match(obj, current[predicate]):
            continue
        echoes.append({
            "drawer_id": drawer.get("drawer_id"),
            "filed_at": _filed_at(drawer),
            "text": drawer.get("text") or "",
            "predicate": predicate,
            "drawer_value": obj or "",
            "current_value": current[predicate],
        })
    echoes.sort(key=lambda echo: echo["drawer_id"] or "")
    return echoes


def find_orphaned_as_written(memory):
    """Facts-room drawers with no relationship predicate in metadata."""
    orphans = []
    for drawer in _room_drawers(memory, FACT_ROOM):
        predicate, _obj = _relationship(drawer)
        if predicate:
            continue
        orphans.append({
            "drawer_id": drawer.get("drawer_id"),
            "text": drawer.get("text") or "",
        })
    orphans.sort(key=lambda orphan: orphan["drawer_id"] or "")
    return orphans


def _field_text(text):
    parts = (text or "").splitlines() or [""]
    if len(parts) == 1:
        return parts[0]
    return "\n" + "\n".join("    " + part for part in parts)


def _section(title, body):
    lines = ["## %s" % title, ""]
    if body:
        lines.extend(body)
    else:
        lines.append("none found")
    lines.append("")
    return lines


def _cluster_lines(clusters):
    lines = []
    for index, cluster in enumerate(clusters, 1):
        if lines:
            lines.append("")
        lines.append("### Cluster %d" % index)
        for member in cluster["members"]:
            label = "- `%s`" % member["drawer_id"]
            if member["drawer_id"] == cluster["keeper_id"]:
                label += " - suggested keeper"
            lines.append(label)
            lines.append("  - filed_at: %s" % (member["filed_at"] or "(none)"))
            lines.append("  - text: %s" % _field_text(member["text"]))
    return lines


def _echo_lines(echoes):
    lines = []
    for echo in echoes:
        if lines:
            lines.append("")
        lines.append("- `%s`" % echo["drawer_id"])
        lines.append("  - predicate: %s" % echo["predicate"])
        lines.append('  - drawer value: "%s"' % echo["drawer_value"])
        lines.append('  - current value: "%s"' % echo["current_value"])
        lines.append("  - text: %s" % _field_text(echo["text"]))
    return lines


def _orphan_lines(orphans):
    lines = []
    for orphan in orphans:
        lines.append("- `%s`" % orphan["drawer_id"])
        lines.append("  - text: %s" % _field_text(orphan["text"]))
    return lines


def format_report(clusters, echoes, orphans):
    """Markdown report. Every section is present. Empty sections say none found."""
    lines = ["# Dream pass", ""]
    lines.extend(_section("Duplicate clusters", _cluster_lines(clusters)))
    lines.extend(_section("Stale echoes", _echo_lines(echoes)))
    lines.extend(_section("Orphaned as-written facts", _orphan_lines(orphans)))
    lines.append(
        "Summary: duplicate clusters: %d; stale echoes: %d; orphaned as-written facts: %d"
        % (len(clusters), len(echoes), len(orphans))
    )
    return "\n".join(lines) + "\n"


def _render(memory):
    return format_report(
        find_duplicate_clusters(memory),
        find_stale_echoes(memory),
        find_orphaned_as_written(memory),
    )


def render_report(memory):
    """The dream-pass report. Does not move drawers or write the palace."""
    return _call_quietly(_render, memory)


def parse_approvals(text):
    """Drawer ids, one per line. Blank lines and # comments are ignored."""
    ids = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.append(stripped)
    return ids


def _apply_one(memory, drawer_id):
    """One approval. Returns (line, move_failed). A refusal is not a failure."""
    try:
        drawers = _drawers_by_id(memory)
    except Exception:
        return ("FAILED %s: could not read drawers" % drawer_id, True)
    drawer = drawers.get(drawer_id)
    if drawer is None:
        return ("REFUSED %s: unknown drawer" % drawer_id, False)
    room = drawer.get("room")
    if room not in _MOVABLE_ROOMS:
        return (
            "REFUSED %s: not in facts or facts-history (currently %s)"
            % (drawer_id, room),
            False,
        )
    predicate = _profile_predicate(drawer)
    if predicate:
        return ("REFUSED %s: profile predicate %s" % (drawer_id, predicate), False)
    if _move_drawer(memory, drawer_id, ABANDONED_ROOM):
        return (
            "%s %s moved to %s" % (drawer_id, room, ABANDONED_ROOM),
            False,
        )
    return (
        "FAILED %s: could not move from %s to %s" % (drawer_id, room, ABANDONED_ROOM),
        True,
    )


def _apply_ids(memory, drawer_ids):
    lines = []
    ok = True
    for drawer_id in drawer_ids:
        line, failed = _apply_one(memory, drawer_id)
        lines.append(line)
        if failed:
            ok = False
    return lines, ok


def apply_approvals(memory, drawer_ids):
    """Move approved ids to facts-abandoned, in file order.

    Unknown ids, drawers outside facts and facts-history, and profile
    predicates are refused. One refusal or failed move does not stop the
    rest. The boolean is True when every non-refused move succeeded.
    """
    return _call_quietly(_apply_ids, memory, drawer_ids)


def _read_text(path, label):
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        sys.stderr.write("Could not read %s: %s\n" % (label, exc))
        return None


def _open_default_memory():
    captured = io.StringIO()
    with redirect_stdout(captured):
        memory = open_memory()
    leaked = captured.getvalue()
    if leaked:
        sys.stderr.write(leaked)
        if not leaked.endswith("\n"):
            sys.stderr.write("\n")
    return memory


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Report duplicate and stale TinyTalk facts. "
        "Apply mode retires only approved drawer ids.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Write the report to PATH instead of stdout.",
    )
    parser.add_argument(
        "--apply",
        metavar="APPROVALS_FILE",
        help="Move listed drawer ids to facts-abandoned.",
    )
    return parser.parse_args(argv)


def main(argv=None, memory=None):
    """Report to stdout or --out. --apply APPROVALS_FILE retires those ids."""
    args = _parse_args(argv)
    if args.apply and args.out:
        sys.stderr.write(
            "Use --out to write a report, or --apply to retire drawers, not both.\n"
        )
        return 2
    if memory is None:
        memory = _open_default_memory()
    if args.apply:
        text = _read_text(args.apply, "the approvals file")
        if text is None:
            return 1
        lines, ok = apply_approvals(memory, parse_approvals(text))
        for line in lines:
            print(line)
        return 0 if ok else 1
    report = render_report(memory)
    if args.out:
        try:
            Path(args.out).write_text(report, encoding="utf-8")
        except OSError as exc:
            sys.stderr.write("Could not write the report: %s\n" % exc)
            return 1
        return 0
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
