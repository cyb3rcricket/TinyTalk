import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from provider import ConfigError, ProviderError, build_provider, load_settings
from speech import Afplay, SpeechController, VoiceStudioSpeech, command_kind, speech_settings

MAX_TURNS = 10
# Characters of text in one outgoing request: soul, memory, history, and the
# new message. REPLY_RESERVE_CHARS of that total is left for the reply.
MAX_REQUEST_CHARS = 24000
REPLY_RESERVE_CHARS = 2000
MIN_FACT_SIMILARITY = 0.45  # initial test value

# Shared limits for every memory block added to one request.
MEMORY_LIMITS = (
    "This is not a complete inventory of conversation storage. "
    "Do not claim that other memories or conversations do not exist "
    "merely because they were not included. "
    "Describe retrieved user facts as saved memories, not as information "
    "learned during model training. "
    "Call each included record a retrieved source, not training data."
)

# Exact questions that ask for the whole fact list, not one similar fact.
MEMORY_INTROSPECTION_QUESTIONS = {
    "what do you remember about me",
    "what do you know about me",
    "what have i told you",
    "what facts do you remember about me",
}

# Only these predicates have one current value. Everything else stays additive.
SINGLE_VALUE_PREDICATES = {
    "test_spaceship_name",
    "favorite_vehicle",
    "preferred_name",
    "home_city",
    "current_job",
}

# Current facts, replaced facts, facts whose update did not finish, and
# unfinished updates that a later command for the same relationship replaced.
FACT_ROOM = "facts"
HISTORY_ROOM = "facts-history"
PENDING_ROOM = "facts-pending"
ABANDONED_ROOM = "facts-abandoned"

PREDICATE_ALIASES = {
    "test_spaceship": "test_spaceship_name",
    "spaceship_name": "test_spaceship_name",
}

# Closed set. Ordinary "what is my test spaceship called?" must not match.
# "" means the previous name of the current test spaceship.
TEST_SPACESHIP_HISTORY_QUESTIONS = {
    "what was my test spaceship called before enterprise": "enterprise",
    "what did i call my test spaceship before enterprise": "enterprise",
    "what was the previous name of my test spaceship": "",
    "what was my test spaceship's previous name": "",
    "what was the old name of my test spaceship": "",
    "what was my test spaceship named previously": "",
}

_TEST_SPACESHIP_NAME = re.compile(
    r"\btest spaceship is (?:named|called)\s+([^.\n?!]+)",
    re.IGNORECASE,
)

TRIPLE_INSTRUCTIONS = (
    "Turn one fact into JSON with exactly three string keys: "
    "subject, predicate, object. "
    "If the fact is clearly about the user, set subject to user. "
    "Make predicate simple snake_case, like favorite_vehicle. "
    "Keep object short and literal. "
    "If you are not confident there is one clear triple, "
    'return {"subject":"","predicate":"","object":""}. '
    "Reply with JSON only. "
    'Example: "my favorite vehicle is a CyberTruck" -> '
    '{"subject":"user","predicate":"favorite_vehicle","object":"CyberTruck"}'
)


def load_dotenv(path):
    """Load KEY=VALUE lines that are not already set. Ignores a missing file."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def snake_predicate(text):
    words = []
    current = []
    for ch in text.strip().lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return "_".join(words)


def is_memory_introspection(text):
    return _normalize_question(text) in MEMORY_INTROSPECTION_QUESTIONS


def _normalize_question(text):
    """Lowercase, trim, and treat curly apostrophes as straight ones."""
    cleaned = (text or "").strip().lower()
    for apostrophe in ("\u2019", "\u2018", "\u02bc"):
        cleaned = cleaned.replace(apostrophe, "'")
    return " ".join(cleaned.rstrip("?.!").split())


def test_spaceship_history_question(text):
    """Return the name a previous-name question asks about, or None.

    "" means the previous name of the current test spaceship. A non-empty
    value is the later name in "called before <name>". None means this is
    not one of those questions.
    """
    normalized = _normalize_question(text)
    if normalized not in TEST_SPACESHIP_HISTORY_QUESTIONS:
        return None
    return TEST_SPACESHIP_HISTORY_QUESTIONS[normalized]


def _names_match(left, right):
    return (left or "").strip().casefold() == (right or "").strip().casefold()


def test_spaceship_fact_name(text):
    """Name from a test-spaceship fact. Other spaceship facts do not match."""
    if not text or "test spaceship" not in text.lower():
        return None
    match = _TEST_SPACESHIP_NAME.search(text)
    if not match:
        return None
    name = match.group(1).strip(" \t\"'")
    return name or None


def _instant(value):
    """Parse a full ISO datetime. A date with no time does not order events."""
    if not isinstance(value, str) or "T" not in value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _canon_predicate(predicate):
    cleaned = snake_predicate(predicate or "")
    return PREDICATE_ALIASES.get(cleaned, cleaned)


def _history_edges(graph_rows):
    edges = []
    for row in graph_rows or []:
        if not isinstance(row, dict):
            continue
        subject = row.get("subject")
        if subject not in (None, "", "user"):
            continue
        stored = row.get("predicate") or ""
        predicate = _canon_predicate(stored)
        if predicate != "test_spaceship_name":
            continue
        obj = row.get("object")
        if not isinstance(obj, str) or not obj.strip():
            continue
        if "current" in row and row.get("current") is not None:
            current = bool(row.get("current"))
        else:
            current = not row.get("valid_to")
        edges.append({
            "id": row.get("id"),
            "stored_predicate": stored,
            "predicate": predicate,
            "object": obj.strip(),
            "current": current,
            "valid_from": row.get("valid_from"),
            "valid_to": row.get("valid_to"),
        })
    return edges


def _history_facts(drawers):
    facts = []
    for item in drawers or []:
        if not isinstance(item, dict):
            continue
        room = item.get("room") or (item.get("metadata") or {}).get("room")
        if room not in ("facts", "facts-history"):
            continue
        text = item.get("content_preview")
        if text is None:
            text = item.get("text") or ""
        text = text.strip()
        name = test_spaceship_fact_name(text)
        if not name:
            continue
        filed_at = (item.get("metadata") or {}).get("filed_at")
        if filed_at is None:
            filed_at = item.get("filed_at")
        facts.append({
            "id": item.get("drawer_id"),
            "room": room,
            "text": text,
            "name": name,
            "filed_at": filed_at,
        })
    return facts


def _unique_name(names):
    found = []
    for name in names:
        if name and not any(_names_match(name, earlier) for earlier in found):
            found.append(name)
    if len(found) == 1:
        return found[0]
    return None


def _graph_instant(value):
    """UTC instant for a graph boundary. Naive filing times and date-only values are unusable."""
    parsed = _instant(value)
    if parsed is None or parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _add_candidate(bucket, name, valid_to=None):
    key = name.casefold()
    found = bucket.get(key)
    if found is None:
        found = {"name": name, "valid_to": None}
        bucket[key] = found
    if not valid_to:
        return
    new_at = _graph_instant(valid_to)
    old_at = _graph_instant(found["valid_to"]) if found["valid_to"] else None
    if found["valid_to"] is None or (new_at is not None and (old_at is None or new_at > old_at)):
        found["valid_to"] = valid_to


def _previous_name(candidates, current_edges):
    """The previous name from graph boundaries, not from when the text was filed.

    One earlier name is that previous name unless its graph boundary is after
    the current edge started. Several names are ordered only when every one
    has a usable graph boundary. A tie, or a boundary after the current edge,
    does not establish a previous name.
    """
    if not candidates:
        return None
    limits = []
    for edge in current_edges or []:
        instant = _graph_instant(edge.get("valid_from"))
        if instant is not None:
            limits.append(instant)
    limit = max(limits) if limits else None
    if len(candidates) == 1:
        only = candidates[0]
        instant = _graph_instant(only.get("valid_to"))
        if instant is not None and limit is not None and instant > limit:
            return None
        return only["name"]
    ranked = []
    for candidate in candidates:
        instant = _graph_instant(candidate.get("valid_to"))
        if instant is None:
            return None
        if limit is not None and instant > limit:
            return None
        ranked.append((instant, candidate["name"]))
    ranked.sort(key=lambda item: item[0])
    if len(ranked) >= 2 and ranked[-1][0] == ranked[-2][0]:
        return None
    return ranked[-1][1]


def test_spaceship_history_evidence(graph_rows, drawers, later_name):
    """Decide the previous test-spaceship name from graph rows and fact drawers.

    later_name is "" for "the previous name", or the name in "before <name>".
    Retrieval order is not used as time order.
    """
    edges = _history_edges(graph_rows)
    facts = _history_facts(drawers)
    current_edges = [edge for edge in edges if edge["current"]]
    inactive = [edge for edge in edges if not edge["current"]]
    current_facts = [fact for fact in facts if fact["room"] == "facts"]
    archived = [fact for fact in facts if fact["room"] == "facts-history"]
    current_name = _unique_name(
        [edge["object"] for edge in current_edges]
        + [fact["name"] for fact in current_facts]
    )
    duplicates = [
        edge for edge in inactive
        if current_name and _names_match(edge["object"], current_name)
    ]
    candidates = {}
    for edge in inactive:
        if current_name and _names_match(edge["object"], current_name):
            continue
        _add_candidate(candidates, edge["object"], valid_to=edge.get("valid_to"))
    for fact in archived:
        if current_name and _names_match(fact["name"], current_name):
            continue
        _add_candidate(candidates, fact["name"])
    history_edges = [
        edge for edge in inactive
        if not (current_name and _names_match(edge["object"], current_name))
    ]
    previous = None
    if current_name and candidates:
        previous = _previous_name(list(candidates.values()), current_edges)
    mismatch = bool(later_name) and not (
        current_name and _names_match(current_name, later_name)
    )
    if mismatch:
        previous = None
    return {
        "current_name": current_name,
        "previous_name": previous,
        "current_facts": current_facts,
        "archived": archived,
        "current_edges": current_edges,
        "history_edges": history_edges,
        "duplicates": duplicates,
        "candidate_count": len(candidates),
        "requested_mismatch": mismatch,
    }


def format_test_spaceship_history(evidence):
    """Label the history decision. Omit names that must not be guessed."""
    lines = []
    current = evidence["current_name"]
    previous = evidence["previous_name"]
    if current:
        lines.append("Current name: %s" % current)
    else:
        lines.append("Current name: not established")
    for fact in evidence["current_facts"]:
        if current and _names_match(fact["name"], current):
            lines.append("Current fact: %s" % fact["text"])
    for edge in evidence["current_edges"]:
        if not current or not _names_match(edge["object"], current):
            continue
        lines.append(
            "Current graph: user -> %s -> %s" % (edge["predicate"], edge["object"])
        )
    if previous:
        lines.append("Previous name: %s" % previous)
        archived_hit = [
            fact for fact in evidence["archived"]
            if _names_match(fact["name"], previous)
        ]
        graph_hit = [
            edge for edge in evidence["history_edges"]
            if _names_match(edge["object"], previous)
        ]
        for fact in archived_hit:
            lines.append("Archived fact: %s" % fact["text"])
        if archived_hit and not graph_hit:
            lines.append(
                "The graph has no different earlier name. "
                "This previous name is the archived fact in facts-history."
            )
        elif graph_hit and not archived_hit:
            lines.append(
                "Inactive graph edge: user -> %s -> %s. "
                "No archived fact names it. "
                "The edge is no longer current, so this is the previous name."
                % (graph_hit[0]["predicate"], graph_hit[0]["object"])
            )
        elif archived_hit and graph_hit:
            lines.append(
                "The archived fact and an inactive graph edge both give this name."
            )
        lines.append(
            "Tell the user the previous name was %s. Do not give a rename date."
            % previous
        )
    else:
        lines.append("Previous name: not established")
        if evidence["requested_mismatch"]:
            lines.append(
                "The question asks about a name that is not the current "
                "saved test-spaceship name."
            )
        elif evidence["candidate_count"] > 1:
            lines.append(
                "More than one earlier test-spaceship name is saved, "
                "and the stored times do not establish which name came "
                "immediately before."
            )
        elif evidence["candidate_count"] == 1:
            lines.append(
                "An earlier test-spaceship record is saved, "
                "but its stored time does not establish it as the previous name."
            )
        else:
            lines.append(
                "The saved records do not include an earlier test-spaceship name."
            )
        lines.append(
            "Tell the user that the saved records do not establish a previous name. "
            "Do not guess a name or a date."
        )
    if evidence["duplicates"]:
        stored = []
        for edge in evidence["duplicates"]:
            raw = edge.get("stored_predicate") or ""
            if raw and raw != edge.get("predicate") and raw not in stored:
                stored.append(raw)
        note = ""
        if stored:
            note = (
                " The stored predicate %s was normalized to test_spaceship_name."
                % ", ".join(stored)
            )
        lines.append(
            "An inactive graph edge repeats the current name.%s "
            "It is not a previous name, and its end time is not a rename date."
            % note
        )
    return "\n".join(lines)


def _metadata_present(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in ("", "unknown", "none"):
        return False
    return True


def _source(kind, text, record_id=None, filed_at=None, valid_from=None, valid_to=None):
    """One injected record. Missing id and time fields are omitted."""
    record = {"kind": kind, "text": text}
    if _metadata_present(record_id):
        record["id"] = str(record_id)
    if _metadata_present(filed_at):
        record["filed_at"] = str(filed_at)
    if _metadata_present(valid_from):
        record["valid_from"] = str(valid_from)
    if _metadata_present(valid_to):
        record["valid_to"] = str(valid_to)
    return record


def _preview(text, limit=80):
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit - 3].rstrip() + "..."


def render_source(source):
    """Label one record for the model without rewriting its text."""
    lines = ["[%s]" % source["kind"]]
    if source.get("id"):
        lines.append("id: %s" % source["id"])
    if source.get("filed_at"):
        lines.append(
            "filed: %s (when this record was filed, not necessarily when the fact changed)"
            % source["filed_at"]
        )
    if source.get("valid_from"):
        lines.append(
            "graph valid_from: %s (a stored graph time, not necessarily when the fact changed)"
            % source["valid_from"]
        )
    if source.get("valid_to"):
        lines.append(
            "graph valid_to: %s (a stored graph time, not necessarily when the fact changed)"
            % source["valid_to"]
        )
    lines.append(source.get("text") or "")
    return "\n".join(lines)


def render_sources(sources):
    return "\n\n".join(render_source(source) for source in sources)


def _fact_source(fact, kind):
    return _source(kind, fact.get("text") or "", fact.get("id"), filed_at=fact.get("filed_at"))


def _edge_source(edge):
    predicate = edge.get("stored_predicate") or edge.get("predicate") or ""
    text = "user -> %s -> %s" % (predicate, edge.get("object") or "")
    return _source(
        "knowledge-graph record",
        text,
        edge.get("id"),
        valid_from=edge.get("valid_from"),
        valid_to=edge.get("valid_to"),
    )


def history_sources(evidence):
    """Records named in the history context. Suppressed names stay out."""
    sources = []
    current = evidence.get("current_name")
    previous = evidence.get("previous_name")
    for fact in evidence.get("current_facts") or []:
        if current and _names_match(fact.get("name"), current):
            sources.append(_fact_source(fact, "saved fact"))
    for edge in evidence.get("current_edges") or []:
        if current and _names_match(edge.get("object"), current):
            sources.append(_edge_source(edge))
    if previous:
        for fact in evidence.get("archived") or []:
            if _names_match(fact.get("name"), previous):
                sources.append(_fact_source(fact, "archived fact"))
        for edge in evidence.get("history_edges") or []:
            if _names_match(edge.get("object"), previous):
                sources.append(_edge_source(edge))
    for edge in evidence.get("duplicates") or []:
        sources.append(_edge_source(edge))
    return sources


def _coerce_saved_fact(item):
    if isinstance(item, str):
        return _source("saved fact", item)
    metadata = item.get("metadata") or {}
    text = item.get("text") or item.get("content_preview") or ""
    room = item.get("room") or metadata.get("room")
    if room == "facts-history":
        kind = "archived fact"
    elif _is_unclassified(metadata):
        kind = UNCLASSIFIED_KIND
    else:
        kind = "saved fact"
    filed_at = item.get("filed_at")
    if filed_at is None:
        filed_at = metadata.get("filed_at")
    return _source(kind, text, item.get("drawer_id") or item.get("id"), filed_at=filed_at)


def _coerce_conversation(item):
    if isinstance(item, str):
        return _source("conversation excerpt", item)
    text = item.get("text") or item.get("content_preview") or ""
    filed_at = item.get("filed_at")
    if filed_at is None:
        filed_at = (item.get("metadata") or {}).get("filed_at")
    return _source(
        "conversation excerpt",
        text,
        item.get("drawer_id") or item.get("id"),
        filed_at=filed_at,
    )


def _drawer_sources(memory, room):
    """Every saved fact in one room, with full text and the metadata already stored."""
    if getattr(memory, "list_drawers", None) is None:
        return None
    sources = []
    for item in _room_drawers(memory, room):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        metadata = item.get("metadata") or {}
        sources.append(_coerce_saved_fact({
            "text": text,
            "drawer_id": item.get("drawer_id"),
            "room": room,
            "metadata": metadata,
            "filed_at": metadata.get("filed_at"),
        }))
    return sources


UNFINISHED_KIND = "unfinished memory update"
UNCLASSIFIED_KIND = "unclassified saved text"
UNCLASSIFIED_NOTE = (
    "A record labeled unclassified saved text was kept without a"
    " confirmed relationship. It is not a current single-value fact."
)
UNFINISHED_NOTE = "A record labeled unfinished memory update is not a current fact."


def _unfinished_text(update):
    lines = [
        'Not a current fact: "%s"' % update.get("text"),
        'This "Remember this:" update did not finish.',
    ]
    predicate = update.get("predicate")
    if predicate in SINGLE_VALUE_PREDICATES:
        if update.get("last_value"):
            lines.append("The last completed %s is %s." % (predicate, update["last_value"]))
        else:
            lines.append(
                "The saved records do not establish the current %s. "
                "Do not state one as current." % predicate
            )
    return " ".join(lines)


def unfinished_memory(memory):
    """Sources for facts whose update did not finish, and the updates themselves.

    A memory without unfinished_updates has none. If they cannot be read, one
    source says so, because saved facts may then be out of date.
    """
    reader = getattr(memory, "unfinished_updates", None)
    if reader is None:
        return [], []
    try:
        updates = reader()
    except Exception:
        return [_source(
            UNFINISHED_KIND,
            "TinyTalk could not check for unfinished memory updates, "
            "so saved facts may be out of date.",
        )], []
    sources = [
        _source(UNFINISHED_KIND, _unfinished_text(update), update.get("drawer_id"),
                filed_at=update.get("filed_at"))
        for update in updates
    ]
    return sources, updates


def _from_unfinished_command(item, updates):
    """True for a stored turn that sent "Remember this:" for an unfinished update."""
    text = item if isinstance(item, str) else (item.get("text") or item.get("content_preview") or "")
    text = text.casefold()
    return "remember this:" in text and any(
        (update.get("text") or "").casefold() in text for update in updates if update.get("text")
    )


def _conversation_similarity(hit):
    """A missing or unreadable score does not clear the conversation floor."""
    if not isinstance(hit, dict):
        return 0
    try:
        return float(hit.get("similarity"))
    except (TypeError, ValueError):
        return 0


def _keep_conversation(hit):
    if not isinstance(hit, dict):
        return False
    text = hit.get("text") or hit.get("content_preview") or ""
    return _conversation_similarity(hit) >= MIN_FACT_SIMILARITY and bool(text)


def _command_fact(text):
    """The fact after the first "Remember this:", or None."""
    raw = text or ""
    marker = "remember this:"
    start = raw.casefold().find(marker)
    if start == -1:
        return None
    rest = raw[start + len(marker):].strip()
    if not rest:
        return None
    return rest.splitlines()[0].strip() or None


def _retired_fact_texts(memory):
    """Exact fact text that is historical or abandoned, not current."""
    if memory is None or getattr(memory, "list_drawers", None) is None:
        return set()
    found = set()
    for room in (HISTORY_ROOM, ABANDONED_ROOM):
        try:
            drawers = _room_drawers(memory, room)
        except Exception:
            continue
        for drawer in drawers:
            if drawer.get("text"):
                found.add(drawer["text"])
    return found


def _from_retired_command(item, retired):
    """True when a stored Remember this command names a fact that is no longer current."""
    if not retired:
        return False
    text = item if isinstance(item, str) else (item.get("text") or item.get("content_preview") or "")
    fact = _command_fact(text)
    return bool(fact) and fact in retired


class SessionSources(object):
    """Sources included with the last successful answer. Session memory only."""

    def __init__(self):
        self.records = []

    def replace(self, records):
        self.records = list(records or [])


def format_sources_report(records):
    records = list(records or [])
    if not records:
        return (
            "No saved memories were included with the last successful answer. "
            "That does not mean no memories exist, and it does not mean the "
            "answer came only from model knowledge."
        )
    lines = [
        "These saved memories were included in the context for the last "
        "successful answer. TinyTalk does not claim the model used every one."
    ]
    for number, source in enumerate(records, 1):
        lines.append("%s. %s" % (number, source["kind"]))
        if source.get("id"):
            lines.append("id: %s" % source["id"])
        if source.get("filed_at"):
            lines.append("filed: %s" % source["filed_at"])
            lines.append(
                "This is when the record was filed, not necessarily when the fact changed."
            )
        if source.get("valid_from"):
            lines.append("graph valid_from: %s" % source["valid_from"])
        if source.get("valid_to"):
            lines.append("graph valid_to: %s" % source["valid_to"])
        if source.get("valid_from") or source.get("valid_to"):
            lines.append("A graph time is not necessarily when the fact changed.")
        lines.append(_preview(source.get("text") or ""))
    return "\n".join(lines)


def _load_test_spaceship_records(memory):
    graph_rows = []
    drawers = []
    if memory is not None and getattr(memory, "kg", None) is not None:
        try:
            graph_rows = memory.kg.query_entity("user") or []
        except Exception:
            graph_rows = []
    if memory is None or getattr(memory, "list_drawers", None) is None:
        return graph_rows, drawers
    for room in ("facts", "facts-history"):
        try:
            found = _room_drawers(memory, room)
        except Exception:
            found = []
        for drawer in found:
            metadata = drawer.get("metadata") or {}
            drawers.append({
                "drawer_id": drawer.get("drawer_id"),
                "room": room,
                "text": drawer.get("text") or "",
                "content_preview": drawer.get("text") or "",
                "metadata": metadata,
                "filed_at": metadata.get("filed_at"),
            })
    return graph_rows, drawers


def _unsettled_row(row, names):
    if not isinstance(row, dict) or _canon_predicate(row.get("predicate")) != "test_spaceship_name":
        return False
    if "current" in row and row.get("current") is not None:
        current = bool(row.get("current"))
    else:
        current = not row.get("valid_to")
    return current and any(_names_match(row.get("object"), name) for name in names)


def historical_context_message(user_prompt, memory):
    later_name = test_spaceship_history_question(user_prompt)
    sources = []
    unfinished, updates = unfinished_memory(memory)
    # Only unfinished test-spaceship updates matter here. Their names are not current.
    ship = []
    for source, update in zip(unfinished, updates):
        if update.get("predicate") == "test_spaceship_name":
            name = update.get("object")
        elif not update.get("predicate"):
            name = test_spaceship_fact_name(update.get("text"))
        else:
            name = None
        if name:
            ship.append((source, name, update.get("last_value")))
    if updates:
        unfinished = [source for source, _name, _last in ship]
    unsettled = [name for _source, name, last in ship if not _names_match(name, last)]
    try:
        graph_rows, drawers = _load_test_spaceship_records(memory)
        graph_rows = [row for row in graph_rows if not _unsettled_row(row, unsettled)]
        evidence = test_spaceship_history_evidence(graph_rows, drawers, later_name)
        body = format_test_spaceship_history(evidence)
        sources = history_sources(evidence)
    except Exception:
        body = (
            "Previous name: not established\n"
            "Tell the user that the saved records do not establish a previous name. "
            "Do not guess a name or a date."
        )
        sources = []
    if unfinished:
        body = body + "\n" + "\n".join(
            "Unfinished update, not the current name: %s" % source["text"] for source in unfinished
        )
        sources = sources + unfinished
    if sources:
        body = body + "\n\nRetrieved sources:\n\n" + render_sources(sources)
    message = {
        "role": "system",
        "content": (
            "This context is the saved test-spaceship history for this question. "
            "Answer only from these labels. "
            "These are retrieved sources, not information learned during model training. "
            "Do not mention a fact that is not written here. "
            "Do not give a rename date.\n\n%s" % body
        ),
    }
    return message, sources


def debug_enabled():
    """TINYTALK_DEBUG is off unless set to 1, true, yes, or on."""
    return os.environ.get("TINYTALK_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def debug(message):
    if debug_enabled():
        print(message)


def memory_instruction(what, body):
    """Tell the model what this request included, and what it did not."""
    return what + " " + MEMORY_LIMITS + "\n\n" + body


def load_soul():
    path = Path(__file__).with_name("SOUL.md")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print("Warning: could not read %s (%s). Using the short fallback." % (path, exc))
        return "You are TinyTalk, a helpful local assistant."


def model_messages(soul, request_messages):
    """Same system instructions, memory context, and history for every provider."""
    return [{"role": "system", "content": soul}] + list(request_messages)


def fact_to_triple(fact, provider):
    """Ask the configured model for one triple. None when the JSON is not one clear fact."""
    raw = provider.complete(
        [
            {"role": "system", "content": TRIPLE_INSTRUCTIONS},
            {"role": "user", "content": fact},
        ]
    )
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        return None
    subject = data.get("subject")
    predicate = data.get("predicate")
    obj = data.get("object")
    if not all(isinstance(value, str) and value.strip() for value in (subject, predicate, obj)):
        return None
    subject = subject.strip()
    if subject.lower() in {"i", "me", "my", "myself"}:
        subject = "user"
    predicate = _canon_predicate(predicate)
    obj = obj.strip()
    if not subject or not predicate or not obj:
        return None
    return subject, predicate, obj


class MemoryStore(object):
    """Local MemPalace facts, conversations, and knowledge graph.

    Embeddings stay inside MemPalace (MiniLM or EmbeddingGemma). This store
    does not call Ollama or xAI.
    """

    def __init__(self, palace, palace_path, kg, add_drawer, list_drawers, update_drawer,
                 get_drawer=None):
        self.palace = palace
        self.palace_path = palace_path
        self.kg = kg
        self.add_drawer = add_drawer
        self.list_drawers = list_drawers
        self.update_drawer = update_drawer
        self.get_drawer = get_drawer
        self.provenance = True

    def search_facts(self, query):
        from mempalace.searcher import search_memories

        found = search_memories(
            query=query,
            palace_path=self.palace_path,
            wing="tinytalk",
            room="facts",
            n_results=3,
        )
        accepted = []
        for hit in found.get("results", []):
            similarity = hit.get("similarity", 0)
            keep = similarity >= MIN_FACT_SIMILARITY and bool(hit.get("text"))
            debug("[debug] fact similarity %s: %s" % (
                similarity, "accepted" if keep else "rejected"
            ))
            if keep:
                accepted.append(hit)
        return accepted

    def search_conversations(self, query):
        from mempalace.searcher import search_memories

        found = search_memories(
            query=query,
            palace_path=self.palace_path,
            wing="tinytalk",
            room="conversations",
            n_results=3,
        )
        accepted = []
        for hit in found.get("results", []):
            similarity = _conversation_similarity(hit)
            keep = similarity >= MIN_FACT_SIMILARITY and bool(hit.get("text"))
            debug("[debug] conversation similarity %s: %s" % (
                similarity, "accepted" if keep else "rejected"
            ))
            if keep:
                accepted.append(hit)
        return accepted

    def list_facts(self):
        if self.list_drawers is None:
            return []
        return [
            drawer["text"] for drawer in _room_drawers(self, FACT_ROOM) if drawer.get("text")
        ]

    def save_exchange(self, user_prompt, response, memory_note=None):
        from mempalace.convo_miner import file_conversation_exchange
        from mempalace.palace import get_collection

        # A fact write opens the palace through MemPalace's tool client. The
        # collection object from startup can then fail later upserts, so each
        # turn opens the collection again at the same path.
        text = "User: %s\n\nAssistant: %s" % (user_prompt, response)
        if memory_note:
            text = text + "\n\n" + memory_note
        file_conversation_exchange(
            get_collection(self.palace_path),
            wing="tinytalk",
            room="conversations",
            text=text,
            source_file="tinytalk",
            agent="tinytalk",
        )

    def current_fact_objects(self, subject, predicate):
        return [link["object"] for link in self.current_links(subject, predicate)]

    def current_links(self, subject, predicate):
        """Current graph values for one relationship, each with its fact drawer id.

        query_entity does not return source_drawer_id, so this pages the stored
        rows with KnowledgeGraph.dump_rows. A stored alias such as test_spaceship
        matches the canonical predicate. drawer_id is None for edges written
        before TinyTalk recorded it. Raises if the graph cannot be read.
        """
        names = {row["id"]: row["name"] for row in _graph_rows(self.kg, "entities")}
        return _links_from_rows(names, _graph_rows(self.kg, "triples"), subject, predicate)

    def pending_facts(self):
        """Facts whose "Remember this:" update did not finish. Raises if they cannot be listed."""
        found = []
        for drawer in _room_drawers(self, PENDING_ROOM):
            predicate, obj = _fact_relationship(drawer["metadata"])
            if predicate:
                predicate = _canon_predicate(predicate)
            found.append({
                "drawer_id": drawer["drawer_id"],
                "text": drawer["text"],
                "predicate": predicate,
                "object": obj,
                "filed_at": drawer["metadata"].get("filed_at"),
            })
        return found

    def unfinished_updates(self):
        """Pending facts, each with the last completed value of its relationship.

        last_value is the one current graph value whose drawer is not itself
        pending or abandoned. It is None when the graph does not establish one.
        """
        updates = self.pending_facts()
        names = None
        rows = None
        if self.kg is not None and any(
            update.get("predicate") in SINGLE_VALUE_PREDICATES for update in updates
        ):
            try:
                names = {row["id"]: row["name"] for row in _graph_rows(self.kg, "entities")}
                rows = _graph_rows(self.kg, "triples")
            except Exception:
                names = {}
                rows = []
        for update in updates:
            update["last_value"] = None
            if rows is None or update.get("predicate") not in SINGLE_VALUE_PREDICATES:
                continue
            values = []
            for link in _links_from_rows(names, rows, "user", update["predicate"]):
                room = _drawer_room(self, link["drawer_id"]) if link["drawer_id"] else None
                if room in (PENDING_ROOM, ABANDONED_ROOM):
                    continue
                if not any(_names_match(link["object"], value) for value in values):
                    values.append(link["object"])
            if len(values) == 1:
                update["last_value"] = values[0]
        return updates


def _provenance_supported(kg):
    """True when triples have the source_drawer_id column this repair writes."""
    try:
        with kg._lock:
            rows = kg._conn().execute("PRAGMA table_info(triples)").fetchall()
    except Exception:
        return False
    return any(row[1] == "source_drawer_id" for row in rows)


def _attach_source_drawer(kg, triple_id, drawer_id):
    """Fill a null provenance field on one open triple. True only when one row changes."""
    if kg is None or not triple_id or not drawer_id:
        return False
    try:
        with kg._lock:
            conn = kg._conn()
            with conn:
                updated = conn.execute(
                    "UPDATE triples SET source_drawer_id = ? "
                    "WHERE id = ? AND valid_to IS NULL AND "
                    "(source_drawer_id IS NULL OR source_drawer_id = '')",
                    (drawer_id, triple_id),
                )
                return updated.rowcount == 1
    except Exception:
        return False


def _link_provenance(memory, links, obj, drawer_id):
    """Attach the current fact drawer to an open edge that has no drawer yet.

    Returns an incomplete status when the attach does not happen, or None.
    """
    if not getattr(memory, "provenance", True):
        return None
    targets = [
        link for link in links
        if _names_match(link.get("object"), obj) and not link.get("source_drawer_id")
    ]
    if not targets:
        return None
    try:
        room = _drawer_room(memory, drawer_id)
    except Exception:
        room = None
    if room != FACT_ROOM:
        return _pending(
            "The fact is not in current facts, so its graph link was not recorded. "
            "Repeat the command to finish"
        )
    for link in targets:
        if not _attach_source_drawer(memory.kg, link.get("triple_id"), drawer_id):
            return _pending(
                "The graph link could not be attached to the saved fact. "
                "Repeat the command to finish"
            )
    return None


def _links_from_rows(names, rows, subject, predicate):
    """Open edges whose predicate canonicalizes to the requested one."""
    wanted = _canon_predicate(predicate)
    links = []
    for row in rows:
        if row.get("valid_to") is not None:
            continue
        stored = row.get("predicate") or ""
        if _canon_predicate(stored) != wanted:
            continue
        if (names.get(row.get("subject")) or "").casefold() != (subject or "").casefold():
            continue
        obj = names.get(row.get("object"))
        if obj:
            links.append({
                "object": obj,
                "drawer_id": row.get("source_drawer_id"),
                "triple_id": row.get("id"),
                "source_drawer_id": row.get("source_drawer_id"),
                "stored_predicate": stored,
            })
    return links


def _graph_rows(kg, table):
    rows = []
    after = 0
    while True:
        page = kg.dump_rows(table, after_rowid=after, limit=1000)
        rows.extend(page)
        if len(page) < 1000:
            return rows
        after = page[-1]["_rowid"]


def _import_memory_tools():
    saved_stdout = sys.stdout
    saved_fd = None
    try:
        saved_fd = os.dup(1)
    except OSError:
        saved_fd = None
    try:
        import mempalace.mcp_server as mcp
        return mcp
    finally:
        if saved_fd is not None:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
        sys.stdout = saved_stdout


def _point_tools_at_palace(mcp, palace_path):
    from mempalace.config import MempalaceConfig

    mcp._config = MempalaceConfig(palace_path=palace_path)
    for name in ("_collection_cache", "_client_cache", "_collection_cache_palace"):
        if hasattr(mcp, name):
            setattr(mcp, name, None)


def open_memory(palace_path=None, kg_path=None):
    """Open MemPalace. Pass paths to keep tests off the real palace."""
    palace = None
    resolved = None
    try:
        from mempalace.config import MempalaceConfig
        from mempalace.palace import get_collection

        if palace_path:
            resolved = MempalaceConfig(palace_path=palace_path).palace_path
        else:
            resolved = MempalaceConfig().palace_path
        palace = get_collection(resolved)
    except Exception:
        print("Warning: MemPalace could not start. Continuing without saved memory.")
        palace = None
        resolved = None

    add_drawer = None
    list_drawers = None
    update_drawer = None
    get_drawer = None
    try:
        mcp = _import_memory_tools()
        if palace_path and resolved:
            _point_tools_at_palace(mcp, resolved)
        add_drawer = mcp.tool_add_drawer
        list_drawers = mcp.tool_list_drawers
        update_drawer = mcp.tool_update_drawer
        get_drawer = mcp.tool_get_drawer
    except Exception:
        add_drawer = None
        list_drawers = None
        update_drawer = None
        get_drawer = None

    kg = None
    try:
        from mempalace.knowledge_graph import KnowledgeGraph

        if kg_path:
            kg = KnowledgeGraph(db_path=kg_path)
        else:
            kg = KnowledgeGraph()
    except Exception:
        print("Warning: MemPalace knowledge graph could not start. Continuing without it.")
        kg = None

    memory = MemoryStore(palace, resolved, kg, add_drawer, list_drawers, update_drawer, get_drawer)
    if kg is not None and not _provenance_supported(kg):
        memory.provenance = False
        print(
            "Warning: this knowledge graph has no source_drawer_id column. "
            "TinyTalk will not attach fact drawers to graph edges."
        )
    return memory


def _with_sources(lead, sources, messages):
    if any(source["kind"] == UNCLASSIFIED_KIND for source in sources):
        lead = lead + " " + UNCLASSIFIED_NOTE
    if any(source["kind"] == UNFINISHED_KIND for source in sources):
        lead = lead + " " + UNFINISHED_NOTE
    return [
        {
            "role": "system",
            "content": memory_instruction(lead, render_sources(sources)),
        }
    ] + messages, sources


def build_context(user_prompt, messages, memory):
    """Return request messages and the records actually injected."""
    if test_spaceship_history_question(user_prompt) is not None:
        message, sources = historical_context_message(user_prompt, memory)
        return [message] + messages, sources
    if memory is None or memory.palace is None:
        return messages, []
    unfinished, updates = unfinished_memory(memory)
    if is_memory_introspection(user_prompt):
        sources = []
        try:
            sources = _drawer_sources(memory, "facts")
            if sources is None:
                sources = [
                    _coerce_saved_fact(fact)
                    for fact in (memory.list_facts() or [])
                    if fact
                ]
        except Exception:
            sources = []
        if sources or unfinished:
            return _with_sources(
                "This context contains the saved facts retrieved for this request.",
                sources + unfinished,
                messages,
            )
        return messages, []

    facts = []
    try:
        facts = memory.search_facts(user_prompt)
    except Exception:
        facts = []
    if facts:
        sources = [_coerce_saved_fact(item) for item in facts if item] + unfinished
        return _with_sources(
            "This context contains the saved facts retrieved for this request.",
            sources,
            messages,
        )

    found = []
    try:
        found = memory.search_conversations(user_prompt)
    except Exception:
        found = []
    # A stored "Remember this:" turn for an unfinished update would repeat
    # that update as if it were settled. The unfinished record stands in for it.
    retired = _retired_fact_texts(memory)
    found = [
        item for item in found
        if item and _keep_conversation(item)
        and not _from_unfinished_command(item, updates)
        and not _from_retired_command(item, retired)
    ]
    if found:
        sources = [_coerce_conversation(item) for item in found] + unfinished
        return _with_sources(
            "This context contains excerpts retrieved for this request "
            "from earlier conversations. "
            "These may contain old assistant mistakes. "
            "When memories conflict, prefer explicit factual statements "
            "made by the user over previous assistant responses. "
            "A Remember this command is current evidence only while that fact "
            "is still a saved fact. An excerpt of a replaced, abandoned, "
            "unfinished, or unclassified update does not override a saved fact.",
            sources,
            messages,
        )
    if unfinished:
        return _with_sources(
            "This context lists saved memory updates that did not finish.",
            unfinished,
            messages,
        )
    return messages, []


def context_messages(user_prompt, messages, memory):
    """Memory is added to this request only. It is not stored in messages."""
    request_messages, _sources = build_context(user_prompt, messages, memory)
    return request_messages


class UnsafeReplacement(Exception):
    """The saved records do not show which fact drawer holds the old value."""


def _status(status, message):
    return {"status": status, "message": message}


def _not_saved(reason):
    return _status("not_saved", "Memory: not saved. %s" % reason)


def _pending(detail):
    return _status(
        "incomplete",
        "Memory: saved as written, but not as a current fact. %s." % detail,
    )


def _fact_source_file(predicate=None, obj=None, unclassified=False):
    """Drawer metadata naming the relationship a fact updates.

    Unclassified text was stored without a confirmed relationship. Callers
    must not treat that marker as a current single-value fact.
    """
    if unclassified:
        return "tinytalk?" + urlencode({"unclassified": "1"})
    if not predicate:
        return "tinytalk"
    return "tinytalk?" + urlencode({"predicate": predicate, "object": obj or ""})


def _source_query(metadata):
    source = (metadata or {}).get("source_file") or ""
    if not source.startswith("tinytalk?"):
        return {}
    fields = parse_qs(source[len("tinytalk?"):])
    return {key: values[0] for key, values in fields.items() if values}


def _fact_relationship(metadata):
    fields = _source_query(metadata)
    return fields.get("predicate"), fields.get("object")


def _is_unclassified(metadata):
    return _source_query(metadata).get("unclassified") == "1"


def _drawer_room(memory, drawer_id):
    """Room of one drawer, or None if it does not exist. Raises if it cannot be read."""
    found = memory.get_drawer(drawer_id)
    error = found.get("error") if isinstance(found, dict) else "no result"
    if error:
        if str(error).startswith("Drawer not found"):
            return None
        raise RuntimeError(error)
    return found.get("room")


def _move_drawer(memory, drawer_id, room):
    try:
        moved = memory.update_drawer(drawer_id, room=room)
    except Exception:
        return False
    return isinstance(moved, dict) and bool(moved.get("success"))


def _room_drawers(memory, room):
    """Every tinytalk drawer in one room, with full text. Raises if the list cannot be read."""
    drawers = []
    offset = 0
    while True:
        page = memory.list_drawers(wing="tinytalk", room=room, limit=100, offset=offset)
        if not isinstance(page, dict) or "drawers" not in page:
            raise RuntimeError("could not list %s" % room)
        for item in page["drawers"]:
            text = item.get("content_preview") or ""
            # MemPalace cuts previews at 200 characters and adds "...".
            if len(text) > 200 and text.endswith("..."):
                full = memory.get_drawer(item["drawer_id"])
                if not isinstance(full, dict) or full.get("error"):
                    raise RuntimeError("could not read drawer %s" % item["drawer_id"])
                text = full.get("content") or ""
            drawers.append({
                "drawer_id": item.get("drawer_id"),
                "text": text.strip(),
                "metadata": item.get("metadata") or {},
            })
        offset += len(page["drawers"])
        if not page["drawers"] or offset >= page.get("total", 0):
            return drawers


def _store_drawer(memory, fact, predicate=None, obj=None, room=PENDING_ROOM):
    """Save the verbatim fact in room (facts-pending), unless the same text is already current.

    Returns (drawer_id, room), or None if it could not be saved. The same text
    found in facts-history or facts-abandoned moves to room, so a restated
    value goes through the whole update again. A save into current facts does
    not pull a pending, historical, or abandoned drawer back into facts.
    """
    target = room
    if predicate:
        source = _fact_source_file(predicate, obj)
    elif target == FACT_ROOM:
        source = _fact_source_file(unclassified=True)
    else:
        source = _fact_source_file()
    try:
        if getattr(memory, "list_drawers", None) is not None:
            for drawer in _room_drawers(memory, FACT_ROOM):
                if drawer["text"] == fact:
                    return drawer["drawer_id"], FACT_ROOM
        saved = memory.add_drawer(
            wing="tinytalk",
            room=target,
            content=fact,
            source_file=source,
            added_by="tinytalk",
        )
    except Exception:
        return None
    if not isinstance(saved, dict) or not saved.get("success"):
        return None
    drawer_id = saved.get("drawer_id")
    if saved.get("reason") != "already_exists" or getattr(memory, "get_drawer", None) is None:
        return drawer_id, target
    try:
        room = _drawer_room(memory, drawer_id)
    except Exception:
        return None
    if room in (FACT_ROOM, target):
        return drawer_id, room
    if target == FACT_ROOM and room in (PENDING_ROOM, HISTORY_ROOM, ABANDONED_ROOM):
        return drawer_id, room
    if room is not None and _move_drawer(memory, drawer_id, target):
        return drawer_id, target
    return None


def _save_as_written(memory, fact, detail):
    """No relationship was identified, so the words alone are the whole update."""
    stored = _store_drawer(memory, fact, room=FACT_ROOM)
    if stored is None:
        return _not_saved("MemPalace could not store this fact.")
    _drawer_id, room = stored
    if room == PENDING_ROOM:
        return _pending(
            "This update was already in progress, so it was not saved as a current fact"
        )
    if room != FACT_ROOM:
        return _pending("Those words stay outside current facts, so nothing was replaced")
    return _status("saved_as_written", "Memory: saved as written. %s." % detail)


def _save_unfinished(memory, fact, detail, predicate=None, obj=None):
    """Keep the words outside current facts when the update cannot finish."""
    stored = _store_drawer(memory, fact, predicate, obj)
    if stored is None:
        return _not_saved("MemPalace could not store this fact.")
    if stored[1] == FACT_ROOM:
        return _status(
            "incomplete",
            "Memory: already saved as written, but the update could not be checked. %s."
            % detail,
        )
    return _pending(detail)


def _finish(memory, drawer_id, room, stale):
    """Make the new drawer current, then retire earlier unfinished updates."""
    if room != FACT_ROOM and not _move_drawer(memory, drawer_id, FACT_ROOM):
        return _pending("The fact could not be moved into current facts. Repeat the command to finish")
    left = [
        update.get("object") or "an earlier value"
        for update in stale
        if update["drawer_id"] != drawer_id
        and not _move_drawer(memory, update["drawer_id"], ABANDONED_ROOM)
    ]
    if left:
        return _status(
            "incomplete",
            "Memory: saved, but the unfinished update to %s could not be cleared. "
            "Repeat the command to finish." % ", ".join(left),
        )
    return None


def _old_fact_drawer(memory, fact, predicate, link):
    """(drawer_id, room to move it to) for one replaced value, or None.

    A graph edge names its drawer, so that exact drawer moves. An edge saved
    before TinyTalk recorded the link is accepted only for test_spaceship_name,
    and only when exactly one current fact parses as a test-spaceship name and
    that name is the graph value. Anything else raises UnsafeReplacement.
    """
    old = link["object"]
    if link.get("drawer_id"):
        try:
            room = _drawer_room(memory, link["drawer_id"])
        except Exception:
            raise UnsafeReplacement("The saved fact for %s could not be read" % old)
        if room == FACT_ROOM:
            return link["drawer_id"], HISTORY_ROOM
        if room == PENDING_ROOM:
            return link["drawer_id"], ABANDONED_ROOM
        return None
    if predicate == "test_spaceship_name":
        try:
            drawers = _room_drawers(memory, FACT_ROOM)
        except Exception:
            raise UnsafeReplacement("The saved facts could not be read, so %s was not replaced" % old)
        named = [d for d in drawers if d["text"] != fact and test_spaceship_fact_name(d["text"])]
        if len(named) == 1 and _names_match(test_spaceship_fact_name(named[0]["text"]), old):
            return named[0]["drawer_id"], HISTORY_ROOM
    raise UnsafeReplacement(
        "The saved value %s is not linked to one saved fact, so it was not replaced" % old
    )


def _save_additive(memory, fact, triple):
    stored = _store_drawer(memory, fact)
    if stored is None:
        return _not_saved("MemPalace could not store this fact.")
    drawer_id, room = stored
    try:
        memory.kg.add_triple(
            triple[0], triple[1], triple[2], source_file="tinytalk", source_drawer_id=drawer_id
        )
    except Exception:
        return _pending("The knowledge graph update failed. Repeat the command to finish")
    debug("[debug] KG: %s -> %s -> %s" % triple)
    finished = _finish(memory, drawer_id, room, [])
    if finished:
        return finished
    if room == FACT_ROOM:
        return _status("already_saved", "Memory: already saved.")
    return _status("saved", "Memory: saved.")


def _drawer_text(memory, drawer_id):
    """Full text of one drawer. Raises if it cannot be read."""
    found = memory.get_drawer(drawer_id)
    error = found.get("error") if isinstance(found, dict) else "no result"
    if error:
        raise RuntimeError(error)
    return (found.get("content") or "").strip()


def _canonical_drawer_id(memory, links, obj):
    """Facts drawer an open link already names for this value, if there is one."""
    for link in links:
        if not _names_match(link.get("object"), obj):
            continue
        drawer_id = link.get("drawer_id")
        if not drawer_id:
            continue
        try:
            room = _drawer_room(memory, drawer_id)
        except Exception:
            continue
        if room == FACT_ROOM:
            return drawer_id
    return None


def _predicate_fact_drawers(memory, predicate):
    """Current facts whose metadata names this single-value predicate."""
    found = []
    for drawer in _room_drawers(memory, FACT_ROOM):
        stored, _obj = _fact_relationship(drawer["metadata"])
        if stored and _canon_predicate(stored) == predicate:
            found.append(drawer)
    return found


def _retire_other_wordings(memory, predicate, keep_ids):
    """Move other current drawers for this predicate into fact history.

    Returns the texts that could not be moved. Drawers are chosen by
    relationship metadata, not by words inside the fact.
    """
    failed = []
    for drawer in _predicate_fact_drawers(memory, predicate):
        if drawer["drawer_id"] in keep_ids:
            continue
        if not _move_drawer(memory, drawer["drawer_id"], HISTORY_ROOM):
            failed.append(drawer["text"] or drawer["drawer_id"])
    return failed


def _wording_incomplete(obj):
    return _status(
        "incomplete",
        "Memory: already saved, but another wording for %s could not be moved. "
        "Repeat the command to finish." % obj,
    )


def _already_current(memory, predicate, obj, canonical_id):
    """The graph already names this value. Do not store a second current fact."""
    try:
        failed = _retire_other_wordings(memory, predicate, {canonical_id})
    except Exception:
        return _wording_incomplete(obj)
    if failed:
        return _wording_incomplete(obj)
    return _status(
        "already_saved",
        "Memory: already saved. The current %s is %s. "
        "That wording was not stored as a second current fact." % (predicate, obj),
    )


def _resume_pending(memory, fact):
    """Triple stored on a matching pending drawer, or None when extraction should run.

    A read failure returns None so the caller keeps the existing extraction path.
    """
    try:
        pending = memory.pending_facts()
    except Exception:
        return None
    matches = [
        update for update in pending
        if update.get("text") == fact and update.get("predicate") and update.get("object")
    ]
    if len(matches) != 1:
        return None
    update = matches[0]
    obj = update["object"]
    if obj.casefold() not in fact.casefold():
        return None
    return ("user", _canon_predicate(update["predicate"]), obj)


def _save_single_value(memory, fact, triple):
    """Replace one single-value fact.

    Order: read the graph and earlier unfinished updates, decide which drawers
    to move, save the new drawer in facts-pending, archive the old drawer,
    supersede in the graph, move the new drawer into facts, then move earlier
    unfinished updates to facts-abandoned. Each step runs only if the one
    before it worked. Until the last steps finish, the new words stay in
    facts-pending, outside current facts. Repeating the command continues.
    """
    subject, predicate, obj = triple
    try:
        links = memory.current_links(subject, predicate)
        stale = [update for update in memory.pending_facts() if update["predicate"] == predicate]
    except Exception:
        return _save_unfinished(
            memory, fact,
            "The current value could not be read, so nothing was replaced. "
            "Repeat the command to finish",
            predicate, obj,
        )
    olds = [link for link in links if not _names_match(link["object"], obj)]
    canonical_id = None
    if not olds:
        canonical_id = _canonical_drawer_id(memory, links, obj)
        if canonical_id:
            try:
                current_text = _drawer_text(memory, canonical_id)
            except Exception:
                return _save_unfinished(
                    memory, fact,
                    "The saved fact could not be read, so nothing was replaced. "
                    "Repeat the command to finish",
                    predicate, obj,
                )
            if current_text != fact:
                return _already_current(memory, predicate, obj, canonical_id)
    moves = []
    for link in olds:
        try:
            move = _old_fact_drawer(memory, fact, predicate, link)
        except UnsafeReplacement as exc:
            return _save_unfinished(memory, fact, str(exc), predicate, obj)
        if move:
            moves.append(move)
    if olds:
        try:
            move_ids = {drawer_id for drawer_id, _to_room in moves}
            for drawer in _predicate_fact_drawers(memory, predicate):
                if drawer["drawer_id"] not in move_ids:
                    moves.append((drawer["drawer_id"], HISTORY_ROOM))
                    move_ids.add(drawer["drawer_id"])
        except Exception:
            return _save_unfinished(
                memory, fact,
                "The saved facts could not be read, so nothing was replaced. "
                "Repeat the command to finish",
                predicate, obj,
            )

    stored = _store_drawer(memory, fact, predicate, obj)
    if stored is None:
        return _not_saved("MemPalace could not store this fact.")
    drawer_id, room = stored
    old_names = ", ".join(link["object"] for link in olds)
    for old_id, to_room in moves:
        if old_id != drawer_id and not _move_drawer(memory, old_id, to_room):
            return _pending(
                "The earlier fact for %s could not be archived. Repeat the command to finish"
                % old_names
            )
    try:
        for link in olds:
            memory.kg.supersede(
                subject, link.get("stored_predicate") or predicate, link["object"], obj,
                source_file="tinytalk", source_drawer_id=drawer_id,
            )
        if not links:
            memory.kg.add_triple(
                subject, predicate, obj, source_file="tinytalk", source_drawer_id=drawer_id
            )
    except Exception:
        if olds:
            return _pending(
                "The knowledge graph still lists %s. Repeat the command to finish" % old_names
            )
        return _pending("The knowledge graph update failed. Repeat the command to finish")
    debug("[debug] KG: %s -> %s -> %s" % triple)
    finished = _finish(memory, drawer_id, room, stale)
    if finished:
        return finished
    if not olds and canonical_id:
        try:
            failed = _retire_other_wordings(memory, predicate, {canonical_id, drawer_id})
        except Exception:
            failed = ["unreadable"]
        if failed:
            return _wording_incomplete(obj)
    linked = _link_provenance(memory, links, obj, drawer_id)
    if linked:
        return linked
    if olds:
        return _status("replaced", "Memory: saved. Replaced %s." % old_names)
    if room == FACT_ROOM and links and not [u for u in stale if u["drawer_id"] != drawer_id]:
        return _status("already_saved", "Memory: already saved.")
    return _status("saved", "Memory: saved.")


def remember_fact(user_prompt, memory, provider):
    """Save a "Remember this:" fact and report what happened.

    Returns {"status", "message"}, or None when the line is not a non-empty
    "Remember this:" command. status is saved, replaced, already_saved,
    saved_as_written (no relationship identified, so only the words were
    stored), not_saved, or incomplete. An incomplete update keeps its words in
    facts-pending, which recall does not treat as a current fact.
    """
    if not user_prompt.lower().startswith("remember this:"):
        return None
    fact = user_prompt[len("remember this:"):].strip()
    if not fact:
        return None
    if memory is None or memory.palace is None or getattr(memory, "add_drawer", None) is None:
        return _not_saved("Saved memory is unavailable in this session.")
    if memory.kg is None:
        return _save_as_written(
            memory, fact, "The knowledge graph is unavailable, so nothing was checked or replaced"
        )

    triple = _resume_pending(memory, fact)
    if triple is None:
        try:
            triple = fact_to_triple(fact, provider)
        except ProviderError as exc:
            print("Warning: could not extract a knowledge-graph triple: %s" % exc)
            triple = None
        except Exception:
            triple = None
        if triple is None:
            return _save_as_written(
                memory, fact,
                "No clear fact could be extracted, so the knowledge graph was not updated "
                "and nothing was replaced",
            )
    if triple[1] not in SINGLE_VALUE_PREDICATES:
        return _save_additive(memory, fact, triple)
    if triple[0].casefold() != "user" or triple[2].casefold() not in fact.casefold():
        return _save_unfinished(
            memory, fact, "The extracted fact did not clearly match the text, so nothing was replaced"
        )
    return _save_single_value(memory, fact, triple)


def request_char_limit():
    """Total characters for one request. A positive env value overrides the default."""
    raw = os.environ.get("TINYTALK_MAX_REQUEST_CHARS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return MAX_REQUEST_CHARS


def _request_chars(soul, messages):
    total = len(soul or "")
    for message in messages:
        total += len((message or {}).get("content") or "")
    return total


def _drop_oldest_turn(history):
    """Remove the oldest user/assistant turn, or one leftover message."""
    if (
        len(history) >= 2
        and history[0].get("role") == "user"
        and history[1].get("role") == "assistant"
    ):
        return history[2:]
    return history[1:]


def _memory_lead(content):
    """The instruction in front of MEMORY_LIMITS, without kind notes."""
    limits_at = content.find(MEMORY_LIMITS)
    if limits_at == -1:
        return None
    lead = content[:limits_at].rstrip()
    for sentence in (UNFINISHED_NOTE, UNCLASSIFIED_NOTE):
        if lead.endswith(sentence):
            lead = lead[: -len(sentence)].rstrip()
    return lead


def _shrink_memory_message(message, kept, omitted, limit):
    """Rebuild one memory block with trailing records removed."""
    note = ""
    if omitted:
        note = (
            "\n\n%d retrieved records were omitted so this request stays within %d characters."
            % (omitted, limit)
        )
    content = message.get("content") or ""
    lead = _memory_lead(content)
    if lead is not None:
        rebuilt, _sources = _with_sources(lead, kept, [])
        text = rebuilt[0]["content"] + note
        return {"role": "system", "content": text}
    marker = "\n\nRetrieved sources:\n\n"
    if marker in content:
        head = content.split(marker, 1)[0]
        if kept:
            text = head + marker + render_sources(kept)
        else:
            text = head
        return {"role": "system", "content": text + note}
    if omitted:
        return {"role": "system", "content": content + note}
    return message


def fit_outgoing(soul, request_messages, sources):
    """Drop oldest history, then trailing records, until the input fits.

    Returns the fitted messages and the records still included, or None when
    the soul and the new message alone cannot fit under the input budget.
    """
    limit = request_char_limit()
    budget = limit - REPLY_RESERVE_CHARS
    request_messages = list(request_messages or [])
    sources = list(sources or [])
    split = 0
    while split < len(request_messages) and request_messages[split].get("role") == "system":
        split += 1
    memory_messages = request_messages[:split]
    chat = request_messages[split:]
    if not chat:
        return None
    memory_message = memory_messages[0] if memory_messages else None
    extras = memory_messages[1:]
    user = chat[-1]
    history = list(chat[:-1])
    kept = list(sources)

    def packed(note_omitted):
        block = None
        if memory_message is not None and (kept or note_omitted):
            if note_omitted:
                block = _shrink_memory_message(memory_message, kept, note_omitted, limit)
            else:
                block = memory_message
        messages = []
        if block is not None:
            messages.append(block)
        messages.extend(extras)
        messages.extend(history)
        messages.append(user)
        return messages

    while True:
        omitted = len(sources) - len(kept)
        messages = packed(omitted)
        if budget >= 1 and _request_chars(soul, messages) <= budget:
            return messages, kept
        if history:
            history = _drop_oldest_turn(history)
            continue
        if kept:
            kept = kept[:-1]
            continue
        bare = extras + [user]
        if budget >= 1 and _request_chars(soul, bare) <= budget:
            return bare, []
        return None


def handle_turn(user_prompt, messages, provider, memory, soul, sources=None):
    """Run one user turn.

    On a provider failure the history is left unchanged and nothing is written
    to MemPalace. Returns the updated message list.
    """
    prior = list(messages)
    messages = list(messages)
    messages.append({"role": "user", "content": user_prompt})
    messages = messages[-(MAX_TURNS * 2):]
    request_messages, included = build_context(user_prompt, messages, memory)
    fitted = fit_outgoing(soul, request_messages, included)
    if fitted is None:
        print(
            "That message is too long for the %d-character request limit. Nothing was saved."
            % request_char_limit()
        )
        return prior
    request_messages, included = fitted
    try:
        response = provider.complete(model_messages(soul, request_messages))
    except ProviderError as exc:
        print("%s Nothing was saved." % exc)
        return prior
    if sources is not None:
        sources.replace(included)

    messages.append({"role": "assistant", "content": response})
    messages = messages[-(MAX_TURNS * 2):]
    print("%s: %s" % (provider.label, response))
    # The reply was written before the fact was stored, so this line, not the
    # reply, says what happened to the memory. The conversation record is
    # saved after that result so later recall can see the final status.
    result = remember_fact(user_prompt, memory, provider)
    if result is not None:
        print(result["message"])
    if memory is not None and memory.palace is not None:
        try:
            if result is not None:
                memory.save_exchange(user_prompt, response, result["message"])
            else:
                memory.save_exchange(user_prompt, response)
        except Exception:
            print("Warning: could not save this turn to MemPalace.")
    print("-" * 30)
    return messages


def greeting(provider):
    if provider.name == "grok":
        return (
            "🤖 Hello! I'm TinyTalk, using Grok through the xAI API. "
            "(Type 'quit' to exit)"
        )
    model = getattr(provider, "model", None) or "Llama"
    return "🤖 Hello! I'm your %s assistant, running locally. (Type 'quit' to exit)" % model


def handle_user_line(user_prompt, messages, provider, memory, soul, speech, sources=None):
    """Route one input line. Speech and /sources do not call the model or memory."""
    lowered = (user_prompt or "").strip().lower()
    if lowered == "/sources":
        records = sources.records if sources is not None else []
        print(format_sources_report(records))
        return messages
    if lowered.startswith("/sources"):
        print("Use /sources to list the saved memories included with the last answer.")
        return messages
    kind = command_kind(user_prompt)
    if kind == "speak":
        speech.speak_last(messages)
        return messages
    if kind == "stop":
        speech.stop()
        return messages
    if kind == "usage":
        print("Use /speak to play the last answer, or /stop to stop playback.")
        return messages
    return handle_turn(user_prompt, messages, provider, memory, soul, sources)


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    try:
        settings = load_settings(os.environ)
        provider = build_provider(settings)
    except ConfigError as exc:
        print("Configuration error: %s" % exc)
        return 1

    base_url, voice_id = speech_settings(os.environ)
    speech = SpeechController(VoiceStudioSpeech(base_url, voice_id), Afplay())
    memory = open_memory()
    sources = SessionSources()
    soul = load_soul()
    print(greeting(provider))
    print(
        "Type /speak to hear the last answer, /stop to stop playback, "
        "or /sources to list memories included with the last answer."
    )
    print("-" * 30)
    run_repl(provider, memory, soul, speech, sources, input)
    return 0


def run_repl(provider, memory, soul, speech, sources, read_line, messages=None):
    """Read chat lines until quit, EOF, or Ctrl-C, then stop speech."""
    if messages is None:
        messages = []
    try:
        while True:
            try:
                user_prompt = read_line("You: ")
            except EOFError:
                print()
                break
            if user_prompt.lower() == "quit":
                print("Goodbye! 👋")
                break
            messages = handle_user_line(
                user_prompt, messages, provider, memory, soul, speech, sources
            )
    except KeyboardInterrupt:
        print()
    finally:
        speech.stop(quiet=True)
    return messages


if __name__ == "__main__":
    sys.exit(main())
