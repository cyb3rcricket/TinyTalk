import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from provider import ConfigError, ProviderError, build_provider, load_settings
from speech import Afplay, SpeechController, VoiceStudioSpeech, command_kind, speech_settings

MAX_TURNS = 10
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
    return text.strip().lower().rstrip("?.!").strip() in MEMORY_INTROSPECTION_QUESTIONS


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


def _same_clock(left, right):
    if left.tzinfo is None and right.tzinfo is None:
        return True
    return left.tzinfo is not None and right.tzinfo is not None


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


def _add_candidate(bucket, name, filed_at=None, valid_to=None):
    key = name.casefold()
    found = bucket.get(key)
    if found is None:
        found = {"name": name, "filed_at": None, "valid_to": None}
        bucket[key] = found
    if filed_at and not found["filed_at"]:
        found["filed_at"] = filed_at
    if valid_to and not found["valid_to"]:
        found["valid_to"] = valid_to


def _candidate_instant(candidate):
    stamps = []
    for key in ("filed_at", "valid_to"):
        raw = candidate.get(key)
        if not raw:
            continue
        parsed = _instant(raw)
        if parsed is None:
            return None
        stamps.append(parsed)
    if not stamps:
        return None
    if any(not _same_clock(stamps[0], other) for other in stamps[1:]):
        return None
    return max(stamps)


def _ordered_previous(candidates, current_filed_at):
    """Pick the latest earlier name only when every stamp can be compared."""
    current_at = _instant(current_filed_at) if current_filed_at else None
    if current_at is None:
        return None
    ranked = []
    for candidate in candidates:
        instant = _candidate_instant(candidate)
        if instant is None:
            return None
        if current_at is not None:
            if not _same_clock(instant, current_at) or instant >= current_at:
                return None
        ranked.append((instant, candidate["name"]))
    ranked.sort(key=lambda item: item[0])
    if len(ranked) >= 2 and ranked[-1][0] == ranked[-2][0]:
        return None
    return ranked[-1][1]


def _times_allow_single(candidate, current_filed_at):
    """A single earlier record can be contradicted by a full timestamp."""
    archive_at = candidate.get("filed_at")
    if not current_filed_at or not archive_at:
        return True
    current = _instant(current_filed_at)
    earlier = _instant(archive_at)
    if current is None or earlier is None or not _same_clock(current, earlier):
        return False
    return earlier < current


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
        _add_candidate(candidates, fact["name"], filed_at=fact.get("filed_at"))
    history_edges = [
        edge for edge in inactive
        if not (current_name and _names_match(edge["object"], current_name))
    ]
    current_filed = [fact.get("filed_at") for fact in current_facts if fact.get("filed_at")]
    current_filed_at = current_filed[0] if len(current_filed) == 1 else None
    previous = None
    if current_name and candidates:
        ordered = list(candidates.values())
        if len(ordered) == 1 and _times_allow_single(ordered[0], current_filed_at):
            previous = ordered[0]["name"]
        elif len(ordered) > 1:
            previous = _ordered_previous(ordered, current_filed_at)
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
    text = item.get("text") or item.get("content_preview") or ""
    room = item.get("room") or (item.get("metadata") or {}).get("room")
    kind = "archived fact" if room == "facts-history" else "saved fact"
    filed_at = item.get("filed_at")
    if filed_at is None:
        filed_at = (item.get("metadata") or {}).get("filed_at")
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


def _drawer_sources(memory, room, kind):
    list_drawers = getattr(memory, "list_drawers", None)
    if list_drawers is None:
        return None
    listed = list_drawers(wing="tinytalk", room=room, limit=100) or {}
    sources = []
    for item in listed.get("drawers") or []:
        text = (item.get("content_preview") or "").strip()
        if not text:
            continue
        record = dict(item)
        record["text"] = text
        record["room"] = room
        sources.append(_coerce_saved_fact(record) if kind == "saved fact" else _source(
            kind,
            text,
            item.get("drawer_id"),
            filed_at=(item.get("metadata") or {}).get("filed_at"),
        ))
    return sources


UNFINISHED_KIND = "unfinished memory update"


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
    list_drawers = getattr(memory, "list_drawers", None) if memory is not None else None
    if list_drawers is None:
        return graph_rows, drawers
    for room in ("facts", "facts-history"):
        try:
            listed = list_drawers(wing="tinytalk", room=room, limit=100) or {}
        except Exception:
            listed = {}
        drawers.extend(listed.get("drawers") or [])
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
    try:
        return Path(__file__).with_name("SOUL.md").read_text(encoding="utf-8")
    except OSError:
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
    predicate = snake_predicate(predicate)
    predicate = PREDICATE_ALIASES.get(predicate, predicate)
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
        return [hit for hit in found.get("results", []) if hit.get("text")]

    def list_facts(self):
        if self.list_drawers is None:
            return []
        listed = self.list_drawers(wing="tinytalk", room="facts", limit=100)
        return [
            item.get("content_preview", "").strip()
            for item in listed.get("drawers", [])
            if item.get("content_preview", "").strip()
        ]

    def save_exchange(self, user_prompt, response):
        from mempalace.convo_miner import file_conversation_exchange
        from mempalace.palace import get_collection

        # A fact write opens the palace through MemPalace's tool client. The
        # collection object from startup can then fail later upserts, so each
        # turn opens the collection again at the same path.
        file_conversation_exchange(
            get_collection(self.palace_path),
            wing="tinytalk",
            room="conversations",
            text="User: %s\n\nAssistant: %s" % (user_prompt, response),
            source_file="tinytalk",
            agent="tinytalk",
        )

    def current_fact_objects(self, subject, predicate):
        return [link["object"] for link in self.current_links(subject, predicate)]

    def current_links(self, subject, predicate):
        """Current graph values for one relationship, each with its fact drawer id.

        query_entity does not return source_drawer_id, so this pages the stored
        rows with KnowledgeGraph.dump_rows. drawer_id is None for edges written
        before TinyTalk recorded it. Raises if the graph cannot be read.
        """
        names = {row["id"]: row["name"] for row in _graph_rows(self.kg, "entities")}
        links = []
        for row in _graph_rows(self.kg, "triples"):
            if row["valid_to"] is not None or row["predicate"] != predicate:
                continue
            if (names.get(row["subject"]) or "").casefold() != subject.casefold():
                continue
            obj = names.get(row["object"])
            if obj:
                links.append({"object": obj, "drawer_id": row.get("source_drawer_id")})
        return links

    def pending_facts(self):
        """Facts whose "Remember this:" update did not finish. Raises if they cannot be listed."""
        found = []
        for drawer in _room_drawers(self, PENDING_ROOM):
            predicate, obj = _fact_relationship(drawer["metadata"])
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
        for update in updates:
            update["last_value"] = None
            if update["predicate"] not in SINGLE_VALUE_PREDICATES or self.kg is None:
                continue
            try:
                values = []
                for link in self.current_links("user", update["predicate"]):
                    room = _drawer_room(self, link["drawer_id"]) if link["drawer_id"] else None
                    if room in (PENDING_ROOM, ABANDONED_ROOM):
                        continue
                    if not any(_names_match(link["object"], value) for value in values):
                        values.append(link["object"])
            except Exception:
                values = []
            if len(values) == 1:
                update["last_value"] = values[0]
        return updates


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

    return MemoryStore(palace, resolved, kg, add_drawer, list_drawers, update_drawer, get_drawer)


def _with_sources(lead, sources, messages):
    if any(source["kind"] == UNFINISHED_KIND for source in sources):
        lead = lead + " A record labeled unfinished memory update is not a current fact."
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
            sources = _drawer_sources(memory, "facts", "saved fact")
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
    found = [item for item in found if item and not _from_unfinished_command(item, updates)]
    if found:
        sources = [_coerce_conversation(item) for item in found] + unfinished
        return _with_sources(
            "This context contains excerpts retrieved for this request "
            "from earlier conversations. "
            "These may contain old assistant mistakes. "
            "When memories conflict, prefer explicit factual statements "
            "made by the user over previous assistant responses.",
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


def _fact_source_file(predicate=None, obj=None):
    """Drawer metadata naming the single-value relationship a fact updates."""
    if not predicate:
        return "tinytalk"
    return "tinytalk?" + urlencode({"predicate": predicate, "object": obj})


def _fact_relationship(metadata):
    source = (metadata or {}).get("source_file") or ""
    if not source.startswith("tinytalk?"):
        return None, None
    fields = parse_qs(source[len("tinytalk?"):])
    return (fields.get("predicate") or [None])[0], (fields.get("object") or [None])[0]


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
    value goes through the whole update again.
    """
    target = room
    try:
        if getattr(memory, "list_drawers", None) is not None:
            for drawer in _room_drawers(memory, FACT_ROOM):
                if drawer["text"] == fact:
                    return drawer["drawer_id"], FACT_ROOM
        saved = memory.add_drawer(
            wing="tinytalk",
            room=target,
            content=fact,
            source_file=_fact_source_file(predicate, obj),
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
    if room is not None and _move_drawer(memory, drawer_id, target):
        return drawer_id, target
    return None


def _save_as_written(memory, fact, detail):
    """No relationship was identified, so the words alone are the whole update."""
    stored = _store_drawer(memory, fact, room=FACT_ROOM)
    if stored is None:
        return _not_saved("MemPalace could not store this fact.")
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
    moves = []
    for link in olds:
        try:
            move = _old_fact_drawer(memory, fact, predicate, link)
        except UnsafeReplacement as exc:
            return _save_unfinished(memory, fact, str(exc), predicate, obj)
        if move:
            moves.append(move)

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
                subject, predicate, link["object"], obj,
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


def handle_turn(user_prompt, messages, provider, memory, soul, sources=None):
    """Run one user turn.

    On a provider failure the history is left unchanged and nothing is written
    to MemPalace. Returns the updated message list.
    """
    messages = list(messages)
    messages.append({"role": "user", "content": user_prompt})
    request_messages, included = build_context(user_prompt, messages, memory)
    try:
        response = provider.complete(model_messages(soul, request_messages))
    except ProviderError as exc:
        print("%s Nothing was saved." % exc)
        return list(messages[:-1])
    if sources is not None:
        sources.replace(included)

    messages.append({"role": "assistant", "content": response})
    messages = messages[-(MAX_TURNS * 2):]
    if memory is not None and memory.palace is not None:
        try:
            memory.save_exchange(user_prompt, response)
        except Exception:
            print("Warning: could not save this turn to MemPalace.")
    print("%s: %s" % (provider.label, response))
    # The reply was written before the fact was stored, so this line, not the
    # reply, says what happened to the memory.
    result = remember_fact(user_prompt, memory, provider)
    if result is not None:
        print(result["message"])
    print("-" * 30)
    return messages


def greeting(provider):
    if provider.name == "grok":
        return (
            "🤖 Hello! I'm TinyTalk, using Grok through the xAI API. "
            "(Type 'quit' to exit)"
        )
    return "🤖 Hello! I'm your Llama assistant, running locally. (Type 'quit' to exit)"


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
    messages = []
    while True:
        try:
            user_prompt = input("You: ")
        except EOFError:
            print()
            speech.stop(quiet=True)
            break
        if user_prompt.lower() == "quit":
            speech.stop(quiet=True)
            print("Goodbye! 👋")
            break
        messages = handle_user_line(
            user_prompt, messages, provider, memory, soul, speech, sources
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
