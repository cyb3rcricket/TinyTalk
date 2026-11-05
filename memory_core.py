"""TinyTalk memory: facts, the current profile, sources, and MemPalace storage.

Moved out of tinytalk without behavior changes. tinytalk re-exports these names.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode

from provider import ProviderError

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

# Stable order for the derived current profile. Same predicates, no new store.
PROFILE_PREDICATES = (
    "preferred_name",
    "home_city",
    "current_job",
    "favorite_vehicle",
    "test_spaceship_name",
)
PROFILE_BUDGET_CHARS = 1500

# Closed set. Other name questions stay on the model plus the profile block.
PREFERRED_NAME_QUESTIONS = {
    "what is my name": "preferred_name",
    "what is my preferred name": "preferred_name",
    "what should you call me": "preferred_name",
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

# Closed wordings, including literal anchors just like "before enterprise".
# Other values and phrasings remain ordinary questions.
PREFERRED_NAME_HISTORY_QUESTIONS = {
    "what was my name before": "",
    "what was my previous name": "",
    "what was my name before willow": "willow",
}
HOME_CITY_HISTORY_QUESTIONS = {
    "what was my home city before": "",
    "what was my previous home city": "",
    "what was my home city before nashville": "nashville",
}
CURRENT_JOB_HISTORY_QUESTIONS = {
    "what was my job before": "",
    "what was my previous job": "",
    "what was my job before engineer": "engineer",
}
FAVORITE_VEHICLE_HISTORY_QUESTIONS = {
    "what was my favorite vehicle before": "",
    "what was my previous favorite vehicle": "",
    "what was my favorite vehicle before cybertruck": "cybertruck",
}
PROFILE_HISTORY_QUESTIONS = {
    "preferred_name": PREFERRED_NAME_HISTORY_QUESTIONS,
    "home_city": HOME_CITY_HISTORY_QUESTIONS,
    "current_job": CURRENT_JOB_HISTORY_QUESTIONS,
    "favorite_vehicle": FAVORITE_VEHICLE_HISTORY_QUESTIONS,
}
HISTORY_LABELS = {
    "preferred_name": "preferred name",
    "home_city": "home city",
    "current_job": "job",
    "favorite_vehicle": "favorite vehicle",
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
    "The supported single-value predicates are preferred_name, home_city, "
    "current_job, favorite_vehicle, and test_spaceship_name. "
    "If you are not confident there is one clear triple, "
    'return {"subject":"","predicate":"","object":""}. '
    "A coworker's name, a ship's name, a hypothetical name, or a legal name "
    "is not preferred_name. Use preferred_name only for the name the user "
    "asks to be called. "
    "Reply with JSON only. "
    'Example: "my favorite vehicle is a CyberTruck" -> '
    '{"subject":"user","predicate":"favorite_vehicle","object":"CyberTruck"}. '
    'Example: "call me Willow" -> '
    '{"subject":"user","predicate":"preferred_name","object":"Willow"}. '
    'Example: "my preferred name is Willow" -> '
    '{"subject":"user","predicate":"preferred_name","object":"Willow"}'
)

_SESSION_NAME = re.compile(
    r"^(?:call me|my preferred name is)\s+(\S(?:.*\S)?)$",
    re.IGNORECASE,
)


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


def profile_history_question(text):
    """Return (predicate, later value) for a closed history wording, or None."""
    normalized = _normalize_question(text)
    if normalized in PREFERRED_NAME_QUESTIONS or normalized in MEMORY_INTROSPECTION_QUESTIONS:
        return None
    later_name = test_spaceship_history_question(text)
    if later_name is not None:
        return "test_spaceship_name", later_name
    for predicate, questions in PROFILE_HISTORY_QUESTIONS.items():
        if normalized in questions:
            return predicate, questions[normalized]
    return None


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


def _history_edges(graph_rows, predicate="test_spaceship_name"):
    edges = []
    for row in graph_rows or []:
        if not isinstance(row, dict):
            continue
        subject = row.get("subject")
        if subject not in (None, "", "user"):
            continue
        stored = row.get("predicate") or ""
        canonical = _canon_predicate(stored)
        if canonical != predicate:
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
            "predicate": canonical,
            "object": obj.strip(),
            "current": current,
            "valid_from": row.get("valid_from"),
            "valid_to": row.get("valid_to"),
        })
    return edges


def _history_facts(drawers, predicate="test_spaceship_name"):
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
        if predicate == "test_spaceship_name":
            # Keep the legacy spaceship parser and its results unchanged.
            name = test_spaceship_fact_name(text)
        else:
            stored, name = _fact_relationship(item.get("metadata") or {})
            if _canon_predicate(stored) != predicate:
                continue
            if not isinstance(name, str):
                continue
            name = name.strip()
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
    return _history_evidence(_history_edges(graph_rows), _history_facts(drawers), later_name)


def profile_history_evidence(graph_rows, drawers, predicate, later_value):
    """Use saved relationship metadata for the four other profile fields."""
    if predicate == "test_spaceship_name":
        return test_spaceship_history_evidence(graph_rows, drawers, later_value)
    if predicate not in PROFILE_HISTORY_QUESTIONS:
        raise ValueError("Unsupported history predicate")
    return _history_evidence(
        _history_edges(graph_rows, predicate), _history_facts(drawers, predicate), later_value,
    )


def _history_evidence(edges, facts, later_name):
    """Shared ordering rules; filing time and retrieval order do not order values."""
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


def format_profile_history(evidence, predicate):
    """Label earlier profile values as historical, with no inferred change date."""
    if predicate == "test_spaceship_name":
        return format_test_spaceship_history(evidence)
    label = HISTORY_LABELS[predicate]
    current = evidence["current_name"]
    previous = evidence["previous_name"]
    lines = ["Current %s: %s" % (label, current or "not established")]
    for fact in evidence["current_facts"]:
        if current and _names_match(fact["name"], current):
            lines.append("Current fact: %s" % fact["text"])
    for edge in evidence["current_edges"]:
        if current and _names_match(edge["object"], current):
            lines.append("Current graph: user -> %s -> %s" % (predicate, edge["object"]))
    lines.append("Previous %s: %s" % (label, previous or "not established"))
    if previous:
        lines.append("This previous value is historical, not current.")
        for fact in evidence["archived"]:
            if _names_match(fact["name"], previous):
                lines.append("Archived fact: %s" % fact["text"])
        for edge in evidence["history_edges"]:
            if _names_match(edge["object"], previous):
                lines.append(
                    "Inactive graph edge: user -> %s -> %s" % (predicate, edge["object"])
                )
        lines.append(
            "Tell the user the previous %s was %s. Do not give a change date."
            % (label, previous)
        )
    else:
        if evidence["requested_mismatch"]:
            lines.append("The question asks about a value that is not the current saved %s." % label)
        elif evidence["candidate_count"] > 1:
            lines.append(
                "More than one earlier value is saved, and the stored times do not "
                "establish which value came immediately before."
            )
        elif evidence["candidate_count"] == 1:
            lines.append(
                "An earlier record is saved, but its stored time does not establish "
                "it as the previous value."
            )
        else:
            lines.append("The saved records do not include an earlier %s." % label)
        lines.append(
            "Tell the user that the saved records do not establish a previous %s. "
            "Do not guess a value or a date." % label
        )
    if evidence["duplicates"]:
        lines.append(
            "An inactive graph edge repeats the current value. It is not a previous "
            "value, and its end time is not a change date."
        )
    return "\n".join(lines)


def _metadata_present(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in ("", "unknown", "none"):
        return False
    return True


def _source(kind, text, record_id=None, filed_at=None, valid_from=None, valid_to=None,
            selection=None, status=None, required=False):
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
    if selection:
        record["selection"] = selection
    if status:
        record["status"] = status
    if required:
        record["required"] = True
    return record


def _preview(text, limit=80):
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit - 3].rstrip() + "..."


def render_source(source):
    """Label one record for the model without rewriting its text."""
    lines = ["[%s]" % source["kind"]]
    if source.get("selection"):
        lines.append("selection: %s" % source["selection"])
    if source.get("status"):
        lines.append("status: %s" % source["status"])
    if source.get("predicate"):
        lines.append("predicate: %s" % source["predicate"])
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
        if source.get("selection"):
            lines.append("selection: %s" % source["selection"])
        if source.get("status"):
            lines.append("status: %s" % source["status"])
        if source.get("predicate"):
            lines.append("predicate: %s" % source["predicate"])
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


def _unsettled_row(row, names, predicate="test_spaceship_name"):
    if not isinstance(row, dict) or _canon_predicate(row.get("predicate")) != predicate:
        return False
    if "current" in row and row.get("current") is not None:
        current = bool(row.get("current"))
    else:
        current = not row.get("valid_to")
    return current and any(_names_match(row.get("object"), name) for name in names)


def historical_context_message(user_prompt, memory):
    question = profile_history_question(user_prompt)
    if question is not None and question[0] != "test_spaceship_name":
        return _profile_historical_context_message(memory, *question)
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


def _profile_historical_context_message(memory, predicate, later_value):
    """The same history read and source path, scoped to one profile predicate."""
    label = HISTORY_LABELS[predicate]
    unfinished, updates = unfinished_memory(memory)
    relevant = [
        (source, update) for source, update in zip(unfinished, updates)
        if update.get("predicate") == predicate
    ]
    unfinished = [source for source, _update in relevant]
    unsettled = [
        update.get("object") for _source, update in relevant
        if not _names_match(update.get("object"), update.get("last_value"))
    ]
    try:
        graph_rows, drawers = _load_test_spaceship_records(memory)
        graph_rows = [row for row in graph_rows if not _unsettled_row(row, unsettled, predicate)]
        evidence = profile_history_evidence(graph_rows, drawers, predicate, later_value)
        body = format_profile_history(evidence, predicate)
        sources = history_sources(evidence)
    except Exception:
        body = (
            "Previous %s: not established\n"
            "Tell the user that the saved records do not establish a previous %s. "
            "Do not guess a value or a date." % (label, label)
        )
        sources = []
    if unfinished:
        body += "\n" + "\n".join(
            "Unfinished update, not the current value: %s" % source["text"]
            for source in unfinished
        )
        sources += unfinished
    if sources:
        body += "\n\nRetrieved sources:\n\n" + render_sources(sources)
    return {
        "role": "system",
        "content": (
            "This context is the saved %s history for this question. "
            "Answer only from these labels. "
            "These are retrieved sources, not information learned during model training. "
            "Do not mention a fact that is not written here. "
            "Do not give a change date.\n\n%s" % (label, body)
        ),
    }, sources


def debug_enabled():
    """TINYTALK_DEBUG is off unless set to 1, true, yes, or on."""
    return os.environ.get("TINYTALK_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def debug(message):
    if debug_enabled():
        print(message)


def memory_instruction(what, body):
    """Tell the model what this request included, and what it did not."""
    return what + " " + MEMORY_LIMITS + "\n\n" + body


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


def _profile_entry(predicate, state, reason=None, **fields):
    """One derived profile field. Absent provenance stays None."""
    entry = {
        "predicate": predicate,
        "value": None,
        "state": state,
        "source_drawer_id": None,
        "graph_record_id": None,
        "source_text": None,
        "recorded_at": None,
        "pending_value": None,
        "pending_drawer_id": None,
        "reason": reason,
    }
    for key, value in fields.items():
        if key in entry:
            entry[key] = value
    return entry


def _unavailable_profile(reason):
    return [
        _profile_entry(predicate, "unavailable", reason)
        for predicate in PROFILE_PREDICATES
    ]


def _metadata_agrees(drawer, predicate, obj):
    """True when stored relationship metadata names this value."""
    metadata = (drawer or {}).get("metadata")
    if _is_unclassified(metadata):
        return False
    stored, stored_obj = _fact_relationship(metadata)
    if not stored or not stored_obj:
        return False
    return _canon_predicate(stored) == predicate and _names_match(stored_obj, obj)


def _distinct_objects(links):
    objects = []
    for link in links or []:
        obj = link.get("object")
        if obj and not any(_names_match(obj, seen) for seen in objects):
            objects.append(obj)
    return objects


def _drawers_by_id(memory):
    """Full drawers keyed by id. Room comes from the listing, not the id text."""
    found = {}
    for room in (FACT_ROOM, PENDING_ROOM, HISTORY_ROOM, ABANDONED_ROOM):
        for drawer in _room_drawers(memory, room):
            item = dict(drawer)
            item["room"] = room
            found[item.get("drawer_id")] = item
    return found


def _collapse_proposals(proposals):
    unique = []
    for proposal in proposals:
        if not any(_names_match(proposal["object"], have["object"]) for have in unique):
            unique.append(proposal)
    return unique


def _classify_profile_predicate(memory, predicate, links, by_id):
    """One predicate from open edges plus pending drawers. Does not search."""
    proposals = []
    for drawer in by_id.values():
        if drawer.get("room") != PENDING_ROOM:
            continue
        stored, obj = _fact_relationship(drawer.get("metadata"))
        if not stored or not obj or _canon_predicate(stored) != predicate:
            continue
        proposals.append({
            "object": obj,
            "drawer_id": drawer.get("drawer_id"),
            "text": drawer.get("text") or "",
            "filed_at": (drawer.get("metadata") or {}).get("filed_at"),
        })

    if len(_distinct_objects(links)) > 1:
        return _profile_entry(
            predicate, "conflict", "Several active values exist.",
            graph_record_id=links[0].get("triple_id"),
        )

    confirmed = []
    last_confirmed = []
    problems = []
    for link in links:
        drawer_id = link.get("drawer_id")
        if not drawer_id:
            problems.append("The graph edge has no fact drawer.")
            continue
        try:
            room = _drawer_room(memory, drawer_id)
        except Exception:
            return _profile_entry(
                predicate, "unavailable", "A linked fact drawer could not be read.",
                graph_record_id=link.get("triple_id"),
            )
        if room is None:
            problems.append("The linked drawer is missing.")
            continue
        drawer = by_id.get(drawer_id)
        if drawer is None or drawer.get("room") != room:
            problems.append("The linked drawer is not in the room the store reported.")
            continue
        agrees = _metadata_agrees(drawer, predicate, link.get("object"))
        record = {
            "value": link.get("object"),
            "source_drawer_id": drawer_id,
            "graph_record_id": link.get("triple_id"),
            "source_text": drawer.get("text") or "",
            "recorded_at": (drawer.get("metadata") or {}).get("filed_at"),
        }
        if room == FACT_ROOM and agrees:
            if not any(item["source_drawer_id"] == drawer_id for item in confirmed):
                confirmed.append(record)
            continue
        if room == FACT_ROOM:
            problems.append("The drawer metadata does not match the graph edge.")
            continue
        if room == HISTORY_ROOM and agrees:
            last_confirmed.append(record)
        if room in (PENDING_ROOM, HISTORY_ROOM, ABANDONED_ROOM):
            problems.append(
                "The graph edge points at %s, not current facts." % room
            )
            if room == PENDING_ROOM and agrees and link.get("object"):
                proposals.append({
                    "object": link.get("object"),
                    "drawer_id": drawer_id,
                    "text": drawer.get("text") or "",
                    "filed_at": record["recorded_at"],
                })
            continue
        problems.append("The linked drawer is not a current fact.")

    if len(confirmed) > 1:
        return _profile_entry(
            predicate, "conflict", "Several current facts exist for this value.",
            graph_record_id=confirmed[0].get("graph_record_id"),
        )
    confirmed_record = confirmed[0] if confirmed else None
    proposals = _collapse_proposals(proposals)
    if len(proposals) > 1:
        return _profile_entry(
            predicate, "conflict", "Several unfinished values exist.",
            graph_record_id=(links[0].get("triple_id") if links else None),
        )
    proposal = proposals[0] if proposals else None

    last = confirmed_record
    if last is None:
        last_values = []
        for record in last_confirmed:
            if not any(_names_match(record["value"], seen) for seen in last_values):
                last_values.append(record["value"])
        if len(last_values) == 1:
            last = last_confirmed[0]

    if proposal is not None:
        fields = {
            "pending_value": proposal["object"],
            "pending_drawer_id": proposal["drawer_id"],
            "source_drawer_id": proposal["drawer_id"],
            "source_text": proposal["text"],
            "recorded_at": proposal.get("filed_at"),
        }
        if last is not None:
            fields["value"] = last["value"]
            fields["graph_record_id"] = last.get("graph_record_id")
        elif links:
            fields["graph_record_id"] = links[0].get("triple_id")
        return _profile_entry(
            predicate, "pending", "An unfinished update is still pending.", **fields
        )

    if confirmed_record is not None and not problems:
        return _profile_entry(predicate, "current", None, **confirmed_record)

    if problems:
        pointed_away = last is not None and all(
            problem.startswith("The graph edge points at")
            for problem in problems
        )
        if pointed_away:
            return _profile_entry(
                predicate, "pending", problems[0],
                value=last["value"],
                source_text=last.get("source_text"),
                source_drawer_id=last.get("source_drawer_id"),
                graph_record_id=last.get("graph_record_id"),
                recorded_at=last.get("recorded_at"),
            )
        return _profile_entry(
            predicate, "conflict", problems[0],
            graph_record_id=(links[0].get("triple_id") if links else None),
            source_drawer_id=(links[0].get("drawer_id") if links else None),
        )

    return _profile_entry(predicate, "missing", "No confirmed saved value.")


def read_current_profile(memory):
    """Rebuild the current profile from the graph and fact drawers.

    One graph read covers every supported predicate. Embeddings are not used.
    A missing graph or an unreadable store is unavailable, not a guess.
    """
    if memory is None or getattr(memory, "kg", None) is None:
        return _unavailable_profile("The knowledge graph is unavailable.")
    if getattr(memory, "list_drawers", None) is None or getattr(memory, "get_drawer", None) is None:
        return _unavailable_profile("Fact drawers are unavailable.")
    try:
        names = {row["id"]: row["name"] for row in _graph_rows(memory.kg, "entities")}
        rows = _graph_rows(memory.kg, "triples")
    except Exception:
        return _unavailable_profile("The knowledge graph could not be read.")
    try:
        by_id = _drawers_by_id(memory)
    except Exception:
        return _unavailable_profile("Fact drawers could not be read.")
    entries = []
    for predicate in PROFILE_PREDICATES:
        links = _links_from_rows(names, rows, "user", predicate)
        entries.append(_classify_profile_predicate(memory, predicate, links, by_id))
    return entries


def _profile_notice(entry):
    predicate = entry["predicate"]
    state = entry["state"]
    reason = entry.get("reason") or ""
    if state == "missing":
        return "No confirmed saved %s." % predicate
    if state == "unavailable":
        return "The saved %s is unavailable. %s" % (predicate, reason)
    if state == "conflict":
        return "The saved records do not settle %s. %s" % (predicate, reason)
    return reason or predicate


def _pending_profile_text(entry):
    return _unfinished_text({
        "text": entry.get("source_text") or entry.get("pending_value") or "",
        "predicate": entry["predicate"],
        "last_value": entry.get("value"),
    })


def _profile_sources(entries, unfinished_sources):
    """Required sources for the derived profile. Pending reuses an existing notice."""
    unfinished_by_id = {
        source.get("id"): source
        for source in unfinished_sources or []
        if source.get("id")
    }
    sources = []
    for entry in entries:
        predicate = entry["predicate"]
        state = entry["state"]
        if state == "current" and entry.get("source_text"):
            source = _source(
                "saved fact",
                entry["source_text"],
                entry.get("source_drawer_id"),
                filed_at=entry.get("recorded_at"),
                selection="profile lookup",
                status="current",
                required=True,
            )
            source["predicate"] = predicate
            sources.append(source)
            continue
        if state == "pending":
            pending_id = entry.get("pending_drawer_id") or entry.get("source_drawer_id")
            existing = unfinished_by_id.get(pending_id) if pending_id else None
            if existing is not None:
                existing["required"] = True
                existing["selection"] = "profile lookup"
                existing["status"] = "pending"
                existing["predicate"] = predicate
                sources.append(existing)
                continue
            source = _source(
                UNFINISHED_KIND,
                _pending_profile_text(entry),
                pending_id,
                filed_at=entry.get("recorded_at"),
                selection="profile lookup",
                status="pending",
                required=True,
            )
            source["predicate"] = predicate
            sources.append(source)
            continue
        source = _source(
            "current profile",
            _profile_notice(entry),
            entry.get("graph_record_id") or entry.get("source_drawer_id"),
            selection="profile lookup",
            status=state,
            required=True,
        )
        source["predicate"] = predicate
        sources.append(source)
    return sources


def profile_budget_chars():
    """Characters for the current-profile block. A positive env value overrides it."""
    raw = os.environ.get("TINYTALK_PROFILE_BUDGET_CHARS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return PROFILE_BUDGET_CHARS


def _budget_profile_sources(sources):
    """Keep whole profile entries. Say so when the budget cannot hold all of them."""
    budget = profile_budget_chars()
    kept = []
    used = 0
    omitted = 0
    for source in sources:
        size = len(render_source(source)) + 2
        if used + size > budget:
            omitted += 1
            continue
        kept.append(source)
        used += size
    if omitted:
        note = _source(
            "current profile",
            "%d current-profile entries did not fit in the %d-character profile budget."
            % (omitted, budget),
            selection="profile lookup",
            status="unavailable",
            required=True,
        )
        kept.append(note)
    return kept


def _dedupe_sources(sources, earlier):
    seen_ids = {source.get("id") for source in earlier if source.get("id")}
    seen_text = {
        (source.get("text") or "").strip()
        for source in earlier
        if (source.get("text") or "").strip()
    }
    kept = []
    for source in sources:
        text = (source.get("text") or "").strip()
        if source.get("id") and source.get("id") in seen_ids:
            continue
        if text and text in seen_text:
            continue
        kept.append(source)
        if source.get("id"):
            seen_ids.add(source["id"])
        if text:
            seen_text.add(text)
    return kept


def _hydrate_conversation(memory, item):
    """Full drawer text when the store can read it. Truncated previews stay opaque."""
    if not isinstance(item, dict):
        return item
    drawer_id = item.get("drawer_id") or item.get("id")
    if not drawer_id or getattr(memory, "get_drawer", None) is None:
        return item
    try:
        found = memory.get_drawer(drawer_id)
    except Exception:
        return item
    if not isinstance(found, dict) or found.get("error") or not found.get("content"):
        return item
    hydrated = dict(item)
    hydrated["text"] = found.get("content")
    return hydrated


def _historical_source(item):
    source = _coerce_conversation(item)
    source["selection"] = "explicit history recall"
    source["status"] = "historical"
    text = source.get("text") or ""
    lowered = text.casefold()
    if "user:" not in lowered or "assistant:" not in lowered:
        source["text"] = (
            "Opaque historical text. Speaker roles could not be recovered.\n" + text
        )
    return source


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
    except Exception:
        return None
    try:
        saved = memory.add_drawer(
            wing="tinytalk",
            room=target,
            content=fact,
            source_file=source,
            added_by="tinytalk",
        )
    except Exception as exc:
        debug(str(exc))
        return None
    if not isinstance(saved, dict) or not saved.get("success"):
        if isinstance(saved, dict) and saved.get("error"):
            debug(saved["error"])
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


def _profile_entry_for(memory, predicate):
    if memory is None or getattr(memory, "palace", None) is None:
        return _profile_entry(
            predicate, "unavailable", "Saved memory is unavailable in this session."
        )
    for entry in read_current_profile(memory):
        if entry["predicate"] == predicate:
            return entry
    return _profile_entry(predicate, "missing", "No confirmed saved value.")


def _direct_sources(entry):
    return [
        source for source in _profile_sources([entry], [])
        if source.get("predicate") == entry.get("predicate")
    ]
