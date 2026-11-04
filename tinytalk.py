import os
import sys
from pathlib import Path

from provider import ConfigError, ProviderError, build_provider, load_settings
from speech import Afplay, SpeechController, VoiceStudioSpeech, command_kind, speech_settings

from memory_core import (
    MIN_FACT_SIMILARITY,
    MEMORY_LIMITS,
    MEMORY_INTROSPECTION_QUESTIONS,
    SINGLE_VALUE_PREDICATES,
    PROFILE_PREDICATES,
    PROFILE_BUDGET_CHARS,
    PREFERRED_NAME_QUESTIONS,
    FACT_ROOM,
    HISTORY_ROOM,
    PENDING_ROOM,
    ABANDONED_ROOM,
    PREDICATE_ALIASES,
    TEST_SPACESHIP_HISTORY_QUESTIONS,
    _TEST_SPACESHIP_NAME,
    TRIPLE_INSTRUCTIONS,
    _SESSION_NAME,
    snake_predicate,
    is_memory_introspection,
    _normalize_question,
    preferred_name_question,
    test_spaceship_history_question,
    _names_match,
    test_spaceship_fact_name,
    _instant,
    _canon_predicate,
    _history_edges,
    _history_facts,
    _unique_name,
    _graph_instant,
    _add_candidate,
    _previous_name,
    test_spaceship_history_evidence,
    format_test_spaceship_history,
    _metadata_present,
    _source,
    _preview,
    render_source,
    render_sources,
    _fact_source,
    _edge_source,
    history_sources,
    _coerce_saved_fact,
    _coerce_conversation,
    _drawer_sources,
    UNFINISHED_KIND,
    UNCLASSIFIED_KIND,
    UNCLASSIFIED_NOTE,
    UNFINISHED_NOTE,
    _unfinished_text,
    unfinished_memory,
    _from_unfinished_command,
    _conversation_similarity,
    _keep_conversation,
    _command_fact,
    _retired_fact_texts,
    _from_retired_command,
    SessionSources,
    format_sources_report,
    _load_test_spaceship_records,
    _unsettled_row,
    historical_context_message,
    debug_enabled,
    debug,
    memory_instruction,
    fact_to_triple,
    MemoryStore,
    _provenance_supported,
    _attach_source_drawer,
    _link_provenance,
    _links_from_rows,
    _graph_rows,
    _import_memory_tools,
    _point_tools_at_palace,
    open_memory,
    _with_sources,
    _profile_entry,
    _unavailable_profile,
    _metadata_agrees,
    _distinct_objects,
    _drawers_by_id,
    _collapse_proposals,
    _classify_profile_predicate,
    read_current_profile,
    _profile_notice,
    _pending_profile_text,
    _profile_sources,
    profile_budget_chars,
    _budget_profile_sources,
    _session_name_request,
    _session_name_note,
    direct_profile_answer,
    _dedupe_sources,
    _semantic_sources,
    _profile_block,
    _deliver_memory,
    _listed_fact_sources,
    build_context,
    _hydrate_conversation,
    _historical_source,
    recall_context,
    context_messages,
    format_memory_report,
    UnsafeReplacement,
    _status,
    _not_saved,
    _pending,
    _fact_source_file,
    _source_query,
    _fact_relationship,
    _is_unclassified,
    _drawer_room,
    _move_drawer,
    _room_drawers,
    _store_drawer,
    _save_as_written,
    _save_unfinished,
    _finish,
    _old_fact_drawer,
    _save_additive,
    _drawer_text,
    _canonical_drawer_id,
    _predicate_fact_drawers,
    _retire_other_wordings,
    _wording_incomplete,
    _already_current,
    _resume_pending,
    _save_single_value,
    remember_fact,
    _profile_entry_for,
    _direct_sources,
)

MAX_TURNS = 10
# Characters of text in one outgoing request: soul, memory, history, and the
# new message. REPLY_RESERVE_CHARS of that total is left for the reply.
MAX_REQUEST_CHARS = 24000
REPLY_RESERVE_CHARS = 2000


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
        if _drop_last_optional(kept):
            continue
        if any(source.get("required") for source in kept):
            return None
        bare = extras + [user]
        if budget >= 1 and _request_chars(soul, bare) <= budget:
            return bare, []
        return None


def _drop_last_optional(kept):
    """Remove the last source that the profile does not require."""
    for index in range(len(kept) - 1, -1, -1):
        if not kept[index].get("required"):
            del kept[index]
            return True
    return False


def _save_turn(memory, user_prompt, response, memory_note=None):
    if memory is None or getattr(memory, "palace", None) is None:
        return
    try:
        if memory_note:
            memory.save_exchange(user_prompt, response, memory_note)
        else:
            memory.save_exchange(user_prompt, response)
    except Exception:
        print("Warning: could not save this turn to MemPalace.")


def _answer_from_profile(user_prompt, messages, provider, memory, sources, predicate):
    """Deterministic preferred-name answer. The chat model is not called."""
    entry = _profile_entry_for(memory, predicate)
    answer = direct_profile_answer(entry, _session_name_request(messages))
    if sources is not None:
        sources.replace(_direct_sources(entry))
    messages.append({"role": "assistant", "content": answer})
    messages = messages[-(MAX_TURNS * 2):]
    print("%s: %s" % (provider.label, answer))
    _save_turn(memory, user_prompt, answer)
    print("-" * 30)
    return messages


def handle_turn(user_prompt, messages, provider, memory, soul, sources=None):
    """Run one user turn.

    On a provider failure the history is left unchanged and nothing is written
    to MemPalace. Returns the updated message list.
    """
    prior = list(messages)
    messages = list(messages)
    messages.append({"role": "user", "content": user_prompt})
    messages = messages[-(MAX_TURNS * 2):]
    if test_spaceship_history_question(user_prompt) is None:
        predicate = preferred_name_question(user_prompt)
        if predicate:
            return _answer_from_profile(
                user_prompt, messages, provider, memory, sources, predicate
            )
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
    if result is not None:
        _save_turn(memory, user_prompt, response, result["message"])
    else:
        _save_turn(memory, user_prompt, response)
    print("-" * 30)
    return messages


def handle_recall(query, messages, provider, memory, soul, sources=None):
    """Answer /recall from historical excerpts. This does not write a fact."""
    prior = list(messages)
    messages = list(messages)
    messages.append({"role": "user", "content": "/recall " + query})
    messages = messages[-(MAX_TURNS * 2):]
    request_messages, included = recall_context(query, messages, memory)
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
    _save_turn(memory, "/recall " + query, response)
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
    if lowered == "/memory":
        if memory is None:
            entries = _unavailable_profile("Saved memory is unavailable in this session.")
        else:
            entries = read_current_profile(memory)
        print(format_memory_report(entries))
        return messages
    if lowered.startswith("/memory"):
        print("Use /memory to show the current profile.")
        return messages
    if lowered == "/recall" or lowered.startswith("/recall "):
        query = user_prompt.strip()[len("/recall"):].strip()
        if not query:
            print("Use /recall followed by what you want from earlier conversations.")
            return messages
        return handle_recall(query, messages, provider, memory, soul, sources)
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
        "/memory to show the current profile, "
        "/recall to search old conversations, "
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
