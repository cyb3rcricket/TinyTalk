# TinyTalk Error Report & Resolution Plan

**Date:** 2026-11-13  
**Repository:** `/Volumes/Backup Plus/GitHub/TinyTalk`  
**Scope:** Static analysis, baseline test verification, and isolated MemPalace reproduction probes (no source code modified).

---

## Baseline Verification

- **Syntax / bytecode compilation:** `.venv/bin/python -m py_compile *.py scripts/*.py tests/*.py` exited `0`.
- **Unit test suite:** `.venv/bin/python -m unittest discover -s tests` — **187 tests ran in 24.822s — `OK`** (0 failures, 0 skipped).
- **Gradio warnings run:** `.venv/bin/python -X tracemalloc=5 -W default -m unittest tests.test_gradio_app -v` — **11 tests ran in 18.548s — `OK`**, emitting `ResourceWarning: unclosed event loop` warnings from upstream `gradio` (`gradio/blocks.py:2970`, `gradio/routes.py:397` -> `gradio/utils.py:124, 133`).

---

## Error Report (Most Severe to Least Severe)

### 1. [High] Retrying an unclassified `"Remember this:"` save leaves stale metadata on the `facts` drawer and puts the profile field into `conflict`

- **Location:** `memory_core.py` (`_store_drawer` L2022–2072, `_metadata_agrees` L1550–1558, `_classify_profile_predicate` L1641–1647)
- **Root Cause:** When triple extraction fails, `_save_as_written` files the drawer into `facts` with `source_file = "tinytalk?unclassified=1"`. If the user later repeats the exact same `"Remember this:"` phrasing and extraction succeeds, `_store_drawer` finds `drawer["text"] == fact` already in `facts` (L2038–2041) and returns `(drawer_id, FACT_ROOM)` immediately without updating the drawer's `source_file` metadata to record `predicate`/`object` and clear `unclassified=1`. `_save_single_value` then links the active KG edge to that drawer and reports `{'status': 'saved', 'message': 'Memory: saved.'}`. On read, `_metadata_agrees` returns `False` because `_is_unclassified(metadata)` is `True`.
- **Verified Reproduction (isolated MemPalace store):**
  1. `remember_fact("Remember this: call me Al", memory, FakeProvider("NONE"))` -> `{'status': 'saved_as_written', ...}`.
  2. `remember_fact("Remember this: call me Al", memory, FakeProvider('{"subject":"user","predicate":"preferred_name","object":"Al"}'))` -> `{'status': 'saved', 'message': 'Memory: saved.'}`.
  3. `read_current_profile(memory)` -> `preferred_name` has `state: 'conflict'`, `value: None`, `reason: 'The drawer metadata does not match the graph edge.'`.

---

### 2. [High] Retrying a failed pending-move update with a paraphrase leaves the active KG edge pointing at `facts-abandoned` (`conflict`)

- **Location:** `memory_core.py` (`_link_provenance` L1360–1372, `_save_single_value` L2294–2372)
- **Root Cause:** If moving a new `facts-pending` drawer into `facts` fails during a profile update, `supersede` can still have set the active KG edge's `source_drawer_id` to that pending drawer (`incomplete`). If the user retries with a different wording (`_resume_pending` only matches exact fact text), `_save_single_value` sees `olds == []`, saves and promotes the new drawer to `facts`, calls `_finish` (moving the stale pending drawer to `facts-abandoned`), and calls `_link_provenance`. Because `_link_provenance` filters to `not link.get("source_drawer_id")` (L1369), it skips updating the active KG edge because `source_drawer_id` is already populated with the abandoned drawer's ID.
- **Verified Reproduction (isolated MemPalace store):**
  1. Save `"Remember this: call me Bob"` -> `'saved'`.
  2. Save `"Remember this: call me Ada"` with `move_drawer(..., room="facts")` failing -> `{'status': 'incomplete', ...}`.
  3. Retry as `"Remember this: my preferred name is Ada"` -> returns `{'status': 'saved', 'message': 'Memory: saved.'}`, while `read_current_profile(memory)` reports `preferred_name` in `state: 'conflict'`, `value: None`, `reason: 'The graph edge points at facts-abandoned, not current facts.'`.

---

### 3. [High] Semantic fact search results lack `metadata`, mislabeling `saved_as_written` fallback text as a confirmed `saved fact` / `current`

- **Location:** `memory_core.py` (`_coerce_saved_fact` L703–718), `tinytalk.py` (`_semantic_sources` L248–257), `mcp_server.py` (`search_memories` L80–89)
- **Root Cause:** `mempalace`'s `search_memories` / `memory.search_facts()` returns hit dicts containing `source_file` / `source_path` (e.g., `'tinytalk?unclassified=1'`), **without** a `'metadata'` dictionary. `_coerce_saved_fact` only inspects `item.get("metadata") or {}` (L706), so `_is_unclassified(metadata)` always evaluates to `False` for search hits. Both `tinytalk._semantic_sources` (marks `status: "current"`) and `mcp_server.search_memories` (emits `label: "semantic fact"`) present unconfirmed fallback text identically to confirmed facts.
- **Verified Reproduction (isolated MemPalace store):**
  - After `remember_fact("Remember this: call me Al", memory, FakeProvider("NONE"))`, `memory.search_facts("call me Al")` returned `'source_file': 'tinytalk?unclassified=1'` with no `'metadata'` key; `_coerce_saved_fact` produced `kind: 'saved fact'`, `_semantic_sources` produced `status: 'current'`, and `mcp_server.search_memories(memory, "call me Al")` returned `semantic_facts: [{'label': 'semantic fact', 'similarity': 1.0, 'text': 'call me Al'}]` while `preferred_name` was `'missing'`.

---

### 4. [Medium] Historical recall and retired-command filtering miss multi-line facts and legacy `test_spaceship_name` metadata

- **Location:** `memory_core.py` (`_command_fact` L833–843, `_history_facts` L269–300), `tinytalk.py` (`recall_context` L347–353), `mcp_server.py` (`recall_thread` L174–183)
- **Causes (static inspection):**
  1. `_command_fact` returns `rest.splitlines()[0].strip() or None` (L843), dropping lines after the first newline, whereas `remember_fact` stores `normalized[len("remember this:"):].strip()` intact. Multi-line facts moved to `facts-history` or `facts-abandoned` will not match `fact in retired`, allowing retired multi-line commands to leak back into `/recall` and `recall_thread`.
  2. `_history_facts` checks `metadata` via `_fact_relationship` for all profile predicates except `"test_spaceship_name"`, where it ignores drawer metadata and relies solely on `test_spaceship_fact_name(text)` (L281–290). Unclassified drawers matching that phrase enter profile history, while paraphrased `test_spaceship_name` drawers that don't match the legacy regex are omitted.
  3. `recall_context` and `recall_thread` run `_from_unfinished_command` / `_from_retired_command` on the search hit preview **before** `_hydrate_conversation` expands long drawer content (`tinytalk.py` L350–352, `mcp_server.py` L177–182), so truncated previews can bypass filtering.

---

### 5. [Medium] `mcp_server.py` `sync_thread` partial-availability misclassification

- **Location:** `mcp_server.py` (`sync_thread` L133–158)
- **Cause (static inspection):** `sync_thread` checks `if memory is None or getattr(memory, "save_exchange", None) is None:` (L133), without checking `getattr(memory, "palace", None) is None` or `callable(memory.save_exchange)` (unlike `recall_thread` at L165 and `remember_fact` at `memory_core.py:2478`). When passed a `MemoryStore` whose `palace` is `None`, `save_exchange` raises inside the `try` block and returns `{"error": "Could not save an exchange.", "saved": 0, "skipped": 0}` instead of `"Saved memory is unavailable."`.

---

### 6. [Medium] `gradio_app.py` repeat-turn detection edge case, unclosed DB/event loops, and unsynchronized shared memory state

- **Location:** `gradio_app.py` (`_appended_assistant` L47–59, `submit_message` L116–156, `create_app` L161–181, `main` L196–203)
- **Causes:**
  1. **Repeat turn at `MAX_TURNS` cap (static):** If `before` is already at the rolling history window cap (`MAX_TURNS * 2` messages) and a new turn produces the exact same sliding list such that `after == before`, `_appended_assistant` returns `None` (L49). `submit_message` then enters the command/error `else` branch (L151–155) and displays raw `captured` stdout including `"Script: ..."` and `"------------------------------"`.
  2. **Unclosed `memory.kg` connection (static):** `gradio_app.main()` creates `app = create_app()` (which opens `memory`) and never closes `memory.kg` when `app.launch()` exits (unlike `mcp_server.main()` at `mcp_server.py:341–346`).
  3. **Concurrency risk (static):** All Gradio requests share one `memory` (`MemoryStore` + `mempalace.mcp_server` tools + SQLite KG connection) across Gradio worker threads without locking.
  4. **Unclosed asyncio event loops (observed in test run):** Instantiating `gr.Blocks` / `App` triggers `ResourceWarning: unclosed event loop` inside `gradio==6.31.0` (`gradio/blocks.py:2970`, `gradio/routes.py:397`).

---

### 7. [Medium] `tinytalk.py` CLI REPL input handling, stale session-name state, and unclosed KG handle

- **Location:** `tinytalk.py` (`_session_name_request` L185–210, `handle_user_line` L696–701, `main` L715–738, `run_repl` L741–762)
- **Causes (static inspection):**
  1. **Stale `_session_name_request` after saving permanent name:** `_session_name_request` skips `"remember this:"` lines (L195–196). If the user previously said `"call me River"` in chat and later saves `"Remember this: call me Willow"`, the earlier `"call me River"` user message remains in `messages`, so `_session_name_request` continues returning `"River"` even after the permanent profile fact was saved to `"Willow"`.
  2. **Blank CLI lines trigger LLM turns:** `run_repl` passes empty/whitespace input directly to `handle_user_line` -> `handle_turn` (L748–757), unlike `gradio_app.py:119–124`.
  3. **Narrow `/recall` prefix check:** L696 checks `lowered == "/recall" or lowered.startswith("/recall ")`, whereas `/sources`, `/memory`, and `/forget` use `lowered.startswith(...)` (L676–695). `/recall\tquery` or `/recallfoo` falls through to `handle_turn` as a normal chat prompt.
  4. **Unclosed `memory.kg`:** `tinytalk.main()` opens `memory = open_memory()` (L726) and never closes `memory.kg` on exit.

---

### 8. [Medium] `speech.py` single-call `os.write` assumption and lock held across process termination

- **Location:** `speech.py` (`Afplay.play` L164–198)
- **Causes (static inspection):**
  1. `os.write(fd, wav_bytes)` (L171) assumes the entire WAV buffer is written in one syscall rather than looping or wrapping `fd` with `os.fdopen(fd, "wb")`.
  2. `_terminate_process(proc)` (up to ~4s wait) runs while holding `self._lock` (L182–185), blocking concurrent `stop()` / `speak_last()` calls.

---

### 9. [Low] `scripts/dream_pass.py` uncaught drawer-read failures, direct env mutation, and unclosed KG handle

- **Location:** `scripts/dream_pass.py` (`_call_quietly` L57–74, `main` L449–480)
- **Causes (static inspection):**
  1. `main()` catches `GraphReadError` around `render_report(memory)` (L469) but does not catch `RuntimeError` from `_room_drawers` (`memory_core.py:2003, 2010`) or missing palace/list failures, producing a raw traceback on partial stores.
  2. `_call_quietly` mutates global `os.environ["TINYTALK_DEBUG"]` directly instead of using `memory_core.scoped_io(output=..., debug=False)`.
  3. `main()` opens default memory via `_open_default_memory()` without closing `memory.kg` before returning.

---

### 10. [Low] Configuration default mismatch between `.env.example` and `provider.py` / `README.md`

- **Location:** `.env.example` (L14–15), `provider.py` (L25), `README.md` (L440–450)
- **Cause:** `provider.py` sets `DEFAULT_OLLAMA_MODEL = "llama3.2:3b"` and `README.md` documents pulling and defaulting to `llama3.2:3b`, whereas `.env.example` sets `OLLAMA_MODEL=phi4-mini` (right below `# llama3.2:3b is the model TinyTalk already used.`). Copying `.env.example` to `.env` switches Ollama to `phi4-mini`.

---

## Recommended Resolution Plan

### Phase 1 — Fix Profile Integrity & Search Classification (`memory_core.py`, `tinytalk.py`, `mcp_server.py`)

1. **Fix `_store_drawer` reuse of `facts` drawers (`memory_core.py` L2037–2043):**
   - When an existing drawer in `FACT_ROOM` matches `fact` and `(predicate, obj)` are supplied, check whether `_is_unclassified(drawer["metadata"])` is `True` or its metadata `(predicate, obj)` disagrees. Update its `source_file` metadata (`_fact_source(predicate, obj)` via `memory.update_drawer`) — or route it through replacement/repair — before returning `(drawer_id, FACT_ROOM)`.
   - Add an isolated test in `tests/test_fact_replacement.py` covering: `Remember this: call me Al` (extraction fails -> `saved_as_written`) followed by `Remember this: call me Al` (extraction succeeds -> `'saved'` and `preferred_name` is `'current'`).

2. **Fix `_link_provenance` stale/abandoned pointer repair (`memory_core.py` L1360–1372):**
   - Allow `_link_provenance` (or a caller-controlled flag from `_save_single_value` when promoting a replacement/paraphrased retry) to update `source_drawer_id` on matching active edges even when `link.get("source_drawer_id")` is non-empty, whenever the existing `source_drawer_id` is in `facts-pending` / `facts-abandoned` / missing (or stale relative to the newly promoted `FACT_ROOM` drawer).
   - Add an isolated test covering: `Bob` -> failed move on `call me Ada` (`incomplete`) -> paraphrased retry `my preferred name is Ada` (`saved` and `preferred_name` ends `'current'` pointing to the `facts` drawer).

3. **Preserve classification on semantic search hits (`memory_core.py` L703–718, `tinytalk.py` L248–257, `mcp_server.py` L80–89):**
   - Update `_is_unclassified` / `_coerce_saved_fact` (and `mcp_server.search_memories`) to check `item.get("source_file")` / `item.get("source_path")` in addition to `item.get("metadata")`, so hits from `tinytalk?unclassified=1` remain `UNCLASSIFIED_KIND` (`"saved as written"`) rather than `"saved fact"` / `"current"`.
   - Add regression tests in `tests/test_memory_recall.py` and `tests/test_mcp_server.py`.

### Phase 2 — Fix Historical Recall & MCP Partial Availability (`memory_core.py`, `tinytalk.py`, `mcp_server.py`)

1. **Multi-line `_command_fact` & pre/post-hydration recall filtering:**
   - Update `_command_fact` (`memory_core.py` L833–843) to preserve the full fact text after `"Remember this:"` (matching `remember_fact`) or fall back cleanly for multi-line commands, and run `_from_unfinished_command` / `_from_retired_command` on the hydrated conversation drawer text in `tinytalk.recall_context` and `mcp_server.recall_thread`.

2. **`_history_facts` metadata consistency:**
   - In `_history_facts` (`memory_core.py` L281–290), prefer `_fact_relationship(metadata)` for `test_spaceship_name` when present and only fall back to `test_spaceship_fact_name(text)` when metadata has no relationship (or guard against unclassified drawers matching accidentally).

3. **`mcp_server.sync_thread` availability check:**
   - Align `sync_thread` (`mcp_server.py` L133–135) with `recall_thread` by checking `memory.palace is None` and `not callable(getattr(memory, "save_exchange", None))` before calling `save_exchange`.

### Phase 3 — Harden CLI, Web UI, Audio, Scripts, and Resource Lifecycle

1. **Close `memory.kg` cleanly across entrypoints:**
   - Wrap `tinytalk.main()`, `gradio_app.main()`, and `scripts/dream_pass.py:main()` in `try ... finally` blocks to close `memory.kg` cleanly (matching `mcp_server.main()`).

2. **`gradio_app.py` fixes:**
   - Serialize `memory` operations inside `gradio_app` with a `threading.Lock` so concurrent Gradio worker threads cannot race on shared `MemoryStore` / `mempalace.mcp_server` state.
   - Detect appended turns reliably without relying solely on `after == before` (e.g., returning turn status/output explicitly or comparing turn markers when at `MAX_TURNS * 2`).

3. **`tinytalk.py` CLI fixes:**
   - Ignore blank/whitespace-only lines in `run_repl` instead of sending them to the provider.
   - Treat `/recall<whitespace>...` and unknown `/recall...` commands consistently with `/memory`, `/sources`, and `/forget`.
   - Clear or supersede any prior unsaved `_session_name_request` when a permanent `"Remember this:"` preferred-name save occurs (e.g., ignore `"call me ..."` messages earlier than the latest `"Remember this:"` preferred-name command).

4. **`speech.py`, `scripts/dream_pass.py`, and `.env.example` cleanup:**
   - Replace `os.write(fd, wav_bytes)` in `Afplay.play` with `with os.fdopen(fd, "wb") as handle: handle.write(wav_bytes)`, and release `self._lock` before calling `_terminate_process(proc)`.
   - Replace direct `TINYTALK_DEBUG` env mutation in `scripts/dream_pass.py` with `memory_core.scoped_io(output=..., debug=False)` and handle `_room_drawers` read errors cleanly in `dream_pass.main()`.
   - Align `.env.example` (`OLLAMA_MODEL=llama3.2:3b`) with `provider.py` and `README.md` (or update `README.md` if `phi4-mini` is the intended example).

### Phase 4 — Verification

- Re-run `.venv/bin/python -m py_compile *.py scripts/*.py tests/*.py` and `.venv/bin/python -m unittest discover -s tests -v`.
- Add targeted unit/integration tests covering every fix in Phases 1–3.
