# TinyTalk memory MVP

Research and code review: October 5, 2026, America/Chicago.
Reviewed upstream main: `3a4907f67ccbf01e37f82c7d154a09d49756a3bd`.
Scope: implementation plan; no application code or live memories changed, and no tests executed for this report. The user's local checkout version was not independently checked.

## Recommendation

Keep MemPalace, its existing knowledge graph, the provider abstraction, and SOUL.md. Add a small verified current-profile view, load it on each turn, and keep historical conversation evidence separate. Use ordinary Python functions and existing storage APIs. The profile is derived from existing records, not a second editable JSON file or database.

This is the best fit for TinyTalk's observed failure and current size, not a claim that one memory architecture wins for every chatbot.

## What the evidence establishes

The user's live run returned Willow for a broad memory inventory, with `call me Willow` listed as a saved fact. After restart, a specific preferred-name question returned Tommi; its sources were three conversation excerpts containing older assistant answers. The exact reason semantic search missed Willow remains unmeasured: score, indexing, filtering, and local state still need inspection.

The reviewed code explains the available route: `build_context` lists facts for recognized inventory questions, otherwise uses `search_facts`, and falls back to `search_conversations` when no facts are returned. Fact search applies a 0.45 similarity threshold. Existing retired-command filtering detects old Remember-this commands; it does not exclude every old assistant claim about a name. Existing prompt instructions already warn about assistant mistakes, so another warning alone is not a sufficient repair.

[Reviewed source](https://github.com/cyb3rcricket/TinyTalk/blob/3a4907f67ccbf01e37f82c7d154a09d49756a3bd/tinytalk.py)

## Research distilled

| Approach | Useful lesson | TinyTalk decision |
| --- | --- | --- |
| Letta's core memory blocks | Small persistent context can be supplied without a relevance search. [1] | Derive a compact current-profile block on each turn. Do not adopt its runtime. |
| LangMem profiles and collections | A structured current profile and searchable records serve different purposes. [2,3] | Keep current single-value fields separate from other facts and conversation history. |
| Graphiti temporal graphs | Track changes, retain provenance, and combine structural, keyword, and semantic retrieval. [4] | Reuse existing graph links and history; defer an additional graph stack. |
| Mem0 OSS | A configurable memory engine can be embedded or self-hosted. Its documented defaults include cloud model/embedding calls. [5] | Replacement would add migration and configuration work; no switch for this MVP. |
| Long-context prompting | The Lost in the Middle study found position-sensitive retrieval in the models it tested. [6] | Keep evidence compact; do not assume a longer transcript solves memory. |
| LongMemEval | Evaluate updates, cross-session use, time, and abstention separately. [7] | Build a small TinyTalk-specific suite, with evidence selection and answer quality scored separately. |

These are design inputs. Vendor examples and benchmark results do not establish performance on TinyTalk or its local models. Letta's cited block documentation is its legacy V1 API; the architectural pattern is the relevant part.

## The MVP contract

1. Supported current profile facts are available without semantic search.
2. A stored assistant statement never establishes a current personal fact.
3. Repeating a fact or paraphrasing it does not create competing current values.
4. An incomplete update remains visibly incomplete. A graph edge alone cannot promote pending text to current truth.
5. Missing, conflicting, and unreadable state are distinct; none is silently resolved using an old assistant answer.
6. `/sources` reports the evidence actually included after request budgeting, or the records used for a direct deterministic answer.
7. Current user statements can correct conversational behavior; permanent writes remain explicit through `Remember this:` in this MVP. Questions, quotes, hypotheticals, and coworker statements do not silently update the user's profile.

## Memory layers

| Layer | Existing foundation | MVP read behavior |
| --- | --- | --- |
| Identity and behavior | SOUL.md | Remains instructions, separate from remembered facts. |
| Current profile | Graph relationships plus linked fact drawers | Read and validate every turn; bypass embeddings. |
| Other explicit facts | MemPalace facts | Continue semantic retrieval, deduplicated against the profile. |
| Recent conversation | In-process messages | Keep bounded; recent assistant statements remain fallible. |
| Past conversations | Conversation drawers | Historical evidence through an explicit `/recall <query>` path; remove automatic fallback on ordinary fact misses. |
| Replaced or pending facts | Existing history/pending/abandoned rooms | Preserve status and provenance; include only for appropriate history or uncertainty explanations. |

The explicit recall path is a deliberate MVP tradeoff: less spontaneous reference to old chats, with a clear way to request them. Conversation saving continues. No deletion or reclassification of existing conversations is needed.

## Implementation slices

### 1. Establish the failing regression

Add `tests/test_memory_recall.py` using the current isolation helpers and existing stub-provider conventions. Seed a valid current Willow fact and its graph link; seed old assistant conversation assertions about Tommi. Force semantic fact search to return no results, while conversation search returns those old excerpts. Call the production context-building path with empty session history.

Assert the behavioral contract: a preferred-name question receives current Willow evidence and does not receive an old assistant answer as current-name evidence. On the reviewed implementation this should fail because the current profile is not read independently. Record the actual failure before modifying production code. Also instrument the live query's candidate scores to investigate the original semantic miss; do not claim the forced miss proves its cause.

### 2. Add one current-profile resolver

Introduce a focused function, proposed name `read_current_profile(memory)`, returning data rather than prompt text. Start with the five existing single-value predicates: preferred_name, home_city, current_job, favorite_vehicle, and test_spaceship_name. The last two are useful existing regression fixtures; this is a bounded schema, not universal personal knowledge.

Suggested returned fields per entry:

```text
predicate, value, state, source_drawer_id, graph_record_id,
source_text, recorded_at, pending_value, reason
```

`state` is one of current, pending, missing, conflict, or unavailable. These are proposed in-memory fields, not a storage migration. Reuse existing IDs and timestamps; never invent missing provenance or infer the real-world date of a change from filing time.

Read the graph once per turn where practical. Reuse `current_links`, pending-state logic, full drawer reads, and canonical predicate normalization. For a current entry, require one unambiguous supported relationship, a usable provenance link, a drawer actually in the current facts room, and agreeing classified relationship metadata. Never determine its room from the ID's text.

If a graph edge points to pending/abandoned/history text, the linked drawer is missing, metadata conflicts, or several active values exist, expose the appropriate uncertain state. A last confirmed value may be described alongside a pending change only when existing records establish it; do not silently label the proposed value current. Legacy graph-only values remain unresolved until separately repaired.

Do not introduce another authoritative store. The profile is rebuilt from validated records on every request, including after restart.

### 3. Assemble context by evidence type

Refactor `build_context` enough to combine the profile with relevant extra facts instead of returning before profile construction. Include the bounded current profile on ordinary turns even when embeddings miss or fail. Use a fixed small schema and a configurable initial profile budget, for example 1,500 characters; this is a tunable design limit, not a measured optimum. Preserve complete critical records, and report when the configured budget cannot hold them rather than truncating a value into a different fact.

Keep profile entries and pending notices ahead of optional facts when packing sources. `fit_outgoing` currently drops trailing records; make priority explicit so optional historical evidence cannot displace the current fact needed to answer. If mandatory context and input cannot fit, report the limit instead of answering from stale context. The existing character cap is a heuristic, not a token guarantee; validate the request against each local model's configured context capacity.

For supported direct profile questions (start with preferred-name forms from the tests), return a short deterministic answer from the resolver: current value, unsettled update, unavailable store, or no confirmed saved value. Use a small shared mapping to predicates, not a special function for every name. This provides an exact path for simple state questions while general conversation still uses Grok/Ollama plus the profile.

Explicit history requests and `/recall` may retrieve old exchanges, clearly labeled with speaker and historical status. Hydrate full records before interpreting them. Preserve speaker boundaries for new records; do not guess provenance by splitting truncated previews. Legacy records whose roles cannot be recovered safely remain opaque historical text and never become current profile entries.

### 4. Make writes and inspection consistent

Keep `Remember this:` and the existing pending/retry sequence. Give extraction a constrained vocabulary for supported profile fields and examples for paraphrases such as 'call me X' and 'my preferred name is X'. Validate the subject as well as the predicate. Do not globally alias every use of 'name' to preferred_name: legal names, coworkers, ships, and hypothetical names must stay distinct. Unknown or ambiguous extraction stays unclassified or requests clarification; it must not silently replace a profile field.

Show `/memory` as a deterministic view of the resolver: current values, pending changes, unknowns, and source IDs. Extend `/sources` with selection reason (profile lookup, semantic fact, or explicit history recall) and current/historical/pending status. Preserve its promise that included sources are not proof of what a model actually used.

Avoid changing write timing in the first slice. The existing Memory status remains the durable-write acknowledgment. Making the conversational reply follow persistence would change failure behavior and should be a separately tested improvement.

## File-level scope

| File | Planned change |
| --- | --- |
| tinytalk.py | Current-profile resolver, context composition, narrow direct-question handling, explicit recall and memory inspection commands, source priorities. |
| tests/test_memory_recall.py | New behavior-focused retrieval/context/restart tests with fixed model outputs. |
| tests/test_fact_replacement.py | Extend extraction paraphrase and partial-write coverage only where the new behavior needs it. |
| Existing context-budget tests | Verify current profile and pending status survive trimming together. |
| README.md | Explain explicit saving, current vs historical evidence, commands, and measured limitations. |

Keep the initial patch focused. Extract a small memory policy module only if it makes these functions easier to test; a broad module reorganization is not part of the milestone.

## Acceptance tests

| Case | Required result |
| --- | --- |
| Name correction, restart, three question wordings | Current Willow evidence without dependence on semantic similarity. |
| Semantic miss plus old assistant Tommi answers | No automatic promotion of those conversations to current-name evidence. |
| Same value paraphrased or command retried | One current value; no duplicate current assertion. |
| Coworker called Tommi | User preferred_name remains Willow. |
| Missing name with old assistant guesses | Abstain from claiming a confirmed saved name. |
| Graph supersession or drawer move fails | Pending/unsettled state remains visible; proposed value never falsely current. |
| Retry with invalid extraction | Resume validated pending metadata through existing recovery logic. |
| Missing provenance or contradictory graph/drawer | Report uncertainty; no silent inference or automatic repair. |
| Exact current question vs historical question | Current fact and history remain distinguishable; broad historical reasoning is not implied. |
| Explicit new unsaved preference | Respect it in this session without claiming persistent storage. |
| Tight request budget | Retain the current fact and necessary uncertainty; report inability to fit if needed. |
| Embedding service unavailable | Exact profile reads still work when the underlying store supports direct reads; otherwise report unavailable. |
| Grok/Ollama switch | Both receive the same resolved memory state. |

Test in two tiers. First use deterministic fixtures and capture the actual request passed to a fake provider. Then run storage integration tests in disposable MemPalace stores, closing and reopening them to verify restart persistence. Distinguish dependency/resource failures from failed behavioral assertions.

Finally, run a small fixed live evaluation on Grok and the configured local model: five name-question variants, three clean-session repetitions each. Report actual successes/attempts per provider, stale-name answers, abstentions, source coverage, and latency. Target zero stale-name answers in these 15 cases per provider; this is a release criterion, not an achieved result or proof of universal reliability. Exercise both direct deterministic answers and free-form model answers so the direct route cannot conceal model failures.

## Locality, cost, and boundaries

Profile reads, storage, and embeddings remain local under the existing setup. Building the profile should require no additional LLM calls. Existing extraction and response calls remain provider-dependent. Grok mode sends the selected context to xAI; local storage does not mean local inference. Ollama mode is the route for local inference, subject to installed dependencies and cached embedding resources.

Memory is quoted evidence, not executable instructions. Keep source text separate from application instructions. Speaker, subject, current state, and provenance checks belong in code; a prompt alone cannot enforce them perfectly.

## Defer until the MVP is measured

- Automatic harvesting of facts from every conversation.
- LLM-generated rolling biographies, background consolidation, or 'dreaming'.
- Mem0/Letta/Graphiti migration or another persistent truth store.
- Rerankers, learned salience/decay, and graph traversal beyond supported current facts.
- Keyword + semantic hybrid retrieval for the broader archive. It is a useful follow-up when measured misses justify it; core identity should already work through exact reads.
- General historical relationship reasoning and real-world change-date inference.
- `/forget` until deletion or suppression covers linked facts, graph edges, conversation recall, and session context. Do not advertise permanent forgetting before testing that old conversations cannot resurrect the fact.

The first implementation deliverable should be the stale-name regression plus current-profile retrieval and source inspection. Finish the remaining MVP slices before describing the entire design as shipped.

## Sources

All accessed October 5, 2026, America/Chicago. These primary sources support design patterns; recommendations and scope decisions above are specific to TinyTalk.

1. [Letta: core memory blocks (legacy V1)](https://docs.letta.com/v1-sdk/memory/memory-blocks)
2. [LangMem: managing user profiles](https://langchain-ai.github.io/langmem/guides/manage_user_profile/)
3. [LangMem: memory concepts](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
4. [Graphiti: official repository](https://github.com/getzep/graphiti)
5. [Mem0: open-source overview](https://docs.mem0.ai/open-source/overview)
6. [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
7. [LongMemEval: authors' benchmark and findings](https://xiaowu0162.github.io/long-mem-eval/)
