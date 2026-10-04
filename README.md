# TinyTalk

TinyTalk started as me rebuilding one of my old college chatbot projects mostly because I wanted to refresh myself on some of the "under the hood" stuff.

At first it was basically:

> local Llama + Ollama + terminal = chatbot

That was supposed to be the project.

Then I started wondering what "memory" actually means for something like this.

Then identity.

Then old memories versus current facts.

Then structured knowledge.

And, well... here we are.

TinyTalk is now a small local AI experiment built around a pretty simple question:

**What happens if you stop treating an AI conversation like one giant pile of text and start separating identity, memory, history, and knowledge into different systems?**

I'm deliberately trying to keep it small enough that I can still open the code and understand what the thing is actually doing.

No giant agent framework. No 47 layers of abstraction.

Just me poking at the machinery.

---

## What it does right now

TinyTalk can answer with `llama3.2:3b` locally through Ollama, or with Grok through the xAI API. The choice is configuration. Soul, short-term history, and MemPalace stay the same either way.

On top of that, I've slowly added a few different kinds of continuity.

### `SOUL.md`

TinyTalk has a separate identity file called `SOUL.md`.

It describes things like:

- who TinyTalk is
- how it should communicate
- how it should treat uncertainty
- how it should think about memory
- how it should think about its own knowledge

The important part is that the Soul is **not memory**.

It isn't retrieved by similarity and it doesn't disappear when older chat turns fall out of context.

It gets loaded as TinyTalk's system instructions every time it answers.

The rough idea is:

```text
Memory = what TinyTalk has learned

Soul = who TinyTalk is supposed to be
```

---

## Short-term context

TinyTalk keeps the last 10 conversation turns active:

```python
MAX_TURNS = 10
```

Once the conversation gets longer than that, older turns fall out of the live context.

That's intentional.

I wanted short-term conversation and long-term memory to be two different things instead of just feeding the model an endlessly growing transcript.

---

## Persistent conversation memory

Completed conversations are saved through MemPalace in:

```text
tinytalk/conversations
```

TinyTalk can search those old exchanges later when something from a previous conversation looks relevant.

So restarting the program clears the immediate chat window, but it doesn't necessarily mean TinyTalk forgot everything that happened before.

---

## Explicit facts

There's also a stronger kind of memory.

If I say:

```text
Remember this: my test spaceship is named Enterprise
```

TinyTalk stores that separately in:

```text
tinytalk/facts
```

These are things I explicitly told it to remember, so TinyTalk treats them as stronger evidence than something it happens to recover from an old conversation.

For normal questions, it searches those facts semantically.

Right now the minimum similarity score is:

```python
MIN_FACT_SIMILARITY = 0.45
```

If a memory doesn't clear that threshold, TinyTalk doesn't inject it into the answer.

That came from discovering that "technically related" memories are not necessarily **usefully related** memories.

---

## "What do you remember about me?"

This turned into its own problem.

Semantic search works pretty well for:

```text
What is my test spaceship called?
```

because that question is semantically close to:

```text
my test spaceship is named Enterprise
```

But this:

```text
What do you remember about me?
```

isn't particularly similar to any one fact.

So TinyTalk now recognizes a few broad memory questions and directly reads the current fact list instead of trying to similarity-search it.

That list is only the saved facts retrieved for that request. The instruction says so. It also says the list is not a complete inventory of stored conversations, that other memories may exist even when they were not included, and that these facts are saved memories rather than something the model learned in training. The same limits are attached when a similar fact or an old conversation excerpt is included.

Each included record is labelled as a saved fact, a conversation excerpt, an archived fact, or a knowledge-graph record. The original text stays as it was stored. An id or a timestamp is shown only when the record already has one. A filing time or a graph time is not a date the real-world fact changed.

TinyTalk does not print those labels after every answer. `/sources` prints the records included with the last successful answer: the type, any stored id, and a short preview. It shows the sources supplied with that answer. It does not prove which sources the model used. If none were included, it says that, without claiming that no memories exist or that the answer came only from model knowledge. `/sources` does not call the model or save a memory. A failed reply leaves the previous list in place. `/speak` and `/stop` leave it in place too. The list lasts for this terminal session only.

That sounds obvious in hindsight.

It was not obvious before I watched it fail. 😂

---

## Facts can change

This was the next rabbit hole.

Imagine I tell TinyTalk:

```text
my test spaceship is named Picklewagon
```

Then later:

```text
my test spaceship is named Serenity
```

Then:

```text
my test spaceship is named Enterprise
```

Those shouldn't necessarily become three equally valid current facts.

For certain single-value relationships, TinyTalk now understands the idea of:

```text
old value -> historical
new value -> current
```

Current explicit facts live in:

```text
tinytalk/facts
```

Superseded ones can move into:

```text
tinytalk/facts-history
```

The goal isn't to erase the past.

It's to stop confusing **what used to be true** with **what TinyTalk currently believes is true**.

---

## Knowledge graph

Explicit facts also get interpreted into simple structured relationships.

Something like:

```text
my test spaceship is named Enterprise
```

can become:

```text
user -> test_spaceship_name -> Enterprise
```

TinyTalk stores those relationships in MemPalace's knowledge graph.

This created another fun problem: small language models don't always name relationships the same way.

One run might produce:

```text
test_spaceship
```

another:

```text
spaceship_name
```

Those two are the same idea, so TinyTalk normalizes them to:

```text
test_spaceship_name
```

`imaginary_spaceship_name` stays its own relationship. An imaginary spaceship does not replace the test spaceship just because the names look related.

That was one of those moments where the model wasn't exactly *wrong*.

The software just needed to stop assuming the model would describe the same idea with the same words every single time.

The knowledge graph currently helps manage structured facts and supersession.

TinyTalk does **not** use the graph as a general answer source. A few exact questions are the exception. TinyTalk recognizes these wordings, ignoring capitalization, extra spaces, a trailing question mark, and straight or curly apostrophes:

```text
What was my test spaceship called before Enterprise?
What did I call my test spaceship before Enterprise?
What was the previous name of my test spaceship?
What was my test spaceship's previous name?
What was the old name of my test spaceship?
What was my test spaceship named previously?
```

For those, it reads the current and inactive `test_spaceship_name` edges and the `tinytalk/facts` and `tinytalk/facts-history` rooms. Stored predicates pass through the same aliases as new facts. The labels name the current value and, when the records support one, the previous name. An inactive edge that repeats the current name is not a previous name. An archived test-spaceship fact fills in when the graph has no different earlier name. A separate imaginary-spaceship fact is not part of this relationship. Filing times are used only when they put the earlier names in order, and TinyTalk does not turn them into a rename date. Search order is not time order. If the records do not establish one previous name, the label says that. These questions do not replace saved facts. Every other question still uses the normal fact search.

---

## The current mental model

This is roughly how I'm thinking about TinyTalk now:

```text
                    SOUL.md
                       |
                       v
                   TinyTalk
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
   Recent Chat      MemPalace    Knowledge Graph
                       |
               +-------+--------+
               |                |
               v                v
             Facts        Conversations
               |
        +------+------+
        |             |
        v             v
     Current        History
```

Or, in plain English:

```text
Soul
= who TinyTalk is supposed to be

Context
= what we're talking about right now

Facts
= things I explicitly told it

Fact history
= things that used to be true

Conversations
= what was actually said

Knowledge graph
= structured relationships TinyTalk has learned
```

They're related.

They aren't the same thing.

That's kind of the whole experiment.

---

# Running it

You'll need:

- Python 3
- MemPalace 3.10+
- the packages in `requirements.txt`

Ollama mode also needs Ollama and Llama 3.2 3B. Grok mode needs an xAI API key instead. It does not need Ollama for generation.

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the example environment file and edit it locally. `.env` is gitignored. `.env.example` only has placeholders.

```bash
cp .env.example .env
```

TinyTalk reads `.env` on startup and does not override variables you already exported. You can also export the variables yourself and skip the file.

## Ollama (the default)

Pull the model:

```bash
ollama pull llama3.2:3b
```

`.env`:

```text
TINYTALK_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

`OLLAMA_MODEL` defaults to `llama3.2:3b` when it is unset. `OLLAMA_BASE_URL` defaults to `http://localhost:11434/v1`.

`TINYTALK_DEBUG` defaults to off. Set it to `1`, `true`, `yes`, or `on` to print fact-similarity scores and knowledge-graph updates. Save warnings and API errors stay visible either way. The startup line still names the provider in use.

Run it:

```bash
python3 tinytalk.py
```

You'll get:

```text
🤖 Hello! I'm your Llama assistant, running locally. (Type 'quit' to exit)
Type /speak to hear the last answer, /stop to stop playback, or /sources to list memories included with the last answer.
------------------------------
You:
```

Leaving `TINYTALK_PROVIDER` unset does the same thing. Ollama mode does not read `XAI_API_KEY` and does not open a Grok client.

## Grok

Grok mode calls the xAI Responses API (`POST https://api.x.ai/v1/responses`) with the OpenAI Python SDK. That is the API xAI documents as the primary text interface. The verified model ID to put in `XAI_MODEL` is `grok-4.7`. That string is the API model ID from the Grok 4.7 docs. It is not a Grok Build label, and reasoning effort such as "medium" is not a model name. TinyTalk does not send a reasoning-effort override, so the model uses its own default.

Create a key on the [xAI console API keys page](https://console.x.ai/team/default/api-keys). Billing for that key is separate from anything Ollama does on your machine. xAI charges per token. Check the current prices in the [xAI models docs](https://docs.x.ai/developers/models) before you leave it running.

`.env`:

```text
TINYTALK_PROVIDER=grok
XAI_API_KEY=
XAI_MODEL=grok-4.7
```

Put the real key only in `.env` or in your shell. Do not commit it, paste it into the README, or put it in browser code. This project has no browser app.

Or, without a file:

```bash
TINYTALK_PROVIDER=grok XAI_API_KEY="your-key" XAI_MODEL="grok-4.7" python3 tinytalk.py
```

You'll get:

```text
🤖 Hello! I'm TinyTalk, using Grok through the xAI API. (Type 'quit' to exit)
Type /speak to hear the last answer, /stop to stop playback, or /sources to list memories included with the last answer.
------------------------------
You:
```

What gets sent: each reply includes `SOUL.md`, any memory context TinyTalk decided to inject for that turn, and the trimmed recent conversation (`MAX_TURNS = 10`). What stays local: MemPalace drawers, the knowledge graph, and embeddings. Embeddings are MemPalace's local MiniLM or EmbeddingGemma model, not Ollama and not xAI.

TinyTalk keeps the transcript itself. Requests set `store` to false and do not send `previous_response_id`, so a Grok conversation is not continued from xAI's stored response history. Each request is plain text generation. It does not send `tools`, `tool_choice`, or `search_parameters`. Setting `tool_choice` without tools returns HTTP 400, and `search_parameters` is the deprecated live-search field that returns HTTP 410. TinyTalk does not enable web search or X search. A failed or empty model response is printed and is not saved as an assistant turn. TinyTalk will not silently switch back to Ollama if Grok fails.

Each request uses a 10 second connect timeout and a 120 second read timeout. The OpenAI SDK's own retries are turned off (`max_retries=0`). TinyTalk retries a connection failure, timeout, rate limit, or server error itself, for three attempts total, with a short backoff. The same limit applies to Ollama.

## Spoken answers

Spoken output is optional. Typed chat works the same way if VoiceStudio is not running.

TinyTalk does not start VoiceStudio. Start that backend yourself and leave it on `http://127.0.0.1:3900`. The saved voice is the design profile `TinyTalk — Curious Little Computer` (`5e49413d`, engine `omnivoice`, seed 42). `/speak` sends that profile id. It does not send the voice name `default`.

`.env` (both optional; these are the defaults):

```text
VOICESTUDIO_BASE_URL=http://127.0.0.1:3900
VOICESTUDIO_VOICE_ID=5e49413d
```

Then:

```bash
python3 tinytalk.py
```

After a successful assistant reply:

- `/speak` speaks that reply. TinyTalk prints `Speaking the last answer…` and generates the audio in the background, so you can keep typing. When the clip is ready it plays through macOS `afplay`.
- A second `/speak` while a clip is already being made or played does not start another one.
- `/stop` stops playback. If generation is still in flight, TinyTalk discards the audio when it arrives and does not play it. VoiceStudio may still finish that request on the server; stopping playback does not cancel it.
- `quit` also stops playback.

`/speak` and `/stop` do not call Grok or Ollama, do not add a conversation turn, and do not save a memory. They speak only the last assistant message shown in the chat. The separate fact-extraction request is not spoken. They also leave the `/sources` list unchanged.

The speech client is separate from the chat client. It allows 10 seconds to connect and 300 seconds for the whole request, and it does not retry. It does not send `XAI_API_KEY`. If VoiceStudio is down, times out, or returns an error, TinyTalk prints that and leaves text chat usable. The first clip after a cold model load can take several minutes. Later clips are faster while the model stays loaded.

## Troubleshooting

Configuration errors stop startup. They look like `Configuration error: ...`.

- `TINYTALK_PROVIDER must be 'ollama' or 'grok'` — the variable is set to something else.
- `Grok mode needs XAI_API_KEY` or `XAI_MODEL` — Grok was selected and one of those is missing. Ollama is not used as a fallback.
- HTTP 401 — xAI rejected the key. Create or copy a key from the console. TinyTalk never prints the key.
- HTTP 403 — the key or team is not allowed to call that model.
- HTTP 404 — `XAI_MODEL` is not a model ID this team can use. `grok-4.7` is the verified ID.
- HTTP 429 — the team rate limit was hit after the retries. Wait and try again. Limits are per model in the xAI console.
- Could not reach the xAI API — DNS, network, or a timeout. The request was not retried forever.
- Could not reach Ollama — start Ollama and confirm `ollama pull llama3.2:3b` has finished. Generation in Grok mode does not need this. Saving and searching memory does not either, unless MemPalace itself has not finished downloading its local embedding model.
- Empty response — the model returned no assistant text, or a response that was not `completed`. Nothing from that turn is written to history or MemPalace.

And that's TinyTalk.

For now.

---

# Things that are still weird

This is very much an experiment, and there are some rough edges.

A few of the interesting ones:

- Llama 3.2 3B is fast and tiny, but sometimes gets weird with structured output or explanations.
- Predicate normalization only covers relationships I've explicitly accounted for.
- Only some facts are currently treated as single-value facts.
- Conversation retrieval and explicit-fact retrieval still take separate paths.
- TinyTalk occasionally exposes too much of its internal prompt plumbing when talking about its memories.
- The knowledge graph isn't queried during normal conversation. Previous-name questions about the test spaceship are the exception.
- There is no live web research system.
- Memory storage and graph updates aren't transactional.

I'm not really trying to hide those edges.

Finding them is half the reason I'm building this.

---

# Roadmap

This isn't meant to be a rigid product roadmap.

It's more of a **things I want to poke at next** list.

## Done

- [x] Ollama and Grok share one chat loop. Ollama is the default. Grok is optional, and TinyTalk does not silently fall back if Grok fails.
- [x] Memories persist in MemPalace. Recall works, and replacing a single-value fact keeps the old value. That path is verified.
- [x] The old test-spaceship chain is cleaned up. Enterprise is current, Serenity is archived, and Picklewagon stays a separate imaginary-spaceship fact.
- [x] Memory answers say what was retrieved and what was not. `TINYTALK_DEBUG` is optional and off by default.
- [x] `/speak` and `/stop` use the local VoiceStudio profile. Live playback is verified.
- [x] The previous-name questions below return Serenity. Asking what the test spaceship is called now still returns Enterprise.
- [x] `imaginary_spaceship_name` stays its own predicate. It no longer replaces `test_spaceship_name`.
- [x] Retrieved facts, conversations, archived facts, and graph records carry source labels and any metadata already stored. `/sources` lists the records included with the last successful answer. It makes no model call and saves no memory. Record dates are not presented as real-world change dates. The list shows supplied sources; it does not prove which sources the model used. Live Grok checks passed: Enterprise is current, Serenity is the previous test-spaceship name, and Picklewagon is excluded from that historical context.

Only these wordings count as that history. "My spaceship" without "test" does not:

```text
What was my test spaceship called before Enterprise?
What did I call my test spaceship before Enterprise?
What was the previous name of my test spaceship?
What was my test spaceship's previous name?
What was the old name of my test spaceship?
What was my test spaceship named previously?
```

Anything else is an ordinary question. This is not a general historical memory, and it is not a timeline.

## Next

- [ ] Broader history questions, and other facts that change over time.
- [ ] Stronger memory provenance than the `/sources` list, and tighter coordination between explicit facts, older conversations, and the graph.
- [ ] Compare models on the same Soul and memories, then add more providers the same way.
- [ ] Speech speed, streaming, and eventually voice input.
- [ ] Version `SOUL.md`. If TinyTalk ever proposes a change to its Soul, I have to approve it first.

Ollama and Grok already share the loop. TinyTalk still owns `SOUL.md`, the last 10 turns, and MemPalace. The model is only the part that writes the next reply.

I'd like TinyTalk itself to eventually be:

```text
Soul + Memory + Context + Knowledge
```

while the model underneath it can change.

Maybe Llama.

Maybe Qwen.

Maybe Gemini.

Maybe something that doesn't exist yet.

The model shouldn't have to **be** TinyTalk.

It should just be one of the brains TinyTalk can use.

If Llama, Qwen, and Gemini all get:

```text
SOUL.md
+
the same memories
+
the same recent conversation
```

and I ask:

```text
Who are you?
```

how much of "TinyTalk" survives the model swap?

I want to find out.

## Research mode

I've also been sketching out an optional command like:

```text
/research <question>
```

The idea would be to let TinyTalk do current web research with real source citations while keeping its personal memory system local.

Research would get its own storage:

```text
tinytalk/research
```

and would keep things like:

```text
question
timestamp
answer
sources
claim -> source relationships
```

I'm putting this on hold until I find a cloud/search path I actually like that can stay at $0.

I don't want to bolt something onto the project just because I can.

---

# Why I'm building this

I'm not really trying to make another ChatGPT.

There are plenty of those.

I'm more interested in the stuff underneath it.

What actually counts as memory?

What makes an assistant feel continuous from one conversation to the next?

If something it remembers becomes wrong, should it overwrite it or remember that it **used to** be true?

Is identity part of the model?

The prompt?

The memory?

Some combination of all of them?

What happens if you swap the model but keep everything else?

I don't have some giant master plan for TinyTalk.

I'm just following the interesting questions as they show up.

Every time I think:

> okay, that's probably enough

something breaks in a way that makes me want to understand one more layer.

So we'll see where it goes.
