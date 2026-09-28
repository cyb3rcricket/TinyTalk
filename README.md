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

TinyTalk currently runs `llama3.2:3b` locally through Ollama.

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

and another:

```text
imaginary_spaceship_name
```

So TinyTalk now normalizes those into a canonical relationship:

```text
test_spaceship_name
```

That was one of those moments where the model wasn't exactly *wrong*.

The software just needed to stop assuming the model would describe the same idea with the same words every single time.

The knowledge graph currently helps manage structured facts and supersession.

TinyTalk does **not** yet use the graph as a general answer source.

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
- Ollama
- Llama 3.2 3B
- MemPalace 3.10+
- the packages in `requirements.txt`

Pull the model:

```bash
ollama pull llama3.2:3b
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run it:

```bash
python3 tinytalk.py
```

You'll get:

```text
🤖 Hello! I'm your Llama assistant, running locally. (Type 'quit' to exit)
------------------------------
You:
```

And that's TinyTalk.

For now.

---

# Things that are still weird

This is very much an experiment, and there are some rough edges.

A few of the interesting ones:

- Llama 3.2 3B is fast and tiny, but sometimes gets weird with structured output or explanations.
- Predicate normalization only covers relationships I've explicitly accounted for.
- Only some facts are currently treated as single-value facts.
- Some of my old test memories still need a one-time cleanup.
- Conversation retrieval and explicit-fact retrieval still take separate paths.
- TinyTalk occasionally exposes too much of its internal prompt plumbing when talking about its memories.
- The knowledge graph isn't queried during normal conversation yet.
- There is no live web research system.
- Memory storage and graph updates aren't transactional.

I'm not really trying to hide those edges.

Finding them is half the reason I'm building this.

---

# Roadmap

This isn't meant to be a rigid product roadmap.

It's more of a **things I want to poke at next** list.

## Next

- [ ] Clean up the old Picklewagon -> Serenity -> Enterprise test-memory chain
- [ ] Verify current vs. historical facts end-to-end
- [ ] Improve memory provenance so TinyTalk knows whether something came from:
  - an explicit fact
  - an old conversation
  - recent context
  - its base model knowledge
- [ ] Stop TinyTalk from saying things like "I was trained on this" when it actually retrieved something from memory
- [ ] Gate or remove the temporary debug output once the memory system settles down

## After that

- [ ] Let the knowledge graph answer historical questions

Example:

```text
What was my spaceship called before Enterprise?
```

- [ ] Improve coordination between:
  - explicit facts
  - conversation memory
  - graph knowledge

- [ ] Make the model/provider swappable

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

## Further out

A few things I'm curious about:

- [ ] letting TinyTalk propose changes to its own Soul
- [ ] requiring human approval before any Soul change
- [ ] versioning `SOUL.md`
- [ ] stronger source/provenance tracking for memories
- [ ] research recall and freshness
- [ ] better temporal knowledge
- [ ] seeing how different models behave when given the exact same Soul and memory

That last one is especially interesting to me.

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
