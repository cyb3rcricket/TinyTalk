# TinyTalk

TinyTalk is a small local conversational AI project for exploring how
identity, memory, conversation, and structured knowledge can work together.

It started as a simple Llama chatbot running locally through Ollama.

It has gradually become an experiment in what happens when a small language
model is given several distinct layers of continuity instead of treating
everything as one giant chat history.

TinyTalk is intentionally small and inspectable. The goal is not to turn it
into a giant agent framework. The goal is to understand what is happening
under the hood.

---

## What TinyTalk Has Now

TinyTalk currently combines several different kinds of context and memory.

### Soul

`SOUL.md` contains TinyTalk's persistent identity and behavioral instructions.

It answers questions like:

- Who is TinyTalk?
- How should it communicate?
- How should it treat uncertainty?
- How should it think about memory and knowledge?

The Soul is loaded once when TinyTalk starts and is sent as the first system
message on every normal chat request.

It is deliberately separate from conversation history and memory.

### Short-Term Conversation Context

TinyTalk keeps the most recent conversation turns in memory while it is
running.

```text
MAX_TURNS = 10

Older turns are trimmed so the active conversation does not grow forever.
Restarting TinyTalk clears this short-term context.
Persistent Conversations
Every completed exchange can be stored in MemPalace under:
tinytalk/conversations

These are verbatim conversation records that can be searched later when
relevant.
This allows TinyTalk to recover useful context from previous runs.
Explicit Facts
The command:
Remember this: ...

stores an explicit fact separately from ordinary conversation history.
Example:
Remember this: my test spaceship is named Enterprise

Explicit facts are stored under:
tinytalk/facts

Normal questions retrieve these facts using semantic similarity.
TinyTalk currently requires a fact similarity score of at least:
0.45

before the fact is injected into the model's context.
This prevents unrelated memories from appearing in arbitrary conversations.
Memory Introspection
Semantic search works well for specific questions such as:
What is my test spaceship called?

but poorly for broad questions such as:
What do you remember about me?

Those questions are not necessarily similar to any one saved fact.
TinyTalk therefore has a separate memory-introspection path for a small set
of explicit questions, including:
What do you remember about me?
What do you know about me?
What have I told you?
What facts do you remember about me?

Instead of similarity-searching one fact, TinyTalk lists the current facts
room directly.
Fact History and Supersession
Some facts represent values that can change over time.
For example:
my test spaceship is named Picklewagon

might later become:
my test spaceship is named Serenity

and later:
my test spaceship is named Enterprise

For selected single-valued predicates, TinyTalk can treat the newer value as
a replacement rather than allowing all values to remain current.
Superseded verbatim facts can be moved to:
tinytalk/facts-history

while the newest value remains in:
tinytalk/facts

This preserves history without treating old information as current truth.
Knowledge Graph
When an explicit fact is saved, TinyTalk also attempts to convert it into a
structured triple:
subject -> predicate -> object

For example:
user -> test_spaceship_name -> Enterprise

The graph is powered by MemPalace's KnowledgeGraph.
TinyTalk also normalizes some model-generated predicate names into canonical
forms.
For example:
test_spaceship
spaceship_name
imaginary_spaceship_name

all normalize to:
test_spaceship_name

This keeps application logic from depending on the language model choosing
the exact same label every time.
The graph currently helps TinyTalk manage structured fact state and
supersession. It is not yet used as a general question-answering source.
Architecture
Conceptually, TinyTalk currently looks like this:
                       SOUL.md
                          |
                          v
                      TinyTalk
                          |
             +------------+------------+
             |            |            |
             v            v            v
       Recent Context   MemPalace   Knowledge Graph
                          |
                  +-------+--------+
                  |                |
                  v                v
                Facts        Conversations
                  |
          +-------+--------+
          |                |
          v                v
       Current        Fact History
          |
          +----------------------+
                                 |
                                 v
                           Local LLM
                         via Ollama

A useful mental model is:
Soul
= who TinyTalk is

Context
= what is happening right now

Facts
= what the user explicitly told TinyTalk

Fact History
= what used to be true

Conversations
= what was actually said

Knowledge Graph
= structured relationships and temporal state

Running TinyTalk
TinyTalk currently uses Ollama and llama3.2:3b.
Requirements
- Python 3
- Ollama
- Llama 3.2 3B
- MemPalace 3.10+
- Python packages from requirements.txt
Install the model:
ollama pull llama3.2:3b

Install Python dependencies:
pip install -r requirements.txt

Run TinyTalk:
python3 tinytalk.py

You should see:
🤖 Hello! I'm your Llama assistant, running locally. (Type 'quit' to exit)
------------------------------
You:

Type:
quit

to exit.
Example Memory Flow
Save a fact:
You: Remember this: my test spaceship is named Enterprise

Later, ask:
You: What is my test spaceship called?

TinyTalk searches the explicit facts room and only uses facts that clear the
similarity threshold.
Ask instead:
You: What do you remember about me?

TinyTalk recognizes this as a memory-introspection question and lists the
current saved facts instead of relying on semantic similarity.
Why Separate These Layers?
A language model does not automatically have one unified concept of
"memory."
Different problems require different mechanisms.
A recent conversation turn is not the same thing as a durable fact.
A durable fact is not the same thing as the original sentence that created it.
A structured graph relationship is not the same thing as either of those.
And none of them are the same thing as the instructions describing who
TinyTalk should be.
Keeping those layers separate makes TinyTalk easier to understand, inspect,
debug, and experiment with.
Current Limitations
TinyTalk is still deliberately experimental.
Some current limitations include:
- The local model is only Llama 3.2 3B and sometimes produces inconsistent
  structured labels or awkward explanations.
- Predicate normalization currently covers only a small explicit alias set.
- Only a small explicit set of predicates are treated as single-valued.
- Fact-history migration for older test data is still manual.
- Conversation-memory fallback and explicit fact retrieval are still separate
  retrieval paths.
- TinyTalk sometimes describes retrieved memory too literally, such as
  referring to internal prompt sections.
- The knowledge graph is not yet used to answer ordinary questions.
- There is currently no web research capability.
- content_preview returned by some MemPalace listing operations is limited
  to 200 characters.
- Memory writes and graph updates are not transactional.
These are useful constraints for the project because they expose where memory
systems become more complicated than simply storing text.
Roadmap
The roadmap is intentionally conservative. TinyTalk should stay small enough
that its behavior can still be understood by reading the code.
Near Term
Clean Up Existing Test Memory
Perform a one-time migration of the spaceship test facts:
Picklewagon -> historical
Serenity -> historical
Enterprise -> current

The goal is to verify that:
What is my test spaceship called?

returns Enterprise, while older names remain available as history.
Improve Memory Provenance
Make injected memory clearer to the model.
TinyTalk should understand the distinction between:
saved user fact
retrieved conversation
recent conversation
model training knowledge

This should reduce responses such as:
"I was trained on this fact"

when the information was actually retrieved from MemPalace.
Verify Fact Supersession End-to-End
Test:
Remember this: ...

across:
- first value
- duplicate value
- changed value
- historical value
and verify both the verbatim fact store and knowledge graph remain coherent.
Reduce Development Debug Noise
Once memory behavior is stable, remove or gate temporary lines such as:
[debug] fact similarity ...
[debug] KG ...

while keeping an optional way to inspect memory behavior during development.
Medium Term
Model / Provider Switching
Decouple TinyTalk from one hard-coded model.
Possible brains may include:
Ollama / local models
Gemini
other free or low-cost APIs

TinyTalk's identity should remain:
Soul + Memory + Context + Knowledge

while the language model becomes an interchangeable inference engine.
Historical Memory Questions
Use the temporal knowledge graph to answer questions such as:
What was my test spaceship called before Enterprise?

without treating superseded information as current.
Better Retrieval Coordination
Explore allowing relevant information from multiple sources to contribute to
one answer:
explicit facts
+
conversation memory
+
structured graph state

while preserving clear provenance.
Optional Research Mode
Potential command:
/research <question>

The goal would be current, grounded web research with preserved source
citations.
Research should remain separate from user facts:
tinytalk/research

A research record should eventually preserve:
query
timestamp
model
answer
sources
claim-to-source mappings

Research mode will remain on hold until a genuinely useful $0 cloud/search
path is confirmed.
Longer-Term Experiments
Soul Evolution
Allow TinyTalk to propose changes to SOUL.md without modifying it silently.
Conceptually:
TinyTalk notices recurring mismatch
        |
        v
proposed Soul change
        |
        v
user approval
        |
        v
versioned SOUL.md update

The user should always remain in control of identity changes.
Soul History
Potentially preserve versions such as:
soul/
├── SOUL.md
└── history/
    ├── 0001-original.md
    ├── 0002-more-concise.md
    └── 0003-more-curious.md

This would make personality evolution inspectable rather than invisible.
Stronger Memory Provenance
Eventually every retrieved piece of context could carry information such as:
source
timestamp
memory type
current/historical status
confidence

TinyTalk could then reason not only about a fact, but about where that fact
came from.
Research Recall and Freshness
Stored research could eventually support:
What did we find out about X?

while detecting when externally sourced information may be stale enough to
research again.
Project Philosophy
TinyTalk is not trying to become the biggest assistant.
It is trying to make the machinery understandable.
Every new feature should ideally answer a question about how AI systems work:
How does identity persist?

How does memory differ from conversation history?

How should old facts be replaced without erasing history?

How does an AI know what it remembers?

How should structured knowledge and verbatim memory interact?

How can different models share the same identity and memory?

If a feature makes those questions harder to understand without adding
something meaningful, it probably does not belong in TinyTalk.
Status
TinyTalk is an active experimental project.
Current focus:
Soul
+
local conversation
+
persistent memory
+
fact history
+
temporal structured knowledge

Cloud models, research, and more advanced agent behavior are intentionally
secondary to keeping the core system small, understandable, and useful.