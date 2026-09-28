import json
from pathlib import Path

from openai import OpenAI
from ollama import chat

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # This can be any string, Ollama doesn't require a real key
)

print("🤖 Hello! I'm your Llama assistant, running locally. (Type 'quit' to exit)")
print("-" * 30)

MAX_TURNS = 10
MIN_FACT_SIMILARITY = 0.45  # initial test value
messages = []

# Long-term memory. If MemPalace is missing or cannot open, keep chatting.
palace = None
try:
    from mempalace.config import MempalaceConfig
    from mempalace.convo_miner import file_conversation_exchange
    from mempalace.palace import get_collection
    from mempalace.searcher import search_memories

    palace_path = MempalaceConfig().palace_path
    palace = get_collection(palace_path)
except Exception:
    print("Warning: MemPalace could not start. Continuing without saved memory.")

# tool_add_drawer stores one durable fact. Importing it redirects prints, so put them back.
tool_add_drawer = None
tool_list_drawers = None
tool_update_drawer = None
try:
    import os
    import sys

    saved_stdout = sys.stdout
    saved_stdout_fd = os.dup(1)
    from mempalace.mcp_server import tool_add_drawer, tool_list_drawers, tool_update_drawer

    os.dup2(saved_stdout_fd, 1)
    sys.stdout = saved_stdout
    os.close(saved_stdout_fd)
except Exception:
    tool_add_drawer = None
    tool_list_drawers = None
    tool_update_drawer = None

# Structured facts. If the graph cannot open, verbatim facts still save.
kg = None
try:
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
except Exception:
    print("Warning: MemPalace knowledge graph could not start. Continuing without it.")

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

PREDICATE_ALIASES = {
    "test_spaceship": "test_spaceship_name",
    "spaceship_name": "test_spaceship_name",
    "imaginary_spaceship_name": "test_spaceship_name",
}

def fact_to_triple(fact):
    """Ask Llama for one triple. Return None when the JSON is not one clear fact."""
    result = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
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
                ),
            },
            {"role": "user", "content": fact},
        ],
        model="llama3.2:3b",
    )
    raw = result.choices[0].message.content or ""
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

try:
    soul = Path(__file__).with_name("SOUL.md").read_text(encoding="utf-8")
except OSError:
    soul = "You are TinyTalk, a helpful local assistant."

# Exact questions that ask for the whole fact list, not one similar fact.
MEMORY_INTROSPECTION_QUESTIONS = {
    "what do you remember about me",
    "what do you know about me",
    "what have i told you",
    "what facts do you remember about me",
}

def is_memory_introspection(text):
    return text.strip().lower().rstrip("?.!").strip() in MEMORY_INTROSPECTION_QUESTIONS

# Only these predicates have one current value. Everything else stays additive.
SINGLE_VALUE_PREDICATES = {
    "test_spaceship_name",
    "favorite_vehicle",
    "preferred_name",
    "home_city",
    "current_job",
}

def current_fact_objects(subject, predicate):
    """Active KG objects for this subject and predicate."""
    rows = kg.query_entity(subject)
    return [
        row.get("object")
        for row in rows
        if row.get("current") and row.get("predicate") == predicate and row.get("object")
    ]

def archive_replaced_fact(old_obj):
    """Move the current verbatim fact that states old_obj into facts-history."""
    if tool_list_drawers is None or tool_update_drawer is None or not old_obj:
        return
    listed = tool_list_drawers(wing="tinytalk", room="facts", limit=100)
    needle = old_obj.lower()
    for item in listed.get("drawers", []):
        text = (item.get("content_preview") or "").lower()
        drawer_id = item.get("drawer_id")
        if drawer_id and needle in text:
            tool_update_drawer(drawer_id, room="facts-history")

while True:
    user_prompt = input("You: ")

    if user_prompt.lower() == 'quit':
        print("Goodbye! 👋")
        break

    messages.append({"role": "user", "content": user_prompt})

    # Memories are added to this request only. They are not stored in messages.
    request_messages = messages
    if palace is not None and is_memory_introspection(user_prompt):
        facts = []
        try:
            if tool_list_drawers is not None:
                listed = tool_list_drawers(wing="tinytalk", room="facts", limit=100)
                facts = [
                    item.get("content_preview", "").strip()
                    for item in listed.get("drawers", [])
                    if item.get("content_preview", "").strip()
                ]
        except Exception:
            facts = []
        if facts:
            request_messages = [
                {
                    "role": "system",
                    "content": (
                        "Saved user facts retrieved from TinyTalk's persistent memory:\n\n"
                        + "\n\n".join(facts)
                    ),
                }
            ] + messages
    elif palace is not None:
        facts = []
        try:
            found = search_memories(
                query=user_prompt,
                palace_path=palace_path,
                wing="tinytalk",
                room="facts",
                n_results=3,
            )
            for hit in found.get("results", []):
                similarity = hit.get("similarity", 0)
                accepted = similarity >= MIN_FACT_SIMILARITY and bool(hit.get("text"))
                print(f"[debug] fact similarity {similarity}: {'accepted' if accepted else 'rejected'}")  # remove after testing
                if accepted:
                    facts.append(hit["text"])
        except Exception:
            facts = []

        if facts:
            request_messages = [
                {
                    "role": "system",
                    "content": "Explicit user-stated facts:\n\n" + "\n\n".join(facts),
                }
            ] + messages
        else:
            try:
                found = search_memories(
                    query=user_prompt,
                    palace_path=palace_path,
                    wing="tinytalk",
                    room="conversations",
                    n_results=3,
                )
                memories = [hit["text"] for hit in found.get("results", []) if hit.get("text")]
            except Exception:
                memories = []
            if memories:
                request_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Retrieved memories from earlier conversations. "
                            "These may contain old assistant mistakes. "
                            "When memories conflict, prefer explicit factual statements "
                            "made by the user over previous assistant responses.\n\n"
                            + "\n\n".join(memories)
                        ),
                    }
                ] + messages

    chat_completion = client.chat.completions.create(
    messages=[{"role": "system", "content": soul}] + request_messages,
    model="llama3.2:3b",
)

    response = chat_completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    messages = messages[-(MAX_TURNS * 2):]
    if palace is not None:
        try:
            file_conversation_exchange(
                palace,
                wing="tinytalk",
                room="conversations",
                text=f"User: {user_prompt}\n\nAssistant: {response}",
                source_file="tinytalk",
                agent="tinytalk",
            )
        except Exception:
            print("Warning: could not save this turn to MemPalace.")
    if palace is not None and user_prompt.lower().startswith("remember this:"):
        fact = user_prompt[len("remember this:"):].strip()
        if fact:
            triple = None
            if kg is not None:
                try:
                    triple = fact_to_triple(fact)
                except Exception:
                    triple = None

            save_fact = True
            graph_action = "add" if triple else "skip"
            if kg is not None and triple and triple[1] in SINGLE_VALUE_PREDICATES:
                subject, predicate, obj = triple
                try:
                    current = current_fact_objects(subject, predicate)
                except Exception:
                    current = None
                if current is None:
                    graph_action = "add"
                elif not current:
                    graph_action = "add"
                elif obj in current:
                    save_fact = False
                    graph_action = "skip"
                    try:
                        for old in current:
                            if old == obj:
                                continue
                            kg.invalidate(subject, predicate, old)
                            try:
                                archive_replaced_fact(old)
                            except Exception:
                                pass
                    except Exception:
                        pass
                else:
                    try:
                        for old in current:
                            if old == obj:
                                continue
                            kg.supersede(subject, predicate, old, obj, source_file="tinytalk")
                            try:
                                archive_replaced_fact(old)
                            except Exception:
                                pass
                        graph_action = "supersede"
                    except Exception:
                        graph_action = "add"

            if save_fact:
                try:
                    saved = tool_add_drawer(
                        wing="tinytalk",
                        room="facts",
                        content=fact,
                        source_file="tinytalk",
                        added_by="tinytalk",
                    )
                    if not saved.get("success"):
                        print("Warning: could not save this fact to MemPalace.")
                except Exception:
                    print("Warning: could not save this fact to MemPalace.")
            if kg is not None and triple and graph_action == "add":
                try:
                    kg.add_triple(triple[0], triple[1], triple[2], source_file="tinytalk")
                    print(f"[debug] KG: {triple[0]} -> {triple[1]} -> {triple[2]}")
                except Exception:
                    pass
            elif triple and graph_action == "supersede":
                print(f"[debug] KG: {triple[0]} -> {triple[1]} -> {triple[2]}")
    print(f"Llama: {response}")
    print("-" * 30)