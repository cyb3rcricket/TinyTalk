from openai import OpenAI
from ollama import chat

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # This can be any string, Ollama doesn't require a real key
)

print("🤖 Hello! I'm your Llama assistant, running locally. (Type 'quit' to exit)")
print("-" * 30)

MAX_TURNS = 10
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
try:
    import os
    import sys

    saved_stdout = sys.stdout
    saved_stdout_fd = os.dup(1)
    from mempalace.mcp_server import tool_add_drawer

    os.dup2(saved_stdout_fd, 1)
    sys.stdout = saved_stdout
    os.close(saved_stdout_fd)
except Exception:
    tool_add_drawer = None

while True:
    user_prompt = input("You: ")

    if user_prompt.lower() == 'quit':
        print("Goodbye! 👋")
        break

    messages.append({"role": "user", "content": user_prompt})

    # Memories are added to this request only. They are not stored in messages.
    request_messages = messages
    if palace is not None:
        facts = []
        try:
            found = search_memories(
                query=user_prompt,
                palace_path=palace_path,
                wing="tinytalk",
                room="facts",
                n_results=3,
            )
            facts = [hit["text"] for hit in found.get("results", []) if hit.get("text")]
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
    messages=request_messages,
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
    print(f"Llama: {response}")
    print("-" * 30)