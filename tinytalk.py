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

    palace = get_collection(MempalaceConfig().palace_path)
except Exception:
    print("Warning: MemPalace could not start. Continuing without saved memory.")

while True:
    user_prompt = input("You: ")

    if user_prompt.lower() == 'quit':
        print("Goodbye! 👋")
        break

    messages.append({"role": "user", "content": user_prompt})

    chat_completion = client.chat.completions.create(
    messages=messages,
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
    print(f"Llama: {response}")
    print("-" * 30)