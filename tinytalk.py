# 1. Use the 'openai' library to connect to Ollama
from openai import OpenAI

# 2. Initialize the client to point to your local Ollama server
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # This can be any string, Ollama doesn't require a real key
)

# A welcome message to the user
print("🤖 Hello! I'm your Llama assistant, running locally. (Type 'quit' to exit)")
print("-" * 30)

# The rest of your code stays the same!
while True:
    user_prompt = input("You: ")

    if user_prompt.lower() == 'quit':
        print("Goodbye! 👋")
        break 

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        # 3. Change the model name to the one you have in Ollama
        model="llama3.1",
    )

    response = chat_completion.choices[0].message.content
    print(f"Llama: {response}")
    print("-" * 30)