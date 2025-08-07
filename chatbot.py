import os
from groq import Groq
import tiktoken

class Chatbot:
    def __init__(self, model="llama3-8b-8192", temperature=0.7, max_tokens=100, token_budget=1000, system_prompt="You are a fed up and sassy assistant who hates answering questions. Reply within 50-80 words."):
        self.client = Groq()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_budget = token_budget
        self.messages = [{"role": "system", "content": system_prompt}]
        self.encoding = self._get_encoding()

    def _get_encoding(self):
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            print(f"Warning: No tokenizer found for model '{self.model}'. Falling back to 'cl100k_base'.")
            return tiktoken.get_encoding("cl100k_base")
        
    def _count_tokens(self, text):
        return len(self.encoding.encode(text))

    def _total_tokens_used(self):
        try:
            return sum(self._count_tokens(msg["content"]) for msg in self.messages)
        except Exception as e:
            print(f"[token count error]: {e}")
            return 0

    def _enforce_token_budget(self):
        try:
            while self._total_tokens_used() > self.token_budget:
                if len(self.messages) <= 2: # Keep system prompt at index 0 in messages
                    break
                self.messages.pop(1) # Remove oldest user prompt
        except Exception as e:
            print(f"[token budget error]: {e}")

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})

        self._enforce_token_budget()
        return reply

api_key = os.getenv("chatbot_api")
if not api_key:
    raise ValueError("No API key found. Set your API key in your PC's environment variables.")

os.environ["Groq_API_key"] = api_key

bot = Chatbot()

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break
    response = bot.chat(user_input)
    print("Assistant:", response)
    print("Current tokens used:", bot._total_tokens_used())