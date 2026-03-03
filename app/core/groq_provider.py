from app.core.llm_provider import LLMProvider
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class GroqProvider(LLMProvider):

     
    def __init__(self):
        super().__init__()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(self, context: str, question: str) -> str:

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ]
        )
            
        content = response.choices[0].message.content

        return content if content else ''
  
