import os
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = OpenAIClient(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def chat_with_groq(prompt: str) -> str:
    """
    Sends a prompt to Groq's LLaMA3-70B model and returns the response.
    Handles errors gracefully.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You summarize document themes with citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Groq Error] {str(e)}")
        return f"Groq API error: {str(e)}"
