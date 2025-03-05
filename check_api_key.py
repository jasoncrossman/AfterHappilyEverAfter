import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    models = openai.models.list()
    print("✅ API key is working perfectly!")
except Exception as e:
    print(f"❌ Something went wrong: {e}")
