import os
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()
key = os.environ.get("GEMINI_API_KEY")
print("key loaded:", bool(key), "| starts with AIza:", key.startswith("AIza") if key else None)
print("repr (check for hidden whitespace):", repr(key))

client = genai.Client(api_key=key, vertexai=False)
resp = client.models.generate_content(model="gemini-flash-latest", contents="Say hello in one sentence.")
print(resp.text)