from dotenv import load_dotenv
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

response = llm.invoke("Hello")

print(response)