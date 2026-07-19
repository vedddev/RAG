from openai import OpenAI
import os
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key=os.getenv("NVIDIA_API_KEY")
)

stream = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")