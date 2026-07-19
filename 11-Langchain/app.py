import requests
import json
import gradio as gr

url = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json"
}

history = []

def generate_response(prompt):
    history.append(f"User: {prompt}")

    final_prompt = "\n".join(history)

    data = {
        "model": "Veronica",
        "prompt": final_prompt,
        "stream": False
    }

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(data)
    )

    if response.status_code == 200:
        data = response.json()
        actual_response = data["response"]

        history.append(f"Assistant: {actual_response}")

        return actual_response

    return f"Error: {response.text}"


interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter your prompt..."
    ),
    outputs="text",
    title="Veronica AI"
)

interface.launch()
