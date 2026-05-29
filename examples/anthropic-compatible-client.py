import httpx


payload = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 256,
    "messages": [
        {
            "role": "user",
            "content": "Summarize RouteLabs Router in one sentence.",
        }
    ],
}

response = httpx.post("http://127.0.0.1:8000/v1/messages", json=payload, timeout=30.0)
response.raise_for_status()

for block in response.json()["content"]:
    if block["type"] == "text":
        print(block["text"])
