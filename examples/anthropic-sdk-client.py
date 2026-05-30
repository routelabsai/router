from anthropic import Anthropic


client = Anthropic(
    base_url="http://127.0.0.1:8000",
    api_key="not-needed-for-local-dev",
)


response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[
        {
            "role": "user",
            "content": "Summarize RouteLabs Router in one sentence.",
        }
    ],
)

for block in response.content:
    if getattr(block, "type", None) == "text":
        print(block.text)
