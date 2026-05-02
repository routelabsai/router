from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed-for-local-dev",
)


response = client.chat.completions.create(
    model="route-auto",
    messages=[
        {
            "role": "user",
            "content": (
                "Summarize this in one sentence: RouteLabs Router chooses between "
                "local and cloud models based on privacy, cost, latency, and task complexity."
            ),
        }
    ],
)

print(response)
