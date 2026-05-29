from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed-for-local-dev",
)


response = client.responses.create(
    model="route-auto",
    input=(
        "Summarize this in one sentence: RouteLabs Router chooses between "
        "local and cloud models based on privacy, cost, latency, and task complexity."
    ),
)

print(response.output_text)
