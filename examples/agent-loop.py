import json
from typing import Any

from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed-for-local-dev",
)


def get_weather(city: str) -> str:
    weather = {
        "Chicago": "58 F and windy",
        "New York": "64 F and cloudy",
        "San Francisco": "61 F and foggy",
    }
    return weather.get(city, "unknown")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]

AVAILABLE_TOOLS = {
    "get_weather": get_weather,
}


def run_agent(prompt: str) -> Any:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    while True:
        response = client.chat.completions.create(
            model="route-auto",
            messages=messages,
            tools=TOOLS,
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            return response

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
            }
        )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            result = AVAILABLE_TOOLS[tool_name](**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result,
                }
            )


if __name__ == "__main__":
    final_response = run_agent("What is the weather in Chicago today?")
    print(final_response.choices[0].message.content)
