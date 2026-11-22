import httpx

from routelabs_router.adapters.anthropic import AnthropicChatAdapter
from routelabs_router.adapters.ollama import OllamaChatAdapter
from routelabs_router.adapters.openai_compatible import OpenAICompatibleChatAdapter
from routelabs_router.config import DEFAULT_CONFIG
from routelabs_router.models import ChatCompletionRequest, ChatMessage


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class CapturingHTTPClient:
    def __init__(self, response_payload: dict) -> None:
        self.response_payload = response_payload
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(
        self,
        url: str,
        json: dict | None = None,
        headers: dict | None = None,
    ) -> FakeResponse:
        self.calls.append(("POST", url, json, headers))
        return FakeResponse(self.response_payload)


def test_openai_compatible_adapter_passes_structured_output_and_common_fields(
    monkeypatch,
) -> None:
    fake = CapturingHTTPClient(
        {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-4.1-mini",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    adapter = OpenAICompatibleChatAdapter(
        DEFAULT_CONFIG.providers.cloud.openai_compatible.model_copy(
            update={"api_key": "test-key"}
        )
    )
    adapter.complete(
        ChatCompletionRequest(
            model="route-auto",
            messages=[ChatMessage(role="user", content="Return JSON")],
            response_format={"type": "json_object"},
            temperature=0.2,
            top_p=0.9,
            max_tokens=64,
            stop=["END"],
            seed=7,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        ),
        model="gpt-4.1-mini",
    )

    _, url, payload, headers = fake.calls[0]
    assert url.endswith("/chat/completions")
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 64
    assert payload["stop"] == ["END"]
    assert payload["seed"] == 7
    assert payload["frequency_penalty"] == 0.1
    assert payload["presence_penalty"] == 0.2
    assert headers["Authorization"].startswith("Bearer ")


def test_ollama_adapter_maps_json_schema_and_common_fields(monkeypatch) -> None:
    fake = CapturingHTTPClient(
        {
            "message": {"content": "{\"answer\":\"ok\"}"},
            "model": "qwen3:4b",
            "prompt_eval_count": 1,
            "eval_count": 1,
            "done_reason": "stop",
        }
    )
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    adapter = OllamaChatAdapter(DEFAULT_CONFIG.providers.local.ollama)
    adapter.complete(
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Return structured JSON")],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            },
            temperature=0.2,
            top_p=0.9,
            max_tokens=64,
            stop="END",
            seed=7,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        ),
        model="qwen3:4b",
    )

    _, url, payload, _ = fake.calls[0]
    assert url.endswith("/api/chat")
    assert payload["format"] == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    assert payload["options"]["temperature"] == 0.2
    assert payload["options"]["top_p"] == 0.9
    assert payload["options"]["num_predict"] == 64
    assert payload["options"]["stop"] == "END"
    assert payload["options"]["seed"] == 7
    assert payload["options"]["frequency_penalty"] == 0.1
    assert payload["options"]["presence_penalty"] == 0.2


def test_anthropic_adapter_maps_messages_tools_and_usage(monkeypatch) -> None:
    fake = CapturingHTTPClient(
        {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello"},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "Chicago"},
                },
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }
    )
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    adapter = AnthropicChatAdapter(
        DEFAULT_CONFIG.providers.cloud.anthropic.model_copy(
            update={"api_key": "test-key"}
        )
    )
    result = adapter.complete(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="Be concise"),
                ChatMessage(role="user", content="What's the weather?"),
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="auto",
            max_tokens=128,
        ),
        model="claude-sonnet-4-20250514",
    )

    _, url, payload, headers = fake.calls[0]
    assert url.endswith("/messages")
    assert payload["system"] == "Be concise"
    assert payload["tools"][0]["name"] == "get_weather"
    assert payload["tool_choice"] == {"type": "auto"}
    assert payload["max_tokens"] == 128
    assert headers["x-api-key"] == "test-key"
    assert result.content == "Hello"
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0]["function"]["name"] == "get_weather"
    assert result.usage.total_tokens == 20


def _anthropic_adapter(monkeypatch, response: dict | None = None):
    fake = CapturingHTTPClient(
        response
        or {
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-sonnet-5-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)
    adapter = AnthropicChatAdapter(
        DEFAULT_CONFIG.providers.cloud.anthropic.model_copy(update={"api_key": "k"})
    )
    return adapter, fake


def test_anthropic_adapter_converts_content_parts_and_developer_role(monkeypatch) -> None:
    adapter, fake = _anthropic_adapter(monkeypatch)
    adapter.complete(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="developer", content="Be terse."),
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,QUJD"},
                        },
                    ],
                ),
            ]
        ),
        model="claude-sonnet-5-5",
    )
    payload = fake.calls[0][2]
    assert payload["system"] == "Be terse."
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
                },
            ],
        }
    ]
    assert payload["max_tokens"] == 16000


def test_anthropic_adapter_drops_sampling_and_forced_tools_for_current_models(
    monkeypatch,
) -> None:
    adapter, fake = _anthropic_adapter(monkeypatch)
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.2,
        top_p=0.9,
        tools=[{"type": "function", "function": {"name": "ls", "parameters": {}}}],
        tool_choice="required",
    )
    adapter.complete(request, model="claude-sonnet-5-5")
    current = fake.calls[-1][2]
    assert "temperature" not in current and "top_p" not in current
    assert current["tool_choice"] == {"type": "auto"}

    adapter.complete(request, model="claude-sonnet-4-5-20250929")
    legacy = fake.calls[-1][2]
    assert legacy["temperature"] == 0.2
    assert legacy["tool_choice"] == {"type": "any"}


def test_anthropic_adapter_maps_max_tokens_stop_reason_to_length(monkeypatch) -> None:
    adapter, _ = _anthropic_adapter(
        monkeypatch,
        {
            "content": [{"type": "text", "text": "partial"}],
            "model": "claude-sonnet-5-5",
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    result = adapter.complete(ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")]))
    assert result.finish_reason == "length"


def test_openai_adapter_uses_max_completion_tokens_for_reasoning_models(monkeypatch) -> None:
    fake = CapturingHTTPClient(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "model": "m"}
    )
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)
    adapter = OpenAICompatibleChatAdapter(
        DEFAULT_CONFIG.providers.cloud.openai_compatible.model_copy(update={"api_key": "k"})
    )
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="developer", content="Be terse."),
            ChatMessage(role="user", content="hi"),
        ],
        max_completion_tokens=50,
        temperature=0.3,
    )

    adapter.complete(request, model="gpt-5-mini")
    reasoning = fake.calls[-1][2]
    assert reasoning["max_completion_tokens"] == 50
    assert "max_tokens" not in reasoning and "temperature" not in reasoning
    assert reasoning["messages"][0]["role"] == "developer"

    adapter.complete(request, model="qwen3-4b-instruct")
    local = fake.calls[-1][2]
    assert local["max_tokens"] == 50 and local["temperature"] == 0.3
    assert local["messages"][0]["role"] == "system"
    assert "tool_name" not in local["messages"][1]


def test_ollama_adapter_forwards_data_url_images(monkeypatch) -> None:
    fake = CapturingHTTPClient({"message": {"content": "ok"}, "model": "qwen3:4b"})
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)
    adapter = OllamaChatAdapter(DEFAULT_CONFIG.providers.local.ollama)
    adapter.complete(
        ChatCompletionRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                    ],
                )
            ]
        )
    )
    message = fake.calls[0][2]["messages"][0]
    assert message == {"role": "user", "content": "describe", "images": ["QUJD"]}
