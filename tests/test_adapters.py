import httpx

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
