from fastapi.testclient import TestClient

from routelabs_router.adapters.base import ProviderExecutionError
from routelabs_router.config import DEFAULT_CONFIG
from routelabs_router.models import (
    ChatCompletionRequest,
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsUsage,
    ProviderEmbeddingResult,
    ProviderResult,
)
from routelabs_router.router import RouterEngine
from routelabs_router.server.app import create_app
from routelabs_router.service import ChatService


class FakeProvider:
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        return ProviderResult(
            content=f"echo: {request.messages[-1].content}",
            model=model or "fake-local-model",
        )

    def embed(
        self, request: EmbeddingsRequest, model: str | None = None
    ) -> ProviderEmbeddingResult:
        inputs = request.input if isinstance(request.input, list) else [request.input]
        return ProviderEmbeddingResult(
            data=[
                EmbeddingObject(index=index, embedding=[0.1 + index, 0.2 + index])
                for index, _ in enumerate(inputs)
            ],
            model=model or "fake-local-embedding-model",
            usage=EmbeddingsUsage(prompt_tokens=len(inputs), total_tokens=len(inputs)),
        )


class FakeCloudProvider:
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        return ProviderResult(
            content="cloud: architecture answer",
            model=model or "fake-cloud-model",
        )

    def embed(
        self, request: EmbeddingsRequest, model: str | None = None
    ) -> ProviderEmbeddingResult:
        inputs = request.input if isinstance(request.input, list) else [request.input]
        return ProviderEmbeddingResult(
            data=[
                EmbeddingObject(index=index, embedding=[1.1 + index, 1.2 + index])
                for index, _ in enumerate(inputs)
            ],
            model=model or "fake-cloud-embedding-model",
            usage=EmbeddingsUsage(prompt_tokens=len(inputs), total_tokens=len(inputs)),
        )


class FakeAnthropicCloudProvider:
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        return ProviderResult(
            content="anthropic: architecture answer",
            model=model or "claude-sonnet-4-20250514",
        )


class ValidJSONCloudProvider(FakeCloudProvider):
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        return ProviderResult(
            content='{"answer":"ok"}',
            model=model or "fake-cloud-model",
        )


class WeakLocalProvider:
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        return ProviderResult(
            content="I don't see any document or enough information to answer confidently.",
            model=model or "weak-local-model",
        )


class ToolCallingProvider:
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        return ProviderResult(
            content="",
            model=model or "tool-local-model",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Chicago\"}",
                    },
                }
            ],
        )


class FailingLocalProvider:
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        raise ProviderExecutionError("ollama", "connection refused")

    def embed(
        self, request: EmbeddingsRequest, model: str | None = None
    ) -> ProviderEmbeddingResult:
        raise ProviderExecutionError("ollama", "connection refused")


class InvalidJSONLocalProvider:
    def __init__(self, content: str) -> None:
        self.content = content

    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        return ProviderResult(
            content=self.content,
            model=model or "invalid-local-model",
        )


class MixedLocalProvider:
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        prompt = request.messages[-1].content.lower()
        if "design architecture" in prompt or "multi-step" in prompt:
            return ProviderResult(
                content="I don't see enough information to answer confidently.",
                model=model or "mixed-local-model",
            )
        return ProviderResult(
            content=(
                "RouteLabs Router is a local-first runtime that chooses between "
                "local and cloud models using policy, verification, and privacy rules."
            ),
            model=model or "mixed-local-model",
        )


def test_route_endpoint_returns_provider_metadata() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/route",
        json={"task": "summarize this document", "private": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["target"] == "local"
    assert data["provider"] == "ollama"
    assert "provider_available" in data
    assert "provider_status" in data


def test_chat_completions_uses_local_provider() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "summarize this document"}],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "echo: summarize this document"
    assert data["route"]["provider"] == "ollama"
    assert data["trace"]["escalated"] is False


def test_responses_endpoint_uses_local_provider() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "input": "summarize this document",
            "model": "route-auto",
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["output_text"] == "echo: summarize this document"
    assert data["output"][0]["type"] == "message"
    assert data["route"]["provider"] == "ollama"
    assert data["trace"]["escalated"] is False


def test_anthropic_messages_endpoint_uses_local_provider() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "summarize this document"}],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["content"][0]["text"] == "echo: summarize this document"
    assert data["role"] == "assistant"
    assert data["route"]["provider"] == "ollama"


def test_anthropic_messages_endpoint_returns_tool_use_blocks() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": ToolCallingProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 128,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "input_schema": {"type": "object"},
                }
            ],
            "messages": [{"role": "user", "content": "What's the weather in Chicago?"}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert any(block["type"] == "tool_use" for block in data["content"])
    assert data["stop_reason"] == "tool_use"


def test_responses_endpoint_supports_message_items_and_instructions() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "instructions": "Answer in one sentence.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "summarize this document"}],
                }
            ],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["output_text"] == "echo: summarize this document"


def test_responses_endpoint_returns_function_calls() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": ToolCallingProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "input": "What is the weather in Chicago?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert any(item["type"] == "function_call" for item in data["output"])


def test_chat_completions_exposes_agent_tool_trace_for_mcp_tools() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": ToolCallingProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Edit the repo config file"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__filesystem__write_file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "mcp__filesystem__write_file"},
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    agent_tools = data["trace"]["agent_tools"]
    assert agent_tools["detected"] is True
    assert agent_tools["mcp_like"] is True
    assert agent_tools["approval_required"] is True
    assert agent_tools["risk_level"] == "high"
    assert data["route"]["agent_tools"] == agent_tools


def test_route_endpoint_accepts_declared_agent_tools() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/route",
        json={
            "task": "Search tickets before answering",
            "tool_names": ["mcp__zendesk__search_tickets"],
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_tools"]["detected"] is True
    assert data["agent_tools"]["mcp_like"] is True
    assert data["provider"] == "ollama"


def test_responses_endpoint_accepts_top_level_input_text_items() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "input_text",
                    "text": "summarize this document",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["output_text"] == "echo: summarize this document"


def test_responses_endpoint_rejects_empty_input_lists() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post("/v1/responses", json={"input": []})

    assert response.status_code == 422
    assert "responses input must include" in response.json()["detail"]


def test_responses_stream_text_chunks() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "route-auto",
            "stream": True,
            "input": "summarize this document",
            "private": False,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "\"type\": \"response.created\"" in body
    assert "\"type\": \"response.output_text.delta\"" in body
    assert "\"type\": \"response.output_text.done\"" in body
    assert "\"type\": \"response.completed\"" in body
    assert "data: [DONE]" in body


def test_responses_stream_tool_calls() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": ToolCallingProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "stream": True,
            "input": "What's the weather in Chicago?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.text
    assert "\"type\": \"response.function_call_arguments.done\"" in body
    assert "\"get_weather\"" in body
    assert "\"type\": \"response.completed\"" in body
    assert "data: [DONE]" in body


def test_anthropic_messages_stream_text_and_stop_events() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 128,
            "stream": True,
            "messages": [{"role": "user", "content": "summarize this document"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: message_start" in body
    assert "event: content_block_delta" in body
    assert "event: message_stop" in body


def test_chat_completions_can_fall_back_to_anthropic_cloud_provider() -> None:
    config = DEFAULT_CONFIG.model_copy(deep=True)
    config.providers.cloud.default = "anthropic"
    config.providers.cloud.anthropic.api_key = "anthropic-test-key"
    service = ChatService(
        config,
        router=RouterEngine(config),
        providers={"ollama": WeakLocalProvider(), "anthropic": FakeAnthropicCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "design architecture for a multi-step agent",
                }
            ],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["route"]["provider"] == "anthropic"
    assert data["choices"][0]["message"]["content"] == "anthropic: architecture answer"
    assert data["trace"]["escalated"] is True


def test_chat_completions_fall_back_to_cloud_when_local_structured_output_is_invalid() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={
            "ollama": InvalidJSONLocalProvider("not json"),
            "openai-compatible": ValidJSONCloudProvider(),
        },
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "route-auto",
            "messages": [{"role": "user", "content": "Return a JSON object"}],
            "response_format": {"type": "json_object"},
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["route"]["provider"] == "openai-compatible"
    assert data["trace"]["attempts"][0]["outcome"] == "failure"
    assert "invalid structured output" in data["trace"]["attempts"][0]["reason"]


def test_chat_completions_return_clear_error_when_structured_output_is_invalid_and_no_fallback_exists() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": InvalidJSONLocalProvider("not json")},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Return a JSON object"}],
            "response_format": {"type": "json_object"},
            "private": True,
        },
    )

    assert response.status_code == 503
    assert "invalid structured output" in response.json()["detail"]


def test_chat_completions_reject_schema_mismatch_when_no_fallback_exists() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": InvalidJSONLocalProvider('{"answer":1}')},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Return structured JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
            "private": True,
        },
    )

    assert response.status_code == 503
    assert "$.answer expected string" in response.json()["detail"]


def test_responses_validate_text_format_json_object() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": InvalidJSONLocalProvider("not json")},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "input": "Return a JSON object",
            "text": {"format": {"type": "json_object"}},
            "private": True,
        },
    )

    assert response.status_code == 503
    assert "invalid structured output" in response.json()["detail"]


def test_chat_completions_reject_string_length_and_pattern_schema_mismatches() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": InvalidJSONLocalProvider('{"code":"abc"}')},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Return structured JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "code",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "minLength": 4,
                                "pattern": "^[A-Z]+$",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
            "private": True,
        },
    )

    assert response.status_code == 503
    assert "$.code expected length >= 4" in response.json()["detail"]


def test_chat_completions_reject_numeric_bounds_schema_mismatches() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": InvalidJSONLocalProvider('{"score":0}')},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Return structured JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "score",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                            }
                        },
                        "required": ["score"],
                    },
                },
            },
            "private": True,
        },
    )

    assert response.status_code == 503
    assert "$.score expected value >= 1" in response.json()["detail"]


def test_chat_completions_reject_array_bounds_schema_mismatches() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": InvalidJSONLocalProvider('{"items":[]}')},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Return structured JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "items",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 2,
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["items"],
                    },
                },
            },
            "private": True,
        },
    )

    assert response.status_code == 503
    assert "$.items expected at least 1 item(s)" in response.json()["detail"]


def test_models_endpoint_lists_route_auto_and_configured_models() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider(), "openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}
    assert "route-auto" in model_ids
    assert DEFAULT_CONFIG.providers.local.ollama.model in model_ids
    assert DEFAULT_CONFIG.providers.cloud.openai_compatible.model in model_ids
    assert DEFAULT_CONFIG.providers.local.ollama.embedding_model in model_ids
    assert DEFAULT_CONFIG.providers.cloud.openai_compatible.embedding_model in model_ids
    route_auto = next(
        item for item in response.json()["data"] if item["id"] == "route-auto"
    )
    assert route_auto["source"] == "virtual"


def test_models_endpoint_merges_live_ollama_inventory(monkeypatch) -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider(), "openai-compatible": FakeCloudProvider()},
    )
    monkeypatch.setattr(
        service,
        "_ollama_model_inventory",
        lambda: [
            {"id": DEFAULT_CONFIG.providers.local.ollama.model, "size_bytes": 123},
            {"id": "mistral:7b", "size_bytes": 456},
        ],
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()["data"]
    qwen = next(item for item in data if item["id"] == DEFAULT_CONFIG.providers.local.ollama.model)
    assert qwen["installed"] is True
    assert qwen["size_bytes"] == 123
    mistral = next(item for item in data if item["id"] == "mistral:7b")
    assert mistral["source"] == "installed"


def test_embeddings_use_local_provider_by_default() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/embeddings",
        json={"input": "hello world", "private": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["route"]["provider"] == "ollama"
    assert data["model"] == DEFAULT_CONFIG.providers.local.ollama.embedding_model
    assert len(data["data"]) == 1
    assert data["data"][0]["embedding"] == [0.1, 0.2]


def test_embeddings_fall_back_to_cloud_when_local_provider_fails() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FailingLocalProvider(), "openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/embeddings",
        json={"input": "hello world", "private": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["route"]["provider"] == "openai-compatible"
    assert data["model"] == DEFAULT_CONFIG.providers.cloud.openai_compatible.embedding_model
    assert data["data"][0]["embedding"] == [1.1, 1.2]


def test_embeddings_return_clear_not_configured_error_for_cloud_fallback() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FailingLocalProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/embeddings",
        json={"input": "hello world", "private": False},
    )

    assert response.status_code == 501
    assert "not configured for embeddings" in response.json()["detail"]


def test_healthz_reports_provider_statuses() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "providers" in data
    assert "ollama" in data["providers"]
    assert "available" in data["providers"]["ollama"]
    assert "status" in data["providers"]["ollama"]


def test_healthz_reports_degraded_when_only_cloud_is_available() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["providers"]["ollama"]["available"] is False
    assert data["providers"]["openai-compatible"]["available"] is True


def test_healthz_reports_error_when_no_provider_is_available() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["providers"]["ollama"]["available"] is False
    assert data["providers"]["openai-compatible"]["available"] is False


def test_chat_completions_treats_route_auto_as_router_selected_model() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "route-auto",
            "messages": [{"role": "user", "content": "summarize this document"}],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == DEFAULT_CONFIG.providers.local.ollama.model
    assert data["trace"]["attempts"][0]["outcome"] == "success"


def test_chat_completions_return_tool_calls_without_escalation() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": ToolCallingProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "route-auto",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "messages": [{"role": "user", "content": "What's the weather in Chicago?"}],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert data["trace"]["escalated"] is False


def test_chat_completions_stream_text_chunks() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "route-auto",
            "stream": True,
            "messages": [{"role": "user", "content": "summarize this document"}],
            "private": False,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "chat.completion.chunk" in body
    assert "\"content\": \"echo: \"" in body
    assert "data: [DONE]" in body


def test_chat_completions_stream_tool_calls() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": ToolCallingProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "route-auto",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "messages": [{"role": "user", "content": "What's the weather in Chicago?"}],
            "private": False,
        },
    )

    assert response.status_code == 200
    body = response.text
    assert "\"tool_calls\"" in body
    assert "\"get_weather\"" in body
    assert "data: [DONE]" in body


def test_chat_completions_uses_cloud_provider_for_high_complexity_tasks() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": WeakLocalProvider(), "openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "design architecture for a multi-step agent",
                }
            ],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "cloud: architecture answer"
    assert data["route"]["provider"] == "openai-compatible"
    assert data["trace"]["initial_route"]["provider"] == "ollama"
    assert data["trace"]["escalated"] is True
    assert data["trace"]["verification"]["should_escalate"] is True


def test_chat_completions_falls_back_to_cloud_when_local_provider_fails() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FailingLocalProvider(), "openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "route-auto",
            "messages": [{"role": "user", "content": "summarize this document"}],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["route"]["provider"] == "openai-compatible"
    assert data["trace"]["escalated"] is True
    assert "falling back to cloud" in data["trace"]["escalation_reason"]
    assert data["trace"]["attempts"][0]["outcome"] == "failure"
    assert data["trace"]["attempts"][1]["outcome"] == "success"


def test_chat_completions_returns_local_answer_with_trace_when_cloud_provider_is_unconfigured() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": WeakLocalProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "design architecture for a multi-step agent",
                }
            ],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["route"]["provider"] == "ollama"
    assert data["trace"]["escalated"] is False
    assert "cloud route failed" in data["trace"]["escalation_reason"]


def test_chat_completions_auto_force_local_for_email_like_content() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider(), "openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Email alice@example.com and summarize the customer update.",
                }
            ],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["route"]["provider"] == "ollama"
    assert data["trace"]["privacy"]["detected"] is True
    assert "private_email" in data["trace"]["privacy"]["categories"]


def test_chat_completions_auto_force_local_for_code_like_content() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider(), "openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "def fix_bug(x):\n    import os\n    return x + 1",
                }
            ],
            "private": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["route"]["provider"] == "ollama"
    assert data["trace"]["privacy"]["detected"] is True
    assert "code" in data["trace"]["privacy"]["categories"]


def test_stats_endpoint_tracks_local_cloud_and_escalation_counts() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": MixedLocalProvider(), "openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "summarize this document"}],
            "private": False,
        },
    )
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "design architecture for a multi-step agent",
                }
            ],
            "private": False,
        },
    )
    stats_response = client.get("/v1/stats")

    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["total_requests"] == 2
    assert stats["local_responses"] == 1
    assert stats["cloud_responses"] == 1
    assert stats["escalations"] == 1
    assert stats["verification_checks"] == 2
    assert stats["verification_failures"] == 1
    assert stats["auto_private_requests"] == 0
    assert stats["local_response_rate"] == 0.5
    assert stats["cloud_response_rate"] == 0.5
    assert stats["escalation_rate"] == 0.5
    assert stats["estimated_total_cost_usd"] == 0.0202
    assert stats["estimated_baseline_cloud_cost_usd"] == 0.04
    assert stats["estimated_cost_saved_usd"] == 0.0198
    assert stats["estimated_cloud_requests_avoided"] == 1
    assert stats["chat_requests"] == 2
    assert stats["embedding_requests"] == 0
    assert stats["avg_total_latency_ms"] >= 0
    assert stats["avg_chat_latency_ms"] >= 0


def test_stats_endpoint_tracks_auto_private_requests() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider(), "openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Contact me at alice@example.com about this project update.",
                }
            ],
            "private": False,
        },
    )
    stats_response = client.get("/v1/stats")

    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["private_requests"] == 1
    assert stats["auto_private_requests"] == 1


def test_logs_endpoint_returns_recent_route_entries() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": MixedLocalProvider(), "openai-compatible": FakeCloudProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Contact alice@example.com about this release note.",
                }
            ],
            "private": False,
        },
    )
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "design architecture for a multi-step agent",
                }
            ],
            "private": False,
        },
    )

    logs_response = client.get("/v1/logs")

    assert logs_response.status_code == 200
    entries = logs_response.json()["entries"]
    assert len(entries) == 2
    latest = entries[0]
    earlier = entries[1]
    assert latest["trace"]["final_route"]["provider"] in {"ollama", "openai-compatible"}
    assert "request_id" in latest
    assert latest["request_kind"] == "chat"
    assert "estimated_request_cost_usd" in latest
    assert "total_latency_ms" in latest
    assert "task_preview" in latest
    assert earlier["trace"]["privacy"]["detected"] is True


def test_embeddings_requests_are_recorded_in_stats_and_logs() -> None:
    service = ChatService(
        DEFAULT_CONFIG,
        router=RouterEngine(DEFAULT_CONFIG),
        providers={"ollama": FakeProvider()},
    )
    app = create_app(service=service)
    client = TestClient(app)

    client.post(
        "/v1/embeddings",
        json={"input": "hello world", "private": False},
    )

    stats_response = client.get("/v1/stats")
    logs_response = client.get("/v1/logs")

    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["total_requests"] == 1
    assert stats["chat_requests"] == 0
    assert stats["embedding_requests"] == 1
    assert stats["avg_embedding_latency_ms"] >= 0

    assert logs_response.status_code == 200
    latest = logs_response.json()["entries"][0]
    assert latest["request_kind"] == "embeddings"
    assert latest["trace"]["attempts"][0]["duration_ms"] >= 0
