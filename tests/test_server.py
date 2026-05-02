from fastapi.testclient import TestClient

from routelabs_router.config import DEFAULT_CONFIG
from routelabs_router.models import ChatCompletionRequest, ProviderResult
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


class FakeCloudProvider:
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        return ProviderResult(
            content="cloud: architecture answer",
            model=model or "fake-cloud-model",
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


def test_chat_completions_uses_cloud_provider_for_high_complexity_tasks() -> None:
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


def test_chat_completions_returns_clear_error_when_cloud_provider_is_unconfigured() -> None:
    app = create_app()
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

    assert response.status_code == 501
    assert "not configured" in response.json()["detail"]
