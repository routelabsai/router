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


class WeakLocalProvider:
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        return ProviderResult(
            content="I don't see any document or enough information to answer confidently.",
            model=model or "weak-local-model",
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
    assert "no cloud provider is configured" in data["trace"]["escalation_reason"]


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
