import httpx

from routelabs_router.client import RouteLabsClient


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeHTTPClient:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def request(self, method: str, url: str, json: dict | None = None) -> FakeResponse:
        self.calls.append((method, url, json))
        return FakeResponse(
            {"method": method, "url": url, "json": json, "status": "ok"}
        )


def test_client_route(monkeypatch) -> None:
    fake = FakeHTTPClient()
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    client = RouteLabsClient("http://example.test")
    response = client.route("summarize text", private=True)

    assert response["method"] == "POST"
    assert response["url"] == "http://example.test/v1/route"
    assert response["json"] == {"task": "summarize text", "private": True}


def test_client_route_supports_agent_role(monkeypatch) -> None:
    fake = FakeHTTPClient()
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    client = RouteLabsClient("http://example.test")
    response = client.route("Implement a parser fix", agent_role="coding")

    assert response["json"] == {
        "task": "Implement a parser fix",
        "private": False,
        "agent_role": "coding",
    }


def test_client_chat_stats_logs_and_health(monkeypatch) -> None:
    fake = FakeHTTPClient()
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    client = RouteLabsClient("http://example.test")
    client.chat([{"role": "user", "content": "hello"}], private=False)
    client.responses("hello from responses")
    client.messages([{"role": "user", "content": "hello from anthropic"}])
    client.embeddings("hello world")
    client.models()
    client.stats()
    client.logs()
    client.health()

    assert fake.calls[0][1] == "http://example.test/v1/chat/completions"
    assert fake.calls[1][1] == "http://example.test/v1/responses"
    assert fake.calls[2][1] == "http://example.test/v1/messages"
    assert fake.calls[3][1] == "http://example.test/v1/embeddings"
    assert fake.calls[4][1] == "http://example.test/v1/models"
    assert fake.calls[5][1] == "http://example.test/v1/stats"
    assert fake.calls[6][1] == "http://example.test/v1/logs"
    assert fake.calls[7][1] == "http://example.test/healthz"


def test_client_chat_supports_agent_loop_fields(monkeypatch) -> None:
    fake = FakeHTTPClient()
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    client = RouteLabsClient("http://example.test")
    client.chat(
        [{"role": "user", "content": "What's the weather in Chicago?"}],
        model="route-auto",
        agent_role="planner",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="auto",
    )

    payload = fake.calls[0][2]
    assert payload["model"] == "route-auto"
    assert payload["agent_role"] == "planner"
    assert payload["tools"][0]["function"]["name"] == "get_weather"
    assert payload["tool_choice"] == "auto"


def test_client_responses_supports_openai_style_fields(monkeypatch) -> None:
    fake = FakeHTTPClient()
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    client = RouteLabsClient("http://example.test")
    client.responses(
        [{"role": "user", "content": "Return valid JSON"}],
        model="route-auto",
        agent_role="reflection",
        instructions="Be concise",
        text={"format": {"type": "json_object"}},
        max_output_tokens=120,
    )

    method, url, payload = fake.calls[0]
    assert method == "POST"
    assert url == "http://example.test/v1/responses"
    assert payload["agent_role"] == "reflection"
    assert payload["instructions"] == "Be concise"
    assert payload["text"]["format"]["type"] == "json_object"
    assert payload["max_output_tokens"] == 120


def test_client_messages_supports_anthropic_style_fields(monkeypatch) -> None:
    fake = FakeHTTPClient()
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    client = RouteLabsClient("http://example.test")
    client.messages(
        [{"role": "user", "content": "Hello Claude"}],
        model="claude-sonnet-4-20250514",
        agent_role="planner",
        system="Be concise",
        max_tokens=256,
        tool_choice={"type": "auto"},
    )

    method, url, payload = fake.calls[0]
    assert method == "POST"
    assert url == "http://example.test/v1/messages"
    assert payload["agent_role"] == "planner"
    assert payload["system"] == "Be concise"
    assert payload["max_tokens"] == 256
    assert payload["tool_choice"]["type"] == "auto"
