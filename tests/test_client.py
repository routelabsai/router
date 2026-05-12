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


def test_client_chat_stats_logs_and_health(monkeypatch) -> None:
    fake = FakeHTTPClient()
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake)

    client = RouteLabsClient("http://example.test")
    client.chat([{"role": "user", "content": "hello"}], private=False)
    client.embeddings("hello world")
    client.models()
    client.stats()
    client.logs()
    client.health()

    assert fake.calls[0][1] == "http://example.test/v1/chat/completions"
    assert fake.calls[1][1] == "http://example.test/v1/embeddings"
    assert fake.calls[2][1] == "http://example.test/v1/models"
    assert fake.calls[3][1] == "http://example.test/v1/stats"
    assert fake.calls[4][1] == "http://example.test/v1/logs"
    assert fake.calls[5][1] == "http://example.test/healthz"
