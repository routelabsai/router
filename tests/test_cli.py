import uvicorn

from routelabs_router import cli
from routelabs_router.models import HealthResponse, ProviderHealth


def test_start_command_uses_config_defaults(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(app, host, port, reload) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload
        captured["app"] = app

    class FakeService:
        def __init__(self, config) -> None:
            self.config = config

        def health(self) -> HealthResponse:
            return HealthResponse(
                status="ok",
                providers={
                    "ollama": ProviderHealth(available=True, status="ready"),
                    "openai-compatible": ProviderHealth(
                        available=False, status="not_configured"
                    ),
                },
            )

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "ChatService", FakeService)
    monkeypatch.setattr(
        "sys.argv",
        ["router", "start", "--config", "./config/router.yaml"],
    )

    cli.main()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
    assert captured["reload"] is False


def test_start_command_allows_overrides(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(app, host, port, reload) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload

    class FakeService:
        def __init__(self, config) -> None:
            self.config = config

        def health(self) -> HealthResponse:
            return HealthResponse(
                status="degraded",
                providers={
                    "ollama": ProviderHealth(available=False, status="unreachable"),
                    "openai-compatible": ProviderHealth(
                        available=True, status="configured"
                    ),
                },
            )

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "ChatService", FakeService)
    monkeypatch.setattr(
        "sys.argv",
        [
            "router",
            "start",
            "--config",
            "./config/router.yaml",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--reload",
        ],
    )

    cli.main()

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000
    assert captured["reload"] is True


def test_start_command_prints_actionable_startup_warnings(
    monkeypatch, capsys
) -> None:
    def fake_run(app, host, port, reload) -> None:
        return None

    class FakeService:
        def __init__(self, config) -> None:
            self.config = config

        def health(self) -> HealthResponse:
            return HealthResponse(
                status="error",
                providers={
                    "ollama": ProviderHealth(available=False, status="unreachable"),
                    "openai-compatible": ProviderHealth(
                        available=False, status="not_configured"
                    ),
                },
            )

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "ChatService", FakeService)
    monkeypatch.setattr(
        "sys.argv",
        ["router", "start", "--config", "./config/router.yaml"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "RouteLabs status: error" in output
    assert "Local provider (ollama): unreachable" in output
    assert "Cloud provider (openai-compatible): not_configured" in output
    assert "Start Ollama with `ollama serve`" in output
    assert "Set `OPENAI_API_KEY`" in output
    assert "no execution path is currently available" in output
