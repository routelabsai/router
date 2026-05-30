import uvicorn
import yaml

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

        def list_models(self):
            class Response:
                data = []

            return Response()

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

        def list_models(self):
            class Response:
                data = []

            return Response()

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

        def list_models(self):
            class Response:
                data = []

            return Response()

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


def test_start_command_warns_when_configured_ollama_model_is_missing(
    monkeypatch, capsys
) -> None:
    def fake_run(app, host, port, reload) -> None:
        return None

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

        def list_models(self):
            class Response:
                data = [
                    type(
                        "Model",
                        (),
                        {
                            "id": "route-auto",
                            "provider": "routelabs",
                            "source": "virtual",
                            "status": "ready",
                            "installed": True,
                        },
                    )()
                ]

            return Response()

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "ChatService", FakeService)
    monkeypatch.setattr(
        "sys.argv",
        ["router", "start", "--config", "./config/router.yaml"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "configured Ollama model 'qwen3:4b' is not installed yet" in output


def test_doctor_command_prints_runtime_report(monkeypatch, capsys) -> None:
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

        def list_models(self):
            class Response:
                data = []

            return Response()

    monkeypatch.setattr(cli, "ChatService", FakeService)
    monkeypatch.setattr(
        "sys.argv",
        ["router", "doctor", "--config", "./config/router.yaml"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "RouteLabs Doctor" in output
    assert "Status: degraded" in output
    assert "Local provider (ollama): unreachable" in output
    assert "Configured local chat model:" in output
    assert "Action: run `ollama serve`" in output


def test_models_command_prints_known_models(monkeypatch, capsys) -> None:
    class FakeService:
        def __init__(self, config) -> None:
            self.config = config

        def list_models(self):
            class Response:
                data = [
                    type(
                        "Model",
                        (),
                        {
                            "id": "route-auto",
                            "provider": "routelabs",
                            "source": "virtual",
                            "status": "ready",
                            "installed": True,
                        },
                    )(),
                    type(
                        "Model",
                        (),
                        {
                            "id": "qwen3:4b",
                            "provider": "ollama",
                            "source": "installed",
                            "status": "installed",
                            "installed": True,
                        },
                    )(),
                ]

            return Response()

    monkeypatch.setattr(cli, "ChatService", FakeService)
    monkeypatch.setattr(
        "sys.argv",
        ["router", "models", "--config", "./config/router.yaml"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "RouteLabs Models" in output
    assert "route-auto" in output
    assert "qwen3:4b" in output


def test_quickstart_command_prints_adoption_paths(monkeypatch, capsys) -> None:
    class FakeService:
        def __init__(self, config) -> None:
            self.config = config

        def health(self) -> HealthResponse:
            return HealthResponse(
                status="degraded",
                providers={
                    "ollama": ProviderHealth(available=False, status="unreachable"),
                    "openai-compatible": ProviderHealth(
                        available=False, status="not_configured"
                    ),
                },
            )

        def list_models(self):
            class Response:
                data = [
                    type(
                        "Model",
                        (),
                        {
                            "id": "route-auto",
                            "provider": "routelabs",
                            "source": "virtual",
                            "status": "ready",
                            "installed": True,
                        },
                    )()
                ]

            return Response()

    monkeypatch.setattr(cli, "ChatService", FakeService)
    monkeypatch.setattr(
        "sys.argv",
        ["router", "quickstart", "--config", "./config/router.yaml"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "RouteLabs Quickstart" in output
    assert "OpenAI-compatible client setup" in output
    assert "Anthropic-compatible client setup" in output
    assert "ollama serve" in output
    assert "OPENAI_API_KEY" in output


def test_start_command_uses_anthropic_env_hint_when_anthropic_is_default_cloud(
    monkeypatch, capsys
) -> None:
    def fake_run(app, host, port, reload) -> None:
        return None

    class FakeService:
        def __init__(self, config) -> None:
            self.config = config

        def health(self) -> HealthResponse:
            return HealthResponse(
                status="degraded",
                providers={
                    "ollama": ProviderHealth(available=True, status="ready"),
                    "anthropic": ProviderHealth(
                        available=False, status="not_configured"
                    ),
                },
            )

        def list_models(self):
            class Response:
                data = []

            return Response()

    original = cli.load_config

    def fake_load_config(path):
        config = original(path)
        config.providers.cloud.default = "anthropic"
        return config

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "ChatService", FakeService)
    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(
        "sys.argv",
        ["router", "start", "--config", "./config/router.yaml"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Cloud provider (anthropic): not_configured" in output
    assert "Set `ANTHROPIC_API_KEY`" in output


def test_init_command_creates_config_with_selected_profile_and_cloud(
    monkeypatch, tmp_path, capsys
) -> None:
    output_path = tmp_path / "router.yaml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "router",
            "init",
            "--profile",
            "privacy-first",
            "--cloud",
            "anthropic",
            "--output",
            str(output_path),
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    data = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert data["routing"]["default_mode"] == "privacy-first"
    assert data["providers"]["cloud"]["default"] == "anthropic"
    assert data["providers"]["cloud"]["anthropic"]["api_key_env"] == "ANTHROPIC_API_KEY"
    assert "Created RouteLabs config" in output
    assert "router quickstart --config" in output


def test_init_command_refuses_to_overwrite_without_force(
    monkeypatch, tmp_path, capsys
) -> None:
    output_path = tmp_path / "router.yaml"
    output_path.write_text("existing: true\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "router",
            "init",
            "--output",
            str(output_path),
        ],
    )

    try:
        cli.main()
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 1

    output = capsys.readouterr().out
    assert "Config already exists" in output
