import uvicorn

from routelabs_router import cli


def test_start_command_uses_config_defaults(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(app, host, port, reload) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload
        captured["app"] = app

    monkeypatch.setattr(uvicorn, "run", fake_run)
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

    monkeypatch.setattr(uvicorn, "run", fake_run)
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
