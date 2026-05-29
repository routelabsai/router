import argparse
from pathlib import Path

import uvicorn

from routelabs_router.config import load_config
from routelabs_router.models import RouteRequest
from routelabs_router.router import RouterEngine
from routelabs_router.server.app import create_app
from routelabs_router.service import ChatService


def main() -> None:
    parser = argparse.ArgumentParser(prog="router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    route_parser = subparsers.add_parser("route", help="Inspect a routing decision")
    route_parser.add_argument("--config", default="./config/router.yaml")
    route_parser.add_argument("--task", required=True)
    route_parser.add_argument(
        "--private",
        default="false",
        choices=("true", "false"),
        help="Whether the task contains private data",
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="Inspect runtime readiness and setup gaps"
    )
    doctor_parser.add_argument("--config", default="./config/router.yaml")

    models_parser = subparsers.add_parser(
        "models", help="List configured and discovered models"
    )
    models_parser.add_argument("--config", default="./config/router.yaml")

    quickstart_parser = subparsers.add_parser(
        "quickstart", help="Show the fastest setup path for local, OpenAI, and Anthropic use"
    )
    quickstart_parser.add_argument("--config", default="./config/router.yaml")

    start_parser = subparsers.add_parser("start", help="Start the RouteLabs server")
    start_parser.add_argument("--config", default="./config/router.yaml")
    start_parser.add_argument("--host", default=None)
    start_parser.add_argument("--port", type=int, default=None)
    start_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development",
    )

    args = parser.parse_args()

    if args.command == "route":
        config = load_config(Path(args.config))
        engine = RouterEngine(config)
        decision = engine.decide(
            RouteRequest(task=args.task, private=args.private == "true")
        )
        print(f"route: {decision.target}")
        print(f"provider: {decision.provider}")
        print(f"model: {decision.model}")
        print(f"reason: {decision.reason}")
        print(f"complexity: {decision.complexity}")
        print(f"verify: {decision.verify}")
    elif args.command == "doctor":
        config = load_config(Path(args.config))
        _print_doctor_report(ChatService(config), config)
    elif args.command == "models":
        config = load_config(Path(args.config))
        _print_models(ChatService(config))
    elif args.command == "quickstart":
        config = load_config(Path(args.config))
        _print_quickstart(ChatService(config), config)
    elif args.command == "start":
        config = load_config(Path(args.config))
        _print_startup_status(config)
        app = create_app(Path(args.config))
        uvicorn.run(
            app,
            host=args.host or config.server.host,
            port=args.port or config.server.port,
            reload=args.reload,
        )


def _print_startup_status(config) -> None:
    service = ChatService(config)
    health = service.health()

    print(f"RouteLabs status: {health.status}")

    local_name = config.providers.local.default
    cloud_name = config.providers.cloud.default
    local = health.providers[local_name]
    cloud = health.providers[cloud_name]

    print(f"Local provider ({local_name}): {local.status}")
    print(f"Cloud provider ({cloud_name}): {cloud.status}")
    for model_name in _missing_configured_ollama_models(service, config):
        print(
            f"Warning: configured Ollama model '{model_name}' is not installed yet. "
            f"Pull it with `ollama pull {model_name}`."
        )

    if not local.available:
        print(
            f"Warning: local provider '{local_name}' is not ready. "
            "Start Ollama with `ollama serve` if you want local execution."
        )
    if not cloud.available:
        env_name = _cloud_api_env_name(config)
        print(
            f"Warning: cloud provider '{cloud_name}' is not configured. "
            f"Set `{env_name}` to enable cloud fallback and escalation."
        )
    if health.status == "error":
        print(
            "Warning: no execution path is currently available. "
            "Start a local provider or configure a cloud provider before sending requests."
        )


def _print_doctor_report(service: ChatService, config) -> None:
    health = service.health()
    print("RouteLabs Doctor")
    print(f"Status: {health.status}")

    local_name = config.providers.local.default
    cloud_name = config.providers.cloud.default
    local = health.providers[local_name]
    cloud = health.providers[cloud_name]

    print(f"Local provider ({local_name}): {local.status}")
    print(f"Cloud provider ({cloud_name}): {cloud.status}")
    print(f"Configured local chat model: {config.providers.local.ollama.model}")
    print(
        "Configured local embedding model: "
        f"{config.providers.local.ollama.embedding_model or config.providers.local.ollama.model}"
    )
    print(
        "Configured cloud chat model: "
        f"{config.providers.cloud.openai_compatible.model}"
    )
    print(
        "Configured cloud embedding model: "
        f"{config.providers.cloud.openai_compatible.embedding_model or config.providers.cloud.openai_compatible.model}"
    )

    installed_ollama_models = [
        model for model in service.list_models().data if model.provider == "ollama" and model.source == "installed"
    ]
    if installed_ollama_models:
        print("Installed Ollama models:")
        for model in installed_ollama_models:
            print(f"- {model.id}")
    else:
        print("Installed Ollama models: none detected")

    configured_local_models = {
        config.providers.local.ollama.model,
        config.providers.local.ollama.embedding_model
        or config.providers.local.ollama.model,
    }
    installed_ids = {model.id for model in installed_ollama_models}
    missing = sorted(model for model in configured_local_models if model not in installed_ids)
    if missing:
        print("Missing configured Ollama models:")
        for model in missing:
            print(f"- {model}")
    if not local.available:
        print("Action: run `ollama serve` to enable local execution.")
    if not cloud.available:
        env_name = _cloud_api_env_name(config)
        print(f"Action: set `{env_name}` to enable cloud fallback and escalation.")


def _print_quickstart(service: ChatService, config) -> None:
    health = service.health()
    local_name = config.providers.local.default
    cloud_name = config.providers.cloud.default
    local = health.providers[local_name]
    cloud = health.providers[cloud_name]
    missing_local_models = _missing_configured_ollama_models(service, config)

    print("RouteLabs Quickstart")
    print("Fastest adoption path: start RouteLabs, point your existing client at it, and use `route-auto`.")
    print("")
    print("Current readiness:")
    print(f"- Local provider ({local_name}): {local.status}")
    print(f"- Cloud provider ({cloud_name}): {cloud.status}")
    if missing_local_models:
        print(f"- Missing local models: {', '.join(missing_local_models)}")
    print("")

    print("1. Local-only setup")
    print("   Run:")
    print("   ollama serve")
    for model_name in missing_local_models:
        print(f"   ollama pull {model_name}")
    print("   router start")
    print("")

    print("2. OpenAI-compatible client setup")
    print("   Base URL: http://127.0.0.1:8000/v1")
    print("   Model: route-auto")
    print("   Example endpoint: /v1/chat/completions or /v1/responses")
    print("")

    print("3. Anthropic-compatible client setup")
    print("   Base URL: http://127.0.0.1:8000")
    print("   Model: claude-sonnet-4-20250514 or route-auto")
    print("   Example endpoint: /v1/messages")
    print("")

    if not cloud.available:
        print("Optional cloud fallback")
        print(f"   Set `{_cloud_api_env_name(config)}` to enable the configured cloud provider.")
        print("   Then restart `router start`.")
        print("")

    print("4. First test request")
    print("   curl -X POST http://127.0.0.1:8000/v1/chat/completions \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"model":"route-auto","messages":[{"role":"user","content":"Summarize RouteLabs Router in one sentence."}]}\'')


def _print_models(service: ChatService) -> None:
    models = service.list_models().data
    print("RouteLabs Models")
    print("Configured and discovered models:")
    for model in models:
        source = model.source or "unknown"
        provider = model.provider or "unknown"
        status = model.status or "unknown"
        installed = (
            f", installed={model.installed}" if model.installed is not None else ""
        )
        print(
            f"- {model.id} [{provider}] source={source}, status={status}{installed}"
        )


def _missing_configured_ollama_models(service: ChatService, config) -> list[str]:
    configured_local_models = {
        config.providers.local.ollama.model,
        config.providers.local.ollama.embedding_model
        or config.providers.local.ollama.model,
    }
    installed_ids = {
        model.id
        for model in service.list_models().data
        if model.provider == "ollama" and model.status == "installed"
    }
    return sorted(model for model in configured_local_models if model not in installed_ids)


def _cloud_api_env_name(config) -> str:
    cloud_name = config.providers.cloud.default
    if cloud_name == "anthropic":
        return config.providers.cloud.anthropic.api_key_env or "ANTHROPIC_API_KEY"
    return config.providers.cloud.openai_compatible.api_key_env or "OPENAI_API_KEY"
