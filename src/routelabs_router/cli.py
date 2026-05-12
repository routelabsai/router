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

    if not local.available:
        print(
            f"Warning: local provider '{local_name}' is not ready. "
            "Start Ollama with `ollama serve` if you want local execution."
        )
    if not cloud.available:
        env_name = config.providers.cloud.openai_compatible.api_key_env or "OPENAI_API_KEY"
        print(
            f"Warning: cloud provider '{cloud_name}' is not configured. "
            f"Set `{env_name}` to enable cloud fallback and escalation."
        )
    if health.status == "error":
        print(
            "Warning: no execution path is currently available. "
            "Start a local provider or configure a cloud provider before sending requests."
        )
