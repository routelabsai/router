import argparse
from pathlib import Path

from routelabs_router.config import load_config
from routelabs_router.models import RouteRequest
from routelabs_router.router import RouterEngine


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

    args = parser.parse_args()

    if args.command == "route":
        config = load_config(Path(args.config))
        engine = RouterEngine(config)
        decision = engine.decide(
            RouteRequest(task=args.task, private=args.private == "true")
        )
        print(f"route: {decision.target}")
        print(f"reason: {decision.reason}")
        print(f"complexity: {decision.complexity}")
        print(f"verify: {decision.verify}")
