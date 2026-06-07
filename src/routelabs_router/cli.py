import argparse
from pathlib import Path

import uvicorn
import yaml

from routelabs_router.config import DEFAULT_CONFIG, load_config
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

    demo_parser = subparsers.add_parser(
        "demo", help="Show focused demos for RouteLabs routing behavior"
    )
    demo_subparsers = demo_parser.add_subparsers(dest="demo", required=True)
    agent_tools_parser = demo_subparsers.add_parser(
        "agent-tools",
        help="Show MCP-style tool risk routing without starting a server",
    )
    agent_tools_parser.add_argument("--config", default="./config/router.yaml")
    agent_tools_parser.add_argument(
        "--preset",
        default="filesystem",
        choices=("filesystem", "openclaw", "hermes"),
        help="Prebuilt agent tool-risk scenario to demo",
    )
    agent_tools_parser.add_argument(
        "--task",
        default=None,
        help="Task text to route through the demo",
    )
    agent_tools_parser.add_argument(
        "--tool-name",
        default=None,
        help="Tool name to analyze for MCP-style and approval-risk signals",
    )

    init_parser = subparsers.add_parser(
        "init", help="Create a starter router config from a profile"
    )
    init_parser.add_argument(
        "--profile",
        default="balanced",
        choices=(
            "balanced",
            "local-first",
            "privacy-first",
            "openclaw",
            "hermes-agent",
            "unsloth-local",
        ),
    )
    init_parser.add_argument(
        "--cloud",
        default="openai-compatible",
        choices=("openai-compatible", "anthropic"),
        help="Default cloud provider to scaffold into the config",
    )
    init_parser.add_argument("--output", default="./config/router.yaml")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
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
    elif args.command == "doctor":
        config = load_config(Path(args.config))
        _print_doctor_report(ChatService(config), config)
    elif args.command == "models":
        config = load_config(Path(args.config))
        _print_models(ChatService(config))
    elif args.command == "quickstart":
        config = load_config(Path(args.config))
        _print_quickstart(ChatService(config), config)
    elif args.command == "demo" and args.demo == "agent-tools":
        config = load_config(Path(args.config))
        scenario = _agent_tools_demo_scenario(args.preset)
        _print_agent_tools_demo(
            config=config,
            task=args.task or scenario["task"],
            tool_name=args.tool_name or scenario["tool_name"],
            preset=args.preset,
        )
    elif args.command == "init":
        _write_init_config(
            profile=args.profile,
            cloud=args.cloud,
            output_path=Path(args.output),
            force=args.force,
        )
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


def _agent_tools_demo_scenario(preset: str) -> dict[str, str]:
    scenarios = {
        "filesystem": {
            "task": "Edit the repo config file",
            "tool_name": "mcp__filesystem__write_file",
        },
        "openclaw": {
            "task": "OpenClaw agent wants to deploy a fix and write a file",
            "tool_name": "mcp__openclaw__shell_exec",
        },
        "hermes": {
            "task": "Hermes agent wants to search memory and send a Slack update",
            "tool_name": "mcp__hermes__send_message",
        },
    }
    return scenarios[preset]


def _print_agent_tools_demo(
    config,
    task: str,
    tool_name: str,
    preset: str,
) -> None:
    engine = RouterEngine(config)
    decision = engine.decide(
        RouteRequest(
            task=task,
            private=False,
            tool_names=[tool_name],
            tool_count=1,
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
    )
    agent_tools = decision.agent_tools

    print("RouteLabs Agent Tool Demo")
    print("Local-first runtime control for agent tool use")
    print("")
    print(f"Preset: {preset}")
    print(f"Task: {task}")
    print(f"Tool: {tool_name}")
    print("")
    print(f"Route: {decision.target}")
    print(f"Provider: {decision.provider}")
    print(f"Model: {decision.model}")
    print(f"Reason: {decision.reason}")
    print(f"Verify: {decision.verify}")

    if agent_tools is None:
        print("Agent tools: none detected")
        return

    print("")
    print("Agent tool trace:")
    print(f"- Detected: {agent_tools.detected}")
    print(f"- MCP-style: {agent_tools.mcp_like}")
    print(f"- Risk level: {agent_tools.risk_level}")
    print(f"- Approval required: {agent_tools.approval_required}")
    if agent_tools.approval_reason:
        print(f"- Approval reason: {agent_tools.approval_reason}")
    if agent_tools.reasons:
        print("- Trace reasons:")
        for reason in agent_tools.reasons:
            print(f"  - {reason}")


def _write_init_config(
    profile: str,
    cloud: str,
    output_path: Path,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(
            f"Config already exists at {output_path}. "
            "Re-run with `--force` to overwrite it."
        )
        raise SystemExit(1)

    profile_path = Path("./config/profiles") / f"{profile}.yaml"
    base = DEFAULT_CONFIG.model_dump()
    profile_data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge_dicts(base, profile_data)
    merged["providers"]["cloud"]["default"] = cloud
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(merged, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Created RouteLabs config at {output_path}")
    print(f"Profile: {profile}")
    print(f"Default cloud provider: {cloud}")
    print("")
    print("Next steps:")
    print("1. Start your local runtime with `ollama serve`.")
    print("2. Pull local models if needed:")
    print(f"   ollama pull {merged['providers']['local']['ollama']['model']}")
    embedding_model = merged["providers"]["local"]["ollama"].get("embedding_model")
    if embedding_model:
        print(f"   ollama pull {embedding_model}")
    print(f"3. Set `{_cloud_api_env_name_from_provider_name(cloud, merged)}` if you want cloud fallback.")
    print(f"4. Run `router quickstart --config {output_path}`.")
    print(f"5. Run `router start --config {output_path}`.")


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


def _cloud_api_env_name_from_provider_name(cloud_name: str, config_dict: dict) -> str:
    if cloud_name == "anthropic":
        return (
            config_dict.get("providers", {})
            .get("cloud", {})
            .get("anthropic", {})
            .get("api_key_env")
            or "ANTHROPIC_API_KEY"
        )
    return (
        config_dict.get("providers", {})
        .get("cloud", {})
        .get("openai_compatible", {})
        .get("api_key_env")
        or "OPENAI_API_KEY"
    )


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge_dicts(base[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    main()
