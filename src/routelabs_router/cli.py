import argparse
from importlib.resources import files
from pathlib import Path

import uvicorn
import yaml

from routelabs_router.config import DEFAULT_CONFIG, load_config
from routelabs_router.hardware import detect_machine_profile, recommend_local_model
from routelabs_router.models import RouteRequest
from routelabs_router.router import RouterEngine
from routelabs_router.service import ChatService


def main() -> None:
    parser = argparse.ArgumentParser(prog="router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    route_parser = subparsers.add_parser("route", help="Inspect a routing decision")
    route_parser.add_argument("--config", default="./config/router.yaml")
    route_parser.add_argument("--task", required=True)
    route_parser.add_argument(
        "--agent-role",
        default=None,
        help="Route through a configured agent role such as planner, coding, vision, or reflection",
    )
    route_parser.add_argument(
        "--private",
        default="false",
        choices=("true", "false"),
        help="Whether the task contains private data",
    )
    route_parser.add_argument(
        "--allow-fallbacks",
        default="true",
        choices=("true", "false"),
        help="Whether cloud fallback and verification escalation are allowed",
    )
    route_parser.add_argument(
        "--max-cloud-cost-usd",
        type=float,
        default=None,
        help="Block cloud fallback if the configured request cost is above this cap",
    )
    route_parser.add_argument(
        "--tool-name",
        action="append",
        default=[],
        help="Declare a tool name for agent/MCP-style risk inspection",
    )
    route_parser.add_argument(
        "--tool-description",
        action="append",
        default=[],
        metavar="NAME=DESCRIPTION",
        help="Declare tool metadata for prompt-injection risk inspection",
    )
    route_parser.add_argument(
        "--tool-choice",
        default=None,
        help="Declare a requested tool choice such as auto, required, or a tool name",
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="Inspect runtime readiness and setup gaps"
    )
    doctor_parser.add_argument("--config", default="./config/router.yaml")

    models_parser = subparsers.add_parser(
        "models", help="List configured and discovered models"
    )
    models_parser.add_argument("--config", default="./config/router.yaml")

    subparsers.add_parser(
        "profiles", help="List starter config profiles available for router init"
    )

    quickstart_parser = subparsers.add_parser(
        "quickstart", help="Show the fastest setup path for local, OpenAI, and Anthropic use"
    )
    quickstart_parser.add_argument("--config", default="./config/router.yaml")

    recommend_parser = subparsers.add_parser(
        "recommend", help="Recommend local models for this machine"
    )
    recommend_subparsers = recommend_parser.add_subparsers(
        dest="recommendation",
        required=True,
    )
    local_model_parser = recommend_subparsers.add_parser(
        "local-model",
        help="Recommend an Ollama model based on local CPU/GPU/RAM",
    )
    local_model_parser.add_argument(
        "--workload",
        default="general",
        choices=("general", "coding", "agent"),
        help="Optimize the recommendation for a workload",
    )

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
    agent_roles_parser = demo_subparsers.add_parser(
        "agent-roles",
        help="Show role-aware routing for planner, coding, vision, and reflection",
    )
    agent_roles_parser.add_argument("--config", default="./config/router.yaml")
    agent_roles_parser.add_argument(
        "--role",
        default=None,
        help="Show one configured role instead of all roles",
    )
    agent_roles_parser.add_argument(
        "--task",
        default="Route this workflow step through the right local agent lane",
        help="Task text to route through each role",
    )

    init_parser = subparsers.add_parser(
        "init", help="Create a starter router config from a profile"
    )
    init_parser.add_argument(
        "--profile",
        default="balanced",
        choices=_available_profile_names(),
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
        try:
            tool_descriptions = _parse_tool_descriptions(args.tool_description)
        except ValueError as exc:
            parser.error(str(exc))
        service = ChatService(config)
        decision = service.inspect_route(
            RouteRequest(
                task=args.task,
                agent_role=args.agent_role,
                private=args.private == "true",
                allow_fallbacks=args.allow_fallbacks == "true",
                max_cloud_cost_usd=args.max_cloud_cost_usd,
                tool_names=args.tool_name,
                tool_descriptions=tool_descriptions,
                tool_count=len(args.tool_name) if args.tool_name else None,
                tool_choice=args.tool_choice,
            )
        )
        _print_route_decision(decision)
    elif args.command == "doctor":
        config = load_config(Path(args.config))
        _print_doctor_report(ChatService(config), config)
    elif args.command == "models":
        config = load_config(Path(args.config))
        _print_models(ChatService(config))
    elif args.command == "profiles":
        _print_profiles()
    elif args.command == "quickstart":
        config = load_config(Path(args.config))
        _print_quickstart(ChatService(config), config)
    elif args.command == "recommend" and args.recommendation == "local-model":
        _print_local_model_recommendation(workload=args.workload)
    elif args.command == "demo" and args.demo == "agent-tools":
        config = load_config(Path(args.config))
        scenario = _agent_tools_demo_scenario(args.preset)
        _print_agent_tools_demo(
            config=config,
            task=args.task or scenario["task"],
            tool_name=args.tool_name or scenario["tool_name"],
            preset=args.preset,
        )
    elif args.command == "demo" and args.demo == "agent-roles":
        config = load_config(Path(args.config))
        _print_agent_roles_demo(
            config=config,
            task=args.task,
            role=args.role,
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
        from routelabs_router.server.app import create_app

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
    print(f"Policy engine: {health.policy_engine.status}")

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
        if local_name == "ollama":
            print(
                f"Warning: local provider '{local_name}' is not ready. "
                "Start Ollama with `ollama serve` if you want local execution."
            )
        else:
            print(f"Warning: local provider '{local_name}' is not ready.")
            for action in _local_runtime_actions(config):
                print(f"Action: {action}")
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
    print(f"Policy engine: {health.policy_engine.status}")

    local_name = config.providers.local.default
    cloud_name = config.providers.cloud.default
    local = health.providers[local_name]
    cloud = health.providers[cloud_name]

    print(f"Local provider ({local_name}): {local.status}")
    print(f"Cloud provider ({cloud_name}): {cloud.status}")
    print(f"Configured local chat model: {_local_chat_model(config)}")
    print("Configured local embedding model: " f"{_local_embedding_model(config)}")
    print(
        "Configured cloud chat model: "
        f"{config.providers.cloud.openai_compatible.model}"
    )
    print(
        "Configured cloud embedding model: "
        f"{config.providers.cloud.openai_compatible.embedding_model or config.providers.cloud.openai_compatible.model}"
    )
    role_bindings = _agent_role_model_bindings(config)
    if role_bindings:
        print("Configured agent role models:")
        for role_name, target, provider, model, verify in role_bindings:
            print(
                f"- {role_name}: {target}/{provider}/{model}, "
                f"verify={verify}"
            )

    installed_ollama_models = [
        model
        for model in service.list_models().data
        if model.provider == "ollama" and model.source == "installed"
    ]
    if installed_ollama_models:
        print("Installed Ollama models:")
        for model in installed_ollama_models:
            print(f"- {model.id}")
    else:
        print("Installed Ollama models: none detected")

    configured_local_models = set(_configured_ollama_model_ids(config))
    installed_ids = {model.id for model in installed_ollama_models}
    missing = sorted(
        model for model in configured_local_models if model not in installed_ids
    )
    if missing:
        print("Missing configured Ollama models:")
        for model in missing:
            print(f"- {model}")
    if not local.available:
        for action in _local_runtime_actions(config):
            print(f"Action: {action}")
    if not cloud.available:
        env_name = _cloud_api_env_name(config)
        print(f"Action: set `{env_name}` to enable cloud fallback and escalation.")


def _print_route_decision(decision) -> None:
    print(f"route: {decision.target}")
    print(f"provider: {decision.provider}")
    print(f"model: {decision.model}")
    print(f"reason: {decision.reason}")
    print(f"complexity: {decision.complexity}")
    print(f"verify: {decision.verify}")
    print(f"policy_engine: {decision.policy_engine}")
    print(f"policy_engine_status: {decision.policy_engine_status}")
    if decision.agent_role is not None:
        print(f"agent_role: {decision.agent_role}")
    if decision.provider_available is not None:
        print(f"provider_available: {decision.provider_available}")
    if decision.provider_status is not None:
        print(f"provider_status: {decision.provider_status}")
    if decision.fallback_available is not None:
        print(f"fallback_available: {decision.fallback_available}")
    if decision.fallback_status is not None:
        print(f"fallback_status: {decision.fallback_status}")

    agent_tools = decision.agent_tools
    if agent_tools is None or not agent_tools.detected:
        return

    print("")
    print("agent_tools:")
    print(f"- detected: {agent_tools.detected}")
    print(f"- mcp_like: {agent_tools.mcp_like}")
    print(f"- risk_level: {agent_tools.risk_level}")
    print(f"- approval_required: {agent_tools.approval_required}")
    if agent_tools.tool_names:
        print(f"- tool_names: {', '.join(agent_tools.tool_names)}")
    if agent_tools.trusted_tool_names:
        print(f"- trusted_tool_names: {', '.join(agent_tools.trusted_tool_names)}")
    if agent_tools.suspicious_tool_names:
        print(
            "- suspicious_tool_names: "
            + ", ".join(agent_tools.suspicious_tool_names)
        )
    if agent_tools.approval_reason:
        print(f"- approval_reason: {agent_tools.approval_reason}")
    if agent_tools.reasons:
        print("- reasons:")
        for reason in agent_tools.reasons:
            print(f"  - {reason}")


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
    for step in _local_runtime_setup_steps(config):
        print(f"   {step}")
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


def _print_local_model_recommendation(workload: str) -> None:
    profile = detect_machine_profile()
    recommendation = recommend_local_model(profile, workload=workload)

    print("RouteLabs Local Model Recommendation")
    print(f"Workload: {workload}")
    print("")
    print("Detected machine:")
    print(f"- OS: {profile.os_name}")
    print(f"- Architecture: {profile.arch}")
    print(f"- CPU cores: {profile.cpu_count}")
    print(f"- RAM: {_format_optional_gb(profile.memory_gb)}")
    print(f"- Accelerator: {profile.accelerator}")
    if profile.gpu_name:
        print(f"- GPU: {profile.gpu_name}")
    if profile.gpu_memory_gb is not None:
        print(f"- GPU memory: {_format_optional_gb(profile.gpu_memory_gb)}")
    print("")
    print("Recommended local model:")
    print(f"- Model: {recommendation.model}")
    print(f"- Embeddings: {recommendation.embedding_model}")
    print(f"- Profile: {recommendation.profile}")
    print(f"- Reason: {recommendation.reason}")
    print("")
    print("Run:")
    for command in recommendation.pull_commands:
        print(f"  {command}")
    print("")
    print("Config hint:")
    for key, value in recommendation.config_hint.items():
        print(f"  {key}: {value}")


def _format_optional_gb(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.1f} GB"


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


def _print_agent_roles_demo(
    config,
    task: str,
    role: str | None = None,
) -> None:
    engine = RouterEngine(config)
    role_names = _selected_agent_role_names(config, role)

    print("RouteLabs Agent Role Demo")
    print("Role-aware local-first routing for agent workflow steps")
    print("")
    print(f"Task: {task}")

    if not role_names:
        if role:
            print(f"No configured agent role named '{role}'.")
        else:
            print("No configured agent roles.")
        return

    print("")
    print("Role routes:")
    for role_name in role_names:
        decision = engine.decide(
            RouteRequest(
                task=task,
                agent_role=role_name,
            )
        )
        print(
            f"- {role_name}: {decision.target}/{decision.provider}/"
            f"{decision.model}, verify={decision.verify}"
        )
        print(f"  reason: {decision.reason}")


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

    base = DEFAULT_CONFIG.model_dump()
    profile_data = _load_profile_data(profile)
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
    for step in _init_local_runtime_steps(merged):
        print(step)
    cloud_config = _cloud_provider_config_from_name(cloud, merged)
    if cloud_config.get("requires_api_key", True):
        print(
            f"3. Set `{_cloud_api_env_name_from_provider_name(cloud, merged)}` "
            "if you want cloud fallback."
        )
    else:
        print(
            "3. Start your OpenAI-compatible proxy if you want cloud fallback "
            f"({cloud_config.get('base_url', 'configured base_url')})."
        )
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


def _print_profiles() -> None:
    print("RouteLabs Profiles")
    print("Starter configs available for `router init --profile`:")
    for profile_name in _available_profile_names():
        profile = _load_profile_data(profile_name)
        merged_profile = _deep_merge_dicts(DEFAULT_CONFIG.model_dump(), profile)
        default_mode = merged_profile.get("routing", {}).get(
            "default_mode", "balanced"
        )
        local = merged_profile.get("providers", {}).get("local", {})
        local_provider = local.get("default", "ollama")
        local_model = _profile_local_model(merged_profile, local_provider)
        roles = profile.get("agents", {}).get("roles", {})
        role_count = len(roles) if isinstance(roles, dict) else 0
        role_suffix = f", roles={role_count}" if role_count else ""
        print(
            f"- {profile_name}: mode={default_mode}, "
            f"local={local_provider}/{local_model}{role_suffix}"
        )


def _profile_local_model(profile: dict, local_provider: str) -> str:
    local = profile.get("providers", {}).get("local", {})
    provider = local.get(local_provider, {}) if isinstance(local, dict) else {}
    if isinstance(provider, dict):
        return str(provider.get("model", "default"))
    return "default"


def _load_profile_data(profile: str) -> dict:
    profile_filename = f"{profile}.yaml"
    profile_path = Path("./config/profiles") / profile_filename
    if profile_path.exists():
        return yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}

    profile_resource = files("routelabs_router.profiles").joinpath(profile_filename)
    return yaml.safe_load(profile_resource.read_text(encoding="utf-8")) or {}


def _available_profile_names() -> list[str]:
    names = set()
    source_profiles = Path("./config/profiles")
    if source_profiles.exists():
        names.update(path.stem for path in source_profiles.glob("*.yaml"))
    package_profiles = files("routelabs_router.profiles")
    names.update(
        path.name.removesuffix(".yaml")
        for path in package_profiles.iterdir()
        if path.name.endswith(".yaml")
    )
    return sorted(names)


def _missing_configured_ollama_models(service: ChatService, config) -> list[str]:
    if config.providers.local.default != "ollama":
        return []
    configured_local_models = set(_configured_ollama_model_ids(config))
    installed_ids = {
        model.id
        for model in service.list_models().data
        if model.provider == "ollama" and model.status == "installed"
    }
    return sorted(
        model for model in configured_local_models if model not in installed_ids
    )


def _configured_ollama_model_ids(config) -> list[str]:
    model_ids = {
        config.providers.local.ollama.model,
        config.providers.local.ollama.embedding_model
        or config.providers.local.ollama.model,
    }
    for role in config.agents.roles.values():
        if _agent_role_uses_ollama(config, role):
            model_ids.add(role.model)
    return sorted(model for model in model_ids if model)


def _agent_role_uses_ollama(config, role) -> bool:
    target = role.target
    provider = role.provider or (
        config.providers.local.default
        if target == "local"
        else config.providers.cloud.default
    )
    return target == "local" and provider == "ollama"


def _agent_role_model_bindings(config) -> list[tuple[str, str, str, str, bool]]:
    bindings = []
    for role_name, role in sorted(config.agents.roles.items()):
        target = role.target
        provider = role.provider or (
            config.providers.local.default
            if target == "local"
            else config.providers.cloud.default
        )
        bindings.append((role_name, target, provider, role.model, role.verify))
    return bindings


def _selected_agent_role_names(config, role: str | None) -> list[str]:
    configured = sorted(config.agents.roles)
    if role is None:
        preferred = ["router", "planner", "coding", "vision", "reflection"]
        ordered = [name for name in preferred if name in config.agents.roles]
        ordered.extend(name for name in configured if name not in ordered)
        return ordered
    role_name = role.strip().lower()
    return [role_name] if role_name in config.agents.roles else []


def _local_chat_model(config) -> str:
    local_name = config.providers.local.default
    if local_name == "llamacpp":
        return config.providers.local.llamacpp.model
    return config.providers.local.ollama.model


def _local_embedding_model(config) -> str:
    local_name = config.providers.local.default
    if local_name == "llamacpp":
        return (
            config.providers.local.llamacpp.embedding_model
            or config.providers.local.llamacpp.model
        )
    return (
        config.providers.local.ollama.embedding_model
        or config.providers.local.ollama.model
    )


def _local_runtime_actions(config) -> list[str]:
    local_name = config.providers.local.default
    if local_name == "llamacpp":
        base_url = config.providers.local.llamacpp.base_url
        return [
            "start your OpenAI-compatible local server "
            f"(llama.cpp, LM Studio, or vLLM) at `{base_url}`"
        ]
    return ["run `ollama serve` to enable local execution"]


def _local_runtime_setup_steps(config) -> list[str]:
    local_name = config.providers.local.default
    if local_name == "llamacpp":
        base_url = config.providers.local.llamacpp.base_url
        return [
            "start your OpenAI-compatible local server "
            f"(llama.cpp, LM Studio, or vLLM) at {base_url}"
        ]
    return ["ollama serve"]


def _cloud_api_env_name(config) -> str:
    cloud_name = config.providers.cloud.default
    if cloud_name == "anthropic":
        return config.providers.cloud.anthropic.api_key_env or "ANTHROPIC_API_KEY"
    return config.providers.cloud.openai_compatible.api_key_env or "OPENAI_API_KEY"


def _cloud_api_env_name_from_provider_name(cloud_name: str, config_dict: dict) -> str:
    provider = _cloud_provider_config_from_name(cloud_name, config_dict)
    if cloud_name == "anthropic":
        return provider.get("api_key_env") or "ANTHROPIC_API_KEY"
    return provider.get("api_key_env") or "OPENAI_API_KEY"


def _cloud_provider_config_from_name(cloud_name: str, config_dict: dict) -> dict:
    key = "anthropic" if cloud_name == "anthropic" else "openai_compatible"
    provider = config_dict.get("providers", {}).get("cloud", {}).get(key, {})
    return provider if isinstance(provider, dict) else {}


def _parse_tool_descriptions(values: list[str]) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--tool-description must use NAME=DESCRIPTION format"
            )
        name, description = value.split("=", 1)
        name = name.strip()
        description = description.strip()
        if not name or not description:
            raise ValueError(
                "--tool-description must include both NAME and DESCRIPTION"
            )
        descriptions[name] = description
    return descriptions


def _init_local_runtime_steps(config_dict: dict) -> list[str]:
    local = config_dict.get("providers", {}).get("local", {})
    if not isinstance(local, dict):
        return ["1. Start your local runtime.", "2. Confirm your local model is ready."]

    local_name = local.get("default", "ollama")
    if local_name == "llamacpp":
        provider = local.get("llamacpp", {})
        provider = provider if isinstance(provider, dict) else {}
        base_url = provider.get("base_url", "http://127.0.0.1:8080/v1")
        model = provider.get("model", "configured local model")
        return [
            "1. Start your OpenAI-compatible local server "
            f"(llama.cpp, LM Studio, or vLLM) at `{base_url}`.",
            f"2. Confirm it exposes model `{model}` from `/v1/models`.",
        ]

    provider = local.get("ollama", {})
    provider = provider if isinstance(provider, dict) else {}
    steps = ["1. Start your local runtime with `ollama serve`."]
    steps.append("2. Pull local models if needed:")
    model = provider.get("model")
    if model:
        steps.append(f"   ollama pull {model}")
    embedding_model = provider.get("embedding_model")
    if embedding_model:
        steps.append(f"   ollama pull {embedding_model}")
    return steps


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
