from fnmatch import fnmatch

from routelabs_router.config import Config
from routelabs_router.models import AgentToolTrace, RouteDecision, RouteRequest


class RouterEngine:
    def __init__(self, config: Config) -> None:
        self.config = config

    def decide(self, request: RouteRequest) -> RouteDecision:
        complexity = classify_complexity(request.task)
        agent_tools = analyze_agent_tools(
            task=request.task,
            tool_names=request.tool_names,
            tool_descriptions=request.tool_descriptions,
            tool_count=request.tool_count,
            tool_choice=request.tool_choice,
            approval_required_patterns=(
                self.config.policies.tools.approval_required_patterns
            ),
            review_recommended_patterns=(
                self.config.policies.tools.review_recommended_patterns
            ),
            trusted_tool_patterns=self.config.policies.tools.trusted_tool_patterns,
        )

        if request.private and self.config.routing.prefer_local_for_private:
            return RouteDecision(
                target="local",
                provider=self.config.providers.local.default,
                model=self._provider_model("local", self.config.providers.local.default),
                reason="privacy policy prefers local execution for private tasks",
                complexity=complexity,
                verify=agent_tools.approval_required or complexity != "low",
                agent_tools=agent_tools,
            )

        if agent_tools.detected:
            return RouteDecision(
                target="local",
                provider=self.config.providers.local.default,
                model=self._provider_model("local", self.config.providers.local.default),
                reason=_agent_tool_route_reason(agent_tools),
                complexity=complexity,
                verify=agent_tools.approval_required or complexity != "low",
                agent_tools=agent_tools,
            )

        if complexity == "high":
            return RouteDecision(
                target="local",
                provider=self.config.providers.local.default,
                model=self._provider_model("local", self.config.providers.local.default),
                reason="high-complexity tasks start local and rely on verification before escalation",
                complexity=complexity,
                verify=True,
                agent_tools=agent_tools,
            )

        return RouteDecision(
            target="local",
            provider=self.config.providers.local.default,
            model=self._provider_model("local", self.config.providers.local.default),
            reason="task is suitable for local-first execution",
            complexity=complexity,
            verify=complexity == "medium",
            agent_tools=agent_tools,
        )

    def _provider_model(self, target: str, provider: str) -> str:
        if target == "local":
            if provider == "ollama":
                return self.config.providers.local.ollama.model
            if provider == "llamacpp":
                return self.config.providers.local.llamacpp.model
        if target == "cloud" and provider == "openai-compatible":
            return self.config.providers.cloud.openai_compatible.model
        if target == "cloud" and provider == "anthropic":
            return self.config.providers.cloud.anthropic.model
        return "unknown"


def classify_complexity(task: str) -> str:
    lowered = task.lower()

    high_signals = [
        "prove",
        "design architecture",
        "multi-step",
        "research",
        "analyze tradeoffs",
        "debug production",
    ]
    if any(signal in lowered for signal in high_signals):
        return "high"

    medium_signals = [
        "summarize",
        "classify",
        "extract",
        "rewrite",
        "compare",
    ]
    if any(signal in lowered for signal in medium_signals):
        return "medium"

    return "low"


def analyze_agent_tools(
    task: str,
    tool_names: list[str] | None = None,
    tool_descriptions: dict[str, str] | None = None,
    tool_count: int | None = None,
    tool_choice: str | dict[str, object] | None = None,
    approval_required_patterns: list[str] | None = None,
    review_recommended_patterns: list[str] | None = None,
    trusted_tool_patterns: list[str] | None = None,
) -> AgentToolTrace:
    names = _dedupe_tool_names(tool_names or [])
    inferred_count = tool_count if tool_count is not None else len(names)
    lowered_task = task.lower()
    trusted_names = [
        name
        for name in names
        if _matches_any_policy_pattern(name, trusted_tool_patterns or [])
    ]
    untrusted_names = [name for name in names if name not in trusted_names]
    all_lowered_names = [name.lower() for name in names]
    lowered_names = [name.lower() for name in untrusted_names]
    descriptions = _clean_tool_descriptions(tool_descriptions or {})
    metadata_findings = _metadata_risk_findings(descriptions)
    reasons: list[str] = []

    mcp_like = (
        any(_looks_mcp_like(name) for name in all_lowered_names)
        or "mcp" in lowered_task
    )
    if mcp_like:
        reasons.append("MCP-style tool context detected")

    if names:
        reasons.append(f"{len(names)} declared tool(s): {', '.join(names[:5])}")
    elif inferred_count:
        reasons.append(f"{inferred_count} declared tool(s)")
    if trusted_names:
        reasons.append(
            "trusted tool policy matched: " + ", ".join(trusted_names[:5])
        )
    if metadata_findings:
        reasons.append(
            "suspicious tool metadata detected: "
            + ", ".join(
                f"{name} matched '{signal}'"
                for name, signal in metadata_findings[:3]
            )
        )

    if _tool_choice_forces_tool(tool_choice):
        reasons.append("tool_choice requires tool use")

    approval_required_patterns = (
        approval_required_patterns or DEFAULT_APPROVAL_REQUIRED_PATTERNS
    )
    review_recommended_patterns = (
        review_recommended_patterns or DEFAULT_REVIEW_RECOMMENDED_PATTERNS
    )
    high_signal = _first_policy_signal(
        values=[*lowered_names, lowered_task],
        patterns=approval_required_patterns,
    )
    metadata_signal = metadata_findings[0][1] if metadata_findings else None
    if metadata_signal is not None:
        high_signal = f"tool_metadata:{metadata_signal}"
    review_signal = None
    if high_signal is None:
        review_signal = _first_policy_signal(
            values=[*lowered_names, lowered_task],
            patterns=review_recommended_patterns,
        )

    risky_signal = high_signal or review_signal
    approval_required = risky_signal is not None
    approval_reason = None
    risk_level = "none"
    if approval_required:
        risk_level = "high" if high_signal is not None else "medium"
        approval_reason = f"tool policy matched '{risky_signal}'"
        reasons.append(f"approval recommended by tool policy '{risky_signal}'")

    detected = bool(
        names
        or inferred_count
        or mcp_like
        or metadata_findings
        or _tool_choice_forces_tool(tool_choice)
    )
    if detected and risk_level == "none":
        risk_level = "low"

    return AgentToolTrace(
        detected=detected,
        tool_count=max(inferred_count, len(names)),
        tool_names=names,
        trusted_tool_names=trusted_names,
        suspicious_tool_names=[name for name, _ in metadata_findings],
        mcp_like=mcp_like,
        metadata_risk_detected=bool(metadata_findings),
        approval_required=approval_required,
        approval_reason=approval_reason,
        risk_level=risk_level,
        reasons=reasons,
    )


DEFAULT_APPROVAL_REQUIRED_PATTERNS = [
    "delete",
    "remove",
    "drop",
    "reset",
    "write",
    "edit",
    "commit",
    "push",
    "deploy",
    "merge",
    "purchase",
    "payment",
    "pay",
    "transfer",
    "email",
    "send",
    "shell",
    "exec",
    "terminal",
]

DEFAULT_REVIEW_RECOMMENDED_PATTERNS = [
    "read",
    "search",
    "fetch",
    "browser",
    "web",
    "database",
    "db",
    "filesystem",
    "file",
    "calendar",
    "issue",
    "ticket",
]

TOOL_METADATA_RISK_PATTERNS = [
    "ignore previous",
    "ignore all previous",
    "ignore the user",
    "ignore system",
    "developer message",
    "system prompt",
    "hidden instruction",
    "do not disclose",
    "do not reveal",
    "secretly",
    "exfiltrate",
    "leak",
    "credential",
    "api key",
    "private key",
    "bypass",
    "disable safety",
    "jailbreak",
]


def _agent_tool_route_reason(agent_tools: AgentToolTrace) -> str:
    if agent_tools.approval_required:
        return "agent tool request starts local so approval-sensitive actions stay inspectable"
    if agent_tools.mcp_like:
        return "MCP-style agent request starts local with tool-use trace visibility"
    return "tool-using agent request starts local with route and tool trace visibility"


def _dedupe_tool_names(tool_names: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for name in tool_names:
        cleaned = str(name).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _clean_tool_descriptions(tool_descriptions: dict[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for name, description in tool_descriptions.items():
        clean_name = str(name).strip()
        clean_description = str(description).strip()
        if clean_name and clean_description:
            cleaned[clean_name] = clean_description
    return cleaned


def _metadata_risk_findings(
    tool_descriptions: dict[str, str],
) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for name, description in tool_descriptions.items():
        lowered = description.lower()
        for pattern in TOOL_METADATA_RISK_PATTERNS:
            if pattern in lowered:
                findings.append((name, pattern))
                break
    return findings


def _looks_mcp_like(name: str) -> bool:
    return name.startswith("mcp__") or name.startswith("mcp_") or "__" in name


def _tool_choice_forces_tool(tool_choice: str | dict[str, object] | None) -> bool:
    if isinstance(tool_choice, str):
        return tool_choice not in {"auto", "none"}
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {"function", "tool", "required"}:
            return True
        function = tool_choice.get("function")
        return isinstance(function, dict) and bool(function.get("name"))
    return False


def _first_policy_signal(values: list[str], patterns: list[str]) -> str | None:
    for value in values:
        for pattern in patterns:
            if _policy_pattern_matches(value=value, pattern=pattern):
                return pattern
    return None


def _matches_any_policy_pattern(value: str, patterns: list[str]) -> bool:
    return any(
        _policy_pattern_matches(value=value, pattern=pattern)
        for pattern in patterns
    )


def _policy_pattern_matches(value: str, pattern: str) -> bool:
    lowered_value = value.lower()
    lowered_pattern = pattern.lower()
    return fnmatch(lowered_value, lowered_pattern) or lowered_pattern in lowered_value
