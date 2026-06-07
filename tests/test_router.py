from routelabs_router.config import DEFAULT_CONFIG
from routelabs_router.models import RouteRequest
from routelabs_router.router import (
    RouterEngine,
    analyze_agent_tools,
    classify_complexity,
)


def test_classify_complexity_low() -> None:
    assert classify_complexity("say hi") == "low"


def test_classify_complexity_medium() -> None:
    assert classify_complexity("summarize this report") == "medium"


def test_classify_complexity_high() -> None:
    assert classify_complexity("research and analyze tradeoffs") == "high"


def test_private_requests_prefer_local() -> None:
    engine = RouterEngine(DEFAULT_CONFIG)
    decision = engine.decide(RouteRequest(task="research strategy", private=True))
    assert decision.target == "local"
    assert decision.provider == "ollama"


def test_high_complexity_defaults_to_cloud() -> None:
    engine = RouterEngine(DEFAULT_CONFIG)
    decision = engine.decide(RouteRequest(task="design architecture for a multi-step system"))
    assert decision.target == "local"
    assert decision.provider == "ollama"
    assert decision.verify is True


def test_analyze_agent_tools_detects_mcp_and_approval_risk() -> None:
    trace = analyze_agent_tools(
        task="Use the repo tool to edit a file",
        tool_names=["mcp__filesystem__write_file"],
        tool_choice={
            "type": "function",
            "function": {"name": "mcp__filesystem__write_file"},
        },
    )

    assert trace.detected is True
    assert trace.mcp_like is True
    assert trace.approval_required is True
    assert trace.risk_level == "high"


def test_tool_requests_start_local_with_agent_trace() -> None:
    engine = RouterEngine(DEFAULT_CONFIG)
    decision = engine.decide(
        RouteRequest(
            task="Search customer tickets",
            tool_names=["mcp__zendesk__search_tickets"],
            tool_choice="auto",
        )
    )

    assert decision.target == "local"
    assert decision.agent_tools is not None
    assert decision.agent_tools.detected is True
    assert decision.agent_tools.mcp_like is True
    assert "tool" in decision.reason


def test_tool_policy_patterns_can_require_approval() -> None:
    config = DEFAULT_CONFIG.model_copy(deep=True)
    config.policies.tools.approval_required_patterns = ["mcp__billing__*"]
    config.policies.tools.review_recommended_patterns = []
    engine = RouterEngine(config)

    decision = engine.decide(
        RouteRequest(
            task="Lookup the latest invoice",
            tool_names=["mcp__billing__lookup_invoice"],
        )
    )

    assert decision.agent_tools is not None
    assert decision.agent_tools.approval_required is True
    assert decision.agent_tools.risk_level == "high"
    assert decision.agent_tools.approval_reason == (
        "tool policy matched 'mcp__billing__*'"
    )


def test_trusted_tool_patterns_suppress_tool_name_risk() -> None:
    config = DEFAULT_CONFIG.model_copy(deep=True)
    config.policies.tools.trusted_tool_patterns = ["mcp__filesystem__write_file"]
    engine = RouterEngine(config)

    decision = engine.decide(
        RouteRequest(
            task="Summarize the draft",
            tool_names=["mcp__filesystem__write_file"],
        )
    )

    assert decision.agent_tools is not None
    assert decision.agent_tools.mcp_like is True
    assert decision.agent_tools.trusted_tool_names == ["mcp__filesystem__write_file"]
    assert decision.agent_tools.approval_required is False
    assert decision.agent_tools.risk_level == "low"
