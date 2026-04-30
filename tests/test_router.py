from routelabs_router.config import DEFAULT_CONFIG
from routelabs_router.models import RouteRequest
from routelabs_router.router import RouterEngine, classify_complexity


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
    assert decision.target == "cloud"
    assert decision.provider == "openai-compatible"
