from routelabs_router.config import DEFAULT_CONFIG
from routelabs_router.policy import LocalPolicyEngine


def test_local_policy_engine_classifies_privacy_and_complexity() -> None:
    engine = LocalPolicyEngine(DEFAULT_CONFIG)

    preflight = engine.evaluate(
        "Research and analyze tradeoffs for alice@example.com",
        explicitly_private=False,
    )

    assert preflight.engine == "local-heuristic"
    assert preflight.status == "ready"
    assert preflight.complexity == "high"
    assert preflight.privacy.detected is True
    assert "private_email" in preflight.privacy.categories


def test_local_policy_engine_is_ready_without_model_runtime() -> None:
    engine = LocalPolicyEngine(DEFAULT_CONFIG)

    available, status = engine.readiness()

    assert available is True
    assert status == "ready"
