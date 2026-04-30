from routelabs_router.config import Config
from routelabs_router.models import RouteDecision, RouteRequest


class RouterEngine:
    def __init__(self, config: Config) -> None:
        self.config = config

    def decide(self, request: RouteRequest) -> RouteDecision:
        complexity = classify_complexity(request.task)

        if request.private and self.config.routing.prefer_local_for_private:
            return RouteDecision(
                target="local",
                reason="privacy policy prefers local execution for private tasks",
                complexity=complexity,
                verify=complexity != "low",
            )

        if complexity == "high":
            return RouteDecision(
                target="cloud",
                reason="high-complexity tasks default to stronger remote models",
                complexity=complexity,
                verify=True,
            )

        return RouteDecision(
            target="local",
            reason="task is suitable for local-first execution",
            complexity=complexity,
            verify=complexity == "medium",
        )


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
