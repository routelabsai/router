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
                provider=self.config.providers.local.default,
                model=self._provider_model("local", self.config.providers.local.default),
                reason="privacy policy prefers local execution for private tasks",
                complexity=complexity,
                verify=complexity != "low",
            )

        if complexity == "high":
            return RouteDecision(
                target="local",
                provider=self.config.providers.local.default,
                model=self._provider_model("local", self.config.providers.local.default),
                reason="high-complexity tasks start local and rely on verification before escalation",
                complexity=complexity,
                verify=True,
            )

        return RouteDecision(
            target="local",
            provider=self.config.providers.local.default,
            model=self._provider_model("local", self.config.providers.local.default),
            reason="task is suitable for local-first execution",
            complexity=complexity,
            verify=complexity == "medium",
        )

    def _provider_model(self, target: str, provider: str) -> str:
        if target == "local":
            if provider == "ollama":
                return self.config.providers.local.ollama.model
            if provider == "llamacpp":
                return self.config.providers.local.llamacpp.model
        if target == "cloud" and provider == "openai-compatible":
            return self.config.providers.cloud.openai_compatible.model
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
