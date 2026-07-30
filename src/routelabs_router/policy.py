from dataclasses import dataclass

from routelabs_router.config import Config
from routelabs_router.models import PrivacyDetectionResult
from routelabs_router.privacy import HeuristicPrivacyDetector


@dataclass(frozen=True)
class PolicyPreflight:
    complexity: str
    privacy: PrivacyDetectionResult
    engine: str
    status: str


class LocalPolicyEngine:
    def __init__(
        self,
        config: Config,
        privacy_detector: HeuristicPrivacyDetector | None = None,
    ) -> None:
        self.config = config
        self.name = config.policies.engine
        self.privacy_detector = privacy_detector or HeuristicPrivacyDetector()

    def evaluate(self, text: str, explicitly_private: bool = False) -> PolicyPreflight:
        _, status = self.readiness()
        return PolicyPreflight(
            complexity=classify_task_complexity(text),
            privacy=self.privacy_detector.evaluate(
                text,
                explicitly_private=explicitly_private,
            ),
            engine=self.name,
            status=status,
        )

    def readiness(self) -> tuple[bool, str]:
        if self.name == "local-heuristic":
            return True, "ready"
        return False, f"unknown policy engine '{self.name}'"


def classify_task_complexity(task: str) -> str:
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
