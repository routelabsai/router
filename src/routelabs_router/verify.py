from routelabs_router.models import ProviderResult, VerificationResult


UNCERTAINTY_SIGNALS = [
    "i don't know",
    "i am not sure",
    "i'm not sure",
    "cannot determine",
    "can't determine",
    "please provide",
    "need more context",
    "don't see any",
    "do not see any",
    "not enough information",
    "cannot access",
]


class HeuristicVerifier:
    def evaluate(self, task: str, complexity: str, result: ProviderResult) -> VerificationResult:
        content = result.content.strip()
        lowered = content.lower()

        signals: list[str] = []
        if not content:
            signals.append("empty_response")
        if any(signal in lowered for signal in UNCERTAINTY_SIGNALS):
            signals.append("weak_grounding")
        if complexity == "high" and len(content) < 120:
            signals.append("thin_response_for_high_complexity")
        if complexity == "medium" and len(content) < 40:
            signals.append("thin_response_for_medium_complexity")

        grounded = "weak_grounding" not in signals and "empty_response" not in signals
        confidence = _confidence_for(complexity, signals)
        should_escalate = complexity in {"medium", "high"} and bool(signals)
        passed = not should_escalate

        if passed:
            reason = "verification accepted the local response"
        else:
            reason = "verification found weak grounding or low-confidence signals"

        return VerificationResult(
            passed=passed,
            confidence=confidence,
            grounded=grounded,
            hallucination_signals=signals,
            reason=reason,
            should_escalate=should_escalate,
        )


def _confidence_for(complexity: str, signals: list[str]) -> float:
    base_confidence = {
        "low": 0.9,
        "medium": 0.76,
        "high": 0.64,
    }.get(complexity, 0.7)
    penalty = 0.18 * len(signals)
    return max(0.05, round(base_confidence - penalty, 2))
