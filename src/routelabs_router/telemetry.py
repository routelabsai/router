from threading import Lock

from routelabs_router.config import TelemetryCostConfig
from routelabs_router.models import DecisionTrace, RouterStats, RouterStatsResponse


class InMemoryTelemetry:
    def __init__(self, costs: TelemetryCostConfig) -> None:
        self._stats = RouterStats()
        self._costs = costs
        self._lock = Lock()

    def record(self, trace: DecisionTrace, is_private: bool) -> None:
        with self._lock:
            self._stats.total_requests += 1
            self._stats.estimated_baseline_cloud_cost_usd = _round_cost(
                self._stats.estimated_baseline_cloud_cost_usd
                + self._costs.cloud_request_cost_usd
            )
            if trace.final_route.target == "local":
                self._stats.local_responses += 1
                self._stats.estimated_total_cost_usd = _round_cost(
                    self._stats.estimated_total_cost_usd
                    + self._costs.local_request_cost_usd
                )
                self._stats.estimated_cloud_requests_avoided += 1
            else:
                self._stats.cloud_responses += 1
                self._stats.estimated_total_cost_usd = _round_cost(
                    self._stats.estimated_total_cost_usd
                    + self._costs.cloud_request_cost_usd
                )

            if trace.escalated:
                self._stats.escalations += 1

            if trace.verification is not None:
                self._stats.verification_checks += 1
                if not trace.verification.passed:
                    self._stats.verification_failures += 1

            if is_private:
                self._stats.private_requests += 1

            self._stats.estimated_cost_saved_usd = _round_cost(
                self._stats.estimated_baseline_cloud_cost_usd
                - self._stats.estimated_total_cost_usd
            )

    def snapshot(self) -> RouterStatsResponse:
        with self._lock:
            stats = self._stats
            return RouterStatsResponse(
                total_requests=stats.total_requests,
                local_responses=stats.local_responses,
                cloud_responses=stats.cloud_responses,
                escalations=stats.escalations,
                verification_checks=stats.verification_checks,
                verification_failures=stats.verification_failures,
                private_requests=stats.private_requests,
                estimated_total_cost_usd=stats.estimated_total_cost_usd,
                estimated_baseline_cloud_cost_usd=stats.estimated_baseline_cloud_cost_usd,
                estimated_cost_saved_usd=stats.estimated_cost_saved_usd,
                estimated_cloud_requests_avoided=stats.estimated_cloud_requests_avoided,
                local_response_rate=stats.local_response_rate,
                cloud_response_rate=stats.cloud_response_rate,
                escalation_rate=stats.escalation_rate,
                verification_failure_rate=stats.verification_failure_rate,
            )


def _round_cost(value: float) -> float:
    return round(value, 6)
