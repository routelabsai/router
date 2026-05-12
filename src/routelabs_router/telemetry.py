from collections import deque
from threading import Lock

from routelabs_router.config import TelemetryCostConfig
from routelabs_router.models import (
    DecisionTrace,
    RouteLogEntry,
    RouteLogResponse,
    RouterStats,
    RouterStatsResponse,
)


class InMemoryTelemetry:
    def __init__(self, costs: TelemetryCostConfig, max_log_entries: int = 100) -> None:
        self._stats = RouterStats()
        self._costs = costs
        self._lock = Lock()
        self._logs: deque[RouteLogEntry] = deque(maxlen=max_log_entries)
        self._total_latency_ms = 0.0
        self._chat_latency_ms = 0.0
        self._embedding_latency_ms = 0.0
        self._local_latency_ms = 0.0
        self._cloud_latency_ms = 0.0
        self._completion_tokens_per_second_sum = 0.0
        self._completion_tokens_per_second_samples = 0

    def record(
        self,
        request_id: str,
        request_kind: str,
        task_preview: str,
        trace: DecisionTrace,
        is_private: bool,
        auto_private: bool,
        total_latency_ms: float | None = None,
        completion_tokens_per_second: float | None = None,
    ) -> None:
        with self._lock:
            self._stats.total_requests += 1
            if request_kind == "chat":
                self._stats.chat_requests += 1
            elif request_kind == "embeddings":
                self._stats.embedding_requests += 1
            baseline_cloud_cost = self._costs.cloud_request_cost_usd
            self._stats.estimated_baseline_cloud_cost_usd = _round_cost(
                self._stats.estimated_baseline_cloud_cost_usd + baseline_cloud_cost
            )
            if trace.final_route.target == "local":
                self._stats.local_responses += 1
                request_cost = self._costs.local_request_cost_usd
                self._stats.estimated_total_cost_usd = _round_cost(
                    self._stats.estimated_total_cost_usd + request_cost
                )
                self._stats.estimated_cloud_requests_avoided += 1
            else:
                self._stats.cloud_responses += 1
                request_cost = self._costs.cloud_request_cost_usd
                self._stats.estimated_total_cost_usd = _round_cost(
                    self._stats.estimated_total_cost_usd + request_cost
                )

            if trace.escalated:
                self._stats.escalations += 1

            if trace.verification is not None:
                self._stats.verification_checks += 1
                if not trace.verification.passed:
                    self._stats.verification_failures += 1

            if is_private:
                self._stats.private_requests += 1
            if auto_private:
                self._stats.auto_private_requests += 1

            if total_latency_ms is not None:
                self._total_latency_ms += total_latency_ms
                if request_kind == "chat":
                    self._chat_latency_ms += total_latency_ms
                elif request_kind == "embeddings":
                    self._embedding_latency_ms += total_latency_ms
                if trace.final_route.target == "local":
                    self._local_latency_ms += total_latency_ms
                else:
                    self._cloud_latency_ms += total_latency_ms

            if completion_tokens_per_second is not None:
                self._completion_tokens_per_second_sum += completion_tokens_per_second
                self._completion_tokens_per_second_samples += 1

            self._stats.estimated_cost_saved_usd = _round_cost(
                self._stats.estimated_baseline_cloud_cost_usd
                - self._stats.estimated_total_cost_usd
            )
            self._logs.appendleft(
                RouteLogEntry(
                    request_id=request_id,
                    request_kind=request_kind,
                    task_preview=task_preview,
                    private=is_private,
                    auto_private=auto_private,
                    estimated_request_cost_usd=_round_cost(request_cost),
                    estimated_baseline_cloud_cost_usd=_round_cost(baseline_cloud_cost),
                    estimated_cost_saved_usd=_round_cost(
                        baseline_cloud_cost - request_cost
                    ),
                    total_latency_ms=_round_millis(total_latency_ms),
                    completion_tokens_per_second=_round_rate(
                        completion_tokens_per_second
                    ),
                    trace=trace,
                )
            )

    def snapshot(self) -> RouterStatsResponse:
        with self._lock:
            stats = self._stats
            return RouterStatsResponse(
                total_requests=stats.total_requests,
                chat_requests=stats.chat_requests,
                embedding_requests=stats.embedding_requests,
                local_responses=stats.local_responses,
                cloud_responses=stats.cloud_responses,
                escalations=stats.escalations,
                verification_checks=stats.verification_checks,
                verification_failures=stats.verification_failures,
                private_requests=stats.private_requests,
                auto_private_requests=stats.auto_private_requests,
                estimated_total_cost_usd=stats.estimated_total_cost_usd,
                estimated_baseline_cloud_cost_usd=stats.estimated_baseline_cloud_cost_usd,
                estimated_cost_saved_usd=stats.estimated_cost_saved_usd,
                estimated_cloud_requests_avoided=stats.estimated_cloud_requests_avoided,
                local_response_rate=stats.local_response_rate,
                cloud_response_rate=stats.cloud_response_rate,
                escalation_rate=stats.escalation_rate,
                verification_failure_rate=stats.verification_failure_rate,
                avg_total_latency_ms=_average(self._total_latency_ms, stats.total_requests),
                avg_chat_latency_ms=_average(self._chat_latency_ms, stats.chat_requests),
                avg_embedding_latency_ms=_average(
                    self._embedding_latency_ms, stats.embedding_requests
                ),
                avg_local_latency_ms=_average(
                    self._local_latency_ms, stats.local_responses
                ),
                avg_cloud_latency_ms=_average(
                    self._cloud_latency_ms, stats.cloud_responses
                ),
                avg_completion_tokens_per_second=_average(
                    self._completion_tokens_per_second_sum,
                    self._completion_tokens_per_second_samples,
                ),
            )

    def recent_logs(self) -> RouteLogResponse:
        with self._lock:
            return RouteLogResponse(entries=list(self._logs))


def _round_cost(value: float) -> float:
    return round(value, 6)


def _round_millis(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def _round_rate(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def _average(total: float, count: int) -> float:
    if count == 0:
        return 0.0
    return round(total / count, 3)
