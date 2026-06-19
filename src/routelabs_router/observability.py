from typing import Any

from routelabs_router.config import OpenTelemetryConfig
from routelabs_router.models import DecisionTrace


class OpenTelemetryRouteTracer:
    def __init__(self, config: OpenTelemetryConfig) -> None:
        self.config = config
        self.available = False
        self.status = "disabled"
        self._tracer: Any = None

        if not config.enabled:
            return

        try:
            from opentelemetry import trace
        except ImportError:
            self.status = "opentelemetry_api_not_installed"
            return

        self._tracer = trace.get_tracer("routelabs_router")
        self.available = True
        self.status = "enabled"

    def record(
        self,
        request_id: str,
        request_kind: str,
        task_preview: str,
        trace: DecisionTrace,
        is_private: bool,
        auto_private: bool,
    ) -> None:
        if self._tracer is None:
            return

        summary = trace.summary
        route = trace.final_route
        span_name = f"routelabs.route.{request_kind}"
        with self._tracer.start_as_current_span(span_name) as span:
            span.set_attribute("gen_ai.operation.name", request_kind)
            span.set_attribute("gen_ai.provider.name", route.provider)
            span.set_attribute("gen_ai.request.model", route.model)
            span.set_attribute("routelabs.request.id", request_id)
            span.set_attribute("routelabs.route.target", route.target)
            span.set_attribute(
                "routelabs.route.initial_target",
                trace.initial_route.target,
            )
            span.set_attribute("routelabs.route.escalated", trace.escalated)
            span.set_attribute("routelabs.route.private", is_private)
            span.set_attribute("routelabs.route.auto_private", auto_private)
            span.set_attribute("routelabs.route.attempt_count", len(trace.attempts))

            if trace.total_latency_ms is not None:
                span.set_attribute(
                    "routelabs.route.total_latency_ms",
                    trace.total_latency_ms,
                )
            if trace.completion_tokens_per_second is not None:
                span.set_attribute(
                    "routelabs.route.completion_tokens_per_second",
                    trace.completion_tokens_per_second,
                )
            if trace.escalation_reason:
                span.set_attribute(
                    "routelabs.route.escalation_reason",
                    trace.escalation_reason,
                )
            if trace.privacy is not None:
                span.set_attribute("routelabs.privacy.detected", trace.privacy.detected)
                span.set_attribute(
                    "routelabs.privacy.categories",
                    ",".join(trace.privacy.categories),
                )
            if trace.verification is not None:
                span.set_attribute(
                    "routelabs.verification.passed",
                    trace.verification.passed,
                )
                span.set_attribute(
                    "routelabs.verification.should_escalate",
                    trace.verification.should_escalate,
                )
                span.set_attribute(
                    "routelabs.verification.confidence",
                    trace.verification.confidence,
                )
            if trace.agent_tools is not None:
                span.set_attribute(
                    "routelabs.agent_tools.detected",
                    trace.agent_tools.detected,
                )
                span.set_attribute(
                    "routelabs.agent_tools.risk_level",
                    trace.agent_tools.risk_level,
                )
                span.set_attribute(
                    "routelabs.agent_tools.approval_required",
                    trace.agent_tools.approval_required,
                )
            if summary is not None:
                span.set_attribute("routelabs.summary.headline", summary.headline)
                span.set_attribute("routelabs.summary.verification", summary.verification)
                span.set_attribute(
                    "routelabs.summary.agent_tool_risk",
                    summary.agent_tool_risk,
                )
            if self.config.include_task_preview:
                span.set_attribute("routelabs.task.preview", task_preview)
