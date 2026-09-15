import json
import re
import time
import uuid

import httpx
from fastapi import HTTPException

from routelabs_router.adapters.base import (
    ChatProvider,
    EmbeddingProvider,
    ProviderExecutionError,
)
from routelabs_router.adapters.anthropic import AnthropicChatAdapter
from routelabs_router.adapters.ollama import OllamaChatAdapter
from routelabs_router.adapters.openai_compatible import OpenAICompatibleChatAdapter
from routelabs_router.config import Config
from routelabs_router.models import (
    AgentToolTrace,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicUsage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DecisionTrace,
    DecisionSummary,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthResponse,
    ModelCard,
    ModelsListResponse,
    ProviderHealth,
    ProviderEmbeddingResult,
    ProviderResult,
    ProviderAttempt,
    RedactionTrace,
    ResponsesFunctionCall,
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
    RouteDecision,
    RouteRequest,
)
from routelabs_router.observability import OpenTelemetryRouteTracer
from routelabs_router.policy import LocalPolicyEngine
from routelabs_router.privacy import HeuristicPrivacyDetector, redact_sensitive_text
from routelabs_router.router import RouterEngine
from routelabs_router.telemetry import InMemoryTelemetry
from routelabs_router.verify import HeuristicVerifier


class ChatService:
    def __init__(
        self,
        config: Config,
        router: RouterEngine | None = None,
        providers: dict[str, ChatProvider] | None = None,
        verifier: HeuristicVerifier | None = None,
        telemetry: InMemoryTelemetry | None = None,
        privacy_detector: HeuristicPrivacyDetector | None = None,
        policy_engine: LocalPolicyEngine | None = None,
        route_tracer: OpenTelemetryRouteTracer | None = None,
    ) -> None:
        self.config = config
        if policy_engine is None and privacy_detector is not None:
            policy_engine = LocalPolicyEngine(
                config,
                privacy_detector=privacy_detector,
            )
        self.policy_engine = (
            policy_engine
            or getattr(router, "policy_engine", None)
            or LocalPolicyEngine(config, privacy_detector=privacy_detector)
        )
        if router is not None:
            router.policy_engine = self.policy_engine
        self.router = router or RouterEngine(config, policy_engine=self.policy_engine)
        self.providers = providers if providers is not None else self._default_providers()
        self.verifier = verifier or HeuristicVerifier()
        self.telemetry = telemetry or InMemoryTelemetry(config.telemetry.costs)
        self.privacy_detector = self.policy_engine.privacy_detector
        self.route_tracer = route_tracer or OpenTelemetryRouteTracer(
            config.telemetry.opentelemetry
        )

    def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        started = time.perf_counter()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        task = _task_from_messages(request)
        preflight = self.policy_engine.evaluate(
            task,
            explicitly_private=request.private,
        )
        privacy = preflight.privacy
        effective_private = request.private or privacy.forced_local
        initial_route = self.router.decide(
            _route_request_from_chat(request, task, effective_private)
        )
        trace = DecisionTrace(
            privacy=privacy,
            agent_tools=initial_route.agent_tools,
            initial_route=initial_route,
            final_route=initial_route,
        )
        requested_model = _requested_model_override(request.model)
        result = self._execute_route_or_fallback(
            request=request,
            route=initial_route,
            trace=trace,
            requested_model=requested_model,
            allow_cross_target_fallback=(not effective_private and request.allow_fallbacks),
            max_cloud_cost_usd=request.max_cloud_cost_usd,
        )

        if (
            initial_route.verify
            and trace.final_route.target == "local"
            and not result.tool_calls
        ):
            verification = self.verifier.evaluate(
                task=task,
                complexity=initial_route.complexity,
                result=result,
            )
            trace.verification = verification

            if (
                verification.should_escalate
                and request.allow_fallbacks
                and not effective_private
            ):
                cloud_route = self._cloud_route(
                    initial_route.complexity,
                    agent_tools=trace.agent_tools,
                    agent_role=initial_route.agent_role,
                )
                trace.escalated = True
                trace.escalation_reason = verification.reason
                try:
                    result = self._execute_route(
                        request=request,
                        route=cloud_route,
                        trace=trace,
                        requested_model=requested_model,
                        max_cloud_cost_usd=request.max_cloud_cost_usd,
                    )
                    trace.final_route = cloud_route
                except HTTPException as exc:
                    trace.escalated = False
                    trace.escalation_reason = (
                        "verification requested escalation but the cloud route failed: "
                        f"{exc.detail}"
                    )
            elif verification.should_escalate:
                if effective_private:
                    trace.escalation_reason = (
                        "verification requested escalation but privacy policy forced "
                        "local execution"
                    )
                else:
                    trace.escalation_reason = (
                        "verification requested escalation but request disabled fallbacks"
                    )

        response = _build_chat_response(
            request_id=request_id,
            route=trace.final_route,
            result=result,
            trace=trace,
        )
        total_latency_ms = _elapsed_ms(started)
        completion_tokens_per_second = _completion_tokens_per_second(
            response.usage.completion_tokens, total_latency_ms
        )
        trace.total_latency_ms = total_latency_ms
        trace.completion_tokens_per_second = completion_tokens_per_second
        trace.summary = _build_decision_summary(trace)
        self.telemetry.record(
            request_id=request_id,
            request_kind="chat",
            task_preview=_task_preview(task),
            trace=trace,
            is_private=effective_private,
            auto_private=privacy.detected and not request.private,
            total_latency_ms=total_latency_ms,
            completion_tokens_per_second=completion_tokens_per_second,
        )
        self.route_tracer.record(
            request_id=request_id,
            request_kind="chat",
            task_preview=_task_preview(task),
            trace=trace,
            is_private=effective_private,
            auto_private=privacy.detected and not request.private,
        )
        return response

    def create_response(self, request: ResponsesRequest) -> ResponsesResponse:
        chat_request = _responses_request_to_chat_request(request)
        chat_response = self.create_chat_completion(chat_request)
        return _build_responses_response(chat_response)

    def create_anthropic_message(
        self, request: AnthropicMessagesRequest
    ) -> AnthropicMessagesResponse:
        chat_request = _anthropic_request_to_chat_request(request)
        chat_response = self.create_chat_completion(chat_request)
        return _build_anthropic_response(chat_response)

    def inspect_route(self, request: RouteRequest) -> RouteDecision:
        decision = self.router.decide(request)

        if decision.provider in self.providers:
            available, status = self._provider_readiness(decision.provider)
        else:
            available, status = False, "unreachable"

        cloud_provider = self.config.providers.cloud.default
        if cloud_provider in self.providers:
            fallback_available, fallback_status = self._provider_readiness(
                cloud_provider
            )
        else:
            fallback_available, fallback_status = False, "unreachable"
        if not request.allow_fallbacks:
            fallback_available = False
            fallback_status = "disabled_by_request"
        if fallback_available and not self._max_cloud_cost_allows_request(
            request.max_cloud_cost_usd
        ):
            fallback_available = False
            fallback_status = "max_cloud_cost_exceeded"
        if fallback_available and not self._cloud_budget_allows_request():
            fallback_available = False
            fallback_status = "budget_exhausted"
        if decision.target == "cloud":
            fallback_available = None
            fallback_status = None
        payload = decision.model_dump()
        payload.update(
            provider_available=available,
            provider_status=status,
            fallback_available=fallback_available,
            fallback_status=fallback_status,
            policy_engine=self.policy_engine.name,
            policy_engine_status=self.policy_engine.readiness()[1],
        )
        return RouteDecision(**payload)

    def create_embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        started = time.perf_counter()
        request_id = f"embd-{uuid.uuid4().hex[:24]}"
        task = _task_from_embedding_input(request.input)
        preflight = self.policy_engine.evaluate(
            task,
            explicitly_private=request.private,
        )
        privacy = preflight.privacy
        effective_private = request.private or privacy.forced_local
        route = self.router.decide(
            RouteRequest(
                task=task,
                private=effective_private,
                allow_fallbacks=request.allow_fallbacks,
                max_cloud_cost_usd=request.max_cloud_cost_usd,
                strip_pii=request.strip_pii,
            )
        )
        requested_model = _requested_model_override(request.model)
        trace = DecisionTrace(
            privacy=privacy,
            agent_tools=route.agent_tools,
            initial_route=route,
            final_route=route,
        )
        final_route, result = self._execute_embeddings_route_or_fallback(
            request=request,
            route=route,
            trace=trace,
            requested_model=requested_model,
            allow_cross_target_fallback=(not effective_private and request.allow_fallbacks),
            max_cloud_cost_usd=request.max_cloud_cost_usd,
        )
        trace.final_route = final_route
        total_latency_ms = _elapsed_ms(started)
        trace.total_latency_ms = total_latency_ms
        trace.summary = _build_decision_summary(trace)
        self.telemetry.record(
            request_id=request_id,
            request_kind="embeddings",
            task_preview=_task_preview(task),
            trace=trace,
            is_private=effective_private,
            auto_private=privacy.detected and not request.private,
            total_latency_ms=total_latency_ms,
        )
        self.route_tracer.record(
            request_id=request_id,
            request_kind="embeddings",
            task_preview=_task_preview(task),
            trace=trace,
            is_private=effective_private,
            auto_private=privacy.detected and not request.private,
        )
        return EmbeddingsResponse(
            data=result.data,
            model=result.model,
            usage=result.usage,
            route=RouteDecision(**{**final_route.model_dump(), "model": result.model, "verify": False}),
        )

    def list_models(self) -> ModelsListResponse:
        ollama_inventory = self._ollama_model_inventory()
        installed_lookup = {
            item["id"]: item for item in ollama_inventory
        }
        llamacpp_inventory = self._openai_compatible_model_inventory("llamacpp")
        llamacpp_lookup = {item["id"]: item for item in llamacpp_inventory}
        models = [
            ModelCard(
                id="route-auto",
                owned_by="routelabs",
                provider="routelabs",
                source="virtual",
                installed=True,
                status="ready",
            ),
            ModelCard(
                id=self.config.providers.local.ollama.model,
                owned_by="routelabs-local",
                provider="ollama",
                source="configured",
                installed=self.config.providers.local.ollama.model in installed_lookup,
                status=(
                    "installed"
                    if self.config.providers.local.ollama.model in installed_lookup
                    else "configured"
                ),
                size_bytes=installed_lookup.get(
                    self.config.providers.local.ollama.model, {}
                ).get("size_bytes"),
            ),
            ModelCard(
                id=self.config.providers.local.ollama.embedding_model
                or self.config.providers.local.ollama.model,
                owned_by="routelabs-local",
                provider="ollama",
                source="configured",
                installed=(
                    (
                        self.config.providers.local.ollama.embedding_model
                        or self.config.providers.local.ollama.model
                    )
                    in installed_lookup
                ),
                status=(
                    "installed"
                    if (
                        self.config.providers.local.ollama.embedding_model
                        or self.config.providers.local.ollama.model
                    )
                    in installed_lookup
                    else "configured"
                ),
                size_bytes=installed_lookup.get(
                    self.config.providers.local.ollama.embedding_model
                    or self.config.providers.local.ollama.model,
                    {},
                ).get("size_bytes"),
            ),
            ModelCard(
                id=self.config.providers.local.llamacpp.model,
                owned_by="routelabs-local",
                provider="llamacpp",
                source="configured",
                installed=self.config.providers.local.llamacpp.model in llamacpp_lookup,
                status=(
                    "ready"
                    if self.config.providers.local.llamacpp.model in llamacpp_lookup
                    else "configured"
                ),
            ),
            ModelCard(
                id=self.config.providers.local.llamacpp.embedding_model
                or self.config.providers.local.llamacpp.model,
                owned_by="routelabs-local",
                provider="llamacpp",
                source="configured",
                installed=(
                    (
                        self.config.providers.local.llamacpp.embedding_model
                        or self.config.providers.local.llamacpp.model
                    )
                    in llamacpp_lookup
                ),
                status=(
                    "ready"
                    if (
                        self.config.providers.local.llamacpp.embedding_model
                        or self.config.providers.local.llamacpp.model
                    )
                    in llamacpp_lookup
                    else "configured"
                ),
            ),
            ModelCard(
                id=self.config.providers.cloud.openai_compatible.model,
                owned_by="routelabs-cloud",
                provider="openai-compatible",
                source="configured",
                installed=self._openai_compatible_cloud_configured(),
                status=(
                    "configured"
                    if self._openai_compatible_cloud_configured()
                    else "not_configured"
                ),
            ),
            ModelCard(
                id=self.config.providers.cloud.openai_compatible.embedding_model
                or self.config.providers.cloud.openai_compatible.model,
                owned_by="routelabs-cloud",
                provider="openai-compatible",
                source="configured",
                installed=self._openai_compatible_cloud_configured(),
                status=(
                    "configured"
                    if self._openai_compatible_cloud_configured()
                    else "not_configured"
                ),
            ),
            ModelCard(
                id=self.config.providers.cloud.anthropic.model,
                owned_by="routelabs-cloud",
                provider="anthropic",
                source="configured",
                installed=self.config.providers.cloud.anthropic.api_key is not None,
                status=(
                    "configured"
                    if self.config.providers.cloud.anthropic.api_key
                    else "not_configured"
                ),
            ),
        ]
        models.extend(
            self._agent_role_model_cards(
                installed_lookup=installed_lookup,
                llamacpp_lookup=llamacpp_lookup,
            )
        )
        models.extend(
            ModelCard(
                id=item["id"],
                owned_by="routelabs-local",
                provider="ollama",
                source="installed",
                installed=True,
                status="installed",
                size_bytes=item.get("size_bytes"),
            )
            for item in ollama_inventory
        )
        models.extend(
            ModelCard(
                id=item["id"],
                owned_by="routelabs-local",
                provider="llamacpp",
                source="discovered",
                installed=True,
                status="ready",
            )
            for item in llamacpp_inventory
        )
        deduped: list[ModelCard] = []
        seen: dict[str, int] = {}
        for model in models:
            if model.id in seen:
                index = seen[model.id]
                existing = deduped[index]
                deduped[index] = existing.model_copy(
                    update={
                        "installed": model.installed
                        if model.installed is not None
                        else existing.installed,
                        "status": model.status or existing.status,
                        "size_bytes": model.size_bytes or existing.size_bytes,
                        "provider": existing.provider or model.provider,
                        "source": existing.source or model.source,
                    }
                )
                continue
            seen[model.id] = len(deduped)
            deduped.append(model)
        return ModelsListResponse(data=deduped)

    def _agent_role_model_cards(
        self,
        installed_lookup: dict[str, dict[str, object]],
        llamacpp_lookup: dict[str, dict[str, object]],
    ) -> list[ModelCard]:
        cards: list[ModelCard] = []
        for role_name, role in sorted(self.config.agents.roles.items()):
            target = role.target
            provider = role.provider or (
                self.config.providers.local.default
                if target == "local"
                else self.config.providers.cloud.default
            )
            installed, status, size_bytes = self._agent_role_model_status(
                provider=provider,
                model=role.model,
                installed_lookup=installed_lookup,
                llamacpp_lookup=llamacpp_lookup,
            )
            cards.append(
                ModelCard(
                    id=role.model,
                    owned_by=(
                        "routelabs-local"
                        if target == "local"
                        else "routelabs-cloud"
                    ),
                    provider=provider,
                    source=f"agent_role:{role_name}",
                    installed=installed,
                    status=status,
                    size_bytes=size_bytes,
                )
            )
        return cards

    def _agent_role_model_status(
        self,
        provider: str,
        model: str,
        installed_lookup: dict[str, dict[str, object]],
        llamacpp_lookup: dict[str, dict[str, object]],
    ) -> tuple[bool | None, str, int | None]:
        if provider == "ollama":
            installed = model in installed_lookup
            raw_size = installed_lookup.get(model, {}).get("size_bytes")
            size_bytes = raw_size if isinstance(raw_size, int) else None
            return (
                installed,
                "installed" if installed else "configured",
                size_bytes,
            )
        if provider == "llamacpp":
            installed = model in llamacpp_lookup
            return installed, "ready" if installed else "configured", None
        if provider == "openai-compatible":
            configured = self._openai_compatible_cloud_configured()
            return configured, "configured" if configured else "not_configured", None
        if provider == "anthropic":
            configured = self.config.providers.cloud.anthropic.api_key is not None
            return configured, "configured" if configured else "not_configured", None
        return None, "unknown", None

    def health(self) -> HealthResponse:
        policy_available, policy_status = self.policy_engine.readiness()

        local_default = self.config.providers.local.default
        cloud_default = self.config.providers.cloud.default

        providers = {
            "policy-engine": ProviderHealth(
                available=policy_available,
                status=policy_status,
            ),
        }

        if local_default in self.providers:
            providers[local_default] = self._provider_health(local_default)
        else:
            providers[local_default] = ProviderHealth(
                available=False,
                status="unavailable",
            )

        if cloud_default in self.providers:
            providers[cloud_default] = self._provider_health(cloud_default)
        elif (
            cloud_default == "openai-compatible"
            and self._openai_compatible_cloud_configured()
        ):
            providers[cloud_default] = ProviderHealth(
                available=True,
                status="configured",
            )
        else:
            providers[cloud_default] = ProviderHealth(
                available=False,
                status="unavailable",
            )

        local_available = providers[local_default].available
        cloud_available = providers[cloud_default].available

        if local_available:
            status = "ok"
        elif cloud_available:
            status = "degraded"
        elif policy_available:
            status = "degraded"
        else:
            status = "error"

        return HealthResponse(
            status=status,
            providers=providers,
            routing_available=policy_available,
            policy_engine=ProviderHealth(
                available=policy_available,
                status=policy_status,
            ),
        )

    def _default_providers(self) -> dict[str, ChatProvider]:
        providers: dict[str, ChatProvider] = {
            "ollama": OllamaChatAdapter(self.config.providers.local.ollama),
            "llamacpp": OpenAICompatibleChatAdapter(
                self.config.providers.local.llamacpp
            ),
        }
        cloud_config = self.config.providers.cloud.openai_compatible
        if self._provider_auth_configured(cloud_config):
            providers["openai-compatible"] = OpenAICompatibleChatAdapter(cloud_config)
        anthropic_config = self.config.providers.cloud.anthropic
        if anthropic_config.api_key:
            providers["anthropic"] = AnthropicChatAdapter(anthropic_config)
        return providers

    def _openai_compatible_cloud_configured(self) -> bool:
        return self._provider_auth_configured(
            self.config.providers.cloud.openai_compatible
        )

    @staticmethod
    def _provider_auth_configured(provider_config) -> bool:
        return bool(provider_config.api_key) or not provider_config.requires_api_key

    def _cloud_route(
        self,
        complexity: str,
        agent_tools: AgentToolTrace | None = None,
        agent_role: str | None = None,
    ) -> RouteDecision:
        return RouteDecision(
            target="cloud",
            provider=self.config.providers.cloud.default,
            model=self._provider_model_for_name(self.config.providers.cloud.default),
            reason="verification requested escalation to a stronger remote model",
            complexity=complexity,
            verify=False,
            agent_role=agent_role,
            agent_tools=agent_tools,
        )

    def get_stats(self):
        return self.telemetry.snapshot()

    def get_recent_logs(self):
        return self.telemetry.recent_logs()

    def _cloud_budget_allows_request(self) -> bool:
        budget = self.config.telemetry.cloud_budget_usd
        if budget is None:
            return True
        projected_cloud_cost = (
            self.telemetry.snapshot().estimated_cloud_cost_usd
            + self.config.telemetry.costs.cloud_request_cost_usd
        )
        return projected_cloud_cost <= budget

    def _max_cloud_cost_allows_request(self, max_cloud_cost_usd: float | None) -> bool:
        if max_cloud_cost_usd is None:
            return True
        return self.config.telemetry.costs.cloud_request_cost_usd <= max_cloud_cost_usd

    def _max_cloud_cost_error(self, max_cloud_cost_usd: float) -> HTTPException:
        next_cost = self.config.telemetry.costs.cloud_request_cost_usd
        return HTTPException(
            status_code=402,
            detail=(
                "cloud request exceeds max_cloud_cost_usd: "
                f"next request costs ${next_cost:.6f}, "
                f"max is ${max_cloud_cost_usd:.6f}"
            ),
        )

    def _cloud_budget_error(self) -> HTTPException:
        budget = self.config.telemetry.cloud_budget_usd
        spent = self.telemetry.snapshot().estimated_cloud_cost_usd
        next_cost = self.config.telemetry.costs.cloud_request_cost_usd
        return HTTPException(
            status_code=402,
            detail=(
                "cloud budget exhausted: "
                f"spent ${spent:.6f}, next request costs ${next_cost:.6f}, "
                f"budget is ${budget:.6f}"
            ),
        )

    def _execute_embeddings_route_or_fallback(
        self,
        request: EmbeddingsRequest,
        route: RouteDecision,
        trace: DecisionTrace,
        requested_model: str | None,
        allow_cross_target_fallback: bool,
        max_cloud_cost_usd: float | None,
    ) -> tuple[RouteDecision, ProviderEmbeddingResult]:
        try:
            return route, self._execute_embeddings_route(
                request=request,
                route=route,
                trace=trace,
                requested_model=requested_model,
                max_cloud_cost_usd=max_cloud_cost_usd,
            )
        except HTTPException as exc:
            if not allow_cross_target_fallback or route.target != "local":
                raise exc
            fallback_route = self._cloud_route(
                route.complexity,
                agent_tools=trace.agent_tools,
                agent_role=route.agent_role,
            )
            trace.escalated = True
            trace.escalation_reason = (
                "local embeddings provider was unavailable, falling back to cloud"
            )
            trace.final_route = fallback_route
            return fallback_route, self._execute_embeddings_route(
                request=request,
                route=fallback_route,
                trace=trace,
                requested_model=requested_model,
                max_cloud_cost_usd=max_cloud_cost_usd,
            )

    def _execute_embeddings_route(
        self,
        request: EmbeddingsRequest,
        route: RouteDecision,
        trace: DecisionTrace,
        requested_model: str | None,
        max_cloud_cost_usd: float | None = None,
    ) -> ProviderEmbeddingResult:
        if route.target == "cloud" and not self._max_cloud_cost_allows_request(
            max_cloud_cost_usd
        ):
            raise self._max_cloud_cost_error(max_cloud_cost_usd or 0)
        if route.target == "cloud" and not self._cloud_budget_allows_request():
            raise self._cloud_budget_error()

        provider = self.providers.get(route.provider)
        if provider is None:
            raise HTTPException(
                status_code=501,
                detail=f"provider '{route.provider}' is not configured for embeddings",
            )
        if not isinstance(provider, EmbeddingProvider):
            raise HTTPException(
                status_code=501,
                detail=f"provider '{route.provider}' does not support embeddings",
            )

        model_name = requested_model or self._provider_embedding_model(route.provider)
        retries = max(1, self._provider_retries(route.provider))
        last_error: str | None = None
        provider_request = _embedding_request_for_route_with_redaction(
            request=request,
            route=route,
            trace=trace,
            strip_pii=self._should_strip_pii_for_cloud(request.strip_pii),
        )
        for attempt_number in range(1, retries + 1):
            attempt_started = time.perf_counter()
            try:
                result = provider.embed(provider_request, model=model_name)
                trace.attempts.append(
                    ProviderAttempt(
                        provider=route.provider,
                        model=result.model,
                        target=route.target,
                        outcome="success",
                        reason=f"attempt {attempt_number}",
                        duration_ms=_elapsed_ms(attempt_started),
                    )
                )
                return result
            except ProviderExecutionError as exc:
                last_error = exc.reason
                trace.attempts.append(
                    ProviderAttempt(
                        provider=route.provider,
                        model=model_name,
                        target=route.target,
                        outcome="failure",
                        reason=f"attempt {attempt_number}: {exc.reason}",
                        duration_ms=_elapsed_ms(attempt_started),
                    )
                )

        status_code = 502 if route.target == "cloud" else 503
        raise HTTPException(
            status_code=status_code,
            detail=f"{route.provider} embeddings failed after {retries} attempt(s): {last_error or 'unknown error'}",
        )

    def _execute_route_or_fallback(
        self,
        request: ChatCompletionRequest,
        route: RouteDecision,
        trace: DecisionTrace,
        requested_model: str | None,
        allow_cross_target_fallback: bool,
        max_cloud_cost_usd: float | None,
    ) -> ProviderResult:
        try:
            return self._execute_route(
                request=request,
                route=route,
                trace=trace,
                requested_model=requested_model,
                max_cloud_cost_usd=max_cloud_cost_usd,
            )
        except HTTPException as exc:
            if not allow_cross_target_fallback or route.target != "local":
                raise exc
            fallback_route = self._cloud_route(
                route.complexity,
                agent_tools=trace.agent_tools,
                agent_role=route.agent_role,
            )
            trace.escalated = True
            trace.escalation_reason = "local provider was unavailable, falling back to cloud"
            trace.final_route = fallback_route
            return self._execute_route(
                request=request,
                route=fallback_route,
                trace=trace,
                requested_model=requested_model,
                max_cloud_cost_usd=max_cloud_cost_usd,
            )

    def _execute_route(
        self,
        request: ChatCompletionRequest,
        route: RouteDecision,
        trace: DecisionTrace,
        requested_model: str | None,
        max_cloud_cost_usd: float | None = None,
    ) -> ProviderResult:
        if route.target == "cloud" and not self._max_cloud_cost_allows_request(
            max_cloud_cost_usd
        ):
            raise self._max_cloud_cost_error(max_cloud_cost_usd or 0)
        if route.target == "cloud" and not self._cloud_budget_allows_request():
            raise self._cloud_budget_error()

        provider = self.providers.get(route.provider)
        if provider is None:
            raise HTTPException(
                status_code=501,
                detail=f"provider '{route.provider}' is not configured",
            )

        model_name = requested_model or route.model
        retries = max(1, self._provider_retries(route.provider))
        last_error: str | None = None
        provider_request = _chat_request_for_route_with_redaction(
            request=request,
            route=route,
            trace=trace,
            strip_pii=self._should_strip_pii_for_cloud(request.strip_pii),
        )
        for attempt_number in range(1, retries + 1):
            attempt_started = time.perf_counter()
            try:
                result = provider.complete(provider_request, model=model_name)
                _validate_structured_output(
                    result=result,
                    response_format=request.response_format,
                )
                trace.attempts.append(
                    ProviderAttempt(
                        provider=route.provider,
                        model=result.model,
                        target=route.target,
                        outcome="success",
                        reason=f"attempt {attempt_number}",
                        duration_ms=_elapsed_ms(attempt_started),
                    )
                )
                return result
            except ProviderExecutionError as exc:
                last_error = exc.reason
                trace.attempts.append(
                    ProviderAttempt(
                        provider=route.provider,
                        model=model_name,
                        target=route.target,
                        outcome="failure",
                        reason=f"attempt {attempt_number}: {exc.reason}",
                        duration_ms=_elapsed_ms(attempt_started),
                    )
                )

        status_code = 502 if route.target == "cloud" else 503
        raise HTTPException(
            status_code=status_code,
            detail=f"{route.provider} failed after {retries} attempt(s): {last_error or 'unknown error'}",
        )

    def _provider_retries(self, provider_name: str) -> int:
        if provider_name == "ollama":
            return self.config.providers.local.ollama.max_retries
        if provider_name == "llamacpp":
            return self.config.providers.local.llamacpp.max_retries
        if provider_name == "openai-compatible":
            return self.config.providers.cloud.openai_compatible.max_retries
        if provider_name == "anthropic":
            return self.config.providers.cloud.anthropic.max_retries
        return 1

    def _should_strip_pii_for_cloud(self, request_override: bool | None) -> bool:
        if request_override is not None:
            return request_override
        return self.config.policies.privacy.scrub_before_cloud

    def _provider_embedding_model(self, provider_name: str) -> str:
        if provider_name == "ollama":
            return (
                self.config.providers.local.ollama.embedding_model
                or self.config.providers.local.ollama.model
            )
        if provider_name == "llamacpp":
            return (
                self.config.providers.local.llamacpp.embedding_model
                or self.config.providers.local.llamacpp.model
            )
        if provider_name == "openai-compatible":
            return (
                self.config.providers.cloud.openai_compatible.embedding_model
                or self.config.providers.cloud.openai_compatible.model
            )
        if provider_name == "anthropic":
            return self.config.providers.cloud.anthropic.model
        return "unknown"

    def _provider_model_for_name(self, provider_name: str) -> str:
        if provider_name == "ollama":
            return self.config.providers.local.ollama.model
        if provider_name == "llamacpp":
            return self.config.providers.local.llamacpp.model
        if provider_name == "openai-compatible":
            return self.config.providers.cloud.openai_compatible.model
        if provider_name == "anthropic":
            return self.config.providers.cloud.anthropic.model
        return "unknown"

    def _provider_health(self, provider_name: str) -> ProviderHealth:
        available, status = self._provider_readiness(provider_name)
        return ProviderHealth(available=available, status=status)

    def _ollama_model_inventory(self) -> list[dict[str, object]]:
        provider = self.providers.get("ollama")
        if provider is not None and not isinstance(provider, OllamaChatAdapter):
            return []
        base_url = self.config.providers.local.ollama.base_url.rstrip("/")
        try:
            with httpx.Client(timeout=1.0) as client:
                response = client.get(f"{base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError):
            return []

        models = data.get("models", [])
        inventory: list[dict[str, object]] = []
        for item in models:
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            inventory.append(
                {
                    "id": name,
                    "size_bytes": int(item.get("size", 0) or 0) or None,
                }
            )
        return inventory

    def _openai_compatible_model_inventory(
        self,
        provider_name: str,
    ) -> list[dict[str, object]]:
        if provider_name == "llamacpp":
            provider = self.providers.get(provider_name)
            if provider is not None and not isinstance(
                provider, OpenAICompatibleChatAdapter
            ):
                return []
            base_url = self.config.providers.local.llamacpp.base_url.rstrip("/")
        elif provider_name == "openai-compatible":
            provider = self.providers.get(provider_name)
            if provider is not None and not isinstance(
                provider, OpenAICompatibleChatAdapter
            ):
                return []
            if not self._openai_compatible_cloud_configured():
                return []
            base_url = self.config.providers.cloud.openai_compatible.base_url.rstrip("/")
        else:
            return []

        try:
            with httpx.Client(timeout=1.0) as client:
                response = client.get(f"{base_url}/models")
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError):
            return []

        inventory: list[dict[str, object]] = []
        for item in data.get("data", []):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                inventory.append({"id": model_id})
        return inventory

    def _provider_readiness(self, provider_name: str) -> tuple[bool, str]:
        if provider_name == "ollama":
            provider = self.providers.get(provider_name)
            if provider is not None and not isinstance(provider, OllamaChatAdapter):
                return True, "ready"
            base_url = self.config.providers.local.ollama.base_url.rstrip("/")
            try:
                with httpx.Client(timeout=1.0) as client:
                    response = client.get(f"{base_url}/api/tags")
                    response.raise_for_status()
                return True, "ready"
            except httpx.HTTPError:
                return False, "unreachable"

        if provider_name == "llamacpp":
            provider = self.providers.get(provider_name)
            if provider is not None and not isinstance(
                provider, OpenAICompatibleChatAdapter
            ):
                return True, "ready"
            base_url = self.config.providers.local.llamacpp.base_url.rstrip("/")
            try:
                with httpx.Client(timeout=1.0) as client:
                    response = client.get(f"{base_url}/models")
                    response.raise_for_status()
                return True, "ready"
            except httpx.HTTPError:
                return False, "unreachable"

        if provider_name == "openai-compatible":
            provider = self.providers.get(provider_name)
            if provider is not None and not isinstance(
                provider, OpenAICompatibleChatAdapter
            ):
                return True, "configured"
            if not self._openai_compatible_cloud_configured():
                return False, "not_configured"
            return True, "configured"

        if provider_name == "anthropic":
            provider = self.providers.get(provider_name)
            if provider is not None and not isinstance(provider, AnthropicChatAdapter):
                return True, "configured"
            if not self.config.providers.cloud.anthropic.api_key:
                return False, "not_configured"
            return True, "configured"

        return False, "unknown"


def _task_from_messages(request: ChatCompletionRequest) -> str:
    user_messages = [
        message.content for message in request.messages if message.role == "user"
    ]
    if user_messages:
        return user_messages[-1]
    return request.messages[-1].content


def _route_request_from_chat(
    request: ChatCompletionRequest,
    task: str,
    private: bool,
) -> RouteRequest:
    tool_names = _tool_names_from_tools(request.tools)
    return RouteRequest(
        task=task,
        private=private,
        allow_fallbacks=request.allow_fallbacks,
        max_cloud_cost_usd=request.max_cloud_cost_usd,
        strip_pii=request.strip_pii,
        tool_names=tool_names,
        tool_descriptions=_tool_descriptions_from_tools(request.tools),
        tool_count=len(request.tools or []),
        tool_choice=request.tool_choice,
        agent_role=request.agent_role,
    )


def _chat_request_for_route_with_redaction(
    request: ChatCompletionRequest,
    route: RouteDecision,
    trace: DecisionTrace,
    strip_pii: bool,
) -> ChatCompletionRequest:
    if route.target != "cloud" or not strip_pii:
        return request

    replacement_count = 0
    categories: list[str] = []
    messages: list[ChatMessage] = []
    for message in request.messages:
        content = message.content
        if content is not None:
            redacted = redact_sensitive_text(content)
            content = redacted.text
            replacement_count += redacted.replacement_count
            categories = _merge_categories(categories, redacted.categories)
        tool_calls = _redact_json_strings(message.tool_calls, categories)
        categories = _merge_categories(categories, tool_calls[1])
        replacement_count += tool_calls[2]
        messages.append(
            message.model_copy(
                update={
                    "content": content,
                    "tool_calls": tool_calls[0],
                }
            )
        )

    tools, tool_categories, tool_replacements = _redact_json_strings(
        request.tools,
        categories,
    )
    categories = _merge_categories(categories, tool_categories)
    replacement_count += tool_replacements

    _record_redaction_trace(
        trace=trace,
        categories=categories,
        replacement_count=replacement_count,
    )
    if replacement_count == 0:
        return request

    return request.model_copy(update={"messages": messages, "tools": tools})


def _embedding_request_for_route_with_redaction(
    request: EmbeddingsRequest,
    route: RouteDecision,
    trace: DecisionTrace,
    strip_pii: bool,
) -> EmbeddingsRequest:
    if route.target != "cloud" or not strip_pii:
        return request

    inputs = request.input if isinstance(request.input, list) else [request.input]
    redacted_inputs: list[str] = []
    categories: list[str] = []
    replacement_count = 0
    for item in inputs:
        redacted = redact_sensitive_text(item)
        redacted_inputs.append(redacted.text)
        categories = _merge_categories(categories, redacted.categories)
        replacement_count += redacted.replacement_count

    _record_redaction_trace(
        trace=trace,
        categories=categories,
        replacement_count=replacement_count,
    )
    if replacement_count == 0:
        return request

    redacted_value = (
        redacted_inputs if isinstance(request.input, list) else redacted_inputs[0]
    )
    return request.model_copy(update={"input": redacted_value})


def _redact_json_strings(
    value: object,
    existing_categories: list[str],
) -> tuple[object, list[str], int]:
    categories = list(existing_categories)
    replacement_count = 0

    if isinstance(value, str):
        redacted = redact_sensitive_text(value)
        categories = _merge_categories(categories, redacted.categories)
        return redacted.text, categories, redacted.replacement_count

    if isinstance(value, list):
        redacted_items: list[object] = []
        for item in value:
            redacted_item, categories, count = _redact_json_strings(item, categories)
            redacted_items.append(redacted_item)
            replacement_count += count
        return redacted_items, categories, replacement_count

    if isinstance(value, dict):
        redacted_dict: dict[object, object] = {}
        for key, item in value.items():
            redacted_item, categories, count = _redact_json_strings(item, categories)
            redacted_dict[key] = redacted_item
            replacement_count += count
        return redacted_dict, categories, replacement_count

    return value, categories, 0


def _merge_categories(existing: list[str], new_categories: list[str]) -> list[str]:
    merged = list(existing)
    for category in new_categories:
        if category not in merged:
            merged.append(category)
    return merged


def _record_redaction_trace(
    trace: DecisionTrace,
    categories: list[str],
    replacement_count: int,
) -> None:
    if replacement_count <= 0:
        return
    trace.redaction = RedactionTrace(
        applied=True,
        categories=categories,
        replacement_count=replacement_count,
        reason="scrubbed sensitive text before cloud execution",
    )


def _tool_names_from_tools(tools: list[dict[str, object]] | None) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        name = tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            function_name = function.get("name")
            if isinstance(function_name, str) and function_name:
                names.append(function_name)
    return names


def _tool_descriptions_from_tools(
    tools: list[dict[str, object]] | None,
) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for tool in tools or []:
        name = tool.get("name")
        description = tool.get("description")
        if isinstance(name, str) and isinstance(description, str) and description:
            descriptions[name] = description
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            function_name = function.get("name")
            function_description = function.get("description")
            if (
                isinstance(function_name, str)
                and isinstance(function_description, str)
                and function_description
            ):
                descriptions[function_name] = function_description
    return descriptions


def _build_chat_response(
    request_id: str,
    route: RouteDecision,
    result: ProviderResult,
    trace: DecisionTrace,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=result.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=result.content,
                    tool_calls=result.tool_calls,
                ),
                finish_reason=result.finish_reason,
            )
        ],
        usage=result.usage,
        route=route,
        trace=trace,
    )


def _build_decision_summary(trace: DecisionTrace) -> DecisionSummary:
    route = trace.final_route
    reason = trace.escalation_reason or route.reason
    privacy = "not_detected"
    if trace.privacy is not None and trace.privacy.detected:
        privacy = "detected:" + ",".join(trace.privacy.categories)

    verification = "not_run"
    if trace.verification is not None:
        verification = "passed" if trace.verification.passed else "failed"
        if trace.verification.should_escalate:
            verification = (
                "failed_escalated"
                if trace.escalated
                else "failed_escalation_unavailable"
            )

    agent_tool_risk = "none"
    if trace.agent_tools is not None and trace.agent_tools.detected:
        agent_tool_risk = trace.agent_tools.risk_level

    headline = _decision_headline(
        initial_target=trace.initial_route.target,
        final_target=route.target,
        provider=route.provider,
        model=route.model,
        escalated=trace.escalated,
        reason=reason,
    )
    return DecisionSummary(
        headline=headline,
        target=route.target,
        provider=route.provider,
        model=route.model,
        reason=reason,
        agent_role=route.agent_role,
        privacy=privacy,
        verification=verification,
        agent_tool_risk=agent_tool_risk,
        attempts=len(trace.attempts),
    )


def _decision_headline(
    initial_target: str,
    final_target: str,
    provider: str,
    model: str,
    escalated: bool,
    reason: str,
) -> str:
    if escalated and initial_target == "local" and final_target == "cloud":
        if "unavailable" in reason or "falling back" in reason:
            return f"Fell back to cloud on {provider}/{model}"
        return f"Escalated to cloud on {provider}/{model}"
    if final_target == "local":
        return f"Stayed local on {provider}/{model}"
    return f"Used cloud on {provider}/{model}"


def _build_responses_response(
    chat_response: ChatCompletionResponse,
) -> ResponsesResponse:
    choice = chat_response.choices[0]
    message_id = f"msg_{chat_response.id}"
    output: list[ResponsesOutputMessage | ResponsesFunctionCall] = [
        ResponsesOutputMessage(
            id=message_id,
            content=(
                [ResponsesOutputText(text=choice.message.content or "")]
                if choice.message.content
                else []
            ),
        )
    ]
    for tool_call in choice.message.tool_calls or []:
        function = tool_call.get("function", {})
        output.append(
            ResponsesFunctionCall(
                id=tool_call.get("id", f"fc_{chat_response.id}"),
                call_id=tool_call.get("id", f"fc_{chat_response.id}"),
                name=function.get("name", "unknown"),
                arguments=function.get("arguments", "{}"),
            )
        )

    return ResponsesResponse(
        id=f"resp_{chat_response.id}",
        created_at=chat_response.created,
        model=chat_response.model,
        output=output,
        output_text=choice.message.content or "",
        usage=ResponsesUsage(
            input_tokens=chat_response.usage.prompt_tokens,
            output_tokens=chat_response.usage.completion_tokens,
            total_tokens=chat_response.usage.total_tokens,
        ),
        route=chat_response.route,
        trace=chat_response.trace,
    )


def _build_anthropic_response(
    chat_response: ChatCompletionResponse,
) -> AnthropicMessagesResponse:
    choice = chat_response.choices[0]
    content: list[dict[str, object]] = []
    if choice.message.content:
        content.append({"type": "text", "text": choice.message.content})
    for tool_call in choice.message.tool_calls or []:
        function = tool_call.get("function", {})
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id", f"toolu_{chat_response.id}"),
                "name": function.get("name", "unknown"),
                "input": _parse_json_string(function.get("arguments", "{}")),
            }
        )
    return AnthropicMessagesResponse(
        id=f"msg_{chat_response.id}",
        content=content,
        model=chat_response.model,
        stop_reason=(
            "tool_use" if choice.finish_reason == "tool_calls" else choice.finish_reason
        ),
        stop_sequence=None,
        usage=AnthropicUsage(
            input_tokens=chat_response.usage.prompt_tokens,
            output_tokens=chat_response.usage.completion_tokens,
        ),
        route=chat_response.route,
        trace=chat_response.trace,
    )


def _task_preview(task: str, limit: int = 120) -> str:
    compact = " ".join(task.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _task_from_embedding_input(value: str | list[str]) -> str:
    if isinstance(value, str):
        return value
    return " ".join(value)


def _requested_model_override(model: str | None) -> str | None:
    if model is None:
        return None
    if model in {"route-auto", "auto"}:
        return None
    return model


def _responses_request_to_chat_request(
    request: ResponsesRequest,
) -> ChatCompletionRequest:
    messages = _messages_from_responses_input(request.input)
    if request.instructions:
        messages.insert(0, ChatMessage(role="system", content=request.instructions))
    response_format = request.text.format if request.text else None
    return ChatCompletionRequest(
        messages=messages,
        model=request.model,
        agent_role=request.agent_role,
        private=request.private,
        allow_fallbacks=request.allow_fallbacks,
        max_cloud_cost_usd=request.max_cloud_cost_usd,
        strip_pii=request.strip_pii,
        stream=request.stream,
        response_format=response_format,
        tools=request.tools,
        tool_choice=request.tool_choice,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stop=request.stop,
        seed=request.seed,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
    )


def _anthropic_request_to_chat_request(
    request: AnthropicMessagesRequest,
) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []
    if request.system is not None:
        system_text = _extract_responses_content(request.system)
        if system_text:
            messages.append(ChatMessage(role="system", content=system_text))
    for message in request.messages:
        messages.extend(_chat_messages_from_anthropic_message(message))
    return ChatCompletionRequest(
        messages=messages,
        model=request.model,
        agent_role=request.agent_role,
        private=request.private,
        allow_fallbacks=request.allow_fallbacks,
        max_cloud_cost_usd=request.max_cloud_cost_usd,
        strip_pii=request.strip_pii,
        stream=request.stream,
        tools=_map_anthropic_tools_to_openai(request.tools),
        tool_choice=_map_anthropic_tool_choice_to_openai(request.tool_choice),
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop_sequences,
    )


def _messages_from_responses_input(value: str | list[object]) -> list[ChatMessage]:
    if isinstance(value, str):
        return [ChatMessage(role="user", content=value)]

    messages: list[ChatMessage] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_text":
            messages.append(
                ChatMessage(
                    role="user",
                    content=_extract_responses_content([item]),
                )
            )
            continue
        if "role" in item:
            messages.append(
                ChatMessage(
                    role=str(item.get("role", "user")),
                    content=_extract_responses_content(item.get("content")),
                )
            )
            continue
        if item.get("type") == "message":
            messages.append(
                ChatMessage(
                    role=str(item.get("role", "user")),
                    content=_extract_responses_content(item.get("content")),
                )
            )

    if not messages:
        raise HTTPException(
            status_code=422,
            detail="responses input must include at least one user or message item",
        )
    return messages


def _extract_responses_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part)
    return str(value)


def _chat_messages_from_anthropic_message(
    message: object,
) -> list[ChatMessage]:
    if not hasattr(message, "role") or not hasattr(message, "content"):
        return []
    role = str(getattr(message, "role"))
    content = getattr(message, "content")
    if isinstance(content, str):
        return [ChatMessage(role=role, content=content)]
    if not isinstance(content, list):
        return [ChatMessage(role=role, content=str(content))]

    text_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []
    messages: list[ChatMessage] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", "tool_use"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
        elif block_type == "tool_result":
            tool_content = block.get("content")
            messages.append(
                ChatMessage(
                    role="tool",
                    content=_extract_responses_content(tool_content),
                    tool_call_id=str(block.get("tool_use_id", "")) or None,
                )
            )

    if text_parts or tool_calls:
        messages.insert(
            0,
            ChatMessage(
                role=role,
                content="\n".join(text_parts) if text_parts else "",
                tool_calls=tool_calls or None,
            ),
        )
    return messages


def _map_anthropic_tools_to_openai(
    tools: list[dict[str, object]] | None,
) -> list[dict[str, object]] | None:
    if tools is None:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        }
        for tool in tools
    ]


def _map_anthropic_tool_choice_to_openai(
    tool_choice: object,
) -> str | dict[str, object] | None:
    if tool_choice is None:
        return None
    if hasattr(tool_choice, "type"):
        choice_type = getattr(tool_choice, "type")
        choice_name = getattr(tool_choice, "name", None)
    elif isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        choice_name = tool_choice.get("name")
    else:
        return None
    if choice_type in {"auto", "any"}:
        return str(choice_type)
    if choice_type == "tool" and isinstance(choice_name, str) and choice_name:
        return {"type": "function", "function": {"name": choice_name}}
    return None


def _parse_json_string(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _completion_tokens_per_second(
    completion_tokens: int, total_latency_ms: float
) -> float | None:
    if completion_tokens <= 0 or total_latency_ms <= 0:
        return None
    return round(completion_tokens / (total_latency_ms / 1000), 3)


def _validate_structured_output(
    result: ProviderResult,
    response_format: dict[str, object] | None,
) -> None:
    if response_format is None or result.tool_calls:
        return

    response_type = response_format.get("type")
    if response_type not in {"json_object", "json_schema"}:
        return

    try:
        parsed = json.loads(result.content or "")
    except json.JSONDecodeError as exc:
        raise ProviderExecutionError(
            "structured-output",
            f"invalid structured output: provider returned non-JSON content ({exc.msg})",
        ) from exc

    if response_type == "json_object":
        if not isinstance(parsed, dict):
            raise ProviderExecutionError(
                "structured-output",
                "invalid structured output: expected a JSON object",
            )
        return

    json_schema = response_format.get("json_schema", {})
    schema = (
        json_schema.get("schema")
        if isinstance(json_schema, dict)
        else None
    )
    if not isinstance(schema, dict):
        return
    error = _validate_json_schema_value(parsed, schema, path="$")
    if error is not None:
        raise ProviderExecutionError(
            "structured-output",
            f"invalid structured output: {error}",
        )


def _validate_json_schema_value(
    value: object,
    schema: dict[str, object],
    path: str,
) -> str | None:
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        type_errors = [
            _validate_json_schema_value(value, {**schema, "type": item}, path)
            for item in expected_type
        ]
        if all(error is not None for error in type_errors):
            return type_errors[0]
        return None

    if expected_type == "object":
        if not isinstance(value, dict):
            return f"{path} expected object"
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    return f"{path}.{key} is required"
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key not in value or not isinstance(child_schema, dict):
                    continue
                error = _validate_json_schema_value(
                    value[key],
                    child_schema,
                    f"{path}.{key}",
                )
                if error is not None:
                    return error
            if schema.get("additionalProperties") is False:
                allowed = {key for key in properties.keys() if isinstance(key, str)}
                for key in value.keys():
                    if key not in allowed:
                        return f"{path}.{key} is not allowed"
        return None

    if expected_type == "array":
        if not isinstance(value, list):
            return f"{path} expected array"
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return f"{path} expected at least {min_items} item(s)"
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            return f"{path} expected at most {max_items} item(s)"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _validate_json_schema_value(
                    item,
                    item_schema,
                    f"{path}[{index}]",
                )
                if error is not None:
                    return error
        return None

    if expected_type == "string":
        if not isinstance(value, str):
            return f"{path} expected string"
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            return f"{path} expected length >= {min_length}"
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            return f"{path} expected length <= {max_length}"
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    return f"{path} does not match pattern {pattern!r}"
            except re.error:
                return f"{path} uses invalid regex pattern {pattern!r}"
    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{path} expected integer"
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{path} expected number"
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            return f"{path} expected boolean"
    elif expected_type == "null":
        if value is not None:
            return f"{path} expected null"

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"{path} must be one of {enum}"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            return f"{path} expected value >= {minimum}"
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            return f"{path} expected value <= {maximum}"
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            return f"{path} expected value > {exclusive_minimum}"
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
            return f"{path} expected value < {exclusive_maximum}"

    return None
