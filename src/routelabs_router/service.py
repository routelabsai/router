import time
import uuid

import httpx
from fastapi import HTTPException

from routelabs_router.adapters.base import (
    ChatProvider,
    EmbeddingProvider,
    ProviderExecutionError,
)
from routelabs_router.adapters.ollama import OllamaChatAdapter
from routelabs_router.adapters.openai_compatible import OpenAICompatibleChatAdapter
from routelabs_router.config import Config
from routelabs_router.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DecisionTrace,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthResponse,
    ModelCard,
    ModelsListResponse,
    ProviderHealth,
    ProviderEmbeddingResult,
    ProviderResult,
    ProviderAttempt,
    RouteDecision,
    RouteRequest,
)
from routelabs_router.privacy import HeuristicPrivacyDetector
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
    ) -> None:
        self.config = config
        self.router = router or RouterEngine(config)
        self.providers = providers or self._default_providers()
        self.verifier = verifier or HeuristicVerifier()
        self.telemetry = telemetry or InMemoryTelemetry(config.telemetry.costs)
        self.privacy_detector = privacy_detector or HeuristicPrivacyDetector()

    def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        started = time.perf_counter()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        task = _task_from_messages(request)
        privacy = self.privacy_detector.evaluate(task, explicitly_private=request.private)
        effective_private = request.private or privacy.forced_local
        initial_route = self.router.decide(
            RouteRequest(task=task, private=effective_private)
        )
        trace = DecisionTrace(
            privacy=privacy, initial_route=initial_route, final_route=initial_route
        )
        requested_model = _requested_model_override(request.model)
        result = self._execute_route_or_fallback(
            request=request,
            route=initial_route,
            trace=trace,
            requested_model=requested_model,
            allow_cross_target_fallback=not effective_private,
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

            if verification.should_escalate:
                cloud_route = self._cloud_route(initial_route.complexity)
                trace.escalated = True
                trace.escalation_reason = verification.reason
                try:
                    result = self._execute_route(
                        request=request,
                        route=cloud_route,
                        trace=trace,
                        requested_model=requested_model,
                    )
                    trace.final_route = cloud_route
                except HTTPException as exc:
                    trace.escalated = False
                    trace.escalation_reason = (
                        "verification requested escalation but the cloud route failed: "
                        f"{exc.detail}"
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
        return response

    def inspect_route(self, request: RouteRequest) -> RouteDecision:
        decision = self.router.decide(request)
        available, status = self._provider_readiness(decision.provider)
        fallback_available, fallback_status = self._provider_readiness(
            self.config.providers.cloud.default
        )
        if decision.target == "cloud":
            fallback_available = None
            fallback_status = None
        payload = decision.model_dump()
        payload.update(
            provider_available=available,
            provider_status=status,
            fallback_available=fallback_available,
            fallback_status=fallback_status,
        )
        return RouteDecision(**payload)

    def create_embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        started = time.perf_counter()
        request_id = f"embd-{uuid.uuid4().hex[:24]}"
        task = _task_from_embedding_input(request.input)
        privacy = self.privacy_detector.evaluate(task, explicitly_private=request.private)
        effective_private = request.private or privacy.forced_local
        route = self.router.decide(RouteRequest(task=task, private=effective_private))
        requested_model = _requested_model_override(request.model)
        trace = DecisionTrace(
            privacy=privacy,
            initial_route=route,
            final_route=route,
        )
        final_route, result = self._execute_embeddings_route_or_fallback(
            request=request,
            route=route,
            trace=trace,
            requested_model=requested_model,
            allow_cross_target_fallback=not effective_private,
        )
        trace.final_route = final_route
        total_latency_ms = _elapsed_ms(started)
        trace.total_latency_ms = total_latency_ms
        self.telemetry.record(
            request_id=request_id,
            request_kind="embeddings",
            task_preview=_task_preview(task),
            trace=trace,
            is_private=effective_private,
            auto_private=privacy.detected and not request.private,
            total_latency_ms=total_latency_ms,
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
                installed=None,
                status="configured",
            ),
            ModelCard(
                id=self.config.providers.local.llamacpp.embedding_model
                or self.config.providers.local.llamacpp.model,
                owned_by="routelabs-local",
                provider="llamacpp",
                source="configured",
                installed=None,
                status="configured",
            ),
            ModelCard(
                id=self.config.providers.cloud.openai_compatible.model,
                owned_by="routelabs-cloud",
                provider="openai-compatible",
                source="configured",
                installed=self.config.providers.cloud.openai_compatible.api_key is not None,
                status=(
                    "configured"
                    if self.config.providers.cloud.openai_compatible.api_key
                    else "not_configured"
                ),
            ),
            ModelCard(
                id=self.config.providers.cloud.openai_compatible.embedding_model
                or self.config.providers.cloud.openai_compatible.model,
                owned_by="routelabs-cloud",
                provider="openai-compatible",
                source="configured",
                installed=self.config.providers.cloud.openai_compatible.api_key is not None,
                status=(
                    "configured"
                    if self.config.providers.cloud.openai_compatible.api_key
                    else "not_configured"
                ),
            ),
        ]
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

    def health(self) -> HealthResponse:
        providers = {
            self.config.providers.local.default: self._provider_health(
                self.config.providers.local.default
            ),
            self.config.providers.cloud.default: self._provider_health(
                self.config.providers.cloud.default
            ),
        }
        local_default = self.config.providers.local.default
        cloud_default = self.config.providers.cloud.default
        local_available = providers[local_default].available
        cloud_available = providers[cloud_default].available

        if local_available:
            status = "ok"
        elif cloud_available:
            status = "degraded"
        else:
            status = "error"

        return HealthResponse(status=status, providers=providers)

    def _default_providers(self) -> dict[str, ChatProvider]:
        providers: dict[str, ChatProvider] = {
            "ollama": OllamaChatAdapter(self.config.providers.local.ollama),
        }
        cloud_config = self.config.providers.cloud.openai_compatible
        if cloud_config.api_key:
            providers["openai-compatible"] = OpenAICompatibleChatAdapter(cloud_config)
        return providers

    def _cloud_route(self, complexity: str) -> RouteDecision:
        return RouteDecision(
            target="cloud",
            provider=self.config.providers.cloud.default,
            model=self.config.providers.cloud.openai_compatible.model,
            reason="verification requested escalation to a stronger remote model",
            complexity=complexity,
            verify=False,
        )

    def get_stats(self):
        return self.telemetry.snapshot()

    def get_recent_logs(self):
        return self.telemetry.recent_logs()

    def _execute_embeddings_route_or_fallback(
        self,
        request: EmbeddingsRequest,
        route: RouteDecision,
        trace: DecisionTrace,
        requested_model: str | None,
        allow_cross_target_fallback: bool,
    ) -> tuple[RouteDecision, ProviderEmbeddingResult]:
        try:
            return route, self._execute_embeddings_route(
                request=request,
                route=route,
                trace=trace,
                requested_model=requested_model,
            )
        except HTTPException as exc:
            if not allow_cross_target_fallback or route.target != "local":
                raise exc
            fallback_route = self._cloud_route(route.complexity)
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
            )

    def _execute_embeddings_route(
        self,
        request: EmbeddingsRequest,
        route: RouteDecision,
        trace: DecisionTrace,
        requested_model: str | None,
    ) -> ProviderEmbeddingResult:
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
        for attempt_number in range(1, retries + 1):
            attempt_started = time.perf_counter()
            try:
                result = provider.embed(request, model=model_name)
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
    ) -> ProviderResult:
        try:
            return self._execute_route(
                request=request,
                route=route,
                trace=trace,
                requested_model=requested_model,
            )
        except HTTPException as exc:
            if not allow_cross_target_fallback or route.target != "local":
                raise exc
            fallback_route = self._cloud_route(route.complexity)
            trace.escalated = True
            trace.escalation_reason = "local provider was unavailable, falling back to cloud"
            trace.final_route = fallback_route
            return self._execute_route(
                request=request,
                route=fallback_route,
                trace=trace,
                requested_model=requested_model,
            )

    def _execute_route(
        self,
        request: ChatCompletionRequest,
        route: RouteDecision,
        trace: DecisionTrace,
        requested_model: str | None,
    ) -> ProviderResult:
        provider = self.providers.get(route.provider)
        if provider is None:
            raise HTTPException(
                status_code=501,
                detail=f"provider '{route.provider}' is not configured",
            )

        model_name = requested_model or route.model
        retries = max(1, self._provider_retries(route.provider))
        last_error: str | None = None
        for attempt_number in range(1, retries + 1):
            attempt_started = time.perf_counter()
            try:
                result = provider.complete(request, model=model_name)
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
        return 1

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

        if provider_name == "openai-compatible":
            provider = self.providers.get(provider_name)
            if provider is not None and not isinstance(
                provider, OpenAICompatibleChatAdapter
            ):
                return True, "configured"
            if not self.config.providers.cloud.openai_compatible.api_key:
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


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _completion_tokens_per_second(
    completion_tokens: int, total_latency_ms: float
) -> float | None:
    if completion_tokens <= 0 or total_latency_ms <= 0:
        return None
    return round(completion_tokens / (total_latency_ms / 1000), 3)
