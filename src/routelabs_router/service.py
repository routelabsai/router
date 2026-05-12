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
        self.telemetry.record(
            request_id=request_id,
            task_preview=_task_preview(task),
            trace=trace,
            is_private=effective_private,
            auto_private=privacy.detected and not request.private,
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
        task = _task_from_embedding_input(request.input)
        privacy = self.privacy_detector.evaluate(task, explicitly_private=request.private)
        effective_private = request.private or privacy.forced_local
        route = self.router.decide(RouteRequest(task=task, private=effective_private))
        requested_model = _requested_model_override(request.model)
        final_route, result = self._execute_embeddings_route_or_fallback(
            request=request,
            route=route,
            requested_model=requested_model,
            allow_cross_target_fallback=not effective_private,
        )
        return EmbeddingsResponse(
            data=result.data,
            model=result.model,
            usage=result.usage,
            route=RouteDecision(**{**final_route.model_dump(), "model": result.model, "verify": False}),
        )

    def list_models(self) -> ModelsListResponse:
        models = [
            ModelCard(id="route-auto", owned_by="routelabs"),
            ModelCard(
                id=self.config.providers.local.ollama.model,
                owned_by="routelabs-local",
            ),
            ModelCard(
                id=self.config.providers.local.ollama.embedding_model
                or self.config.providers.local.ollama.model,
                owned_by="routelabs-local",
            ),
            ModelCard(
                id=self.config.providers.local.llamacpp.model,
                owned_by="routelabs-local",
            ),
            ModelCard(
                id=self.config.providers.local.llamacpp.embedding_model
                or self.config.providers.local.llamacpp.model,
                owned_by="routelabs-local",
            ),
            ModelCard(
                id=self.config.providers.cloud.openai_compatible.model,
                owned_by="routelabs-cloud",
            ),
            ModelCard(
                id=self.config.providers.cloud.openai_compatible.embedding_model
                or self.config.providers.cloud.openai_compatible.model,
                owned_by="routelabs-cloud",
            ),
        ]
        deduped: list[ModelCard] = []
        seen: set[str] = set()
        for model in models:
            if model.id in seen:
                continue
            seen.add(model.id)
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
        return HealthResponse(status="ok", providers=providers)

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
        requested_model: str | None,
        allow_cross_target_fallback: bool,
    ) -> tuple[RouteDecision, ProviderEmbeddingResult]:
        try:
            return route, self._execute_embeddings_route(
                request=request,
                route=route,
                requested_model=requested_model,
            )
        except HTTPException as exc:
            if not allow_cross_target_fallback or route.target != "local":
                raise exc
            fallback_route = self._cloud_route(route.complexity)
            return fallback_route, self._execute_embeddings_route(
                request=request,
                route=fallback_route,
                requested_model=requested_model,
            )

    def _execute_embeddings_route(
        self,
        request: EmbeddingsRequest,
        route: RouteDecision,
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
        for _ in range(retries):
            try:
                return provider.embed(request, model=model_name)
            except ProviderExecutionError as exc:
                last_error = exc.reason

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
            try:
                result = provider.complete(request, model=model_name)
                trace.attempts.append(
                    ProviderAttempt(
                        provider=route.provider,
                        model=result.model,
                        target=route.target,
                        outcome="success",
                        reason=f"attempt {attempt_number}",
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

    def _provider_readiness(self, provider_name: str) -> tuple[bool, str]:
        if provider_name == "ollama":
            base_url = self.config.providers.local.ollama.base_url.rstrip("/")
            try:
                with httpx.Client(timeout=1.0) as client:
                    response = client.get(f"{base_url}/api/tags")
                    response.raise_for_status()
                return True, "ready"
            except httpx.HTTPError:
                return False, "unreachable"

        if provider_name == "openai-compatible":
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
