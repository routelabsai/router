import time
import uuid

from fastapi import HTTPException

from routelabs_router.adapters.base import ChatProvider
from routelabs_router.adapters.ollama import OllamaChatAdapter
from routelabs_router.adapters.openai_compatible import OpenAICompatibleChatAdapter
from routelabs_router.config import Config
from routelabs_router.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DecisionTrace,
    ProviderResult,
    RouteDecision,
    RouteRequest,
)
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
    ) -> None:
        self.config = config
        self.router = router or RouterEngine(config)
        self.providers = providers or self._default_providers()
        self.verifier = verifier or HeuristicVerifier()
        self.telemetry = telemetry or InMemoryTelemetry()

    def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        initial_route = self.router.decide(
            RouteRequest(task=_task_from_messages(request), private=request.private)
        )
        trace = DecisionTrace(initial_route=initial_route, final_route=initial_route)

        provider = self.providers.get(initial_route.provider)
        if provider is None:
            raise HTTPException(
                status_code=501,
                detail=f"provider '{initial_route.provider}' is not configured",
            )

        result = provider.complete(request, model=request.model or initial_route.model)

        if initial_route.verify:
            verification = self.verifier.evaluate(
                task=_task_from_messages(request),
                complexity=initial_route.complexity,
                result=result,
            )
            trace.verification = verification

            if verification.should_escalate:
                cloud_route = self._cloud_route(initial_route.complexity)
                cloud_provider = self.providers.get(cloud_route.provider)
                if cloud_provider is not None:
                    trace.escalated = True
                    trace.escalation_reason = verification.reason
                    trace.final_route = cloud_route
                    result = cloud_provider.complete(
                        request, model=request.model or cloud_route.model
                    )
                else:
                    trace.escalation_reason = (
                        "verification requested escalation but no cloud provider is configured"
                    )

        response = _build_chat_response(trace.final_route, result, trace)
        self.telemetry.record(trace, is_private=request.private)
        return response

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


def _task_from_messages(request: ChatCompletionRequest) -> str:
    user_messages = [
        message.content for message in request.messages if message.role == "user"
    ]
    if user_messages:
        return user_messages[-1]
    return request.messages[-1].content


def _build_chat_response(
    route: RouteDecision, result: ProviderResult, trace: DecisionTrace
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=result.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.content),
                finish_reason=result.finish_reason,
            )
        ],
        usage=result.usage,
        route=route,
        trace=trace,
    )
