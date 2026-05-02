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
    ProviderResult,
    RouteDecision,
    RouteRequest,
)
from routelabs_router.router import RouterEngine


class ChatService:
    def __init__(
        self,
        config: Config,
        router: RouterEngine | None = None,
        providers: dict[str, ChatProvider] | None = None,
    ) -> None:
        self.config = config
        self.router = router or RouterEngine(config)
        self.providers = providers or self._default_providers()

    def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        route = self.router.decide(
            RouteRequest(task=_task_from_messages(request), private=request.private)
        )

        provider = self.providers.get(route.provider)
        if provider is None:
            raise HTTPException(
                status_code=501,
                detail=f"provider '{route.provider}' is not configured",
            )

        result = provider.complete(request, model=request.model or route.model)
        return _build_chat_response(route, result)

    def _default_providers(self) -> dict[str, ChatProvider]:
        providers: dict[str, ChatProvider] = {
            "ollama": OllamaChatAdapter(self.config.providers.local.ollama),
        }
        cloud_config = self.config.providers.cloud.openai_compatible
        if cloud_config.api_key:
            providers["openai-compatible"] = OpenAICompatibleChatAdapter(cloud_config)
        return providers


def _task_from_messages(request: ChatCompletionRequest) -> str:
    user_messages = [
        message.content for message in request.messages if message.role == "user"
    ]
    if user_messages:
        return user_messages[-1]
    return request.messages[-1].content


def _build_chat_response(
    route: RouteDecision, result: ProviderResult
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
    )
