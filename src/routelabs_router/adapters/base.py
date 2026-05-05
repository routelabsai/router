from typing import Protocol

from routelabs_router.models import ChatCompletionRequest, ProviderResult


class ProviderExecutionError(RuntimeError):
    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"{provider}: {reason}")


class ChatProvider(Protocol):
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        """Execute a chat completion request."""
