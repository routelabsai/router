from typing import Protocol

from routelabs_router.models import ChatCompletionRequest, ProviderResult


class ChatProvider(Protocol):
    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        """Execute a chat completion request."""
