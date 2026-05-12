from typing import Protocol, runtime_checkable

from routelabs_router.models import (
    ChatCompletionRequest,
    EmbeddingsRequest,
    ProviderEmbeddingResult,
    ProviderResult,
)


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


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(
        self, request: EmbeddingsRequest, model: str | None = None
    ) -> ProviderEmbeddingResult:
        """Execute an embeddings request."""
