from typing import Any

import httpx

from routelabs_router.adapters.base import ProviderExecutionError
from routelabs_router.config import ProviderConfig
from routelabs_router.models import ChatCompletionRequest, ProviderResult


class OllamaChatAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.timeout = config.timeout_seconds

    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        payload = {
            "model": model or request.model or self.config.model,
            "messages": [message.model_dump() for message in request.messages],
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.config.base_url.rstrip('/')}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderExecutionError("ollama", "request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError("ollama", str(exc)) from exc

        content = _extract_content(data)
        usage = _extract_usage(data)
        resolved_model = data.get("model", payload["model"])

        return ProviderResult(
            content=content,
            model=resolved_model,
            finish_reason="stop",
            usage=usage,
            raw=data,
        )


def _extract_content(data: dict[str, Any]) -> str:
    message = data.get("message", {})
    return str(message.get("content", ""))


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
    completion_tokens = int(data.get("eval_count", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
