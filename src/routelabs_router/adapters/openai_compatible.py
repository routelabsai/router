from typing import Any

import httpx

from routelabs_router.config import ProviderConfig
from routelabs_router.models import ChatCompletionRequest, ProviderResult


class OpenAICompatibleChatAdapter:
    def __init__(self, config: ProviderConfig, timeout: float = 60.0) -> None:
        self.config = config
        self.timeout = timeout

    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        payload = {
            "model": model or request.model or self.config.model,
            "messages": [message.model_dump() for message in request.messages],
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        return ProviderResult(
            content=_extract_content(data),
            model=data.get("model", payload["model"]),
            finish_reason=_extract_finish_reason(data),
            usage=_extract_usage(data),
            raw=data,
        )


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", ""))


def _extract_finish_reason(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return "stop"
    return str(choices[0].get("finish_reason", "stop"))


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
