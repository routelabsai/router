from typing import Any

import httpx

from routelabs_router.adapters.base import ProviderExecutionError
from routelabs_router.config import ProviderConfig
from routelabs_router.models import (
    ChatCompletionRequest,
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsUsage,
    ProviderEmbeddingResult,
    ProviderResult,
)


class OpenAICompatibleChatAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.timeout = config.timeout_seconds

    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        payload = {
            "model": model or request.model or self.config.model,
            "messages": [message.model_dump() for message in request.messages],
        }
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderExecutionError(
                "openai-compatible", "request timed out"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError("openai-compatible", str(exc)) from exc

        return ProviderResult(
            content=_extract_content(data),
            model=data.get("model", payload["model"]),
            finish_reason=_extract_finish_reason(data),
            usage=_extract_usage(data),
            tool_calls=_extract_tool_calls(data),
            raw=data,
        )

    def embed(
        self, request: EmbeddingsRequest, model: str | None = None
    ) -> ProviderEmbeddingResult:
        payload = {
            "model": model or request.model or self.config.embedding_model or self.config.model,
            "input": request.input,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.config.base_url.rstrip('/')}/embeddings",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderExecutionError(
                "openai-compatible", "request timed out"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError("openai-compatible", str(exc)) from exc

        return ProviderEmbeddingResult(
            data=[
                EmbeddingObject(
                    index=int(item.get("index", index)),
                    embedding=[float(value) for value in item.get("embedding", [])],
                )
                for index, item in enumerate(data.get("data", []))
            ],
            model=str(data.get("model", payload["model"])),
            usage=EmbeddingsUsage(
                prompt_tokens=int(data.get("usage", {}).get("prompt_tokens", 0) or 0),
                total_tokens=int(data.get("usage", {}).get("total_tokens", 0) or 0),
            ),
        )


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    return "" if content is None else str(content)


def _extract_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    choices = data.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return None
    return tool_calls


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
