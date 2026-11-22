import re
from typing import Any

import httpx

from routelabs_router.adapters.base import ProviderExecutionError
from routelabs_router.config import ProviderConfig
from routelabs_router.models import (
    ChatCompletionRequest,
    ChatMessage,
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
        model_name = model or request.model or self.config.model
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                _serialize_message(message, keep_developer_role=_is_openai_model(model_name))
                for message in request.messages
            ],
        }
        reasoning_model = _is_openai_reasoning_model(model_name)
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        token_limit = request.output_token_limit
        if token_limit is not None:
            # OpenAI reasoning models reject `max_tokens`; most local
            # OpenAI-compatible servers only understand `max_tokens`.
            payload["max_completion_tokens" if reasoning_model else "max_tokens"] = token_limit
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.seed is not None:
            payload["seed"] = request.seed
        if not reasoning_model:
            # Reasoning models reject non-default sampling parameters.
            for field in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
                value = getattr(request, field)
                if value is not None:
                    payload[field] = value
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


_REASONING_MODEL_PATTERN = re.compile(r"^(?:openai/)?(?:o\d|gpt-5)")


def _is_openai_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL_PATTERN.match(model))


_OPENAI_MODEL_PATTERN = re.compile(r"^(?:openai/)?(?:gpt-(?!oss)|o\d|chatgpt-)")


def _is_openai_model(model: str) -> bool:
    return bool(_OPENAI_MODEL_PATTERN.match(model))


def _serialize_message(
    message: ChatMessage, keep_developer_role: bool = True
) -> dict[str, Any]:
    payload = message.model_dump(exclude_none=True, exclude={"tool_name"})
    payload.setdefault("content", None)
    # Local servers (llama.cpp, LM Studio, vLLM chat templates) commonly
    # reject the newer `developer` role; `system` is the portable equivalent.
    if payload["role"] == "developer" and not keep_developer_role:
        payload["role"] = "system"
    return payload


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
