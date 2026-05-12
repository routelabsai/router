import json
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


class OllamaChatAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.timeout = config.timeout_seconds

    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        payload = {
            "model": model or request.model or self.config.model,
            "messages": [_serialize_ollama_message(message) for message in request.messages],
            "stream": False,
        }
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            payload["options"] = {**payload.get("options", {}), "temperature": request.temperature}
        if request.top_p is not None:
            payload["options"] = {**payload.get("options", {}), "top_p": request.top_p}
        if request.stop is not None:
            payload["options"] = {**payload.get("options", {}), "stop": request.stop}
        if request.max_tokens is not None:
            payload["options"] = {**payload.get("options", {}), "num_predict": request.max_tokens}
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
            finish_reason=_extract_finish_reason(data),
            usage=usage,
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
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.config.base_url.rstrip('/')}/api/embed",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderExecutionError("ollama", "request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError("ollama", str(exc)) from exc

        embeddings = data.get("embeddings") or []
        if embeddings and isinstance(embeddings[0], (int, float)):
            embeddings = [embeddings]
        items = [
            EmbeddingObject(index=index, embedding=[float(value) for value in vector])
            for index, vector in enumerate(embeddings)
        ]
        usage = EmbeddingsUsage(
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            total_tokens=int(data.get("prompt_eval_count", 0) or 0),
        )
        return ProviderEmbeddingResult(
            data=items,
            model=str(data.get("model", payload["model"])),
            usage=usage,
        )


def _extract_content(data: dict[str, Any]) -> str:
    message = data.get("message", {})
    content = message.get("content", "")
    return "" if content is None else str(content)


def _extract_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    message = data.get("message", {})
    raw_tool_calls = message.get("tool_calls") or []
    if not raw_tool_calls:
        return None
    result: list[dict[str, Any]] = []
    for index, tool_call in enumerate(raw_tool_calls):
        function = tool_call.get("function", {})
        result.append(
            {
                "id": tool_call.get("id", f"call_{index}"),
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": _normalize_arguments(function.get("arguments", {})),
                },
            }
        )
    return result


def _extract_finish_reason(data: dict[str, Any]) -> str:
    tool_calls = _extract_tool_calls(data)
    if tool_calls:
        return "tool_calls"
    return str(data.get("done_reason", "stop") or "stop")


def _normalize_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _serialize_ollama_message(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    if message.tool_calls is not None:
        payload["tool_calls"] = [
            {
                "function": {
                    "name": tool_call.get("function", {}).get("name", ""),
                    "arguments": tool_call.get("function", {}).get("arguments", {}),
                }
            }
            for tool_call in message.tool_calls
        ]
    if message.role == "tool" and (message.tool_name or message.name):
        payload["tool_name"] = message.tool_name or message.name
    return payload


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
    completion_tokens = int(data.get("eval_count", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
