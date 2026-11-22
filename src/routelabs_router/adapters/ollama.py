import json
from collections.abc import Iterator
from typing import Any

import httpx

from routelabs_router.adapters.base import ProviderExecutionError
from routelabs_router.config import ProviderConfig
from routelabs_router.streaming import (
    StreamEnd,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
)
from routelabs_router.models import (
    content_text,
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

    def _chat_payload(
        self, request: ChatCompletionRequest, model: str | None, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or request.model or self.config.model,
            "messages": [_serialize_ollama_message(message) for message in request.messages],
            "stream": stream,
        }
        format_value = _map_ollama_format(request.response_format)
        if format_value is not None:
            payload["format"] = format_value
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        options = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop,
            "num_predict": request.output_token_limit,
            "seed": request.seed,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }
        options = {key: value for key, value in options.items() if value is not None}
        if options:
            payload["options"] = options
        return payload

    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        payload = self._chat_payload(request, model, stream=False)
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

    def stream(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> Iterator[StreamEvent]:
        payload = self._chat_payload(request, model, stream=True)
        model_name = str(payload["model"])
        tool_index = 0
        end = StreamEnd(model=model_name)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST", f"{self.config.base_url.rstrip('/')}/api/chat", json=payload
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise ProviderExecutionError(
                            "ollama", f"HTTP {response.status_code}: {response.text[:300]}"
                        )
                    for line in response.iter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise ProviderExecutionError("ollama", str(chunk["error"])[:300])
                        message = chunk.get("message") or {}
                        if message.get("thinking"):
                            yield ThinkingDelta(str(message["thinking"]))
                        if message.get("content"):
                            yield TextDelta(str(message["content"]))
                        for tool_call in message.get("tool_calls") or []:
                            function = tool_call.get("function", {})
                            yield ToolCallDelta(
                                index=tool_index,
                                id=tool_call.get("id") or f"call_{tool_index}",
                                name=function.get("name", ""),
                                arguments=_normalize_arguments(function.get("arguments", {})),
                            )
                            tool_index += 1
                        if chunk.get("done"):
                            end = StreamEnd(
                                finish_reason="tool_calls"
                                if tool_index
                                else str(chunk.get("done_reason") or "stop"),
                                usage=_extract_usage(chunk),
                                model=str(chunk.get("model") or model_name),
                            )
        except httpx.TimeoutException as exc:
            raise ProviderExecutionError("ollama", "request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError("ollama", str(exc)) from exc
        yield end

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


def _map_ollama_format(response_format: dict[str, Any] | None) -> str | dict[str, Any] | None:
    if response_format is None:
        return None
    response_type = response_format.get("type")
    if response_type == "json_object":
        return "json"
    if response_type == "json_schema":
        json_schema = response_format.get("json_schema", {})
        schema = json_schema.get("schema")
        if isinstance(schema, dict):
            return schema
    return None


def _serialize_ollama_message(message: ChatMessage) -> dict[str, Any]:
    role = "system" if message.role == "developer" else message.role
    payload: dict[str, Any] = {"role": role}
    if isinstance(message.content, list):
        payload["content"] = content_text(message.content)
        images = _images_from_parts(message.content)
        if images:
            payload["images"] = images
    elif message.content is not None:
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


def _images_from_parts(parts: list[dict[str, Any]]) -> list[str]:
    """Ollama takes raw base64 images, so only data URLs can be forwarded."""
    images: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if isinstance(url, str) and url.startswith("data:") and "," in url:
            images.append(url.split(",", 1)[1])
    return images
