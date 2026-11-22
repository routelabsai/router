import json
import re
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
    iter_sse_data,
)
from routelabs_router.models import (
    ChatCompletionRequest,
    ChatMessage,
    ProviderResult,
    content_text,
)


class AnthropicChatAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.timeout = config.timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        return headers

    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        payload = _build_anthropic_payload(
            request=request,
            model=model or request.model or self.config.model,
        )
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.config.base_url.rstrip('/')}/messages",
                    json=payload,
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderExecutionError("anthropic", "request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError("anthropic", str(exc)) from exc

        return ProviderResult(
            content=_extract_content(data),
            model=str(data.get("model", payload["model"])),
            finish_reason=_extract_finish_reason(data),
            usage=_extract_usage(data),
            tool_calls=_extract_tool_calls(data),
            raw=data,
        )

    def stream(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> Iterator[StreamEvent]:
        payload = _build_anthropic_payload(
            request=request,
            model=model or request.model or self.config.model,
        )
        payload["stream"] = True
        model_name = str(payload["model"])
        tool_indexes: dict[int, int] = {}
        stop_reason = "end_turn"
        input_tokens = 0
        output_tokens = 0
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.config.base_url.rstrip('/')}/messages",
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise ProviderExecutionError(
                            "anthropic", f"HTTP {response.status_code}: {response.text[:300]}"
                        )
                    for data in iter_sse_data(response.iter_lines()):
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        event_type = event.get("type")
                        if event_type == "error":
                            raise ProviderExecutionError(
                                "anthropic", str(event.get("error"))[:300]
                            )
                        if event_type == "message_start":
                            message = event.get("message") or {}
                            model_name = str(message.get("model") or model_name)
                            usage = message.get("usage") or {}
                            input_tokens = int(usage.get("input_tokens", 0) or 0)
                        elif event_type == "content_block_start":
                            block = event.get("content_block") or {}
                            if block.get("type") == "tool_use":
                                tool_index = len(tool_indexes)
                                tool_indexes[int(event.get("index", 0))] = tool_index
                                yield ToolCallDelta(
                                    index=tool_index,
                                    id=block.get("id"),
                                    name=block.get("name"),
                                )
                        elif event_type == "content_block_delta":
                            delta = event.get("delta") or {}
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                yield TextDelta(str(delta["text"]))
                            elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                                yield ThinkingDelta(str(delta["thinking"]))
                            elif delta.get("type") == "input_json_delta":
                                block_index = int(event.get("index", 0))
                                if block_index in tool_indexes:
                                    yield ToolCallDelta(
                                        index=tool_indexes[block_index],
                                        arguments=str(delta.get("partial_json") or ""),
                                    )
                        elif event_type == "message_delta":
                            delta = event.get("delta") or {}
                            stop_reason = str(delta.get("stop_reason") or stop_reason)
                            usage = event.get("usage") or {}
                            output_tokens = int(usage.get("output_tokens", output_tokens) or 0)
                            input_tokens = int(usage.get("input_tokens", input_tokens) or 0)
        except httpx.TimeoutException as exc:
            raise ProviderExecutionError("anthropic", "request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError("anthropic", str(exc)) from exc
        yield StreamEnd(
            finish_reason=_STOP_REASON_TO_FINISH_REASON.get(stop_reason, stop_reason),
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            model=model_name,
        )


DEFAULT_MAX_TOKENS = 16000

# Claude models released before Opus 4.7 still accept sampling parameters;
# newer models reject non-default temperature/top_p with a 400.
_LEGACY_SAMPLING_PATTERN = re.compile(
    r"^claude-(?:3|instant|2|(?:opus|sonnet|haiku)-4(?:-[0-6])?(?:-\d{8})?$|"
    r"(?:opus|sonnet|haiku)-4-[0-6]-\d{8}$)"
)
# Models that reject forced tool use (`tool_choice` of type any/tool).
_NO_FORCED_TOOL_PATTERN = re.compile(
    r"^claude-(?:fable-5-[1-9]|mythos-5-[1-9]|opus-5-[1-9]|sonnet-5-[1-9]|"
    r"(?:fable|mythos|opus|sonnet)-[6-9])"
)


def _supports_sampling(model: str) -> bool:
    if not model.startswith("claude-"):
        return True  # e.g. a proxy alias; pass parameters through unchanged
    return bool(_LEGACY_SAMPLING_PATTERN.match(model))


def _supports_forced_tool_choice(model: str) -> bool:
    return not _NO_FORCED_TOOL_PATTERN.match(model)


def _build_anthropic_payload(
    request: ChatCompletionRequest,
    model: str,
) -> dict[str, Any]:
    system_blocks = []
    messages = []
    for message in request.messages:
        if message.role in {"system", "developer"}:
            text = content_text(message.content)
            if text:
                system_blocks.append({"type": "text", "text": text})
            continue
        converted = _serialize_message(message)
        if converted is not None:
            messages.append(converted)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.output_token_limit or DEFAULT_MAX_TOKENS,
    }
    if system_blocks:
        payload["system"] = system_blocks if len(system_blocks) > 1 else system_blocks[0]["text"]
    if _supports_sampling(model):
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
    if request.stop is not None:
        payload["stop_sequences"] = (
            request.stop if isinstance(request.stop, list) else [request.stop]
        )
    if request.tools is not None:
        payload["tools"] = _map_tools(request.tools)
    tool_choice = _map_tool_choice(request.tool_choice)
    if (
        tool_choice is not None
        and tool_choice.get("type") in {"any", "tool"}
        and not _supports_forced_tool_choice(model)
    ):
        tool_choice = {"type": "auto"}
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def _serialize_message(message: ChatMessage) -> dict[str, Any] | None:
    if message.role == "tool":
        tool_result = {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id or message.name or "tool_result",
            "content": content_text(message.content),
        }
        return {"role": "user", "content": [tool_result]}

    role = "assistant" if message.role == "assistant" else "user"
    content: list[dict[str, Any]] = _content_blocks(message.content)
    for tool_call in message.tool_calls or []:
        function = tool_call.get("function", {})
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id", "tool_use"),
                "name": function.get("name", ""),
                "input": _parse_tool_arguments(function.get("arguments")),
            }
        )
    if not content:
        return None
    return {"role": role, "content": content}


def _content_blocks(content: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
        elif part_type == "image_url":
            image = _image_block(part.get("image_url"))
            if image is not None:
                blocks.append(image)
    return blocks


def _image_block(image_url: Any) -> dict[str, Any] | None:
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:") and "," in url:
        header, data = url.split(",", 1)
        media_type = header[5:].split(";", 1)[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _map_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped = []
    for tool in tools:
        function = tool.get("function", {})
        mapped.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object"}),
            }
        )
    return mapped


def _map_tool_choice(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "any", "none"}:
            return {"type": tool_choice}
        if tool_choice == "required":
            return {"type": "any"}
        return None
    if tool_choice.get("type") == "function":
        function = tool_choice.get("function", {})
        name = function.get("name")
        if isinstance(name, str) and name:
            return {"type": "tool", "name": name}
    if isinstance(tool_choice.get("type"), str):
        return tool_choice
    return None


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {"value": value}


def _extract_content(data: dict[str, Any]) -> str:
    content = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "")
            if text:
                content.append(str(text))
    return "\n".join(content)


def _extract_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    result = []
    for index, block in enumerate(data.get("content", [])):
        if block.get("type") != "tool_use":
            continue
        result.append(
            {
                "id": block.get("id", f"toolu_{index}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            }
        )
    return result or None


def _extract_finish_reason(data: dict[str, Any]) -> str:
    if _extract_tool_calls(data):
        return "tool_calls"
    stop_reason = data.get("stop_reason")
    return _STOP_REASON_TO_FINISH_REASON.get(str(stop_reason), str(stop_reason or "stop"))


_STOP_REASON_TO_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
