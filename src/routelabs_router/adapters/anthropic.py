import json
from typing import Any

import httpx

from routelabs_router.adapters.base import ProviderExecutionError
from routelabs_router.config import ProviderConfig
from routelabs_router.models import ChatCompletionRequest, ChatMessage, ProviderResult


class AnthropicChatAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.timeout = config.timeout_seconds

    def complete(
        self, request: ChatCompletionRequest, model: str | None = None
    ) -> ProviderResult:
        payload = _build_anthropic_payload(
            request=request,
            model=model or request.model or self.config.model,
        )
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.config.base_url.rstrip('/')}/messages",
                    json=payload,
                    headers=headers,
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


def _build_anthropic_payload(
    request: ChatCompletionRequest,
    model: str,
) -> dict[str, Any]:
    system_blocks = []
    messages = []
    for message in request.messages:
        if message.role == "system":
            if message.content:
                system_blocks.append({"type": "text", "text": message.content})
            continue
        converted = _serialize_message(message)
        if converted is not None:
            messages.append(converted)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.max_tokens or 1024,
    }
    if system_blocks:
        payload["system"] = system_blocks if len(system_blocks) > 1 else system_blocks[0]["text"]
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
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def _serialize_message(message: ChatMessage) -> dict[str, Any] | None:
    if message.role == "tool":
        tool_result = {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id or message.name or "tool_result",
            "content": message.content or "",
        }
        return {"role": "user", "content": [tool_result]}

    role = "assistant" if message.role == "assistant" else "user"
    content: list[dict[str, Any]] = []
    if message.content:
        content.append({"type": "text", "text": message.content})
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
    return {"role": role, "content": content if len(content) > 1 else content}


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
        if tool_choice in {"auto", "any"}:
            return {"type": tool_choice}
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
    if stop_reason == "end_turn":
        return "stop"
    return str(stop_reason or "stop")


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
