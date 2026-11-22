import hmac
import json
import math
import os
import re
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypeVar

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from routelabs_router.config import DEFAULT_CONFIG, Config, load_config
from routelabs_router.models import (
    AnthropicCountTokensResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DecisionTrace,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthResponse,
    ModelsListResponse,
    ResponsesRequest,
    ResponsesResponse,
    RouteDecision,
    RouteLogResponse,
    RouteRequest,
    RouterStatsResponse,
)
from routelabs_router.router import RouterEngine
from routelabs_router.adapters.base import ProviderExecutionError
from routelabs_router.service import (
    PRIVATE_MODEL_ALIASES,
    BufferedChatStream,
    ChatService,
    ChatStream,
    build_anthropic_response,
    build_responses_response,
)
from routelabs_router.streaming import TextDelta, ThinkingDelta, ToolCallDelta

CONFIG_ENV_VAR = "ROUTELABS_CONFIG"

RequestT = TypeVar("RequestT", bound=BaseModel)


def _package_version() -> str:
    try:
        return version("routelabs-router")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "0.0.0"


def create_app(
    config_path: Path | None = None, service: ChatService | None = None
) -> FastAPI:
    resolved_path = config_path or Path("./config/router.yaml")
    if service is not None:
        config = service.config
    elif config_path is None and not resolved_path.exists():
        config = DEFAULT_CONFIG
    else:
        config = load_config(resolved_path)
    engine = RouterEngine(config)
    chat_service = service or ChatService(config, router=engine)
    app = FastAPI(title="RouteLabs Router", version=_package_version())
    _install_api_key_auth(app, config)

    @app.get("/healthz")
    def healthcheck() -> HealthResponse:
        return chat_service.health()

    @app.post("/v1/route", response_model=RouteDecision)
    def route(request: RouteRequest, http_request: Request) -> RouteDecision:
        return chat_service.inspect_route(_apply_routing_headers(request, http_request))

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(
        request: ChatCompletionRequest, http_request: Request, response: Response
    ) -> ChatCompletionResponse | StreamingResponse:
        request = _apply_routing_headers(request, http_request)
        if request.stream:
            stream = chat_service.open_chat_stream(request, _stream_mode(http_request))
            include_usage = bool((request.stream_options or {}).get("include_usage"))
            return _event_stream(
                _stream_chat_completion_chunks(stream, include_usage=include_usage),
                _stream_headers(stream.request_id, stream),
            )
        result = chat_service.create_chat_completion(request)
        response.headers.update(_route_headers(result.id, result.route, result.trace))
        return result

    @app.post("/v1/responses", response_model=ResponsesResponse)
    def responses(
        request: ResponsesRequest, http_request: Request, response: Response
    ) -> ResponsesResponse | StreamingResponse:
        request = _apply_routing_headers(request, http_request)
        if request.stream:
            stream = chat_service.open_response_stream(request, _stream_mode(http_request))
            return _event_stream(
                _stream_responses_chunks(stream),
                _stream_headers(f"resp_{stream.request_id}", stream),
            )
        result = chat_service.create_response(request)
        response.headers.update(_route_headers(result.id, result.route, result.trace))
        return result

    @app.post("/v1/messages", response_model=AnthropicMessagesResponse)
    def anthropic_messages(
        request: AnthropicMessagesRequest, http_request: Request, response: Response
    ) -> AnthropicMessagesResponse | StreamingResponse:
        request = _apply_routing_headers(request, http_request)
        if request.stream:
            stream = chat_service.open_anthropic_stream(request, _stream_mode(http_request))
            return _event_stream(
                _stream_anthropic_messages(stream),
                _stream_headers(f"msg_{stream.request_id}", stream),
            )
        result = chat_service.create_anthropic_message(request)
        response.headers.update(_route_headers(result.id, result.route, result.trace))
        return result

    @app.post("/v1/messages/count_tokens", response_model=AnthropicCountTokensResponse)
    def anthropic_count_tokens(request: dict[str, Any]) -> AnthropicCountTokensResponse:
        # Claude Code and the Anthropic SDKs call this before sending large
        # prompts. The router may serve the request from any provider, so it
        # returns a provider-neutral estimate instead of a tokenizer-exact count.
        return AnthropicCountTokensResponse(input_tokens=_estimate_tokens(request))

    @app.post("/v1/embeddings", response_model=EmbeddingsResponse)
    def embeddings(request: EmbeddingsRequest, http_request: Request) -> EmbeddingsResponse:
        return chat_service.create_embeddings(_apply_routing_headers(request, http_request))

    @app.get("/v1/models", response_model=ModelsListResponse)
    def list_models() -> ModelsListResponse:
        return chat_service.list_models()

    @app.get("/v1/stats", response_model=RouterStatsResponse)
    def stats() -> RouterStatsResponse:
        return chat_service.get_stats()

    @app.get("/v1/logs", response_model=RouteLogResponse)
    def logs() -> RouteLogResponse:
        return chat_service.get_recent_logs()

    return app


def create_app_from_env() -> FastAPI:
    """Factory used by `router start --reload`, which needs an import string."""
    config_path = os.environ.get(CONFIG_ENV_VAR)
    return create_app(Path(config_path) if config_path else None)



# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


def _install_api_key_auth(app: FastAPI, config: Config) -> None:
    env_name = config.server.api_key_env
    expected = os.environ.get(env_name) if env_name else None
    if not expected:
        return

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            supplied = request.headers.get("x-api-key") or ""
            authorization = request.headers.get("authorization") or ""
            if authorization.lower().startswith("bearer "):
                supplied = supplied or authorization[7:].strip()
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "authentication_error",
                            "message": (
                                f"missing or invalid RouteLabs API key; send the value of "
                                f"{env_name} as 'Authorization: Bearer <key>' or 'x-api-key'"
                            ),
                        }
                    },
                )
        return await call_next(request)


# --------------------------------------------------------------------------
# Routing controls carried in HTTP headers
# --------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _parse_bool_header(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise HTTPException(status_code=400, detail=f"header {name} must be true or false")


def _apply_routing_headers(request: RequestT, http_request: Request) -> RequestT:
    """Let clients that cannot add body fields (Claude Code, Cursor, Codex,
    IDE plugins) control routing with headers or a model alias."""
    headers = http_request.headers
    fields = type(request).model_fields
    update: dict[str, Any] = {}

    private = headers.get("x-routelabs-private")
    if private is not None and "private" in fields:
        update["private"] = _parse_bool_header("x-routelabs-private", private)
    if getattr(request, "model", None) in PRIVATE_MODEL_ALIASES:
        update["private"] = True

    allow_fallbacks = headers.get("x-routelabs-allow-fallbacks")
    if allow_fallbacks is not None and "allow_fallbacks" in fields:
        update["allow_fallbacks"] = _parse_bool_header(
            "x-routelabs-allow-fallbacks", allow_fallbacks
        )

    strip_pii = headers.get("x-routelabs-strip-pii")
    if strip_pii is not None and "strip_pii" in fields:
        update["strip_pii"] = _parse_bool_header("x-routelabs-strip-pii", strip_pii)

    max_cost = headers.get("x-routelabs-max-cloud-cost-usd")
    if max_cost is not None and "max_cloud_cost_usd" in fields:
        try:
            parsed = float(max_cost)
        except ValueError:
            parsed = -1.0
        if parsed < 0 or math.isnan(parsed):
            raise HTTPException(
                status_code=400,
                detail="header x-routelabs-max-cloud-cost-usd must be a non-negative number",
            )
        update["max_cloud_cost_usd"] = parsed

    agent_role = headers.get("x-routelabs-agent-role")
    if agent_role and "agent_role" in fields:
        update["agent_role"] = agent_role.strip()

    return request.model_copy(update=update) if update else request


def _route_headers(
    request_id: str, route: RouteDecision, trace: DecisionTrace
) -> dict[str, str]:
    headers = {
        "x-routelabs-request-id": request_id,
        "x-routelabs-target": route.target,
        "x-routelabs-provider": route.provider,
        "x-routelabs-model": route.model,
        "x-routelabs-escalated": "true" if trace.escalated else "false",
    }
    reason = trace.escalation_reason or route.reason
    if reason:
        # Header values must be latin-1; keep the reason short and printable.
        headers["x-routelabs-reason"] = re.sub(r"[^\x20-\x7e]", "?", reason)[:200]
    return headers


def _estimate_tokens(request: dict[str, Any]) -> int:
    payload = {
        key: request.get(key) for key in ("system", "messages", "tools") if request.get(key)
    }
    text = json.dumps(payload, ensure_ascii=False)
    # ~4 characters per token is the usual estimate for English text and code.
    return max(1, math.ceil(len(text) / 4))


# --------------------------------------------------------------------------
# Streaming
#
# Live streams forward provider tokens as they arrive. Buffered streams (used
# when the verifier or a JSON schema must see the whole answer first) replay a
# finished response through the same formatters, byte for byte.
# --------------------------------------------------------------------------


def _stream_mode(http_request: Request) -> str | None:
    value = http_request.headers.get("x-routelabs-stream")
    return value.strip().lower() if value else None


def _stream_headers(response_id: str, stream: ChatStream) -> dict[str, str]:
    headers = _route_headers(response_id, stream.route, stream.trace)
    headers["x-routelabs-streaming"] = stream.mode
    return headers


def _sse(data: dict[str, Any], event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"


def _event_stream(chunks: Iterator[str], headers: dict[str, str]) -> StreamingResponse:
    return StreamingResponse(
        chunks,
        media_type="text/event-stream",
        headers={**headers, "cache-control": "no-cache", "x-accel-buffering": "no"},
    )


def _known_input_tokens(stream: ChatStream) -> int:
    if isinstance(stream, BufferedChatStream):
        return stream.final().usage.prompt_tokens
    return 0


def _stream_chat_completion_chunks(
    stream: ChatStream, include_usage: bool = False
) -> Iterator[str]:
    base = {
        "id": stream.request_id,
        "object": "chat.completion.chunk",
        "created": stream.created,
        "model": stream.model,
    }

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        payload = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if include_usage:
            payload["usage"] = None
        return _sse(payload)

    yield chunk({"role": "assistant", "content": ""})
    try:
        for event in stream.events():
            if isinstance(event, TextDelta):
                yield chunk({"content": event.text})
            elif isinstance(event, ThinkingDelta):
                # De facto field used by DeepSeek, vLLM and llama.cpp clients.
                yield chunk({"reasoning_content": event.text})
            elif isinstance(event, ToolCallDelta):
                tool_call: dict[str, Any] = {"index": event.index}
                function: dict[str, Any] = {"arguments": event.arguments}
                if event.id:
                    tool_call.update(id=event.id, type="function")
                if event.name is not None:
                    function["name"] = event.name
                tool_call["function"] = function
                yield chunk({"tool_calls": [tool_call]})
    except ProviderExecutionError as exc:
        yield _sse({"error": {"type": "provider_error", "message": exc.reason}})
        yield "data: [DONE]\n\n"
        return
    final = stream.final()
    yield chunk({}, finish_reason=final.choices[0].finish_reason)
    if include_usage:
        yield _sse({**base, "choices": [], "usage": final.usage.model_dump()})
    yield "data: [DONE]\n\n"


def _stream_responses_chunks(stream: ChatStream) -> Iterator[str]:
    sequence = 0

    def event(payload: dict[str, Any]) -> str:
        nonlocal sequence
        payload = {**payload, "sequence_number": sequence}
        sequence += 1
        return _sse(payload, event=payload["type"])

    response_id = f"resp_{stream.request_id}"
    message_id = f"msg_{stream.request_id}"
    in_progress = {
        "id": response_id,
        "object": "response",
        "created_at": stream.created,
        "status": "in_progress",
        "model": stream.model,
        "output": [],
        "output_text": "",
    }
    yield event({"type": "response.created", "response": in_progress})
    yield event({"type": "response.in_progress", "response": in_progress})

    empty_part = {"type": "output_text", "text": "", "annotations": []}
    text_started = False
    try:
        for item in stream.events():
            if not isinstance(item, TextDelta):
                continue  # tool calls are emitted whole once the stream ends
            if not text_started:
                text_started = True
                yield event(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    }
                )
                yield event(
                    {
                        "type": "response.content_part.added",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": empty_part,
                    }
                )
            yield event(
                {
                    "type": "response.output_text.delta",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": item.text,
                }
            )
    except ProviderExecutionError as exc:
        yield event({"type": "error", "code": "provider_error", "message": exc.reason})
        yield "data: [DONE]\n\n"
        return

    final = build_responses_response(stream.final())
    for output_index, output_item in enumerate(final.output):
        item_data = output_item.model_dump()
        if output_item.type == "message":
            if not text_started:
                continue
            for done in (
                {
                    "type": "response.output_text.done",
                    "item_id": message_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": final.output_text,
                },
                {
                    "type": "response.content_part.done",
                    "item_id": message_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {**empty_part, "text": final.output_text},
                },
            ):
                yield event(done)
        elif output_item.type == "function_call":
            yield event(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {**item_data, "status": "in_progress", "arguments": ""},
                }
            )
            yield event(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": output_item.id,
                    "output_index": output_index,
                    "delta": item_data.get("arguments", ""),
                }
            )
            yield event(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": output_item.id,
                    "output_index": output_index,
                    "call_id": item_data.get("call_id"),
                    "name": item_data.get("name"),
                    "arguments": item_data.get("arguments", ""),
                }
            )
        else:
            continue
        yield event(
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {**item_data, "status": "completed"},
            }
        )

    yield event({"type": "response.completed", "response": final.model_dump()})
    yield "data: [DONE]\n\n"


def _stream_anthropic_messages(stream: ChatStream) -> Iterator[str]:
    message = {
        "id": f"msg_{stream.request_id}",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": stream.model,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": _known_input_tokens(stream), "output_tokens": 0},
    }
    yield _sse({"type": "message_start", "message": message}, event="message_start")

    block_index = 0
    open_kind: str | None = None  # "thinking" or "text"

    def close_block() -> Iterator[str]:
        nonlocal block_index, open_kind
        if open_kind == "thinking":
            yield _sse(
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "signature_delta", "signature": ""},
                },
                event="content_block_delta",
            )
        if open_kind is not None:
            yield _sse(
                {"type": "content_block_stop", "index": block_index},
                event="content_block_stop",
            )
            block_index += 1
            open_kind = None

    try:
        for item in stream.events():
            if isinstance(item, ThinkingDelta):
                kind, delta = "thinking", {"type": "thinking_delta", "thinking": item.text}
                start_block = {"type": "thinking", "thinking": "", "signature": ""}
            elif isinstance(item, TextDelta):
                kind, delta = "text", {"type": "text_delta", "text": item.text}
                start_block = {"type": "text", "text": ""}
            else:
                continue  # tool_use blocks are emitted whole once the stream ends
            if open_kind != kind:
                yield from close_block()
                open_kind = kind
                yield _sse(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": start_block,
                    },
                    event="content_block_start",
                )
            yield _sse(
                {"type": "content_block_delta", "index": block_index, "delta": delta},
                event="content_block_delta",
            )
    except ProviderExecutionError as exc:
        yield _sse(
            {"type": "error", "error": {"type": "api_error", "message": exc.reason}},
            event="error",
        )
        return

    yield from close_block()

    final = build_anthropic_response(stream.final())
    for block in final.content:
        if block.get("type") != "tool_use":
            continue
        yield _sse(
            {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {**block, "input": {}},
            },
            event="content_block_start",
        )
        yield _sse(
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block.get("input", {})),
                },
            },
            event="content_block_delta",
        )
        yield _sse({"type": "content_block_stop", "index": block_index}, event="content_block_stop")
        block_index += 1

    yield _sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": final.stop_reason, "stop_sequence": final.stop_sequence},
            "usage": {
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            },
        },
        event="message_delta",
    )
    yield _sse({"type": "message_stop"}, event="message_stop")

app = create_app()
