import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from routelabs_router.config import load_config
from routelabs_router.models import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthResponse,
    ModelsListResponse,
    ResponsesRequest,
    ResponsesResponse,
    RouteLogResponse,
    RouteDecision,
    RouteRequest,
    RouterStatsResponse,
)
from routelabs_router.router import RouterEngine
from routelabs_router.service import ChatService


def create_app(
    config_path: Path | None = None, service: ChatService | None = None
) -> FastAPI:
    resolved_path = config_path or Path("./config/router.yaml")
    config = load_config(resolved_path)
    engine = RouterEngine(config)
    chat_service = service or ChatService(config, router=engine)
    app = FastAPI(title="RouteLabs Router", version="0.3.0")

    @app.get("/healthz")
    def healthcheck() -> HealthResponse:
        return chat_service.health()

    @app.post("/v1/route", response_model=RouteDecision)
    def route(request: RouteRequest) -> RouteDecision:
        return chat_service.inspect_route(request)

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse:
        if request.stream:
            response = chat_service.create_chat_completion(request)
            return StreamingResponse(
                _stream_chat_completion_chunks(response),
                media_type="text/event-stream",
            )
        return chat_service.create_chat_completion(request)

    @app.post("/v1/responses", response_model=ResponsesResponse)
    def responses(
        request: ResponsesRequest,
    ) -> ResponsesResponse | StreamingResponse:
        if request.stream:
            response = chat_service.create_response(request)
            return StreamingResponse(
                _stream_responses_chunks(response),
                media_type="text/event-stream",
            )
        return chat_service.create_response(request)

    @app.post("/v1/messages", response_model=AnthropicMessagesResponse)
    def anthropic_messages(
        request: AnthropicMessagesRequest,
    ) -> AnthropicMessagesResponse | StreamingResponse:
        if request.stream:
            response = chat_service.create_anthropic_message(request)
            return StreamingResponse(
                _stream_anthropic_messages(response),
                media_type="text/event-stream",
            )
        return chat_service.create_anthropic_message(request)

    @app.post("/v1/embeddings", response_model=EmbeddingsResponse)
    def embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
        return chat_service.create_embeddings(request)

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


app = create_app()


def _stream_chat_completion_chunks(response: ChatCompletionResponse):
    choice = response.choices[0]
    base = {
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
    }

    if choice.message.tool_calls:
        first_chunk = {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": choice.message.tool_calls,
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"
    else:
        content = choice.message.content or ""
        first = True
        for token in content.split():
            chunk = {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            **({"role": "assistant"} if first else {}),
                            "content": token + " ",
                        },
                        "finish_reason": None,
                    }
                ],
            }
            first = False
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0)

    final_chunk = {
        **base,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": choice.finish_reason,
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _stream_responses_chunks(response: ResponsesResponse):
    message_item = next(
        (item for item in response.output if item.type == "message"),
        None,
    )
    message_item_id = message_item.id if message_item is not None else None

    yield f"data: {json.dumps({'type': 'response.created', 'response': response.model_dump()})}\n\n"

    if response.output_text:
        if message_item_id is not None:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": message_item.model_dump(),
                    }
                )
                + "\n\n"
            )
        for token in response.output_text.split():
            event = {
                "type": "response.output_text.delta",
                "response_id": response.id,
                "item_id": message_item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": token + " ",
            }
            yield f"data: {json.dumps(event)}\n\n"
            time.sleep(0)
        if message_item_id is not None:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "response.output_text.done",
                        "response_id": response.id,
                        "item_id": message_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": response.output_text,
                    }
                )
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": message_item.model_dump(),
                    }
                )
                + "\n\n"
            )

    for index, item in enumerate(response.output):
        if item.type != "function_call":
            continue
        event = {
            "type": "response.function_call_arguments.done",
            "item_id": item.id,
            "output_index": index,
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
        }
        yield f"data: {json.dumps(event)}\n\n"

    completed = {
        "type": "response.completed",
        "response": response.model_dump(),
    }
    yield f"data: {json.dumps(completed)}\n\n"
    yield "data: [DONE]\n\n"


def _stream_anthropic_messages(response: AnthropicMessagesResponse):
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': response.model_dump()})}\n\n"
    for index, block in enumerate(response.content):
        yield (
            "event: content_block_start\n"
            f"data: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': block})}\n\n"
        )
        if block.get("type") == "text":
            text = str(block.get("text", ""))
            for token in text.split():
                yield (
                    "event: content_block_delta\n"
                    f"data: {json.dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': token + ' '}})}\n\n"
                )
                time.sleep(0)
        elif block.get("type") == "tool_use":
            yield (
                "event: content_block_delta\n"
                f"data: {json.dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block.get('input', {}))}})}\n\n"
            )
        yield (
            "event: content_block_stop\n"
            f"data: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"
        )
    yield (
        "event: message_delta\n"
        f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': response.stop_reason, 'stop_sequence': response.stop_sequence}, 'usage': response.usage.model_dump()})}\n\n"
    )
    yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"
