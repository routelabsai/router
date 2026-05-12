import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from routelabs_router.config import load_config
from routelabs_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthResponse,
    ModelsListResponse,
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
    app = FastAPI(title="RouteLabs Router", version="0.2.0")

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
