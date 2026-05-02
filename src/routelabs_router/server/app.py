from pathlib import Path

from fastapi import FastAPI

from routelabs_router.config import load_config
from routelabs_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
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
    app = FastAPI(title="RouteLabs Router", version="0.1.0")

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/route", response_model=RouteDecision)
    def route(request: RouteRequest) -> RouteDecision:
        return engine.decide(request)

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        return chat_service.create_chat_completion(request)

    @app.get("/v1/stats", response_model=RouterStatsResponse)
    def stats() -> RouterStatsResponse:
        return chat_service.get_stats()

    @app.get("/v1/logs", response_model=RouteLogResponse)
    def logs() -> RouteLogResponse:
        return chat_service.get_recent_logs()

    return app


app = create_app()
