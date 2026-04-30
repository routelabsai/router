from pathlib import Path

from fastapi import FastAPI

from routelabs_router.config import load_config
from routelabs_router.models import RouteDecision, RouteRequest
from routelabs_router.router import RouterEngine

CONFIG_PATH = Path("./config/router.yaml")
config = load_config(CONFIG_PATH)
engine = RouterEngine(config)

app = FastAPI(title="RouteLabs Router", version="0.1.0")


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/route", response_model=RouteDecision)
def route(request: RouteRequest) -> RouteDecision:
    return engine.decide(request)
