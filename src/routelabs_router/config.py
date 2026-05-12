import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    base_url: str
    model: str
    embedding_model: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 1


class LocalProvidersConfig(BaseModel):
    default: str
    ollama: ProviderConfig
    llamacpp: ProviderConfig


class CloudProvidersConfig(BaseModel):
    default: str
    openai_compatible: ProviderConfig


class ProvidersConfig(BaseModel):
    local: LocalProvidersConfig
    cloud: CloudProvidersConfig


class ServerConfig(BaseModel):
    host: str
    port: int


class RoutingConfig(BaseModel):
    default_mode: str
    escalate_on_verification_failure: bool
    prefer_local_for_private: bool


class PrivacyPolicyConfig(BaseModel):
    deny_cloud_when_private: bool


class ComplexityPolicyConfig(BaseModel):
    local_max: str


class PoliciesConfig(BaseModel):
    privacy: PrivacyPolicyConfig
    complexity: ComplexityPolicyConfig


class TelemetryCostConfig(BaseModel):
    local_request_cost_usd: float
    cloud_request_cost_usd: float


class TelemetryConfig(BaseModel):
    costs: TelemetryCostConfig


class Config(BaseModel):
    server: ServerConfig
    routing: RoutingConfig
    providers: ProvidersConfig
    policies: PoliciesConfig
    telemetry: TelemetryConfig


DEFAULT_CONFIG = Config.model_validate(
    {
        "server": {"host": "127.0.0.1", "port": 8000},
        "routing": {
            "default_mode": "balanced",
            "escalate_on_verification_failure": True,
            "prefer_local_for_private": True,
        },
        "providers": {
            "local": {
                "default": "ollama",
                "ollama": {
                    "base_url": "http://127.0.0.1:11434",
                    "model": "qwen3:4b",
                    "embedding_model": "embeddinggemma",
                },
                "llamacpp": {
                    "base_url": "http://127.0.0.1:8080",
                    "model": "qwen3-4b-instruct",
                    "embedding_model": "qwen3-embedding",
                },
            },
            "cloud": {
                "default": "openai-compatible",
                "openai_compatible": {
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4.1-mini",
                    "embedding_model": "text-embedding-3-small",
                    "api_key_env": "OPENAI_API_KEY",
                },
            },
        },
        "policies": {
            "privacy": {"deny_cloud_when_private": True},
            "complexity": {"local_max": "medium"},
        },
        "telemetry": {
            "costs": {
                "local_request_cost_usd": 0.0002,
                "cloud_request_cost_usd": 0.02,
            }
        },
    }
)


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    merged = _deep_merge(DEFAULT_CONFIG.model_dump(), raw)
    _inject_provider_secrets(merged)
    return Config.model_validate(merged)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _inject_provider_secrets(config: dict) -> None:
    cloud = config.get("providers", {}).get("cloud", {})
    openai_compatible = cloud.get("openai_compatible")
    if not isinstance(openai_compatible, dict):
        return

    env_name = openai_compatible.get("api_key_env")
    if isinstance(env_name, str) and env_name:
        api_key = os.getenv(env_name)
        if api_key:
            openai_compatible["api_key"] = api_key
