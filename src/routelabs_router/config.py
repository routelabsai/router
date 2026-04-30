from pathlib import Path

import yaml
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    base_url: str
    model: str


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


class Config(BaseModel):
    server: ServerConfig
    routing: RoutingConfig
    providers: ProvidersConfig
    policies: PoliciesConfig


DEFAULT_CONFIG = Config.model_validate(
    {
        "server": {"host": "127.0.0.1", "port": 8787},
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
                },
                "llamacpp": {
                    "base_url": "http://127.0.0.1:8080",
                    "model": "qwen3-4b-instruct",
                },
            },
            "cloud": {
                "default": "openai-compatible",
                "openai_compatible": {
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4.1-mini",
                },
            },
        },
        "policies": {
            "privacy": {"deny_cloud_when_private": True},
            "complexity": {"local_max": "medium"},
        },
    }
)


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    merged = DEFAULT_CONFIG.model_dump()
    merged.update(raw)
    return Config.model_validate(merged)
