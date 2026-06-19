from pathlib import Path

import yaml

from routelabs_router.config import load_config


def test_load_config_deep_merges_nested_values(tmp_path: Path) -> None:
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "local": {
                        "ollama": {
                            "model": "llama3.2:3b",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.providers.local.ollama.model == "llama3.2:3b"
    assert config.providers.local.ollama.base_url == "http://127.0.0.1:11434"
    assert config.providers.local.ollama.requires_api_key is False
    assert config.providers.local.llamacpp.base_url == "http://127.0.0.1:8080/v1"
    assert config.telemetry.cloud_budget_usd is None
    assert config.telemetry.opentelemetry.enabled is False


def test_load_config_reads_cloud_api_key_from_environment(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "cloud": {
                        "openai_compatible": {
                            "api_key_env": "ROUTELABS_TEST_API_KEY",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROUTELABS_TEST_API_KEY", "test-key")

    config = load_config(config_path)

    assert config.providers.cloud.openai_compatible.api_key == "test-key"


def test_load_config_reads_anthropic_api_key_from_environment(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "cloud": {
                        "anthropic": {
                            "api_key_env": "ROUTELABS_ANTHROPIC_TEST_API_KEY",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROUTELABS_ANTHROPIC_TEST_API_KEY", "anthropic-test-key")

    config = load_config(config_path)

    assert config.providers.cloud.anthropic.api_key == "anthropic-test-key"


def test_load_config_deep_merges_tool_policy_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policies": {
                    "tools": {
                        "trusted_tool_patterns": ["mcp__linear__search_*"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.policies.tools.trusted_tool_patterns == ["mcp__linear__search_*"]
    assert "write" in config.policies.tools.approval_required_patterns
    assert "search" in config.policies.tools.review_recommended_patterns


def test_load_config_reads_opentelemetry_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "telemetry": {
                    "opentelemetry": {
                        "enabled": True,
                        "include_task_preview": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.telemetry.opentelemetry.enabled is True
    assert config.telemetry.opentelemetry.include_task_preview is True


def test_load_config_reads_cloud_budget(tmp_path: Path) -> None:
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        yaml.safe_dump({"telemetry": {"cloud_budget_usd": 0.05}}),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.telemetry.cloud_budget_usd == 0.05


def test_load_config_allows_no_key_openai_compatible_proxy(tmp_path: Path) -> None:
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "cloud": {
                        "openai_compatible": {
                            "base_url": "http://127.0.0.1:4000/v1",
                            "requires_api_key": False,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.providers.cloud.openai_compatible.base_url == "http://127.0.0.1:4000/v1"
    assert config.providers.cloud.openai_compatible.requires_api_key is False
    assert config.providers.cloud.openai_compatible.api_key is None
