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
