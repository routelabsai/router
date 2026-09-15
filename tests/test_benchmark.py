import json

import pytest
import yaml

from routelabs_router import cli
from routelabs_router.benchmark import run_policy_benchmark
from routelabs_router.config import DEFAULT_CONFIG


def test_packaged_policy_benchmark_passes() -> None:
    result = run_policy_benchmark(DEFAULT_CONFIG)

    assert result.dataset == "route-policy-smoke-v1"
    assert result.cases == 8
    assert result.passed == result.cases
    assert result.accuracy == 1.0
    assert result.local_route_rate == 1.0
    assert result.verification_rate == 0.75
    assert result.estimated_router_cost_usd == pytest.approx(0.0016)
    assert result.estimated_always_cloud_cost_usd == pytest.approx(0.16)


def test_custom_benchmark_reports_mismatches(tmp_path) -> None:
    dataset = tmp_path / "custom.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "name": "intentional-mismatch",
                "cases": [
                    {
                        "name": "wrong-complexity",
                        "task": "Write hello",
                        "expected": {"complexity": "high"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_policy_benchmark(DEFAULT_CONFIG, dataset)

    assert result.passed == 0
    assert result.results[0].mismatches == [
        "complexity: expected 'high', got 'low'"
    ]


def test_benchmark_cli_prints_machine_readable_results(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "router",
            "benchmark",
            "--config",
            "./config/router.yaml",
            "--json",
        ],
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    assert result["dataset"] == "route-policy-smoke-v1"
    assert result["passed"] == result["cases"]
