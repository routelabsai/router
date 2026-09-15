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


@pytest.mark.parametrize(
    ("case_update", "expected_update", "field_name"),
    [
        ({"private": "false"}, {}, "private"),
        ({"agent_role": "unknown-role"}, {}, "agent_role"),
        ({"tool_choice": "sometimes"}, {}, "tool_choice"),
        ({}, {"target": "edge"}, "expected.target"),
        ({}, {"complexity": "extreme"}, "expected.complexity"),
        ({}, {"verify": "yes"}, "expected.verify"),
        ({}, {"risk_level": "critical"}, "expected.risk_level"),
    ],
)
def test_benchmark_rejects_invalid_case_values_without_exposing_fixture_content(
    tmp_path, case_update, expected_update, field_name
) -> None:
    dataset = tmp_path / "malformed.yaml"
    case = {
        "name": "malformed-case",
        "task": "PRIVATE TASK SENTINEL",
        "tool_names": ["mcp__tickets__lookup"],
        "tool_descriptions": {"mcp__tickets__lookup": "PRIVATE TOOL SENTINEL"},
        "expected": {"target": "local"},
    }
    case.update(case_update)
    case["expected"].update(expected_update)
    dataset.write_text(
        yaml.safe_dump({"name": "malformed", "cases": [case]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        run_policy_benchmark(DEFAULT_CONFIG, dataset)

    message = str(exc_info.value)
    assert "malformed-case" in message
    assert field_name in message
    assert "PRIVATE TASK SENTINEL" not in message
    assert "PRIVATE TOOL SENTINEL" not in message


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
