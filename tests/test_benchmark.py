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


def test_benchmark_accepts_named_tool_choice(tmp_path) -> None:
    dataset = tmp_path / "named-tool.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "name": "named-tool",
                "cases": [
                    {
                        "name": "forced-tool",
                        "task": "Use the lookup tool",
                        "tool_choice": "mcp__tickets__lookup",
                        "expected": {"target": "local"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_policy_benchmark(DEFAULT_CONFIG, dataset)

    assert result.cases == 1


@pytest.mark.parametrize(
    ("case_update", "expected_update", "field_name"),
    [
        ({"private": "false"}, {}, "private"),
        ({"agent_role": "unknown-role"}, {}, "agent_role"),
        ({"tool_choice": "   "}, {}, "tool_choice"),
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


@pytest.mark.parametrize("as_json", [False, True])
def test_benchmark_cli_fails_after_printing_mismatches(
    monkeypatch, capsys, tmp_path, as_json
) -> None:
    dataset = tmp_path / "mismatch.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "name": "mismatch",
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
    arguments = ["router", "benchmark", "--dataset", str(dataset)]
    if as_json:
        arguments.append("--json")
    monkeypatch.setattr("sys.argv", [*arguments, "--fail-on-mismatch"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    if as_json:
        result = json.loads(output)
        assert result["passed"] == 0
        assert result["results"][0]["mismatches"]
    else:
        assert "Cases passed: 0/1" in output
        assert "wrong-complexity" in output


def test_benchmark_cli_keeps_default_success_exit_for_mismatches(
    monkeypatch, capsys, tmp_path
) -> None:
    dataset = tmp_path / "mismatch.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
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
    monkeypatch.setattr(
        "sys.argv", ["router", "benchmark", "--dataset", str(dataset)]
    )

    cli.main()

    assert "Cases passed: 0/1" in capsys.readouterr().out


def test_benchmark_cli_succeeds_with_flag_when_all_cases_match(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["router", "benchmark", "--fail-on-mismatch"]
    )

    cli.main()

    assert "Cases passed: 8/8" in capsys.readouterr().out
