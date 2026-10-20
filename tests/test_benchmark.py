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
    assert result.privacy_cases == 0
    assert result.privacy_detection_accuracy is None


def test_packaged_privacy_benchmark_reports_category_metrics() -> None:
    result = run_policy_benchmark(DEFAULT_CONFIG, suite="privacy")

    assert result.dataset == "privacy-policy-synthetic-v1"
    assert result.cases == 15
    assert result.passed == result.cases
    assert result.privacy_cases == result.cases
    assert result.privacy_detection_accuracy == 1.0
    assert result.privacy_category_precision == 1.0
    assert result.privacy_category_recall == 1.0
    assert {
        category.category for category in result.privacy_categories
    } == {
        "account_number",
        "code",
        "explicit_private_flag",
        "private_email",
        "private_identifier",
        "private_phone",
        "secret",
    }


def test_privacy_expectations_report_category_mismatches(tmp_path) -> None:
    dataset = tmp_path / "privacy-mismatch.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "name": "wrong-category",
                        "task": "Draft a reply to alex@example.invalid",
                        "expected": {
                            "privacy_detected": True,
                            "privacy_categories": ["secret"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_policy_benchmark(DEFAULT_CONFIG, dataset)

    assert result.passed == 0
    assert result.privacy_detection_accuracy == 1.0
    assert result.privacy_category_precision == 0.0
    assert result.privacy_category_recall == 0.0
    assert result.results[0].privacy_categories == ["private_email"]
    assert result.results[0].mismatches == [
        "privacy_categories: expected ['secret'], got ['private_email']"
    ]


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
        ({}, {"privacy_detected": "yes"}, "expected.privacy_detected"),
        ({}, {"privacy_categories": [""]}, "expected.privacy_categories"),
        (
            {},
            {"privacy_detected": False, "privacy_categories": ["secret"]},
            "expected.privacy_categories",
        ),
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


def test_benchmark_cli_runs_packaged_privacy_suite(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["router", "benchmark", "--suite", "privacy", "--fail-on-mismatch"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Dataset: privacy-policy-synthetic-v1" in output
    assert "Privacy detection accuracy: 100.0%" in output
    assert "Privacy category recall: 100.0%" in output


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
