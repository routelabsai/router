from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import yaml

from routelabs_router.config import Config
from routelabs_router.models import RouteRequest
from routelabs_router.router import RouterEngine


@dataclass(frozen=True)
class BenchmarkCaseResult:
    name: str
    passed: bool
    mismatches: list[str]
    target: str
    complexity: str
    verify: bool
    risk_level: str
    privacy_detected: bool
    privacy_categories: list[str]


@dataclass(frozen=True)
class PrivacyCategoryResult:
    category: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None


@dataclass(frozen=True)
class BenchmarkResult:
    dataset: str
    cases: int
    passed: int
    accuracy: float
    local_route_rate: float
    verification_rate: float
    estimated_router_cost_usd: float
    estimated_always_cloud_cost_usd: float
    estimated_savings_vs_cloud_usd: float
    privacy_cases: int
    privacy_detection_accuracy: float | None
    privacy_category_precision: float | None
    privacy_category_recall: float | None
    privacy_categories: list[PrivacyCategoryResult]
    results: list[BenchmarkCaseResult]

    def to_dict(self) -> dict:
        return asdict(self)


def run_policy_benchmark(
    config: Config,
    dataset_path: Path | None = None,
    suite: str = "policy",
) -> BenchmarkResult:
    raw, dataset_name = _load_dataset(dataset_path, suite=suite)
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark dataset must contain a non-empty 'cases' list")

    engine = RouterEngine(config)
    results: list[BenchmarkCaseResult] = []
    local_routes = 0
    verified_routes = 0
    router_cost = 0.0
    privacy_expected: list[set[str]] = []
    privacy_actual: list[set[str]] = []
    privacy_detection_matches = 0

    for index, raw_case in enumerate(cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"benchmark case {index} must be a mapping")
        name = str(raw_case.get("name") or f"case-{index}")
        task = raw_case.get("task")
        expected = raw_case.get("expected")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"benchmark case '{name}' must include a task")
        if not isinstance(expected, dict) or not expected:
            raise ValueError(f"benchmark case '{name}' must include expectations")

        private = raw_case.get("private", False)
        if not isinstance(private, bool):
            _invalid_case_field(name, "private")
        agent_role = raw_case.get("agent_role")
        if agent_role is not None:
            if not isinstance(agent_role, str) or agent_role.strip().lower() not in config.agents.roles:
                _invalid_case_field(name, "agent_role")
        tool_choice = raw_case.get("tool_choice")
        if not _valid_tool_choice(tool_choice):
            _invalid_case_field(name, "tool_choice")
        _validate_expectations(expected, name)

        request = RouteRequest(
            task=task,
            private=private,
            agent_role=agent_role,
            tool_names=_string_list(raw_case.get("tool_names", []), name),
            tool_descriptions=_string_mapping(
                raw_case.get("tool_descriptions", {}), name
            ),
            tool_choice=tool_choice,
        )
        decision = engine.decide(request)
        privacy = engine.policy_engine.evaluate(
            task,
            explicitly_private=private,
        ).privacy
        detected_categories = set(privacy.categories)
        risk_level = (
            decision.agent_tools.risk_level if decision.agent_tools else "none"
        )
        actual = {
            "target": decision.target,
            "complexity": decision.complexity,
            "verify": decision.verify,
            "risk_level": risk_level,
            "privacy_detected": privacy.detected,
            "privacy_categories": sorted(detected_categories),
        }
        mismatches = []
        for field, wanted in expected.items():
            comparable_wanted = (
                sorted(wanted) if field == "privacy_categories" else wanted
            )
            if actual[field] != comparable_wanted:
                mismatches.append(
                    f"{field}: expected {comparable_wanted!r}, got {actual[field]!r}"
                )

        has_privacy_expectation = (
            "privacy_detected" in expected or "privacy_categories" in expected
        )
        if has_privacy_expectation:
            expected_categories = set(expected.get("privacy_categories", []))
            expected_detected = expected.get(
                "privacy_detected", bool(expected_categories)
            )
            privacy_expected.append(expected_categories)
            privacy_actual.append(detected_categories)
            if privacy.detected == expected_detected:
                privacy_detection_matches += 1
        results.append(
            BenchmarkCaseResult(
                name=name,
                passed=not mismatches,
                mismatches=mismatches,
                target=decision.target,
                complexity=decision.complexity,
                verify=decision.verify,
                risk_level=risk_level,
                privacy_detected=privacy.detected,
                privacy_categories=sorted(detected_categories),
            )
        )
        if decision.target == "local":
            local_routes += 1
            router_cost += config.telemetry.costs.local_request_cost_usd
        else:
            router_cost += config.telemetry.costs.cloud_request_cost_usd
        if decision.verify:
            verified_routes += 1

    total = len(results)
    passed = sum(result.passed for result in results)
    always_cloud_cost = total * config.telemetry.costs.cloud_request_cost_usd
    privacy_metrics = _privacy_metrics(privacy_expected, privacy_actual)
    privacy_cases = len(privacy_expected)
    return BenchmarkResult(
        dataset=dataset_name,
        cases=total,
        passed=passed,
        accuracy=passed / total,
        local_route_rate=local_routes / total,
        verification_rate=verified_routes / total,
        estimated_router_cost_usd=router_cost,
        estimated_always_cloud_cost_usd=always_cloud_cost,
        estimated_savings_vs_cloud_usd=always_cloud_cost - router_cost,
        privacy_cases=privacy_cases,
        privacy_detection_accuracy=(
            privacy_detection_matches / privacy_cases if privacy_cases else None
        ),
        privacy_category_precision=privacy_metrics[0],
        privacy_category_recall=privacy_metrics[1],
        privacy_categories=privacy_metrics[2],
        results=results,
    )


def _load_dataset(dataset_path: Path | None, suite: str) -> tuple[dict, str]:
    if dataset_path is not None:
        if not dataset_path.exists():
            raise ValueError(f"benchmark dataset not found: {dataset_path}")
        raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
        name = str(raw.get("name") or dataset_path.stem)
        return raw, name

    packaged_suites = {
        "policy": "policy-routing.yaml",
        "privacy": "privacy-policy.yaml",
    }
    if suite not in packaged_suites:
        raise ValueError(f"unknown packaged benchmark suite: {suite}")
    resource = files("routelabs_router.benchmarks").joinpath(packaged_suites[suite])
    raw = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    return raw, str(raw.get("name") or "policy-routing")


def _invalid_case_field(case_name: str, field_name: str) -> None:
    raise ValueError(f"benchmark case '{case_name}' field '{field_name}' is invalid")


def _valid_tool_choice(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, dict):
        return False

    choice_type = value.get("type")
    if choice_type in {"auto", "none", "required", "any"}:
        return True
    if choice_type == "function":
        function = value.get("function")
        return (
            isinstance(function, dict)
            and isinstance(function.get("name"), str)
            and bool(function["name"].strip())
        )
    if choice_type == "tool":
        name = value.get("name")
        return isinstance(name, str) and bool(name.strip())
    return False


def _validate_expectations(expected: dict, case_name: str) -> None:
    validators = {
        "target": lambda value: isinstance(value, str) and value in {"local", "cloud"},
        "complexity": lambda value: (
            isinstance(value, str) and value in {"low", "medium", "high"}
        ),
        "verify": lambda value: isinstance(value, bool),
        "risk_level": lambda value: (
            isinstance(value, str) and value in {"none", "low", "medium", "high"}
        ),
        "privacy_detected": lambda value: isinstance(value, bool),
        "privacy_categories": lambda value: (
            isinstance(value, list)
            and all(isinstance(item, str) and bool(item.strip()) for item in value)
            and len(value) == len(set(value))
        ),
    }
    for field, value in expected.items():
        validator = validators.get(field)
        if validator is None or not validator(value):
            _invalid_case_field(case_name, f"expected.{field}")

    if (
        expected.get("privacy_detected") is False
        and expected.get("privacy_categories")
    ):
        _invalid_case_field(case_name, "expected.privacy_categories")


def _privacy_metrics(
    expected_cases: list[set[str]],
    actual_cases: list[set[str]],
) -> tuple[float | None, float | None, list[PrivacyCategoryResult]]:
    categories = (
        sorted(set().union(*expected_cases, *actual_cases))
        if expected_cases
        else []
    )
    results: list[PrivacyCategoryResult] = []
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for category in categories:
        true_positives = sum(
            category in expected and category in actual
            for expected, actual in zip(expected_cases, actual_cases)
        )
        false_positives = sum(
            category not in expected and category in actual
            for expected, actual in zip(expected_cases, actual_cases)
        )
        false_negatives = sum(
            category in expected and category not in actual
            for expected, actual in zip(expected_cases, actual_cases)
        )
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        results.append(
            PrivacyCategoryResult(
                category=category,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                precision=_ratio_or_none(
                    true_positives, true_positives + false_positives
                ),
                recall=_ratio_or_none(
                    true_positives, true_positives + false_negatives
                ),
            )
        )

    precision = _ratio_or_none(
        total_true_positives,
        total_true_positives + total_false_positives,
    )
    recall = _ratio_or_none(
        total_true_positives,
        total_true_positives + total_false_negatives,
    )
    return precision, recall, results


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _string_list(value: object, case_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"benchmark case '{case_name}' tool_names must be strings")
    return value


def _string_mapping(value: object, case_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(
            f"benchmark case '{case_name}' tool_descriptions must map strings to strings"
        )
    return value
