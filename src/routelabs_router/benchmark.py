from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import yaml

from routelabs_router.config import Config
from routelabs_router.models import RouteRequest
from routelabs_router.router import RouterEngine

_EXPECTED_FIELDS = {
    "target": str,
    "complexity": str,
    "verify": bool,
    "risk_level": str,
}


@dataclass(frozen=True)
class BenchmarkCaseResult:
    name: str
    passed: bool
    mismatches: list[str]
    target: str
    complexity: str
    verify: bool
    risk_level: str


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
    results: list[BenchmarkCaseResult]

    def to_dict(self) -> dict:
        return asdict(self)


def run_policy_benchmark(
    config: Config,
    dataset_path: Path | None = None,
) -> BenchmarkResult:
    raw, dataset_name = _load_dataset(dataset_path)
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark dataset must contain a non-empty 'cases' list")

    engine = RouterEngine(config)
    results: list[BenchmarkCaseResult] = []
    local_routes = 0
    verified_routes = 0
    router_cost = 0.0

    for index, raw_case in enumerate(cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"benchmark case {index} must be a mapping")
        name = str(raw_case.get("name") or f"case-{index}")
        task = raw_case.get("task")
        expected = raw_case.get("expected")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"benchmark case '{name}' field 'task' must be a non-empty string")
        if not isinstance(expected, dict) or not expected:
            raise ValueError(f"benchmark case '{name}' field 'expected' must be a non-empty mapping")

        private = _optional_bool(raw_case.get("private", False), name, "private")
        agent_role = _optional_string(raw_case.get("agent_role"), name, "agent_role")
        tool_choice = _optional_tool_choice(raw_case.get("tool_choice"), name, "tool_choice")
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
        risk_level = (
            decision.agent_tools.risk_level if decision.agent_tools else "none"
        )
        actual = {
            "target": decision.target,
            "complexity": decision.complexity,
            "verify": decision.verify,
            "risk_level": risk_level,
        }
        unsupported = sorted(set(expected) - set(actual))
        if unsupported:
            raise ValueError(
                f"benchmark case '{name}' field 'expected' has unsupported keys: "
                + ", ".join(unsupported)
            )
        mismatches = [
            f"{field}: expected {wanted!r}, got {actual[field]!r}"
            for field, wanted in expected.items()
            if actual[field] != wanted
        ]
        results.append(
            BenchmarkCaseResult(
                name=name,
                passed=not mismatches,
                mismatches=mismatches,
                target=decision.target,
                complexity=decision.complexity,
                verify=decision.verify,
                risk_level=risk_level,
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
        results=results,
    )


def _load_dataset(dataset_path: Path | None) -> tuple[dict, str]:
    if dataset_path is not None:
        if not dataset_path.exists():
            raise ValueError(f"benchmark dataset not found: {dataset_path}")
        raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
        name = str(raw.get("name") or dataset_path.stem)
        return raw, name

    resource = files("routelabs_router.benchmarks").joinpath("policy-routing.yaml")
    raw = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    return raw, str(raw.get("name") or "policy-routing")


def _field_error(case_name: str, field: str, message: str) -> ValueError:
    return ValueError(f"benchmark case '{case_name}' field '{field}' {message}")


def _optional_bool(value: object, case_name: str, field: str) -> bool:
    if not isinstance(value, bool):
        raise _field_error(case_name, field, "must be a boolean")
    return value


def _optional_string(value: object, case_name: str, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _field_error(case_name, field, "must be a string")
    return value


def _optional_tool_choice(value: object, case_name: str, field: str) -> str | dict | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        return value
    raise _field_error(case_name, field, "must be a string or mapping")


def _validate_expectations(expected: dict, case_name: str) -> None:
    for field, wanted in expected.items():
        allowed = _EXPECTED_FIELDS.get(field)
        if allowed is None:
            continue
        if not isinstance(wanted, allowed):
            raise _field_error(
                case_name,
                f"expected.{field}",
                f"must be a {allowed.__name__}",
            )


def _string_list(value: object, case_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _field_error(case_name, "tool_names", "must be a list of strings")
    return value


def _string_mapping(value: object, case_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise _field_error(
            case_name,
            "tool_descriptions",
            "must map strings to strings",
        )
    return value
