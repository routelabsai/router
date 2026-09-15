# Contributing

Thanks for helping build `RouteLabs Router`.

## Project goals

This project is trying to make hybrid inference systems more practical, transparent, and local-first. We care about:

- clear routing decisions
- reproducible behavior
- privacy-aware defaults
- pragmatic performance engineering

## How to contribute

### 1. Start with an issue

For new features or significant changes, use the relevant issue form first so
we can align on scope. Small fixes with a focused test can go directly to a
pull request.

### 2. Keep changes focused

Small, reviewable pull requests are much easier to merge than broad refactors.

### 3. Preserve decision visibility

If you change routing behavior, make sure the explanation path stays visible in logs or traces.

### 4. Add tests when behavior changes

Routing policy, verification, and fallback logic should be backed by tests whenever practical.

## Development

```bash
conda create -n routelabs-router python=3.11 -y
conda activate routelabs-router
python -m pip install --upgrade pip setuptools wheel
pip install -e '.[dev]'
pytest
router start --reload
router route --task "classify this email"
```

## Environment notes

- use Python `3.11+`
- prefer the `conda` environment above over a system Python on macOS
- if `fastapi.testclient` complains about missing `httpx`, reinstall with `pip install -e '.[dev]'`
- if editable install fails under an older environment, check `python --version` first

## Areas where help is especially welcome

These tracks are deliberately scoped so a contributor can find a concrete
starting point:

- **Benchmark datasets:** add labeled policy cases for coding-agent, privacy,
  and tool-risk workloads under `src/routelabs_router/benchmarks/`. Run them
  with `router benchmark --dataset <path>`.
- **Runtime adapters:** extend the explicit adapter boundary under
  `src/routelabs_router/adapters/` and add fake-provider tests before requiring
  live credentials.
- **Verification:** add verifier strategies behind the interface in
  `src/routelabs_router/verify.py`, including examples that demonstrate when
  escalation should and should not happen.
- **Policy safety:** improve privacy, complexity, and tool-risk detection while
  including both positive and false-positive regression cases.
- **Integrations:** add minimal, executable examples for coding agents and
  OpenAI- or Anthropic-compatible clients under `examples/`.
- **Observability:** improve route traces and metrics without placing prompts,
  credentials, or private content in telemetry by default.

## Pull request checklist

- explain the user-visible behavior and why it belongs in RouteLabs
- add or update tests for behavior changes
- run `pytest`
- run `python scripts/release_smoke.py` when changing packaging, profiles, CLI,
  or installed behavior
- update the changelog for user-visible changes
- keep secrets, production prompts, and customer data out of fixtures and logs
- preserve an inspectable explanation when changing routing decisions

Every pull request runs CI on Python 3.11 and 3.12, plus an installed-wheel
smoke test.

## Code style

- prefer simple interfaces
- keep adapter boundaries explicit
- optimize for readability before cleverness
- avoid hiding routing decisions in magic behavior
