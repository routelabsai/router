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

For new features or significant changes, open an issue first so we can align on scope.

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

- runtime adapters
- benchmarking harnesses
- verification strategies
- policy design
- observability and tracing
- SDKs and integrations

## Code style

- prefer simple interfaces
- keep adapter boundaries explicit
- optimize for readability before cleverness
- avoid hiding routing decisions in magic behavior
