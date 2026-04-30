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
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
uvicorn routelabs_router.server.app:app --reload
router route --task "classify this email"
```

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
