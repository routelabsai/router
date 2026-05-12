# Release Guide

## Current packaging state

`RouteLabs Router` is package-ready enough for:

- PyPI installs
- source installs
- GitHub-based `pip install`
- GitHub Actions releases through Trusted Publishing

The repository includes a GitHub Actions publish workflow using PyPI Trusted Publishing.

## Install modes

### Source install

```bash
pip install -e '.[dev]'
router start --reload
```

### PyPI install

```bash
pip install routelabs-router
router start
```

### GitHub install

```bash
pip install git+https://github.com/routelabsai/router.git
router start
```

## Suggested release checklist

1. Run tests

```bash
pytest
```

2. Verify normal non-editable install

```bash
pip install .
router --help
router start --help
```

3. Bump version in `pyproject.toml`

4. Commit and push the version bump

```bash
git add .
git commit -m "Release vX.Y.Z"
git push origin main
```

5. Build distributions locally as a sanity check

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

6. Create and push the release tag to trigger GitHub Actions publishing

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

7. Verify the published package from a fresh environment

```bash
pip install routelabs-router
router --help
router start --help
```

For the preferred release flow, use GitHub Actions Trusted Publishing.
See [Trusted Publishing Setup](trusted-publishing.md).
