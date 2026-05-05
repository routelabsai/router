# Release Guide

## Current packaging state

`RouteLabs Router` is package-ready enough for:

- source installs
- GitHub-based `pip install`
- future PyPI publication

It is not automatically published to PyPI yet.

The repository now includes a GitHub Actions publish workflow and is ready for PyPI Trusted Publishing setup.

## Install modes

### Source install

```bash
pip install -e '.[dev]'
router start --reload
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

4. Commit and tag

```bash
git tag v0.1.0
git push origin v0.1.0
```

5. Build distributions

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

6. Publish to PyPI

```bash
python -m twine upload dist/*
```

For the longer-term preferred flow, use GitHub Actions Trusted Publishing instead of local `twine upload`.
See [TRUSTED_PUBLISHING.md](TRUSTED_PUBLISHING.md).

## After PyPI publication

Update the README quick install path to:

```bash
pip install routelabs-router
router start
```
