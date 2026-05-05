# Trusted Publishing Setup

This repository supports PyPI Trusted Publishing through GitHub Actions.

## Why this is the preferred release path

Trusted Publishing is better than long-lived API tokens because:

- no PyPI token needs to live in local shells
- no PyPI token needs to be stored in GitHub secrets
- GitHub Actions exchanges a short-lived OIDC identity token with PyPI
- future releases become a tag push or workflow dispatch

## One-time PyPI setup

Before the workflow can publish, configure a Trusted Publisher in PyPI.

On PyPI:

1. Create the project if it does not already exist, or open the existing project.
2. Open the project's `Publishing` settings.
3. Add a GitHub Actions trusted publisher with:
   - owner: `routelabsai`
   - repository: `router`
   - workflow name: `publish.yml`
   - environment name: `pypi`

Official docs:

- PyPI Trusted Publishers overview: https://docs.pypi.org/trusted-publishers/
- Adding a trusted publisher: https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- Publishing with GitHub Actions: https://docs.pypi.org/trusted-publishers/using-a-publisher/

## What the workflow does

The GitHub Actions workflow lives at:

- `.github/workflows/publish.yml`

It:

1. builds `sdist` and `wheel`
2. uploads those build artifacts between jobs
3. publishes to PyPI using `pypa/gh-action-pypi-publish@release/v1`

## How to release

### Option 1: tag push

```bash
git tag v0.1.0
git push origin v0.1.0
```

### Option 2: manual workflow dispatch

Use the GitHub Actions UI to run `Publish to PyPI`.

## Recommended release sequence

1. Bump version in `pyproject.toml`
2. Commit the version bump
3. Push to `main`
4. Create and push a version tag
5. Watch the `Publish to PyPI` workflow
6. Verify `pip install routelabs-router`

## After first successful publish

Update the README primary install path to:

```bash
pip install routelabs-router
router start
```
