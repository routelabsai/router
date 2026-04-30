# Architecture

## Overview

`RouteLabs Router` is designed as a middleware layer between applications and inference providers.

The long-term system has five major responsibilities:

- normalize requests
- evaluate policy constraints
- inspect runtime availability and performance
- execute verification-aware routing
- emit decision telemetry

## Logical flow

1. An application sends a request to `routerd`.
2. The request is normalized into an internal task envelope.
3. The policy engine checks privacy and routing constraints.
4. The classifier estimates task complexity and execution profile.
5. The router selects a target provider and verification plan.
6. An adapter executes the request.
7. Verification may accept the result or trigger escalation.
8. A decision trace is emitted for observability and benchmarking.

## Near-term implementation shape

- `routelabs_router/server`
  - local daemon
- `routelabs_router/cli.py`
  - development CLI
- `routelabs_router/config.py`
  - config model and loading
- `routelabs_router/router.py`
  - route decision engine
- `routelabs_router/models.py`
  - shared request and decision models

## Planned extension points

- `routelabs_router/adapters`
  - runtime-specific inference backends
- `routelabs_router/verify`
  - response validation and escalation hooks
- `routelabs_router/telemetry`
  - metrics, traces, and benchmark outputs
- `routelabs_router/profile`
  - hardware and runtime state inspection

## Design constraints

- local-first by default
- no hidden model switching
- configuration must be understandable by developers
- adapters should remain loosely coupled
- decision traces should be easy to audit
