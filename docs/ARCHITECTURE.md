# Architecture

## Overview

`RouteLabs Router` is designed as a middleware layer between applications and inference providers.

The stronger product framing is:

`RouteLabs Router` is a local-first AI runtime with verification-aware escalation and cost visibility.

The long-term system has five major responsibilities:

- normalize requests
- evaluate policy constraints
- inspect runtime availability and performance
- execute verification-aware routing
- emit decision telemetry

The higher-level goals behind those responsibilities are:

- answer locally first when feasible
- verify before escalating
- protect privacy by policy
- explain every routing decision
- measure cost and latency outcomes

## Logical flow

1. An application sends a request to `routerd`.
2. The request is normalized into an internal task envelope.
3. The policy engine checks privacy and routing constraints.
4. The classifier estimates task complexity and execution profile.
5. The router selects a target provider and verification plan.
6. An adapter executes the request.
7. Verification may accept the result or trigger escalation.
8. A decision trace is emitted for observability and benchmarking.

The ideal future flow is:

1. local model answers first
2. verifier scores grounding, confidence, and hallucination risk
3. escalation only happens if the verifier says the answer is weak
4. traces and metrics capture why that happened

## Current implementation

Today the repository supports:

- `/v1/route` for route inspection
- `/v1/chat/completions` for OpenAI-style chat requests
- `/v1/stats` for simple routing telemetry
- `/v1/logs` for recent request-level route logs
- `Ollama` as the first real execution backend
- generic OpenAI-compatible cloud execution
- heuristic verification and escalation traces
- simple estimated cost accounting
- heuristic privacy detection

The current execution behavior is intentionally conservative but now genuinely hybrid:

- local routes execute through `Ollama`
- cloud routes execute through a generic OpenAI-compatible adapter when an API key is configured
- verification can escalate weak local responses to the cloud when configured
- if verification requests escalation but no cloud provider is configured, the local response is returned with a trace explaining why escalation did not happen
- the routing decision is included in the chat response for transparency
- in-memory telemetry tracks local/cloud outcomes and escalation counts
- telemetry also reports simple estimated savings against an all-cloud baseline
- heuristic privacy detection can force local execution for obvious sensitive or code-like content
- recent route logs expose per-request trace data for debugging and trust

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
- `routelabs_router/service.py`
  - chat execution orchestration
- `routelabs_router/adapters`
  - provider adapters

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
- verification-aware escalation over naive complexity-only escalation
- no hidden model switching
- configuration must be understandable by developers
- adapters should remain loosely coupled
- decision traces should be easy to audit
