# Roadmap

## Vision

Build an open-source inference control plane that can route agent and application workloads across local and cloud models using explicit policy, runtime telemetry, and verification-aware escalation.

## Principles

- local-first when feasible
- cloud when justified
- privacy is a routing constraint, not a suggestion
- verification is part of execution, not an afterthought
- decisions should be inspectable and benchmarkable

## Phase 0: Foundation

- establish repository structure
- define configuration model
- create daemon and CLI entry points
- add decision tracing primitives
- document architecture and contribution flow

## Phase 1: Core routing MVP

- support an OpenAI-style request envelope internally
- add rule-based task classification
- add `Ollama` adapter
- add generic OpenAI-compatible cloud adapter
- route based on:
  - privacy requirement
  - estimated complexity
  - local runtime availability
  - configured cost and latency preferences
- add fallback from local to cloud when verification fails

Current status:
- route inspection endpoint implemented
- `Ollama` local chat execution implemented
- OpenAI-style `/v1/chat/completions` endpoint implemented
- generic OpenAI-compatible cloud execution implemented

## Phase 2: Verification and evaluation

- schema validation hooks
- citation or grounding checks for retrieval tasks
- correctness sampling and self-check prompts
- confidence thresholds per route profile
- benchmarking harness across tasks, models, and hardware classes

## Phase 3: Runtime intelligence

- device profiling for CPU, RAM, GPU, and queue pressure
- quantization-aware model selection
- KV-cache and context-window-aware scheduling
- adaptive routing based on observed latency and failure rates
- routing memory tuned per machine profile

## Phase 4: Agent workflow routing

- step-level policies for:
  - retrieval
  - planning
  - tool use
  - verification
  - final synthesis
- MCP and tool-aware routing
- policy packs for enterprise privacy and compliance

## Phase 5: Product surfaces

- TypeScript SDK
- Python SDK
- desktop control panel
- IDE integration
- browser extension
- mobile companion

## Open questions

- how much routing should remain rule-based vs learned
- how to benchmark trustworthiness without overfitting to a benchmark suite
- how to represent privacy policy in a portable, developer-friendly config format
- how to support heterogeneous local runtimes without leaky abstractions
