# Roadmap

## Vision

Build an open-source inference control plane that can route agent and application workloads across local and cloud models using explicit policy, runtime telemetry, and verification-aware escalation.

Framed more directly:

`RouteLabs Router` should become a local-first AI runtime with verification-aware escalation and cost visibility.

## Principles

- local-first when feasible
- verify before escalating
- cloud when justified
- privacy is a routing constraint, not a suggestion
- verification is part of execution, not an afterthought
- decisions should be inspectable and benchmarkable
- cost and latency outcomes should be measurable

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
- first verification-aware escalation loop implemented
- initial stats endpoint implemented for local/cloud/escalation visibility
- simple estimated cost savings implemented

## Phase 2: Verification and evaluation

- verification-first routing loop:
  - local answer
  - verifier checks grounding, confidence, and hallucination risk
  - escalate only if needed
- schema validation hooks
- citation or grounding checks for retrieval tasks
- correctness sampling and self-check prompts
- confidence thresholds per route profile
- benchmarking harness across tasks, models, and hardware classes

## Phase 3: Cost, latency, and runtime intelligence

- cost and latency dashboard primitives
- local vs cloud savings metrics
- escalation rate metrics
- device profiling for CPU, RAM, GPU, and queue pressure
- quantization-aware model selection
- KV-cache and context-window-aware scheduling
- adaptive routing based on observed latency and failure rates
- routing memory tuned per machine profile

## Phase 4: Privacy and agent workflow routing

- policy-driven privacy routing:
  - PII-sensitive tasks stay local
  - codebase and internal-document tasks prefer local
  - generic tasks may use cloud when justified
- step-level policies for:
  - retrieval
  - planning
  - tool use
  - verification
  - final synthesis
- MCP and tool-aware routing
- policy packs for enterprise privacy and compliance

## Phase 5: Learning router and product surfaces

- learn from past escalations
- learn from user corrections
- tune routing policies from observed outcomes

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
