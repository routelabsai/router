# RouteLabs Router

`RouteLabs Router` is a Python-first, local-first inference control plane for hybrid LLM systems.

It sits between applications and model runtimes, then decides whether a request should run on a local model or a cloud model based on:

- cost
- latency
- task complexity
- privacy policy
- runtime health
- verification signals

The goal is simple: route each step to the cheapest, fastest, safest model that can still be trusted.

## Why this exists

The current ecosystem has strong model runners and strong cloud gateways, but very few tools act as an intelligent middleware layer across both.

Today you can find:

- local runtimes like `Ollama`, `llama.cpp`, `LM Studio`, and `LocalAI`
- training and export stacks like `Unsloth`
- app shells like `AnythingLLM`, `Open WebUI`, and local desktop chat clients
- cloud routing layers like `LiteLLM`, `OpenRouter`, and `Portkey`

What is still missing is a policy-aware scheduler that:

- routes between local and cloud models in real time
- understands privacy and data handling rules
- escalates only when verification fails or confidence is weak
- reasons at the workflow-step level instead of only the full-query level
- adapts to device constraints like memory, latency, and cache pressure

That is the problem this project is designed to solve.

## MVP scope

The first version focuses on a narrow but useful slice:

- OpenAI-compatible chat-style request handling
- local/cloud routing decisions
- adapter-based execution
- verification-aware fallback hooks
- structured telemetry showing why a route was chosen

This repository intentionally starts small. It is a control-plane foundation, not a full chat app.

## Initial architecture

- `routerd`: local HTTP daemon built with `FastAPI`
- `router`: CLI for config inspection and dry-run routing
- `policy`: rule-based engine for privacy, complexity, and fallback decisions
- `adapters`: runtime integrations such as `Ollama`, `llama.cpp`, `LM Studio`, and generic OpenAI-compatible endpoints
- `verify`: response checks that can trigger escalation
- `telemetry`: structured decision logging for benchmarking and future learning

## Current status

This is an early scaffold. The repository already includes:

- project docs
- roadmap
- contribution guide
- Python project metadata
- minimal API server and CLI
- YAML config loading
- basic route decision flow

## Getting started

### Prerequisites

- Python `3.11+`

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run the daemon

```bash
uvicorn routelabs_router.server.app:app --reload
```

### Inspect a routing decision

```bash
router route --task "summarize this document" --private false
```

## Example routing philosophy

- send simple, low-risk tasks to local models first
- prefer local execution when privacy rules require it
- escalate to stronger models when verification or confidence checks fail
- keep the decision trace visible so routing can be audited and improved

## Near-term roadmap

- OpenAI-compatible `/v1/chat/completions` endpoint
- adapters for `Ollama` and generic OpenAI-compatible providers
- policy packs for privacy and cost controls
- task classification and prompt-shape heuristics
- verification hooks and fallback thresholds
- benchmark harness for local vs cloud trade-off analysis

More detail lives in [ROADMAP.md](/Users/saisandeepkantareddy/Downloads/untitled%20folder%202/ROADMAP.md).

## Suggested GitHub setup

- organization: `routelabsai`
- repository: `router`

If those names are taken, alternatives:

- `modelrouter`
- `infrarouter`
- `taskrouter`

## License

This scaffold uses the MIT License. See [LICENSE](/Users/saisandeepkantareddy/Downloads/untitled%20folder%202/LICENSE).
