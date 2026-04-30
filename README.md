# RouteLabs Router

`RouteLabs Router` is a Python-first, local-first inference control plane for hybrid LLM systems.

It gives applications one endpoint that can decide:

- when to stay local
- when to use the cloud
- when privacy should override convenience
- which provider and model should handle the request
- why that decision was made

The goal is simple: route each step to the cheapest, fastest, safest model that can still be trusted.

## Why Use This

Most teams today have one of these problems:

- `Ollama` runs local models well, but it does not decide when a task should stay local versus escalate
- cloud gateways like `LiteLLM` and `OpenRouter` route across hosted APIs, but they are not built around local-first policy decisions
- chat apps can call models, but they usually hide the execution logic instead of exposing it

`RouteLabs Router` is the layer above those tools.

It is for teams who want:

- one API for hybrid local + cloud inference
- transparent routing decisions
- privacy-aware defaults
- provider and model selection that can evolve over time
- a foundation for agentic step-level routing later

## What It Looks Like

```text
app / agent / extension
        |
        v
  RouteLabs Router
        |
        +--> policy + task complexity
        +--> privacy constraints
        +--> provider selection
        +--> verification hooks
        |
        +--> Ollama
        +--> llama.cpp
        +--> cloud provider
```

## Quick Demo

Once the server is running, you can inspect decisions directly:

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task":"summarize a short product description","private":false}'
```

Expected shape:

```json
{
  "target": "local",
  "provider": "ollama",
  "model": "qwen3:4b",
  "reason": "task is suitable for local-first execution",
  "complexity": "medium",
  "verify": true
}
```

And you can send an OpenAI-style chat request:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Summarize this in one sentence: RouteLabs Router chooses between local and cloud models based on privacy, cost, latency, and task complexity."}],
    "private":false
  }'
```

If `Ollama` is running locally, that request executes against your configured local model.

It sits between applications and model runtimes, then decides whether a request should run on a local model or a cloud model based on:

- cost
- latency
- task complexity
- privacy policy
- runtime health
- verification signals

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

## Positioning

| Tool | Core strength | What it does not solve |
| --- | --- | --- |
| `Ollama` | Great local model runtime and API | Hybrid routing and policy decisions |
| `LiteLLM` | Cloud API normalization and routing | Local-first execution strategy |
| `OpenRouter` | Hosted provider access and fallback | On-device privacy-aware control plane |
| `RouteLabs Router` | Policy-aware hybrid local/cloud routing | Full cloud adapter coverage is still in progress |

## MVP scope

The first version focuses on a narrow but useful slice:

- OpenAI-compatible chat-style request handling
- local/cloud routing decisions
- adapter-based execution
- verification-aware fallback hooks
- structured telemetry showing why a route was chosen

This repository intentionally starts small. It is a control-plane foundation, not a full chat app.

## Use Cases

- Local-first copilots that should only escalate when a task gets difficult
- Privacy-sensitive workflows where private data should never leave the device
- Browser or desktop assistants that need one middleware layer above multiple runtimes
- Agent systems that want future step-level routing instead of a single fixed model

## Initial architecture

- `routerd`: local HTTP daemon built with `FastAPI`
- `router`: CLI for config inspection and dry-run routing
- `policy`: rule-based engine for privacy, complexity, and fallback decisions
- `adapters`: runtime integrations such as `Ollama`, `llama.cpp`, `LM Studio`, and generic OpenAI-compatible endpoints
- `verify`: response checks that can trigger escalation
- `telemetry`: structured decision logging for benchmarking and future learning

## Current status

This is an early but working scaffold. The repository already includes:

- project docs
- roadmap
- contribution guide
- Python project metadata
- `FastAPI` server and CLI
- YAML config loading
- route inspection endpoint
- OpenAI-style `/v1/chat/completions` endpoint
- real local execution through `Ollama`
- test coverage for routing and API behavior
- example config profiles
- example curl flows

## Getting started

### Prerequisites

- Python `3.11+`
- `conda` recommended for the smoothest setup on macOS

### Create the environment

```bash
conda create -n routelabs-router python=3.11 -y
conda activate routelabs-router
python -m pip install --upgrade pip setuptools wheel
pip install -e '.[dev]'
```

### Why `conda` is the recommended path

During validation we hit two common issues that `conda` + Python `3.11` resolved cleanly:

- Python `3.9.7` was too old for this project
- older packaging tooling made editable installs unreliable

If you see `requires a different Python: 3.9.7 not in '>=3.11'`, create the `conda` environment above and retry.

### Run tests

```bash
pytest
```

### Optional profile configs

The repo includes starter profiles in [`config/profiles/`](config/profiles):

- `balanced.yaml`
- `local-first.yaml`
- `privacy-first.yaml`

Use one as your active config by copying or merging it into [`config/router.yaml`](config/router.yaml).

### Run the daemon

```bash
uvicorn routelabs_router.server.app:app --reload
```

### Inspect a routing decision

```bash
router route --task "summarize a short product description" --private false
```

### Test the API

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Route inspection:

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task":"summarize a short product description","private":false}'
```

OpenAI-style chat completion:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Summarize this in one sentence: RouteLabs Router chooses between local and cloud models based on privacy, cost, latency, and task complexity."}],
    "private":false
  }'
```

If `Ollama` is running locally, the chat endpoint will execute against your configured local model. If the router decides a task should go to the cloud, the API currently returns `501` until the first cloud adapter is added.

### Run with Ollama

Start `Ollama`, make sure the configured model exists, then run the server:

```bash
ollama serve
ollama pull qwen3:4b
uvicorn routelabs_router.server.app:app --reload
```

The default local provider configuration lives in [`config/router.yaml`](config/router.yaml).

### More examples

- curl walkthrough: [`examples/curl-quickstart.md`](examples/curl-quickstart.md)
- product framing and common scenarios: [`examples/use-cases.md`](examples/use-cases.md)

## Example routing philosophy

- send simple, low-risk tasks to local models first
- prefer local execution when privacy rules require it
- escalate to stronger models when verification or confidence checks fail
- keep the decision trace visible so routing can be audited and improved

## Near-term roadmap

- generic OpenAI-compatible cloud adapter
- policy packs for privacy and cost controls
- task classification and prompt-shape heuristics
- verification hooks and fallback thresholds
- benchmark harness for local vs cloud trade-off analysis

More detail lives in [ROADMAP.md](ROADMAP.md).

## License

This scaffold uses the MIT License. See [LICENSE](LICENSE).
