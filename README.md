# RouteLabs Router

`RouteLabs Router` is a local-first runtime that sits between your app and local/cloud LLMs.

It gives applications one endpoint that can decide:

- when to stay local
- when to use the cloud
- when privacy should override convenience
- which provider and model should handle the request
- why that decision was made
- when verification forced an escalation
- when privacy detection forced local execution

The goal is simple: route each step to the cheapest, fastest, safest model that can still be trusted.

## Who This Is For

This repo is mainly for:

- AI app builders
- local-first power users
- agent and workflow developers
- teams experimenting with privacy-aware and cost-aware inference

If you want a polished end-user chat app, this is not that.
If you want a runtime and routing layer you can plug into your own tools, this is exactly that.

## What This Is

Think of RouteLabs as:

- a local runtime/server you run on your machine
- a Python client you can call from your app
- an OpenAI-compatible endpoint you can place in front of existing clients

It is not primarily:

- a browser extension
- a desktop UI
- a plugin marketplace product

Those may come later, but the current product is a runtime + middleware + API.

## 60-Second Quickstart

Install from PyPI and start the local runtime:

```bash
pip install routelabs-router
router start --reload
```

Then in another terminal:

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task":"summarize a short product description","private":false}'
```

If you want cloud execution too:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Install

### Recommended user install

```bash
pip install routelabs-router
router start
```

### Contributor install

Clone the repo and install from source:

```bash
git clone https://github.com/routelabsai/router.git
cd router
conda create -n routelabs-router python=3.11 -y
conda activate routelabs-router
python -m pip install --upgrade pip setuptools wheel
pip install -e '.[dev]'
router start --reload
```

## Why Use This

Most teams today have one of these problems:

- `Ollama` runs local models well, but it does not decide when a task should stay local versus escalate
- cloud gateways like `LiteLLM` and `OpenRouter` route across hosted APIs, but they are not built around local-first policy decisions
- chat apps can call models, but they usually hide the execution logic instead of exposing it

`RouteLabs Router` is the layer above those tools.

It is for teams who want:

- one API for hybrid local + cloud inference
- verification-aware escalation instead of naive “hard task -> expensive model”
- transparent routing decisions
- privacy-aware defaults
- automatic local preference for obvious sensitive or code-like content
- cost and latency visibility
- provider and model selection that can evolve over time
- a foundation for agentic step-level routing later

For the longer-term product thesis, see [docs/VISION.md](docs/VISION.md).

## How You Use It

There are three practical ways to adopt RouteLabs today.

### 1. As a local runtime/server

Run:

```bash
router start --reload
```

Then point your tools to `http://127.0.0.1:8000`.

### 2. As a Python library client

Use the built-in client:

```python
from routelabs_router import RouteLabsClient

client = RouteLabsClient("http://127.0.0.1:8000")
print(client.route("Summarize a short product description"))
```

### 3. As an OpenAI-compatible endpoint

If you already have code using an OpenAI-style client, point it at RouteLabs via `base_url`.

That is one of the easiest ways to adopt it without rewriting your app.

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

What this tells you:

- the router chose `local`
- it selected `ollama`
- it picked a model
- it marked the request as worth verification

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
If `OPENAI_API_KEY` is set, high-complexity requests can route through the configured OpenAI-compatible cloud provider.
The response includes a trace showing the initial route, verification result, and any escalation.

## Positioning

| Tool | Core strength | What it does not solve |
| --- | --- | --- |
| `Ollama` | Great local model runtime and API | Hybrid routing and policy decisions |
| `LiteLLM` | Cloud API normalization and routing | Local-first execution strategy |
| `OpenRouter` | Hosted provider access and fallback | On-device privacy-aware control plane |
| `RouteLabs Router` | Verification-aware local-first runtime with hybrid routing | Early-stage policy and provider coverage |

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

## Current Status

This is an early but usable product foundation. The repository already includes:

- project docs
- roadmap
- contribution guide
- Python project metadata
- `FastAPI` server and CLI
- YAML config loading
- route inspection endpoint
- OpenAI-style `/v1/chat/completions` endpoint
- real local execution through `Ollama`
- generic OpenAI-compatible cloud execution
- first verification-aware escalation loop
- stats endpoint for local/cloud/escalation visibility
- simple estimated cost savings in stats
- heuristic privacy detection for email/identifier/code-like content
- recent route logs for per-request inspection
- test coverage for routing and API behavior
- example config profiles
- example curl flows

Still early:

- verifiers are heuristic and still early
- cost and latency dashboards are not implemented yet
- privacy detection is heuristic rather than model-based
- learning from user corrections is still future work

## More Docs

- Product vision: [docs/VISION.md](docs/VISION.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)
- Contributor guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Release guide: [docs/RELEASE.md](docs/RELEASE.md)
- PyPI trusted publishing: [docs/TRUSTED_PUBLISHING.md](docs/TRUSTED_PUBLISHING.md)

## Setup And Usage

### Prerequisites

- Python `3.11+`
- `conda` recommended for the smoothest setup on macOS

### Install from PyPI

```bash
conda create -n routelabs-router python=3.11 -y
conda activate routelabs-router
python -m pip install --upgrade pip
pip install routelabs-router
```

### Install from source

Use this path if you want to contribute or modify the router itself.

```bash
git clone https://github.com/routelabsai/router.git
cd router
conda create -n routelabs-router python=3.11 -y
conda activate routelabs-router
python -m pip install --upgrade pip setuptools wheel
pip install -e '.[dev]'
```

### Configure cloud execution

If you want cloud-routed requests to execute instead of returning a configuration error, set:

```bash
export OPENAI_API_KEY=your_api_key_here
```

The default cloud adapter uses the OpenAI-compatible endpoint configured in [`config/router.yaml`](config/router.yaml).

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

### Start the runtime

```bash
router start --reload
```

For explicit host or port overrides:

```bash
router start --host 0.0.0.0 --port 8000 --reload
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

Stats endpoint:

```bash
curl http://127.0.0.1:8000/v1/stats
```

Recent route logs:

```bash
curl http://127.0.0.1:8000/v1/logs
```

### Python client

You can also call the router from Python:

```python
from routelabs_router import RouteLabsClient

client = RouteLabsClient("http://127.0.0.1:8000")

route = client.route("Summarize a short product description")
chat = client.chat(
    [
        {
            "role": "user",
            "content": "Summarize this in one sentence: RouteLabs Router chooses between local and cloud models based on privacy, cost, latency, and task complexity.",
        }
    ]
)
stats = client.stats()
logs = client.logs()
```

There is also a runnable example in [`examples/python-client.py`](examples/python-client.py).

### OpenAI-compatible drop-in example

If you already use the OpenAI Python SDK, you can point it at RouteLabs:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed-for-local-dev",
)

response = client.chat.completions.create(
    model="route-auto",
    messages=[
        {
            "role": "user",
            "content": "Summarize this in one sentence: RouteLabs Router chooses between local and cloud models based on privacy, cost, latency, and task complexity.",
        }
    ],
)
```

See [`examples/openai-compatible-client.py`](examples/openai-compatible-client.py).
You may need to install the OpenAI SDK separately:

```bash
pip install openai
```

The stats response includes simple estimated fields such as:

- `estimated_total_cost_usd`
- `estimated_baseline_cloud_cost_usd`
- `estimated_cost_saved_usd`
- `estimated_cloud_requests_avoided`

OpenAI-style chat completion:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Summarize this in one sentence: RouteLabs Router chooses between local and cloud models based on privacy, cost, latency, and task complexity."}],
    "private":false
  }'
```

If `Ollama` is running locally, the chat endpoint will execute against your configured local model.
If `OPENAI_API_KEY` is set, high-complexity tasks can execute through the configured OpenAI-compatible cloud provider. If it is not set, cloud-routed chat requests return a clear configuration error.
The stats endpoint gives a simple first pass at the eventual cost/latency visibility story by showing how many requests stayed local, how many escalated, and how often verification failed.
It also includes a lightweight savings estimate based on configurable per-request local and cloud cost assumptions.
The logs endpoint exposes recent request-level decisions so users can inspect privacy detection, verification, escalation, final route choice, and estimated per-request cost directly.

### Privacy-aware behavior

The router can now automatically prefer local execution for requests that look like:

- emails or phone-like identifiers
- SSN-like or account-like identifiers
- secret-like tokens
- code-like content

This first version uses lightweight heuristics so it is easy to run locally.
For a more advanced future detector, the project can integrate a model such as `openai/privacy-filter`.

### Run with Ollama

Start `Ollama`, make sure the configured model exists, then run the server:

```bash
ollama serve
ollama pull qwen3:4b
router start --reload
```

The default local provider configuration lives in [`config/router.yaml`](config/router.yaml).

### Hybrid mode example

With both `Ollama` and `OPENAI_API_KEY` configured:

- simple tasks usually run locally
- private tasks prefer local execution
- high-complexity tasks can route to the cloud

Example cloud-leaning route check:

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task":"design architecture for a multi-step agent","private":false}'
```

### More examples

- curl walkthrough: [`examples/curl-quickstart.md`](examples/curl-quickstart.md)
- product framing and common scenarios: [`examples/use-cases.md`](examples/use-cases.md)

## Example Routing Philosophy

- send simple, low-risk tasks to local models first
- prefer local execution when privacy rules require it
- escalate to stronger models when verification or confidence checks fail
- keep the decision trace visible so routing can be audited and improved

## Near-Term Roadmap

- richer verification strategies beyond heuristics
- policy packs for privacy and cost controls
- better task classification and prompt-shape heuristics
- latency-aware telemetry and routing feedback loops
- benchmark harness for local vs cloud trade-off analysis

More detail lives in [ROADMAP.md](ROADMAP.md).

## License

This scaffold uses the MIT License. See [LICENSE](LICENSE).
