# Agent Role Routing

RouteLabs can route a request through a named agent role while keeping the same
OpenAI-compatible API surface.

The `qwen-agent-mesh` profile maps the sketch below onto local Ollama models:

```text
User
  |
Router: qwen3:4b
  |
  +-- planner: gemma3:4b
  +-- coding: devstral:latest
  +-- vision: qwen2.5vl:7b
  |
Reflection: gemma3:4b
```

Create a config from the profile:

```bash
router init --profile qwen-agent-mesh --output ./config/router.yaml
```

Check which role models are ready or missing:

```bash
router doctor --config ./config/router.yaml
```

Preview role routing without starting the server:

```bash
router demo agent-roles --config ./config/router.yaml
router demo agent-roles --config ./config/router.yaml --role coding
```

Run the server with the config:

```bash
router start --config ./config/router.yaml
```

Then choose a lane per request:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "route-auto",
    "agent_role": "coding",
    "messages": [
      {"role": "user", "content": "Write a Python function that validates a route policy."}
    ]
  }'
```

The response `route` and `trace.summary` include the selected `agent_role`, so
logs stay readable when a workflow fans out across planner, coding, vision, and
reflection calls.
