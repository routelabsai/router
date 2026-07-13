# Changelog

## Unreleased

Highlights:

- configurable `agent_role` routing for router, planner, coding, vision, and reflection lanes
- `qwen-agent-mesh` profile for Qwen3, Gemma, Devstral, and Qwen VL local agent meshes
- `router demo agent-roles` for previewing role-aware routing without starting a server
- `router profiles` for listing starter profiles available to `router init`
- `router route --agent-role` for inspecting role-specific route decisions
- `router init --profile qwen-agent-mesh` for scaffolding the agent mesh profile
- `router init --profile` choices are now discovered from the packaged profile set
- starter profiles are now packaged with the CLI so `router init --profile ...` works outside a source checkout
- release tests now verify packaged starter profiles match the source profile files
- installed CLI and ASGI imports no longer require `./config/router.yaml` to exist for non-server commands
- `scripts/release_smoke.py` builds a wheel and smoke-tests the installed CLI from a temp environment
- PyPI publish workflow now runs tests and the installed-wheel smoke before building publish artifacts
- `router doctor`, startup warnings, and `/v1/models` now include configured agent role models
- Python client helpers now accept `agent_role` for route, chat, responses, and messages calls
- agent role traces are preserved through verification escalation and provider fallback
- packaging metadata now uses modern SPDX license fields

Suggested preview test:

```bash
router init --profile qwen-agent-mesh --output ./config/router.yaml
router doctor --config ./config/router.yaml
router demo agent-roles --config ./config/router.yaml
router route --config ./config/router.yaml --task "Implement a parser fix" --agent-role coding
```

## 0.4.0

This release expands RouteLabs Router from an Ollama-first gateway into a more
practical local-first control plane for OpenAI-compatible local runtimes,
agent-tool routing, cost guardrails, and trace visibility.

Highlights:

- local OpenAI-compatible runtime support for `llama.cpp`, LM Studio, vLLM, and similar `/v1` servers
- `llamacpp-local`, `lmstudio-local`, and `litellm-proxy` starter config profiles
- `router recommend local-model` for machine-aware Ollama model recommendations
- richer `router route` output with provider readiness, fallback status, MCP-style tool-risk traces, and suspicious tool metadata detection
- suspicious tool-description metadata detection for prompt-injection-like language, credential exfiltration language, and safety bypass language
- per-request `allow_fallbacks` and `max_cloud_cost_usd` controls across OpenAI-compatible, Responses, Anthropic-compatible, and embeddings requests
- configurable cloud budget guardrail for fallback and verification escalation
- compact decision summaries in traces and recent logs
- optional OpenTelemetry route spans through the `observability` extra
- improved model discovery/readiness for OpenAI-compatible local runtimes and no-key local proxy modes

Suggested upgrade test:

```bash
pip install --upgrade routelabs-router
router recommend local-model
router doctor
router route --task "Search tickets" --tool-name mcp__tickets__search
```

## 0.3.0

This release makes RouteLabs much easier to adopt across the current agent ecosystem by adding both newer OpenAI-style surfaces and an Anthropic-compatible surface.

Highlights:

- `/v1/responses` support for newer OpenAI-style agent clients
- `/v1/messages` support for Anthropic Messages-style clients
- Anthropic cloud provider adapter and Anthropic-first cloud fallback support
- MCP-style agent tool traces with approval-risk hints in route decisions and logs
- configurable tool-risk policy patterns for approval, review, and trusted tools
- `router demo agent-tools` for a zero-setup MCP-style tool-risk trace demo
- OpenClaw and Hermes Agent demo presets, starter profiles, and gateway guides
- README cleanup to surface integrations and remove duplicate chat/stats walkthroughs
- stronger structured-output validation for `json_object` and practical `json_schema` constraints
- semantic streaming events for both `/v1/responses` and `/v1/messages`
- clearer docs and examples for OpenAI-compatible and Anthropic-compatible adoption paths

Suggested upgrade test:

```bash
pip install --upgrade routelabs-router
router doctor
router start
```

## 0.2.0

This release turns RouteLabs Router into a much more usable local-first runtime for real builder workflows.

Highlights:

- OpenAI-compatible chat, embeddings, models, stats, logs, and health endpoints
- verification-aware local-first routing with cloud escalation when needed
- privacy-aware local preference for obvious sensitive or code-like content
- local-to-cloud fallback when providers fail
- tool-calling passthrough and API-layer streaming
- structured-output passthrough and broader OpenAI request compatibility
- `router doctor` for setup checks and actionable environment guidance
- `router models` for configured plus live-discovered local model visibility
- startup warnings for missing providers and missing configured Ollama models
- route traces, recent logs, cost estimates, latency metrics, and chat token-speed visibility
- OpenClaw and Unsloth workflow guides plus starter config profiles

Suggested upgrade test:

```bash
pip install --upgrade routelabs-router
router doctor
router start
```
