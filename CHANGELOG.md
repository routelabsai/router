# Changelog

## 0.8.0

Realtime streaming:

- stream provider tokens to the client as they are generated on `/v1/chat/completions`, `/v1/responses` and `/v1/messages`, for Ollama, OpenAI-compatible servers (llama.cpp, LM Studio, vLLM, OpenAI, LiteLLM) and Anthropic
- add `routing.streaming` (`auto`, `realtime`, `buffered`) and the per-request `x-routelabs-stream` header. `auto` streams live whenever verification could not change the outcome and buffers only when a verified local answer could still escalate
- stream model reasoning (Ollama `thinking`, `reasoning_content`, Anthropic thinking) as OpenAI-style `reasoning_content` deltas and Anthropic `thinking` blocks. With `qwen3:4b` on Ollama, the first visible token now arrives in 0.15s instead of 44.5s
- keep cloud fallback when the local provider fails before the first token, and end the stream with a client-format error event when it fails mid-stream
- run the verifier after realtime streams and record its verdict, telemetry, and traces as for buffered requests
- report the mode in an `x-routelabs-streaming` response header
- fix Ollama ignoring `max_completion_tokens`

## 0.7.0

Client compatibility (found by pointing real SDKs and agent tools at the router):

- fix `router start --reload`, the README quickstart command, which exited immediately because uvicorn cannot reload an app object
- accept OpenAI content-part arrays (`[{"type": "text", ...}, {"type": "image_url", ...}]`) instead of rejecting them with a 422, and forward images to Anthropic and Ollama
- preserve Anthropic image blocks instead of silently dropping them
- stream text without changing it; streaming previously collapsed newlines and indentation, which broke code output
- emit spec-shaped stream events so the official OpenAI and Anthropic SDK accumulators no longer double text, crash on tool-call deltas, or reject duplicate Responses output items
- support Responses API `function_call` / `function_call_output` input items and flat function tools, so multi-turn agent tool loops work on `/v1/responses`
- add `POST /v1/messages/count_tokens` (estimated) for Claude Code and Anthropic SDK clients
- stop forwarding vendor model names such as `claude-sonnet-5-5` to local providers; prefer the Anthropic provider for `claude-*` and the OpenAI provider for `gpt-*` / `o*` when escalating
- add `x-routelabs-*` request headers and the `routelabs/private` model alias for routing control, and `x-routelabs-*` response headers describing the decision
- add optional API-key auth through `ROUTELABS_API_KEY`, plus a startup warning when binding a public host without it
- accept `developer` messages (mapped to `system` for Anthropic, Ollama and local OpenAI-compatible servers), `max_completion_tokens`, and `stream_options.include_usage`
- map stop reasons correctly in both directions (`max_tokens` ⇄ `length`, `end_turn` ⇄ `stop`, `refusal` ⇄ `content_filter`)
- adapt to current model APIs: drop sampling parameters and send `max_completion_tokens` for OpenAI reasoning models, drop sampling parameters and downgrade forced `tool_choice` for Claude models that reject them, and raise the Anthropic default `max_tokens` from 1024 to 16000
- update default cloud models to `gpt-5-mini` and `claude-sonnet-5-5`
- report the installed package version from the API instead of a hard-coded `0.3.0`

- add a packaged synthetic privacy benchmark with detection accuracy, category-level precision/recall, near-miss cases, JSON output, and CI failure support
- detect environment-style API keys, tokens, secrets, and passwords as sensitive content before routing
- add an opt-in `router benchmark --fail-on-mismatch` exit code for CI while preserving benchmark reports
- validate custom benchmark dataset fields with case/field-specific errors that do not echo fixture content
- publish a scoped contributor backlog with good-first-issue candidates, acceptance criteria, and maintainer response expectations
- connect the issue chooser to GitHub Discussions and private vulnerability reporting

## 0.6.0

- add deterministic local policy preflight for route inspection without starting a model runtime
- scrub detected email addresses, phone numbers, identifiers, payment-card patterns, and secrets before allowed cloud fallback or escalation
- expose redaction decisions in request traces and support per-request `strip_pii` overrides
- configure cloud scrubbing across the default and packaged starter profiles
- preserve explicitly supplied provider registries and report missing-provider health accurately, contributed in PR #1
- run tests on Python 3.11 and 3.12 plus an installed-wheel smoke test for every pull request and push to `main`
- add `router benchmark` for reproducible offline policy-routing evaluation, custom YAML datasets, and configured cost comparisons
- prevent ordinary prompts containing action words such as “write” or “edit” from being classified as agent-tool risks when no tool context exists
- add structured issue forms, a pull request checklist, security reporting guidance, and concrete contributor tracks

Suggested upgrade test:

```bash
pip install --upgrade routelabs-router
router init --profile privacy-first --output ./config/router.yaml
router benchmark --config ./config/router.yaml
router route --config ./config/router.yaml --task "Summarize this note for alice@example.com"
```

## 0.5.0

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
