# Changelog

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
