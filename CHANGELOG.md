# Changelog

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
