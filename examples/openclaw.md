# RouteLabs + OpenClaw

This guide shows how to try `RouteLabs Router` as a local-first gateway in front of an OpenClaw setup.

This is the right current framing:

- OpenClaw is the assistant surface and control plane
- RouteLabs is the local/cloud routing layer underneath

## Why this combination is interesting

OpenClaw already has strong provider and onboarding UX.
RouteLabs adds:

- local-first routing
- privacy-aware local preference
- verification-aware escalation
- provider fallback
- logs, stats, and cost/latency visibility

That makes the combination useful if you want OpenClaw to keep using an OpenAI-style model interface while RouteLabs makes the execution decisions.

## Recommended current usage

Use RouteLabs as an OpenAI-compatible backend where your OpenClaw setup allows a custom OpenAI-style provider base URL.

Point that provider to:

```text
http://127.0.0.1:8000/v1
```

and use:

```text
route-auto
```

as the model when you want RouteLabs to choose the actual backend.

## Before you start

1. Start RouteLabs:

```bash
router start --reload
```

2. Check readiness:

```bash
router doctor
```

3. Make sure local models are present if you want true local-first behavior:

```bash
ollama serve
ollama pull qwen3:4b
ollama pull embeddinggemma
```

4. If you want cloud fallback:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## What to test first

Before wiring OpenClaw, validate the RouteLabs side directly:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

Then test chat:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"route-auto",
    "messages":[{"role":"user","content":"Summarize RouteLabs Router in one sentence."}]
  }'
```

## Suggested OpenClaw test path

Once your OpenClaw setup is pointed at RouteLabs:

1. test a simple chat request
2. test a privacy-sensitive prompt
3. test a tool-calling flow
4. inspect RouteLabs logs:

```bash
curl http://127.0.0.1:8000/v1/logs
curl http://127.0.0.1:8000/v1/stats
```

That lets you confirm:

- whether requests stayed local
- whether anything escalated
- whether fallback triggered
- how the latency looked

## Important current note

Treat this as an OpenAI-compatible gateway integration first, not a fully custom OpenClaw-native provider integration.

That means the best current value is:

- faster evaluation
- easier experimentation
- lower cloud usage
- more visible routing behavior

If this proves useful, the next deeper integration would be a more explicit OpenClaw setup guide or provider-specific polish.
