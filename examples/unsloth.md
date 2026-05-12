# RouteLabs + Unsloth

This is the cleanest current RouteLabs story for Unsloth:

1. fine-tune or prepare a model with Unsloth
2. export it to a runtime RouteLabs can sit above
3. use RouteLabs to decide when that local model should run versus when the request should escalate

In other words:

- Unsloth helps you train, optimize, and export
- RouteLabs helps you route, verify, and observe

## Best workflow today

Use Unsloth to export a model to one of these inference paths:

- `Ollama`
- `llama.cpp`
- another OpenAI-compatible local server

Then configure RouteLabs to use that local model as the default local backend.

## Example local-first workflow

### 1. Prepare a model with Unsloth

Follow your normal Unsloth training or inference flow.

After that, export or save your model into a runtime RouteLabs can sit above.

The simplest first path is usually `Ollama`.

### 2. Pull or register the local model

Example:

```bash
ollama list
```

If your exported model is available in Ollama, note the exact model name.

### 3. Set the RouteLabs local model

Update [`config/router.yaml`](../config/router.yaml):

```yaml
providers:
  local:
    default: ollama
    ollama:
      model: your-unsloth-model
      embedding_model: embeddinggemma
```

Then start RouteLabs:

```bash
router start --reload
```

### 4. Verify what RouteLabs sees

```bash
router doctor
router models
```

These commands help confirm:

- RouteLabs can see the configured local model
- the model is actually installed
- cloud fallback is configured or not

### 5. Exercise the local model through RouteLabs

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"route-auto",
    "messages":[{"role":"user","content":"Summarize the role of RouteLabs Router in one sentence."}],
    "private":false
  }'
```

## Why this workflow is useful

By putting RouteLabs above the Unsloth-served or Unsloth-exported model, you get:

- privacy-aware local preference
- verification-aware escalation
- cloud fallback when needed
- logs and request traces
- latency and token-speed visibility

That is a much better operator experience than just “run one local model directly and hope it is enough.”

## Good first use cases

- fine-tuned internal assistant models
- domain-specific local copilots
- lower-cost local-first agent flows
- private document or code workflows

## Recommended first milestone

The easiest strong demo is:

- Unsloth model handles simple/private tasks locally
- RouteLabs escalates only when the request is weak or the provider fails
- `/v1/logs` and `/v1/stats` show what happened

That gives users a visible reason to adopt the combination instead of treating Unsloth and RouteLabs as unrelated tools.
