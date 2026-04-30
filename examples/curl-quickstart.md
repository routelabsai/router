# Curl Quickstart

These examples assume:

- `conda activate routelabs-router`
- `uvicorn routelabs_router.server.app:app --reload`
- `Ollama` is running locally for live chat execution

## 1. Health check

```bash
curl http://127.0.0.1:8000/healthz
```

## 2. Inspect a normal route

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task":"summarize a short product description","private":false}'
```

Expected behavior:

- target should usually be `local`
- provider should usually be `ollama`

## 3. Inspect a hard task

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task":"design architecture for a multi-step agent","private":false}'
```

Expected behavior:

- target should usually be `cloud`
- current chat execution for cloud-routed tasks returns `501` until a cloud adapter is added

## 4. Inspect a private task

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task":"research customer data policy","private":true}'
```

Expected behavior:

- target should prefer `local`

## 5. Run a local chat completion

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Summarize this in one sentence: RouteLabs Router chooses between local and cloud models based on privacy, cost, latency, and task complexity."}],
    "private":false
  }'
```

If `Ollama` is available, the request should return an assistant response and include the route metadata.
