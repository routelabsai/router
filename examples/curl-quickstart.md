# Curl Quickstart

These examples assume:

- `conda activate routelabs-router`
- `uvicorn routelabs_router.server.app:app --reload`
- `Ollama` is running locally for live chat execution
- `OPENAI_API_KEY` is optional for cloud-routed live execution

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
- if `OPENAI_API_KEY` is configured, high-complexity chat execution can run through the cloud provider
- otherwise chat execution returns `501` with a configuration error

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

## 6. Run a cloud-routed chat completion

First:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Design architecture for a multi-step agent that routes private tasks locally and complex tasks to the cloud."}],
    "private":false
  }'
```
