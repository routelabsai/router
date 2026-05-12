# Use Cases

## Local-first assistant

Route simple summarization, extraction, and classification tasks to a local model while reserving stronger remote models for genuinely hard reasoning tasks.

## Privacy-sensitive workflows

Keep tasks marked as private on-device by default. This is useful for internal notes, customer data, or regulated documents.

## Middleware for desktop or browser tools

Put `RouteLabs Router` between your app and model runtimes so your UI can talk to one endpoint while routing behavior evolves underneath.

## Agent step routing

Use this project as the foundation for future workflows where retrieval, planning, tool use, and verification can each be routed differently.

## OpenClaw gateway

Use RouteLabs as an OpenAI-compatible routing layer underneath an OpenClaw setup so the assistant surface stays the same while local/cloud routing becomes visible and policy-driven.

See [openclaw.md](openclaw.md).

## Unsloth-to-runtime routing

Use Unsloth to train or export a model, then put RouteLabs above the local runtime so your fine-tuned model handles simple/private tasks locally and RouteLabs escalates only when needed.

See [unsloth.md](unsloth.md).
