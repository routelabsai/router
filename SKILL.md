# RouteLabs Router SKILL

## Purpose

This repository is building `RouteLabs Router`, a local-first AI runtime with:

- verification-aware escalation
- privacy-aware routing
- cost and latency visibility
- transparent route traces

The project should feel useful to builders before it tries to feel polished to general end users.

## What Users Should See First

User-facing surfaces should emphasize:

- one endpoint above local and cloud LLMs
- local-first execution
- verification-aware escalation
- privacy-aware defaults
- clear route explanations
- simple setup with `Ollama` locally and optional cloud execution

In short, users should quickly understand:

1. what problem this solves
2. how to run it
3. how it differs from just using `Ollama` or a cloud gateway
4. what is real today versus what is still coming

## What Should Stay Deeper In Docs

These are important, but should live in `docs/` rather than dominate the landing page:

- long-form product philosophy
- internal architecture details
- future learning-router plans
- research framing
- implementation nuance around verifier design

The README should stay adoption-focused.
The deeper docs can stay strategy- and architecture-focused.

## Product Truths

The core differentiators to preserve are:

- local model answers first when possible
- verification should decide escalation, not only task complexity
- privacy is a routing constraint
- cost and latency should be measurable
- every route should be explainable

If a proposed feature weakens those ideas, it is probably off-strategy.

## Current User Persona

The repo is currently for:

- AI app builders
- local-first power users
- agent and workflow developers
- teams experimenting with cost/privacy-aware inference

It is not yet a mass-market end-user product.

## Near-Term Priority Order

The next most important work should generally favor:

1. verifier hooks
2. route traces and structured telemetry
3. cost and latency summaries
4. better provider coverage
5. policy-driven privacy detection
6. learned routing improvements

## Communication Rules

When updating docs or demos:

- lead with what works now
- be explicit about what is early
- prefer self-contained demo prompts
- avoid overstating verifier or dashboard features before they exist
- keep the repo honest and easy to trust
