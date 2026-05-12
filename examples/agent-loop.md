# Agent Loop Example

This example shows the simplest useful RouteLabs agent flow:

1. point an OpenAI-compatible client at RouteLabs
2. send a prompt with `tools`
3. if the model returns `tool_calls`, execute them locally
4. append tool results back into the conversation
5. ask RouteLabs for the next step

The runnable script is:

- [examples/agent-loop.py](agent-loop.py)

Run it with:

```bash
python examples/agent-loop.py
```

What it demonstrates:

- OpenAI-compatible `base_url` integration
- `route-auto` model selection
- tool-calling passthrough through RouteLabs
- a minimal agent loop users can adapt to real tools

This is the best first pattern for builders who want to test RouteLabs in an actual agentic workflow instead of only using single-shot chat calls.
