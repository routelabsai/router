# RouteLabs + Hermes Agent

This guide shows how to try `RouteLabs Router` as a local-first model gateway underneath a Hermes Agent setup.

Hermes Agent is interesting for RouteLabs because it is local/open, persistent, and autonomous. Those traits make it a good fit for a routing layer that can keep private work local, escalate only when justified, and expose tool-risk traces for long-running agent workflows.

## Recommended current usage

Use RouteLabs wherever your Hermes setup can point an OpenAI-compatible provider at a custom base URL.

Use:

```text
http://127.0.0.1:8000/v1
```

and set the model to:

```text
route-auto
```

Treat RouteLabs as the model gateway and policy/audit layer. Hermes remains the autonomous agent surface and memory layer.

## Recommended RouteLabs profile

Generate a Hermes-oriented config:

```bash
router init --profile hermes-agent --output ./config/router.yaml --force
```

This profile is local-first and includes policy defaults for common persistent-agent risks:

- sending messages or email
- writing memory or files
- shell or terminal execution
- calendar creation
- memory reads and searches that you explicitly trust

## Before wiring Hermes

Start RouteLabs:

```bash
router start --reload
```

Check readiness:

```bash
router doctor
```

Preview a Hermes-style tool risk trace:

```bash
router demo agent-tools --preset hermes
```

## Direct route check

You can inspect what RouteLabs would do before wiring Hermes:

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "task":"Hermes agent wants to search memory and send a Slack update",
    "tool_names":["mcp__hermes__send_message"],
    "tool_choice":{"type":"function","function":{"name":"mcp__hermes__send_message"}}
  }'
```

Expected:

- route starts local
- `agent_tools.mcp_like` is `true`
- `agent_tools.approval_required` is `true`
- `agent_tools.risk_level` is `high`

## Tool policy tuning

Start with:

```yaml
policies:
  tools:
    approval_required_patterns:
      - "mcp__hermes__send_*"
      - "mcp__hermes__write_*"
      - "mcp__hermes__shell_*"
      - "mcp__hermes__calendar__create*"
    review_recommended_patterns:
      - "mcp__hermes__search_*"
      - "mcp__hermes__calendar__read*"
    trusted_tool_patterns:
      - "mcp__hermes__read_memory"
      - "mcp__hermes__search_memory"
```

Tune these to your actual Hermes tool names. Trusted tools still appear in RouteLabs traces, but their names are ignored for risk matching.

## What to inspect after Hermes traffic

```bash
curl http://127.0.0.1:8000/v1/logs
curl http://127.0.0.1:8000/v1/stats
```

Look for:

- whether private work stayed local
- whether any request escalated
- which model/provider handled the request
- whether tool names triggered approval-risk policy

## Current limitation

This is an OpenAI-compatible gateway integration, not a Hermes-native plugin. RouteLabs does not discover or execute Hermes tools yet. It traces declared tool names and route decisions at the model gateway layer.
