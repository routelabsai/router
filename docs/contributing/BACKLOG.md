# Contributor Backlog

These tasks are intentionally small enough to discuss and review independently.
Before starting, open a feature proposal that links to the relevant task below.
Maintainers should convert these into labeled GitHub issues as capacity becomes
available.

## Good first issues

### Add `--fail-on-mismatch` to `router benchmark`

Suggested labels: `good first issue`, `help wanted`

Why: custom benchmark datasets currently report failed cases but still exit
successfully, which makes them awkward to use as CI quality gates.

Likely files:

- `src/routelabs_router/cli.py`
- `tests/test_benchmark.py`
- `README.md`

Acceptance criteria:

- add an opt-in `--fail-on-mismatch` flag
- exit non-zero when one or more cases mismatch and the flag is present
- retain the current reporting output before exiting
- cover passing and failing datasets in CLI tests
- document a CI usage example

### Add benchmark dataset schema documentation

Suggested labels: `good first issue`, `documentation`

Why: custom YAML datasets are supported, but contributors should not need to
read implementation code to discover the accepted fields.

Likely files:

- `docs/benchmarking.md` (new)
- `README.md`
- `src/routelabs_router/benchmarks/policy-routing.yaml`

Acceptance criteria:

- document every case input and supported expectation field
- include one minimal case and one tool-risk case
- explain configured-cost assumptions and benchmark limitations
- link the guide from the README and contributor guide

### Improve malformed benchmark dataset errors

Suggested labels: `good first issue`, `help wanted`

Why: validation errors should identify the dataset case and invalid field
without exposing the full task or private fixture content.

Likely files:

- `src/routelabs_router/benchmark.py`
- `tests/test_benchmark.py`

Acceptance criteria:

- reject invalid `private`, `agent_role`, `tool_choice`, and expectation values
- include the case name and field name in each error
- avoid echoing task text and tool descriptions in errors
- add parameterized regression tests

## Help wanted

### Build a sanitized privacy-policy evaluation dataset

Suggested labels: `help wanted`

Why: privacy routing needs measurable precision and recall, especially for
false positives that unnecessarily force public tasks local.

Likely files:

- `src/routelabs_router/benchmarks/`
- `src/routelabs_router/privacy.py`
- `tests/test_privacy.py`

Acceptance criteria:

- include synthetic positive and negative cases for every supported category
- include near-miss cases for emails, phones, identifiers, cards, secrets, and
  code-like content
- contain no real credentials, personal information, or customer prompts
- report category-level results without printing sensitive fixture values
- document dataset provenance and limitations

### Add an evaluation interface for verifier strategies

Suggested labels: `help wanted`, `enhancement`

Why: RouteLabs can escalate after heuristic verification, but it cannot yet
compare verifier behavior reproducibly across labeled cases.

Likely files:

- `src/routelabs_router/verify.py`
- `src/routelabs_router/benchmark.py`
- `tests/`

Acceptance criteria:

- define a narrow interface for labeled verifier evaluation
- measure false escalations and missed escalations separately
- ship an offline deterministic fixture set
- keep model-backed judges optional and outside the default test suite
- expose reasons for every verifier decision

### Add a minimal TypeScript client

Suggested labels: `help wanted`, `enhancement`

Why: coding-agent and web-tool integrations frequently run in TypeScript, while
the repository currently ships only a Python client and protocol examples.

Suggested scope:

- a separately packaged TypeScript client for route inspection, chat,
  responses, messages, health, stats, and logs
- no generated SDK framework in the first contribution
- no UI or dashboard work

Acceptance criteria:

- support configurable base URL and request timeout
- preserve typed route and trace metadata
- include Node-based tests with mocked HTTP responses
- include one OpenAI-compatible migration example
- document versioning and publishing without coupling Python and npm releases

## Maintainer rules for backlog issues

- assign one clear owner before work begins
- prefer a focused pull request over combining adjacent tasks
- answer new pull requests within three business days when possible
- state whether behavior changes require changelog and benchmark updates
- close or rescope stale tasks rather than leaving contributors uncertain
