# Security Policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting flow from the repository's
**Security** tab. Do not disclose credentials, private prompts, customer data,
or exploitable details in a public issue.

If private vulnerability reporting is unavailable, open a public issue that
contains no sensitive technical details and asks the maintainers to establish
a private contact channel.

Include affected versions, impact, safe reproduction conditions, and any known
mitigations. We will acknowledge a report before discussing disclosure timing
or a release plan.

## Scope

Security-sensitive areas include:

- privacy classification and forced-local routing
- PII and secret redaction before cloud execution
- provider credentials and configuration loading
- tool-risk detection and approval guidance
- prompt or private-content leakage through logs and telemetry
- request validation across compatible API surfaces
