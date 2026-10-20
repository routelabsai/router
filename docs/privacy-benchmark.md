# Privacy Benchmark

RouteLabs includes a deterministic, offline privacy-policy evaluation suite.
It measures whether synthetic sensitive inputs are detected and whether public
near-misses avoid unnecessary privacy classification.

Run it without a model, GPU, network connection, or API key:

```bash
router benchmark --suite privacy --fail-on-mismatch
```

Use `--json` for CI artifacts or downstream analysis. The report includes:

- detection accuracy across labeled positive and negative cases
- micro-averaged privacy-category precision and recall
- true-positive, false-positive, and false-negative counts by category
- the normal routing, verification, and configured-cost metrics

The suite covers explicit private requests, email-like text, phone-like text,
identifiers, account/card patterns, secret-like tokens, environment-style
secrets, and code-like content. Each positive category has nearby negative
examples where practical.

## Dataset safety and provenance

The packaged `privacy-policy-synthetic-v1` dataset is hand-authored and contains
only synthetic values, reserved example domains, standard payment test data,
and deliberately nonfunctional credential strings. It contains no customer
prompts, production credentials, or real personal information.

Benchmark failures report case names and category labels, not task contents or
tool descriptions. Contributors should preserve that property when adding
cases.

## Limitations

This is a regression suite for the local heuristic detector, not proof of
regulatory compliance or real-world privacy recall. Synthetic examples cannot
represent every locale, identifier format, secret type, language, or
context-dependent disclosure. Treat its scores as a release floor and expand
it with sanitized, reviewed cases rather than production data.

Custom YAML datasets can use these expectation fields:

```yaml
cases:
  - name: synthetic-email
    task: Draft a reply to alex@example.invalid
    expected:
      privacy_detected: true
      privacy_categories: [private_email]
```

`privacy_categories` comparisons are order-independent. If
`privacy_detected` is omitted, the benchmark infers it from whether the
expected category list is empty.
