# Benchmark audit and consolidation

Status: applied and validated locally. Review baseline: PR #2803 at
`f7e864f9422d47150251ff09a35a386c8bea4357`.

## Outcome and scope

The [audit](../audits/benchmarks.md) records the numerical fixes, shared binding
and generation contracts, JSON diagnostic repair, and validation. The work
covers local benchmark generation and sparse-count evaluation, including the
supported large-width limits. It preserves public APIs, canonical schemas,
semantic case IDs, input validation, and circuit definitions.

## Decisions

Reuse the family registry and existing phase and schema helpers. Keep the Python
helper limited to generation and JSON I/O. Family-specific analytic models,
output layouts, and evaluation remain explicit. Compute fidelity from counts
before normalization. Retain numerical limit tests and both BV circuit methods,
with one asymmetric input per method.

The merged QAOA implementation remains the baseline; this change introduces no
new execution service, dependency, or circuit variant. Local checks and their
limits are recorded in the audit.
