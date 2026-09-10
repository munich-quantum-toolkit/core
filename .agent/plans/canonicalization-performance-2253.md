# Canonicalization performance fixes

Status: implementation, validation, performance audit, and complexity review
complete. Base: `1b9d08911cdfce55e801e5b0791b4437c2bd2e4a`.

## Goal and ownership

Address the four cases from the
[audit](../audits/canonicalization-performance-2253.md): redundant CBit load
scans, per-gate parameter bookkeeping, pairwise QTensor commuting, and repeated
validation of shared parameter expressions.

## Decisions

- Reuse only immediately adjacent CBit loads with identical register/index SSA
  values. Preserve existing write and unknown-user barriers.
- Finite gate parameters are a precondition. Keep a direct-constant sanity check
  in operation verification and existing numerical guards at their point of use.
  Do not scan parameter expressions or add program-wide checks, caches, or
  pipeline and exporter validation calls.
- Batch only proven commuting QTensor chains. Preserve group order, slot
  identity, dominance, and linearity; stop at dynamic, same-index, unknown, or
  cross-block dependencies. Include the commuting prefix to avoid repeated
  suffix normalization under a bottom-up greedy walk.
- Keep direct-constant sanity and numerical correctness tests. Remove tests that
  require exhaustive detection of invalid parameter expressions.

## Completion evidence

The native build, 1467 focused C++ tests, two CLI checks, 445 Python MLIR/Qiskit
tests, routine lint, full-file C++ lint, and stub generation pass. Generated
stubs have no tracked changes. No additional release blocker was found in scope.
