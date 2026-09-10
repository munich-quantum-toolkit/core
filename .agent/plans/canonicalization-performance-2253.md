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
- Individual operation verifiers check direct constants. The program validator
  owns deep expression checks, with a cache limited to one call and an iterative
  operand traversal. Compiler and export boundaries invoke it; raw MLIR clients
  must do so explicitly. This changes the standalone operation-verifier
  contract.
- Batch only proven commuting QTensor chains. Preserve group order, slot
  identity, dominance, and linearity; stop at dynamic, same-index, unknown, or
  cross-block dependencies. Include the commuting prefix to avoid repeated
  suffix normalization under a bottom-up greedy walk.
- Preserve meaningful test coverage. Remove the duplicate standalone check in
  the deep-expression test because program validation already verifies MLIR.

## Completion evidence

The native build, 1470 focused C++ tests, two CLI checks, 375 Python Qiskit
tests, routine lint, full-file C++ lint for all 17 changed sources, and stub
generation pass. Generated stubs have no tracked changes. No additional release
blocker was found in scope.
