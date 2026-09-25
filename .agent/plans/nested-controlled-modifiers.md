# Nested modifiers in controlled composites

Status: complete. Stacked on #2565; addresses #2588. Both PRs rebased on
`c92ffa7a6` and reviewed together on 2026-09-25.

## Scope and decisions

Reuse QCO modifier unrolling in multi-control decomposition for inverse bodies
and constant integer powers of operations on disjoint wires. Preserve the width
threshold, native target support, global phase, and wire order. Do not
distribute fractional, runtime, or overlapping composite powers, or repeat large
bodies. Existing simplifications and native synthesis still apply.

Use the existing greedy rewrite traversal and modifier canonicalization. Add no
pipeline pass or early rotation merging. Frontends require no changes.

Unrolling requires verified linear QCO IR. A direct unitary input defined by
another body operation already proves wire overlap, so no wire-ID map is needed.
Iterate inverse bodies directly and use one `unrollModifier` overload family.

## Validation

The original matrix regression failed on #2565 with the reported
unsupported-control error. After rebasing and simplifying the shared helpers,
1,525 native tests and 458 Qiskit translation tests pass: compiler 236,
decomposition 315, MQT transforms 32, QC IR 368, and QCO IR 574. Twelve
additional Qiskit-to-target matrix checks cover nested inverse/power bodies and
open MCMT inverses in both target pipelines. Existing tests retain the OpenQASM
reproducer, controlled phase, reordered operands, native support, width limits,
and unsupported powers. Native validation uses Clang 23 with ThinLTO and Qiskit
2.5.2.

Repository lint and full changed-file C++ lint pass.
