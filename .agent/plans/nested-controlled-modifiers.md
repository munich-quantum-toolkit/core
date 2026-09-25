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

The reduced suite passes 1,524 native tests and 428 Qiskit translation tests:
compiler 233, decomposition 315, MQT transforms 31, QC IR 368, QCO IR 575, and
two native-synthesis checks. Native validation uses Clang 23 with ThinLTO and
Qiskit 2.5.2.

One OpenQASM-to-target matrix test covers plain, inverse, and positive/negative
integer-power bodies through both pipelines, including controlled phase and wire
order. Six MCMT cases cover full-width and repeated-target definitions, open and
mixed control states, and outer modifiers. Three unsupported-power cases isolate
overlap, fractional exponents, and runtime exponents. Width, native-target,
control-scope, and failed-unrolling contracts remain covered. Boundary
assertions permit MLIR's constant hoisting; they do not freeze whole IR. The
native-synthesis tests own symbolic-rotation cost checks.

Before test consolidation, twelve additional Qiskit-to-target matrix checks
passed for nested modifiers and open MCMT inverses. Production code is unchanged
by the test and documentation cleanup.

Repository lint and full changed-file C++ lint pass.
