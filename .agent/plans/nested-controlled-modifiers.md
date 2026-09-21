# Nested modifiers in controlled composites

Status: complete. Stacked on #2565; addresses #2588.

## Scope and decisions

Reuse QCO modifier unrolling in multi-control decomposition for inverse bodies
and constant integer powers of operations on disjoint wires. Preserve the width
threshold, native target support, global phase, and wire order. Do not
distribute fractional, runtime, or overlapping composite powers, or repeat large
bodies. Existing simplifications and native synthesis still apply.

Use the existing greedy rewrite traversal and modifier canonicalization. Add no
pipeline pass or early rotation merging. Frontends require no changes.

## Validation

The new matrix regression fails on #2565 with the reported unsupported-control
error. All 579 affected native tests pass: compiler 232, decomposition 315, and
MQT transforms 32. Tests cover both target pipelines, the OpenQASM reproducer,
controlled phase, reordered operands, native support, width limits, and
unsupported powers.

Repository lint and full changed-file C++ lint pass.
