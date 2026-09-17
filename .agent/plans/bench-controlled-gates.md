# Controlled composite gates

Status: complete.

## Scope and ownership

The versioned circuit adapter must retain the complete definition of controlled
gates whose targets span more wires than their base operation. Generic control
unwrapping applies only when operand widths agree.

Native target compilation and synthesis must unroll composite modifiers before
multi-control decomposition. Reuse the existing modifier pass and cleanup;
preserve nested controls, operand order, and controlled global phases. Merge
one-qubit bodies before unrolling to retain dynamic controlled gates.

## Validation

Exact-unitary import and compiler regressions pass, along with all 229 native
compiler tests and 720 MLIR Python tests. Stub generation, repository lint, and
full changed-file C++ lint pass. All 354 Bench tests pass with frontend
preprocessing removed, including the arithmetic benchmarks.
