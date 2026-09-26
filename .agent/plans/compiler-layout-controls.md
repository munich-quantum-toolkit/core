# Compiler layout provenance and native placement

Status: complete.

## Scope and decisions

`mqt.layout` belongs to the program's sole direct entry point. Copies, MLIR
serialization, and QC/QCO conversion retain it. Transformations invalidate it;
formats without layout support require explicit discard. Module-boundary checks
remain separate from attribute verification because MLIR parsing introduces a
temporary enclosing module. Partial imported Qiskit maps remain partial.

`QCOProgram::compileForTargetWithLayout` owns stack-local tracking state and
returns a detached `MappingResult` only after the complete pipeline and final
linearity verification succeed. Inputs are local entry-block allocations with
compile-time constant sizes. Source tags preserve allocation order and idle
slots through cleanup; the full mapper permutation tracks workspace movement.

Initial and final source mappings contain target site IDs. The full routing
permutation contains target indices in site order. Qiskit export accepts the
snapshot with its target and constructs a complete `TranspileLayout`, including
workspace ancillas. Callers must use the same target and unchanged compiled
program. Native results describe the native input allocations and do not compose
with imported provenance. Discard imported provenance when choosing native input
identity.

The allocation verifier owns source-tag width validation. Tensor shrinking only
remaps the surviving tags. Fixed size means a constant allocation operand; a
dynamic tensor type does not make that size runtime-dependent.

## Validation

Native release builds pass on main and combined with PR #2607. The suites report
3,651 and 3,672 native passes respectively, with one existing skip in each, and
634 Python passes on each build. Stubs and MLIR reference generation succeed.
Repository lint and full changed-file C++ lint pass.

Tests retain one owning oracle per contract where practical. The native full
unitary test covers source and idle-input permutations; Qiskit's
`Operator.from_circuit` covers the exported full physical permutation. Duplicate
Python unitary checks, repeated error matrices, and the large seed/size matrix
are removed. See the [audit](../audits/pr2553-layout-review.md) for results.
