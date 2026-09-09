# Compiler-only control-flow legalization

Status: implemented. Latest local validation: 2026-09-09.

## Scope

Legalize structured QCO/SCF control flow for the selected payload. Producers
normalize CFG branches before target compilation. Scalar operations, measurement
provenance, allocation, functions, and final payload-profile verification remain
separate checks.

The implementation is in
`mlir/lib/Dialect/QCO/Transforms/LegalizePayloadControlFlow.cpp`; compiler tests
are in `mlir/unittests/Compiler/test_compiler_pipeline.cpp`. Public contracts
are in `docs/mlir/target_compilation.md` and the QCO `Passes.td`.

## Decisions

- Keep SCCP and QCO cleanup between unrolling and residual legality checks:
  unrolling exposes constant bounds and branches.
- Reuse MLIR trip counts, zero/one-trip promotion, and full unrolling. Require
  literal bounds, signed-arithmetic safety, and a scaled step that fits the IV
  type. Interpret unsigned bounds with zero extension. Limit the pass to 65,536
  cloned body operations. A temporary constant lets LLVM unroll terminator-only
  state updates under the same budget; cleanup removes it.
- Build switch fallbacks iteratively. Preflight the payload's branch-depth limit
  and a compiler limit of 256 total control-flow levels, including moved case
  bodies. Retained native multiway switches do not use this expansion limit.
- Require explicit quantum iteration arguments and QCO branch state transport.
  Exactly one SSA use does not exclude captures in repeated regions. Keep
  negative fixtures valid under allocation verification so they test this rule.
- Reuse the cached `TargetEnvironment`. Capability IDs remain a compiler
  snapshot; the QDMI adapter and final payload-profile checks stay separate.

## Validation

After rebasing on main `2bd6a88e1`, the LLVM/MLIR 23.1.0 release build passed
all 201 compiler tests. The full native suite passed 3,388 tests with one
optional `QueryJobId` skip. MLIR documentation generation, repository lint, and
whole changed-file C++ lint passed. Focused regressions cover IV values,
cumulative cloning, switch depth, case/default selection, quantum-state
forwarding, and invalid captures.

The loop-boundary fixes pass all 203 compiler tests, including frontend state
permutations, terminator-only induction values, and unsigned bounds above the
signed range of their type.
