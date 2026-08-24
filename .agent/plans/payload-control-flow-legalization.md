# Compiler-only control-flow legalization

Status: independently rebased and locally validated; design remains gated.

## Scope and design gate

Core #2162 follows #2219 without QDMI runtime or adapter ancestry. This is a
non-blocking Core 4.1 candidate, gated by the capability design in Core #2365
and QDMI #523. The rebase preserves the prototype; it does not settle the
provider-neutral capability vocabulary.

Legalize structural control flow against the selected target environment. Retain
supported constructs, lower unsupported static loops and switches where
possible, and fail closed when residual control flow cannot be represented.
Scalar computation, measurement provenance, allocation, functions and final QIR
profile verification remain separate work.

## Implementation

Keep two passes in one source: bounded static-loop unrolling before cleanup,
then dialect conversion for residual branches and loops. Reuse MLIR symbol DCE,
CFG-to-SCF lifting, SCCP, loop unrolling and conversion legality. Preserve the
65,536 cloned-operation limit and widened trip-count guard against overflow.
Reject invalid linear captures; carry quantum values explicitly through regions.

The canonical pipeline retains its validated CompilerTarget parameter for
placement and decomposition. It does not reintroduce removed cleanup passes or
unknown-target fallbacks. Both legalization passes consume the existing cached
TargetEnvironmentAnalysis. Capability names and constraints remain provisional.

## Validation

Build independently on #2219 and run the compiler and full native suites,
including constant-control folding, loop bounds and overflow, conversion of
switches with linear results, unsupported dynamic control and exact constraint
boundaries. Run repository lint, C++ lint and the MLIR documentation build.

The release suite passes: 3,889 tests pass and one existing optional-device test
skips. MLIR documentation and repository lint pass. The current LLVM correctly
represents the full-width loop range; its test now verifies that the loop and
its exact trip count survive instead of requiring the LLVM 22 failure.

Preserve Simon Hofmann's human co-authorship and existing review discussion. No
archive branches or automatic review requests.
