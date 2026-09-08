# Compiler-only control-flow legalization

Status: implemented.

## Scope and release boundary

Core #2162 follows #2219 without QDMI runtime or adapter ancestry and targets
Core 4.0. Core #2365 and QDMI #523 track the separate Core 4.1/QDMI 1.4
adaptation and do not gate this prototype. The rebase preserves the capability
snapshot; human review must still settle the provider-neutral vocabulary.

Legalize structural control flow against the selected target environment. Retain
supported constructs, lower unsupported static loops and switches where
possible, and fail closed when residual control flow cannot be represented.
Scalar computation, measurement provenance, allocation, functions and final QIR
profile verification remain separate work.

## Implementation

Keep two passes in one source: bounded static-loop unrolling before cleanup,
then dialect conversion for residual branches and loops. Reuse MLIR symbol DCE,
SCCP, native static trip counts, loop unrolling and conversion legality. Require
structured QCO/SCF input; producers normalize CFG branches before compilation.
Preserve literal-bound proofs, the 65,536 cloned-operation limit, and signed
arithmetic safety checks for full unrolling. Reject invalid linear captures;
carry quantum values explicitly through regions.

The canonical pipeline receives one selected TargetEnvironment and shares its
prepared target with all passes. It does not reintroduce removed cleanup passes
or unknown-target fallbacks. Both legalization passes consume the existing
cached TargetEnvironmentAnalysis. Capability names and constraints remain
provisional.

Single-case switches require only multiway branching. Cleanup after mapping
remains; the redundant cleanup immediately after control legalization is
removed.

## Validation

The optimized native build passed all 3,217 configured tests, with one existing
optional-device skip. The compiler suite passed all 191 tests, including early
CFG rejection, runtime assertions, single-case quantum and classical switches,
full-width trip counts, unroll bounds, and linear-state constraints. MLIR
documentation, repository lint, and whole changed-file C++ lint passed.

Simon Hofmann's human co-authorship and the existing review history are
preserved. The child commits are restacked on the shared-environment
implementation in #2219.
