# Independent compiler capability prototype

Status: independently rebased and locally validated; contract design remains
gated.

## Scope and release boundary

Core #2219 owns the compiler-only representation of selected program execution
capabilities and the corresponding target environment. It has no dependency on
the QDMI 1.4 adoption branch. Core PR #2162 adds control-flow legalization. Core
PR #2227 is the separate QDMI integration layer. Settled compiler-target and
typed attribute support already landed through #2218, #2323, and #2215.

The remaining payload model is a non-blocking Core 4.1 candidate. Core #2365 and
QDMI #523 must record the contract decisions before this prototype is considered
merge-ready. Rebase mechanics do not settle format identity, operation sets,
execution guarantees, classical capabilities, or opaque-program semantics.

## Preserved behavior

Target inference rejects unknown topology or gate sets. Retain fixed and
variadic operation arities, arbitrary controlled DDSIM gates, and zero-arity
global phase. Retain current QCO linearity checks, reusable-function boundaries,
and SDK input support. Do not reintroduce superseded target-inference commits.

The canonical pipeline keeps target-aware decomposition and deterministic
placement for all-to-all connectivity, with routing only for explicit graphs.
Mapping, native synthesis, and conformance consume the validated module
environment through MLIR's analysis manager. Placement and decomposition retain
their current target-taking factories. The pipeline builder receives the same
validated target that was attached to the module.

## Implementation

The prototype types and cached analysis live in
`mlir/Compiler/TargetEnvironment.h` and `TargetEnvironment.cpp`; typed metadata
belongs to the MQT dialect. `mlir/lib/Compiler/Pipeline.cpp` owns pipeline
execution. Keep `Programs.cpp` focused on the program representation.

Bindings and `mqt-cc` accept a selected environment, while untargeted output
selection stays separate. No unreleased QDMI APIs or provider SDK dependencies
are introduced. Capability records remain prototype vocabulary until the design
tracker settles their semantics.

## Validation

Run the independent release build, compiler/mapping/synthesis/MQT IR tests,
command-line checks, Python compiler and QDMI regressions, generated stubs,
lint, and C++ lint. Explicitly cover missing environments, invalidation after
metadata changes, variadic gates and global phase, unsupported output without
consuming input, and preservation of current linearity checks.

The release build passes, with 3,879 native tests passing and one existing
optional-device test skipping. All 558 targeted Python tests pass with the
superconducting reference device enabled. Stub generation and C++ lint pass.
