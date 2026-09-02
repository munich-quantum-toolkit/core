# Record compiler targets as typed MQT IR

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core represents a compilation target as the context-free C++
`mlir::CompilerTarget` value. After this change, clients can also store and
exchange the same immutable target facts as a typed `#mqt.compilation_target`
MLIR attribute. Converting the C++ value to the attribute and back preserves
names, ordered sites, timing data, connectivity, native operations, and fixed or
variadic operation arities.

The attribute deliberately describes only compiler-target facts. It does not
define a payload format, execution capability, combined target environment, or
module attachment. Those contracts require separate design work.

## Progress

- [x] (2026-09-01 05:48Z) Added typed compiler-target and leaf attributes.
- [x] (2026-09-01 05:48Z) Added lossless C++ materialization and reconstruction.
- [x] (2026-09-01 05:48Z) Added parsing, printing, verification, and conversion
  tests.
- [x] (2026-09-01 05:48Z) Added generated dialect-documentation examples.
- [x] (2026-09-01 06:16Z) Rebased on the final #2218 head and reran focused MLIR
      tests, documentation generation, and repository lint.
- [x] (2026-09-01 06:16Z) Recorded final validation evidence and outcomes.
- [x] (2026-09-01 18:59Z) Restacked the focused commit on the final #2218 target
      model.
- [x] (2026-09-01 19:02Z) Updated the attribute model and conversions for
      complete target facts, zero-arity operations, and variadic operations.
- [x] (2026-09-01 19:07Z) Reran focused MLIR tests, documentation generation,
      C++ lint, and repository lint.

## Surprises & Discoveries

- Observation: The original implementation combined compiler-target facts with
  payload and DLTI contracts. Evidence: the complete original implementation
  required `PayloadEnvAttr`, `TargetEnvAttr`, and `MLIRDLTIDialect`, while the
  compiler-target conversion itself uses none of them.
- Observation: The refreshed base requires MLIR 23.1.0. Evidence: the cached
  MLIR 22.1.8 configuration failed in QCO control-flow interfaces before any
  changed source was compiled; configuring against the installed 23.1.0
  toolchain built every focused target successfully.
- Observation: The final #2218 model treats missing connectivity and operation
  support as target-inference errors and gives operations a fixed or variadic
  arity. Evidence: `CompilerTarget::Connectivity` and `NativeOperations` have no
  unknown state, and `CompilerTarget::Operation::Arity` distinguishes fixed
  widths from inclusive variadic minima.

## Decision Log

- Decision: Keep only `CompilationTargetAttr` and its leaf attributes.
  Rationale: MQT Core 4.0 needs stable compiler-target serialization, while the
  payload capability model is still under design. Date/Author: 2026-09-01, Lukas
  Burgholzer.
- Decision: Do not define a canonical module attribute in this change.
  Rationale: A module attachment would imply ownership and composition rules
  that belong to the deferred target-environment design. Date/Author:
  2026-09-01, Lukas Burgholzer.
- Decision: Keep the existing C++ target enums and map them explicitly to the
  MQT dialect enums. Rationale: The C++ target API remains independent of the
  generated enum types, while the two-state mappings are exhaustive and small.
  Date/Author: 2026-09-01, Lukas Burgholzer.
- Decision: Represent operation arity as `OperationArityAttr` with a fixed or
  variadic kind and one value. Rationale: Fixed zero represents `gphase`, while
  a positive variadic minimum represents a base gate that accepts additional
  controls. Variadic and fixed-zero operations cannot have site tuples because
  one tuple cannot describe all accepted widths. Date/Author: 2026-09-01, Lukas
  Burgholzer.

## Outcomes & Retrospective

The focused implementation records compiler-target facts without adding a
payload-capability or DLTI model. The typed target round trip preserves fixed
zero-arity and positive-minimum variadic operations. The MQT dialect and
compiler tests, generated dialect documentation, C++ lint, and repository lint
all pass.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` declares the immutable C++ target model.
`mlir/lib/Compiler/Target.cpp` validates target data, prepares routing and
synthesis caches, and implements conversions. The MQT dialect is declared in
`mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td` and implemented in
`mlir/lib/Dialect/MQT/IR/MQTDialect.cpp`. TableGen generates the public C++
attribute and enum declarations from that dialect file.

A site is a hardware location identified by a target-defined nonnegative
integer. Its position in the site's array defines the compiler's dense vertex.
Connectivity is all-to-all or an explicit set of undirected couplings.
Native-operation support is unrestricted or an explicit list. An explicit
operation has a fixed width, including zero, or a positive variadic minimum.
These facts must survive every conversion without inference.

## Plan of Work

Define typed leaf attributes for duration units, sites, couplings, ordered site
tuples, and native operations. Compose them in `CompilationTargetAttr`. Give
each attribute a concise generated-documentation example and verify structural
invariants in `MQTDialect.cpp`.

Add `CompilerTarget::materialize(MLIRContext&)` and
`CompilerTarget::create(mqt::CompilationTargetAttr)`. The conversion must copy
only source facts, including each operation's arity kind and value.
Reconstruction must use the existing C++ factories so that normal validation and
derived-cache construction remain centralized.

Test textual parsing and printing in the MQT dialect unit tests. Test C++ to
MLIR to C++ conversion in the compiler-target unit tests, including fixed-zero
and variadic operations, both connectivity kinds, both operation-support kinds,
and invalid reconstructed topology.

## Concrete Steps

Run these commands from the repository root:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittest-mqt-ir
    cmake --build --preset release --target mqt-core-mlir-unittests-compiler
    ./build/release/mlir/unittests/Dialect/MQT/IR/mqt-core-mlir-unittest-mqt-ir
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    cmake --build --preset release --target mlir-doc
    uvx nox -s cpp-lint -- origin/codex/generalize-compiler-target
    uvx nox -s lint

The two test binaries must report no failed tests. The documentation and lint
targets must complete successfully.

## Validation and Acceptance

Acceptance requires parsing and printing a detailed `#mqt.compilation_target`
attribute without structural change. Materializing a detailed `CompilerTarget`,
reconstructing it from the attribute, and materializing it again must produce
the same attribute. The same round trip must preserve all-to-all or explicit
connectivity, unrestricted or explicit operation support, and fixed or variadic
operation arities. Invalid or disconnected reconstructed targets must retain the
existing C++ diagnostics.

Generated dialect documentation must show concrete syntax for the composite
attribute and its leaf records. Repository lint must report no formatting,
spelling, metadata, or generated-file problems.

## Idempotence and Recovery

All builds and tests are repeatable. CMake writes only below `build/`. If a
generated TableGen declaration is stale, rerun the normal CMake build rather
than editing generated files. No migration or destructive data operation is
required.

## Artifacts and Notes

The final diff must not contain payload descriptors, program capabilities,
payload environments, target environments, or an MLIR DLTI dependency.

Validation evidence from 2026-09-01:

    14 tests from MQTIRTest (14 passed)
    145 tests from CompilerTest (145 passed)
    Built target mlir-doc
    uvx nox -s cpp-lint -- origin/codex/generalize-compiler-target: success
    uvx nox -s lint: success

## Interfaces and Dependencies

The public additions are:

    static llvm::Expected<CompilerTarget>
    CompilerTarget::create(mqt::CompilationTargetAttr attribute);

    mqt::CompilationTargetAttr
    CompilerTarget::materialize(MLIRContext& context) const;

The MQT dialect exports `DurationUnitAttr`, `SiteAttr`, `CouplingAttr`,
`SiteTupleAttr`, `OperationArityAttr`, `NativeOperationAttr`,
`CompilationTargetAttr`, `ConnectivityKind`, `NativeOperationsKind`, and
`OperationArityKind`. The implementation uses existing LLVM and MLIR libraries
and adds no third-party dependency.

Revision note: This plan replaces the broader target-environment design with the
compiler-target-only scope selected for MQT Core 4.0.
