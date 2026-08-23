# Record complete compiler target environments in MLIR

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

An MLIR module must carry enough target information to reproduce and inspect a
compilation without hidden C++ pass state. After this change, the module-level
`mqt.target_env` attribute records both the immutable hardware target and the
exact selected payload environment. A textual MLIR round trip and a
`CompilerTarget` round trip demonstrate that no source metadata or derived cache
is lost.

## Progress

- [x] (2026-08-23 17:36Z) Rebased the existing payload-attribute foundation on
  the explicit compiler-target fact model.
- [x] (2026-08-23 17:36Z) Inspected the compiler target, MQT dialect, build
  boundaries, existing payload attributes, and focused tests.
- [x] (2026-08-23 18:04Z) Defined and verified the typed target-environment
  attribute graph.
- [x] (2026-08-23 18:04Z) Added lossless `CompilerTarget` materialization and
  reconstruction.
- [x] (2026-08-23 18:04Z) Added structural, textual round-trip, DLTI-query, and
  C++ target round-trip tests.
- [x] (2026-08-23 18:09Z) Built the focused targets, generated dialect
  documentation, and ran clang-tidy and the full lint session.
- [x] (2026-08-23 18:12Z) Restacked and updated pull request #2215.

## Surprises & Discoveries

- Observation: A single optional capability list cannot retain a known payload
  baseline when provider-specific optional metadata is unknown. Evidence: the
  old `TargetEnvAttr` used list absence for all unknown metadata, so a producer
  could not record baseline capabilities and unknown optional capabilities at
  the same time.
- Observation: The generic DLTI operation query selects one attached query
  interface and can be ambiguous if a module also has a data-layout attribute.
  Evidence: MLIR's DLTI query helper walks attached attributes rather than
  selecting `mqt.target_env` by its canonical name.

## Decision Log

- Decision: `mqt.target_env` contains one typed compilation target and one typed
  payload environment. Rationale: mapping and synthesis depend on hardware
  facts, while control-flow legalization and terminal lowering depend on the
  selected payload; both are required to replay compilation. Date/Author:
  2026-08-23, Codex.
- Decision: Keep `CompilerTarget` as a context-free immutable C++ snapshot and
  convert it at the compiler-to-IR boundary. Rationale: public target objects
  remain cheap to copy and do not depend on an MLIR context. Date/Author:
  2026-08-23, Codex.
- Decision: Store source facts only. Rationale: gate bases, adjacency, shortest
  paths, canonical names, and other caches can be recomputed by the validated
  C++ constructor. Date/Author: 2026-08-23, Codex.
- Decision: Do not connect this change to the work-in-progress QDMI v1.4 API.
  Rationale: the attribute schema is provider-neutral, and the later QDMI
  adapter can map stable descriptors and features into it without making the MQT
  dialect depend on QDMI. Date/Author: 2026-08-23, Codex.
- Decision: Use direct typed lookup by the canonical module attribute name as
  the authoritative access path. Rationale: DLTI remains useful for nested
  extension queries but is not an unambiguous module-level selector when other
  DLTI attributes are present. Date/Author: 2026-08-23, Codex.

## Outcomes & Retrospective

The typed target schema and C++ conversion are implemented and published in pull
request #2215. The focused MQT IR and full compiler suites, dialect
documentation, clang-tidy, and lint pass.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` and `mlir/lib/Compiler/Target.cpp` define
and validate `CompilerTarget`. The value contains ordered sites, optional timing
metadata, connectivity whose state is unknown, all-to-all, or explicit, and
native operations whose state is unknown, unrestricted, or explicit. Its private
storage also contains derived routing and synthesis caches; those caches must
not appear in textual IR.

`mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td` defines shared MQT attributes.
The current payload foundation records an exact descriptor and extensible
capability IDs. `mlir/lib/Dialect/MQT/IR/MQTDialect.cpp` implements structural
verification and enforces that `mqt.target_env` is attached only to a module.
`mlir/unittests/Dialect/MQT/IR/test_mqt_ir.cpp` parses and verifies textual MQT
IR.

A payload descriptor is the exact tuple of format ID, semantic version, profile,
and text or binary encoding. A capability is a positive execution guarantee
scoped to that descriptor. A constraint is a typed upper bound or other
condition on one capability. Different constraints on one capability are
conjunctive. An empty constraint list means unrestricted support for that
capability.

## Plan of Work

Replace the payload-only target attribute graph in
`mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td` with typed attributes for a
payload descriptor, program constraints, program capabilities, a payload
environment, duration units, sites, site tuples, operations, connectivity,
native operations, a compilation target, and the combined target environment.
Use enums for closed knowledge states and string IDs for extensible capabilities
and constraints. Verify descriptor versions as canonical `major.minor.patch`
semantic versions. A payload environment always stores its complete known
effective capability list and separately records whether optional capability
metadata is complete.

The compilation-target attributes must preserve the exact source facts accepted
by `CompilerTarget`. Unknown, unrestricted, and explicit states must remain
distinct even when their associated lists are empty. Verify invalid kind/list
combinations, duplicate IDs and tuples, missing duration units, bad site
references, and non-finite or out-of-range numeric metadata. Let the C++ target
constructor remain the final semantic validator instead of duplicating derived
graph checks in the dialect.

Make the combined target environment implement `DLTIQueryInterface`. Expose the
typed compilation target and payload environment under reserved MQT query keys,
then delegate provider- or dialect-namespaced keys to an optional `dlti.map`.
Reject unnamespaced extension keys and collisions with reserved keys. Keep
direct `ModuleOp::getAttrOfType<TargetEnvAttr>(TargetEnvAttr::name)` lookup as
the compiler contract.

Add conversion methods in `mlir/include/mlir/Compiler/Target.h` and
`mlir/lib/Compiler/Target.cpp`. Materialization creates an
`mqt::CompilationTargetAttr` in a supplied `MLIRContext`. Reconstruction reads
the typed attributes and calls existing validated `CompilerTarget::create`
factories. Link `MQTCompilerTarget` to `MLIRMQTDialect`; do not add QDMI or a
new bridge library.

Extend the MQT IR and compiler target tests. One textual module must contain a
complete target with ordered sparse site IDs, explicit topology, constrained and
unconstrained payload capabilities, site-specific operation metadata, and a
namespaced extension. Additional tests cover unknown and unrestricted target
facts, known-empty versus incomplete optional capability metadata, malformed
semantic versions and target records, module-only placement, DLTI queries, and
all `CompilerTarget` to attribute to `CompilerTarget` round trips.

## Concrete Steps

Run all commands from the repository root. Build the dialect and compiler test
targets with:

    cmake --build --preset release --target mqt-core-mlir-unittest-mqt-ir mqt-core-mlir-unittests-compiler mlir-doc

Run the focused binaries:

    ./build/release/mlir/unittests/Dialect/MQT/IR/mqt-core-mlir-unittest-mqt-ir
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler --gtest_filter='CompilerTarget.*'

Finish with:

    uvx nox -s lint
    git diff --check

All tests must report zero failures. The generated dialect documentation must
build without warnings from the new attribute definitions.

## Validation and Acceptance

Acceptance requires a parsed and printed `mqt.target_env` whose compilation
target and payload environment are structurally equal after round trip. A
context-free `CompilerTarget` with sparse site IDs, explicit couplings, native
operations, timing, and fidelity must produce a compilation-target attribute
that reconstructs to the same public facts. Separate targets with unknown,
unrestricted, and explicit facts must remain different after reconstruction.

Invalid semantic versions, duplicate constraint IDs, invalid kind/list pairs,
unknown site references, timing without a duration unit, unnamespaced extension
keys, and non-module attachment must fail with specific diagnostics. A direct
DLTI query on the attribute must return both reserved MQT values and a nested
extension value.

## Idempotence and Recovery

Builds, tests, formatting, and documentation generation are safe to repeat. The
work is stacked on the compiler-target prerequisite and does not change the QDMI
dependency. Preserve the recorded pre-stack backup ref before rewriting the
published pull-request branch. Use the exact recorded remote commit as the lease
when publication is ready.

## Artifacts and Notes

The payload completeness distinction is observable as follows:

    capabilities = [], optional_capabilities_known = true

means that no optional capability is supported, while:

    capabilities = [baseline capability], optional_capabilities_known = false

keeps the known baseline and states that provider-specific optional metadata is
not available.

## Interfaces and Dependencies

Use ODS `AttrDef` types and generated parsers and printers. Use LLVM containers
inside MLIR implementation code. Use `mlir::DLTIQueryInterface` only as the
query layer; do not implement `DataLayoutSpecInterface` or
`TargetSystemSpecInterface`. Keep format, profile, capability, constraint, and
extension IDs extensible strings. Use typed enums only for payload encoding,
connectivity state, and native-operation state. Add no generic dictionary
snapshot, target technology enum, capability-policy helper, QDMI header, or
derived compiler cache.

Revision note: This plan expands the payload-only pull request after the
compiler target gained explicit knowledge states. It records the complete target
environment and keeps QDMI integration as a later adapter change.
