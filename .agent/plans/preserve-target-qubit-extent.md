# Preserve the physical target extent in QC-to-Qiskit export

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Target compilation maps logical qubits to physical site identifiers. Cleanup
correctly removes unused `qco.static` and `qco.sink` pairs, but an exporter can
then infer a circuit that is narrower than the target. After this change, target
compilation records a compact module attribute with the physical address extent.
Native QC-to-Qiskit export uses that attribute as a minimum circuit width. A
two-qubit program compiled for a five-site dense target therefore exports as a
five-qubit Qiskit circuit without keeping three empty static-qubit operations
alive.

## Progress

- [x] (2026-08-17 07:17Z) Read `AGENTS.md`, `.agent/PLANS.md`, and
      `docs/ai_usage.md`; inspect mapping, QCO-to-QC conversion, Qiskit export,
      and their tests.
- [x] (2026-08-17 07:42Z) Add and propagate the target-qubit-extent module
      metadata from successful mapping through QCO-to-QC conversion.
- [x] (2026-08-17 07:42Z) Make native QC-to-Qiskit export strictly validate and
      apply the extent as a lower bound.
- [x] (2026-08-17 07:42Z) Add focused C++ and Python regression tests for sparse
      and dense targets, lower-bound semantics, and malformed metadata.
- [x] (2026-08-17 07:42Z) Update target-compilation and Qiskit documentation and
      add a separate Unreleased changelog bullet that references PR #2133.
- [x] (2026-08-17 07:42Z) Run focused builds, affected tests, the complete
      documentation build, the full lint session, and diff checks.

## Surprises & Discoveries

- Observation: A target site identifier is not a dense compiler vertex.
  `CompilerTarget::numQubits()` reports the number of sites, while mapped
  `qco.static` operations retain the target-defined identifiers. Evidence: the
  existing `QCOProgramUsesCompactAllToAllPlacement` test uses site identifiers
  2472 and 18449 for a two-site target.
- Observation: The QCO-to-QC conversion rewrites operations in the existing
  module and leaves module attributes intact. The extent can therefore cross the
  conversion without a conversion-specific pattern.
- Observation: `CompilerTarget::Site::create` accepts `INT64_MAX`, but target
  construction currently stores site identifiers in a default
  `llvm::DenseMap<int64_t, ...>`. LLVM reserves `INT64_MAX` and `INT64_MAX - 1`
  as map sentinels, so constructing such a target aborts before mapping.
  Evidence: a boundary regression stopped in `DenseMap.h` with
  `Empty/Tombstone value shouldn't be inserted into map`. Fixing target storage
  is separate from preserving the extent; the producer still converts the
  largest signed site identifier to `uint64_t` before adding one.

## Decision Log

- Decision: Define the extent as one plus the largest target site identifier,
  which is the exclusive upper bound of the physical address space. Rationale:
  the number of target sites cannot represent sparse or nonzero-based site
  identifiers. Date/Author: 2026-08-17 / Codex.
- Decision: Store the extent as the unsigned 64-bit module attribute
  `mqt.target_qubit_extent`. Rationale: target site identifiers are nonnegative
  signed 64-bit values, so their exclusive upper bound can be 2^63. The
  repository already uses the `mqt.*` namespace for metadata shared by QC and
  QCO. Date/Author: 2026-08-17 / Codex.
- Decision: Keep cleanup of unused static/sink pairs unchanged. Rationale: a
  module attribute records one count without retaining dead SSA operations.
  Date/Author: 2026-08-17 / Codex.
- Decision: Apply the metadata as a lower bound before QC resource collection.
  Rationale: later dynamic allocations must start after the physical target
  address space, and observed resources must still be able to increase the
  exported width. Date/Author: 2026-08-17 / Codex.
- Decision: Add a separate bullet to the existing Unreleased Added section and
  reference PR #2133. Rationale: adding this behavior to older
  target-compilation PR references would attribute the new change to the wrong
  pull requests, and this task must not create a new Fixed section. Date/Author:
  2026-08-17 / Codex.

## Outcomes & Retrospective

The mapping pass now records `mqt.target_qubit_extent` as an exact `ui64`
exclusive upper bound after mapping succeeds. QC-to-Qiskit export validates the
attribute and treats it as a minimum width while retaining wider observed
resources. Cleanup still removes unused static/sink pairs, and module cloning
plus QCO-to-QC conversion preserve the metadata without special conversion code.

Validation passed: all 133 compiler-pipeline C++ tests, all 142 tests in the two
affected Python MLIR test files, the complete warning-as-error documentation
build, the repository-wide lint session, `mlir-doc`, and `git diff --check`. The
changelog references PR #2133. The pre-existing `DenseMap` sentinel limitation
for site IDs `INT64_MAX` and `INT64_MAX - 1` remains deliberately out of scope
and should be tracked separately.

## Context and Orientation

`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` implements the target
mapping pass. The pass replaces dynamic QCO allocations with `qco.static`
operations whose integer indices are target site identifiers. The canonical
target pipeline in `mlir/lib/Compiler/TargetCompilation.cpp` runs cleanup after
mapping, so unused `qco.static`/`qco.sink` pairs disappear.

`mlir/lib/Conversion/QCOToQC/QCOToQC.cpp` converts a mapped QCO module to QC in
place. `bindings/mlir/qiskit/QiskitExport.cpp` collects QC resources and creates
a native Qiskit `QuantumCircuit`. Today it derives `numQubits` only from
surviving QC allocations and static sites. The new module attribute supplies a
minimum width when target cleanup removed unused sites.

The term "extent" means an exclusive upper bound on site identifiers. A target
with sites 5, 9, and 17 has three sites and an extent of 18. A dense target with
sites 0 through 4 has five sites and an extent of 5.

## Plan of Work

Add the shared attribute name to `mlir/include/mlir/Dialect/Utils/Utils.h`.
After successful mapping, compute one plus the maximum value from
`CompilerTarget::siteIds()` and attach it to the mapped module as a `ui64`
integer. Do not change static-qubit creation or cleanup.

In `bindings/mlir/qiskit/QiskitExport.cpp`, read the optional module attribute
before collecting QC resources. Reject values that are not a positive `ui64` or
cannot fit in Qiskit's 32-bit qubit count. Initialize the export state with the
validated extent. Existing collection then takes the maximum of this lower bound
and all observed static sites and appends any dynamic resources after it.

Extend `mlir/unittests/Compiler/test_compiler_pipeline.cpp` to prove that sparse
site IDs produce their address extent, the attribute survives QCO-to-QC
conversion, and unused static/sink pairs remain absent. Extend the Python MLIR
tests to compile a two-qubit circuit for a wider target and verify the returned
Qiskit circuit width. Add focused validation tests for malformed metadata.

Document the metadata and export behavior in `docs/mlir/target_compilation.md`
and `docs/mlir/python_compiler_collection.md`. Add a separate bullet to the
existing Unreleased Added section with a PR #2133 reference. Do not attribute
the behavior to older PRs or add a new Fixed section.

## Concrete Steps

Run all commands from the repository root.

Configure and build the release tree when no compatible build exists:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittests-compiler

Run the focused C++ test binary:

    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerPipelineTest.*Target*'

Build the Python extension and run focused Python tests:

    uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    uv run --no-sync pytest test/python/test_mlir.py \
      test/python/test_mlir_qiskit_translation.py -k 'target_extent'

Build the complete documentation and run lint before handoff:

    uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check

Record exact results in this plan. If the local environment cannot build a
required component, record the command and diagnostic instead of claiming a
pass.

Completed validation results:

- `mqt-core-mlir-unittests-compiler --gtest_brief=1`: 133 tests passed. This
  includes the target-QIR path that calls `llvmIR()` and the sparse-site extent
  assertion `18449 + 1 == 18450`.
- `pytest test/python/test_mlir.py test/python/test_mlir_qiskit_translation.py`:
  142 tests passed, including six new focused extent cases.
- `cmake --build build/release --target mlir-doc`: passed. CMake emitted only
  pre-existing developer warnings from `cmake/CleanMLIRDocs.cmake` about the
  `\.` escape.
- `nox --non-interactive -s docs`: passed in four minutes with Sphinx warnings
  treated as errors.
- `nox --non-interactive -s lint`: passed all repository hooks.
- `git diff --check`: passed.

## Validation and Acceptance

Acceptance requires a mapped two-qubit program for a dense five-site target to
contain the `mqt.target_qubit_extent` value 5 after cleanup while containing no
unused static/sink pairs. QCO-to-QC conversion must retain the attribute, and
native Qiskit export must return a circuit whose `num_qubits` is 5.

A sparse target must record one plus its largest site identifier, not its site
count. A manually supplied extent that is smaller than observed QC resources
must not shrink the circuit. A zero, wrongly typed, or Qiskit-unrepresentable
extent must fail with a controlled error before circuit allocation.

Existing compiler and Qiskit translation tests must continue to pass. The
complete documentation build, lint session, and `git diff --check` must pass or
have a recorded environment-specific blocker.

## Idempotence and Recovery

All source edits and test commands are repeatable. Re-running mapping replaces
the module attribute with the extent of the current target. Re-running export
does not consume the QC program. Build artifacts stay under `build/` and remain
untracked. If a build fails, fix the source and rerun the same target; no data
migration or destructive operation is required.

## Artifacts and Notes

The intended mapped module shape is:

    module attributes {mqt.target_qubit_extent = 5 : ui64} {
      // Only live mapped static qubits remain here.
    }

For sparse sites 5, 9, and 17, the expected attribute value is 18.

## Interfaces and Dependencies

`mlir::utils::TARGET_QUBIT_EXTENT_ATTR` must be an `llvm::StringLiteral` equal
to `mqt.target_qubit_extent`. The mapping pass must write an `mlir::IntegerAttr`
with unsigned 64-bit type. No public C++ or Python method is added. Native
Qiskit export consumes the attribute through the existing
`QCProgram.to_qiskit()` path and continues to use the versioned Qiskit C API
translation interface.

Revision note (2026-08-17): Created the self-contained plan after inspecting the
current target mapping and native Qiskit export paths.

Revision note (2026-08-17): Finalized the implementation, documented the
separate changelog-entry decision and `DenseMap` boundary limitation, and
recorded successful validation.
