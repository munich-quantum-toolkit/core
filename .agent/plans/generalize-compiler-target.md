# Generalize compiler target facts

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

The compiler target currently treats missing topology as all-to-all
connectivity, missing native operations as unrestricted support, and names every
quantum resource a qubit. After this change, target descriptions can represent
neutral atoms, trapped ions, photonic modes, spin qubits, and other site-based
systems without claiming facts that a provider did not report. A focused
compiler target test demonstrates the three knowledge states: unknown,
unrestricted, and an explicit list.

## Progress

- [x] (2026-08-23 15:50Z) Inspected the public target API, storage validation,
  mapping, synthesis, QDMI adapter, bindings, and tests.
- [x] (2026-08-23 17:08Z) Added explicit connectivity and native-operation
  knowledge states.
- [x] (2026-08-23 17:08Z) Replaced target qubit-count vocabulary with site count
      and operation arity.
- [x] (2026-08-23 17:12Z) Updated compiler consumers, QDMI construction, public
      bindings, Python tests, and documentation.
- [x] (2026-08-23 17:20Z) Made passes request target facts only when the
  residual program needs them and made QDMI operation applicability fail
  closed when the provider does not report it.
- [x] (2026-08-23 17:25Z) Regenerated bindings and ran focused clang-tidy on
  every changed C++ source and test file.
- [ ] Add the pull request reference to the launch changelog entry.
- [x] (2026-08-23 17:31Z) Ran the compiler, mapping, synthesis, and Python
  tests; regenerated stubs; ran focused clang-tidy, full lint, and the final
  diff checks.
- [ ] Publish the signed pull request.

## Surprises & Discoveries

- Observation: The existing `std::optional<std::vector<...>>` parameters use
  absence to mean unrestricted support, so they cannot represent unknown
  metadata. Evidence: the class comment and `Storage::supportsOperation` in
  `mlir/lib/Compiler/Target.cpp`.
- Observation: Tests must compare `std::optional<bool>` with `true`, `false`, or
  `std::nullopt`; `EXPECT_TRUE` and `EXPECT_FALSE` inspect only whether the
  optional has a value. Evidence: the first focused compiler test run exposed
  this test-only error.
- Observation: QDMI operation site applicability is optional. Treating an
  unavailable site list as global support promoted missing metadata to a native
  operation claim. Evidence: `QDMI_OPERATION_PROPERTY_SITES` defines the valid
  site tuples, while `Operation::getSites()` returns `std::nullopt` when the
  provider does not report the property.

## Decision Log

- Decision: Describe hardware through facts rather than a modality enum.
  Rationale: sites, connectivity, operations, and optional calibration data are
  useful across hardware modalities, while an enum would force technology
  switches into compiler passes. Date/Author: 2026-08-23, Codex.
- Decision: Use explicit unknown, unrestricted, and explicit states for both
  connectivity and native operations. Rationale: missing provider metadata must
  not grant support. Date/Author: 2026-08-23, Codex.
- Decision: Keep this prerequisite free of MLIR target attributes and QDMI
  program features. Rationale: the following target-environment change will
  serialize this validated contract. Date/Author: 2026-08-23, Codex.
- Decision: A pass diagnoses unknown metadata only when a surviving operation
  needs that fact. Rationale: program requirements are stage-relative; a
  classical or single-site program does not need native-operation or topology
  claims. Date/Author: 2026-08-23, Codex.

## Outcomes & Retrospective

The context-free target contract is implemented. The compiler, mapping,
synthesis, and focused Python suites pass. Generated stubs are current. Focused
clang-tidy, full lint, and final diff checks pass. The changelog reference and
publication remain.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` defines the public immutable
`CompilerTarget`. `mlir/lib/Compiler/Target.cpp` validates it and caches routing
and synthesis facts. Mapping and synthesis passes under
`mlir/lib/Dialect/QCO/Transforms/` consume those facts. The QDMI adapter in
`mlir/lib/Compiler/QDMIAdapter.cpp` constructs a target from device metadata.
Tests live in `mlir/unittests/Compiler/test_compiler_target.cpp` and adjacent
mapping and synthesis test directories.

Unknown means the provider did not report enough information. Unrestricted means
every site pair or operation is accepted. Explicit means the target lists the
accepted couplings or operations. A pass that requires unknown information must
emit a diagnostic instead of assuming support.

## Plan of Work

Add small value types to `CompilerTarget` for connectivity and native-operation
support. Each type carries a three-way kind and, for the explicit kind, the
existing vector. Make target construction accept these values and default them
to unknown. Rename target `numQubits()` to `numSites()` and operation
`numQubits()` to `arity()`. Update mapping, synthesis, the QDMI adapter,
bindings, and tests to use the new vocabulary and to handle unknown facts before
querying routes or operation support.

Keep site identifiers, ordered operation site tuples, timing units, T1/T2 data,
and fidelity values unchanged. Add no technology enum and no generic property
container. Add the pull request reference to the existing general Compiler
Collection changelog entry.

## Concrete Steps

From the repository root, edit the target header and implementation, then use
`rg` to update every caller. Build and run:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittests-compiler
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    uvx nox -s lint

Run the focused mapping and synthesis binaries discovered from their CMake
targets when those sources change. All commands are repeatable.

## Validation and Acceptance

Target tests must prove that unknown topology is distinct from all-to-all and
explicit topology, and that unknown native operations are distinct from all and
an explicit list. Existing explicit target mapping and synthesis tests must
still pass. The build must contain no old public `numQubits()` or operation
qubit-count references. `uvx nox -s lint` and `git diff --check` must pass.

## Idempotence and Recovery

Builds and tests are safe to repeat. Preserve unrelated worktree changes. Before
rewriting a published branch, record the remote head and create a backup ref.
Use an exact force-with-lease and verify every signed commit before pushing.

## Artifacts and Notes

The current behavior to replace is summarized by the existing class comment:

    An absent topology means all-to-all connectivity. An absent operation set
    means that every operation is native.

## Interfaces and Dependencies

Use LLVM containers already linked by the compiler. Do not add dependencies. The
public target keeps shared immutable storage. Connectivity and native operation
state are context-free C++ values so the later MLIR attribute layer can
materialize them without making `CompilerTarget` depend on an MLIR context.
