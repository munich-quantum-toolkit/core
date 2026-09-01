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
- [x] (2026-08-23 17:33Z) Added the pull request reference to the existing
  Compiler Collection launch changelog entry.
- [x] (2026-08-23 17:31Z) Ran the compiler, mapping, synthesis, and Python
  tests; regenerated stubs; ran focused clang-tidy, full lint, and the final
  diff checks.
- [x] (2026-08-23 17:33Z) Published the signed change as pull request #2218.
- [x] (2026-08-23 17:55Z) Kept the DDSIM QDMI snapshot fail-closed and made its
      QIR example state the simulator's unrestricted facts explicitly.
- [x] (2026-09-01 05:42Z) Archived the exact published head, rebased the signed
      commits onto current `main`, and reconciled post-branch compiler tests
      with the explicit connectivity and operation states.
- [x] (2026-09-01 05:50Z) Addressed the substantive review findings: preserved
      the full nonnegative site-ID domain, avoided topology reconciliation for
      single-site control flow with unknown connectivity, and documented a
      truthful explicit DDSIM operation set.
- [x] (2026-09-01 06:11Z) Regenerated bindings and ran 139 compiler, 83 mapping,
      25 target-synthesis, and focused Python QIR tests; built the
      documentation; and ran full repository lint plus changed-file C++ lint.
- [x] (2026-09-01 08:41Z) Archived the pre-refresh heads, rebased onto `main` at
      `30bb9d1f8`, adapted the newly merged mapping regression test to the
      three-state target API, and reran the affected builds and tests.
- [x] (2026-09-01 11:23Z) Added an exact versioned DDSIM device marker for the
      all-to-all topology and homogeneous fixed-arity operations that QDMI 1.3
      cannot encode compactly. Restored the direct `CompilerTarget.from_device`
      documentation and end-to-end QIR test.
- [x] (2026-09-01 11:23Z) Diagnosed the Windows ARM failure as two parallel
      `mqt-cc` tests sharing one scratch directory. Isolated the invalid-input
      test and repeated both tests concurrently 100 times.
- [x] (2026-09-01 14:58Z) Archived the pre-integration head and rebased onto the
      merged placement pass. Target compilation now maps explicit connectivity,
      places all-to-all targets, and places unknown connectivity only when no
      non-barrier multi-site operation remains.
- [x] (2026-09-01 15:07Z) Ran the final native, Python, documentation, stub,
      lint, and C++ lint validation and prepared the signed rebased head for
      publication.

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
- Observation: QDMI v1.3 has no compact representation for all-to-all
  connectivity or unrestricted operations. Evidence: the DDSIM device omits both
  optional lists because enumerating all pairs of 65,535 sites is not practical.
- Observation: LLVM dense containers reserve sentinel values inside the key
  domain. Evidence: `DenseMapInfo<int64_t>` reserves the two largest signed
  values, while `CompilerTarget::SiteId` intentionally accepts every nonnegative
  `int64_t` value.
- Observation: Unknown connectivity is sufficient for a program containing no
  non-barrier multi-site operation, including structured control flow. Evidence:
  such a program cannot change its layout, so branch reconciliation would only
  query topology unnecessarily.
- Observation: Placement does not need a coupling graph. Evidence: the merged
  placement pass replaces dynamic allocations with target sites without using
  routing data, while the mapping pass requires explicit connectivity.
- Observation: The Windows ARM failure was a test-data race, not a compiler
  regression. Evidence: the QIR and invalid-input CTest scripts used the same
  output directory while the QIR script removed that directory at startup.

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
- Decision: Use sentinel-free standard containers for site IDs. Rationale: this
  preserves the documented public domain instead of introducing an arbitrary
  range restriction to accommodate an implementation detail. Date/Author:
  2026-09-01, Lukas Burgholzer with Codex assistance.
- Decision: Route explicit connectivity through the mapping pass and route
  all-to-all or unknown connectivity through the placement pass. The placement
  pass rejects unknown connectivity before mutation when a non-barrier
  multi-site operation remains. Rationale: placement needs only target sites,
  while routing needs a coupling graph. Keeping the guard in the placement pass
  also protects direct pass users. Date/Author: 2026-09-01, Lukas Burgholzer
  with Codex assistance.
- Decision: Supersede the call-site DDSIM workaround with an exact namespaced
  marker in the bundled device's first custom property. Rationale: QDMI 1.3
  cannot enumerate the simulator's all-to-all topology or homogeneous operation
  support compactly, while an exact marker lets only an explicit provider claim
  those facts. Date/Author: 2026-09-01, Lukas Burgholzer with Codex assistance.
- Decision: Build on the independently merged topology-free placement pass.
  Rationale: target compilation must still assign static sites when topology is
  unknown, but the placement stage does not need a coupling graph. The separate
  prerequisite kept each change reviewable. Date/Author: 2026-09-01, Lukas
  Burgholzer with Codex assistance.

## Outcomes & Retrospective

The context-free target contract is implemented in pull request #2218. DDSIM now
exposes enough exact metadata for `CompilerTarget.from_device` under QDMI 1.3,
and the direct path compiles and executes QIR. The compiler and focused Python
tests pass. Target compilation uses the merged placement pass for all-to-all and
safe unknown-connectivity programs, while explicit coupling graphs use the
mapping pass. Durable archive branches preserve the published heads from before
the rescope, `main` refresh, and placement integration.

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

Use deterministic placement for all-to-all connectivity and for unknown
connectivity when no non-barrier multi-site operation remains. Use the mapping
pass only for an explicit coupling graph. Validate unknown connectivity in the
placement pass before it changes the program.

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
still pass. Placement must accept barriers and single-site operations with
unknown connectivity, reject a multi-site unitary without changing the input,
and place all-to-all programs compactly. The build must contain no old public
`numQubits()` or operation qubit-count references. `uvx nox -s lint` and
`git diff --check` must pass.

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

Plan revision note (2026-09-01): Updated the plan after the independent
placement pass merged. The final design delegates topology-free work to that
pass and keeps routing confined to explicit connectivity.
