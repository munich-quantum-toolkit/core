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
systems without silently inventing facts. Every compiler target is complete:
connectivity is all-to-all or explicit, and native-operation support is
unrestricted or explicit. Incomplete QDMI metadata fails during target
inference, before a compiler pass runs. Operation capabilities also distinguish
fixed arity from a variadic total width with a minimum, so simulator targets can
retain zero-site global phases and arbitrary controlled forms of their standard
gates.

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
- [x] (2026-09-01 15:39Z) Removed the intermediate unknown target states,
      required complete connectivity and native-operation facts at construction,
      and moved incomplete QDMI metadata failures to target inference.
- [x] (2026-09-01 15:56Z) Generalized homogeneous fixed-arity QDMI operation
      validation beyond two sites so that the SV1 three-site operations remain
      representable, while retaining format-level constructs outside the
      compiler target.
- [x] (2026-09-01 18:44Z) Represented zero-site global phase and DDSIM's
      arbitrary positive controls without enumerating controlled-gate aliases.
- [x] (2026-09-01 18:44Z) Taught target support checks and the QDMI adapter
      about variadic operation widths, then updated bindings, documentation, and
      focused tests.
- [x] (2026-09-01 18:44Z) Prepared the revised #2218 head and split typed target
      serialization and target-aware controlled-operation decomposition into
      direct dependent work without new archive branches.

## Surprises & Discoveries

- Observation: The existing `std::optional<std::vector<...>>` parameters use
  absence to mean unrestricted support, so omission cannot safely represent a
  complete target fact. Evidence: the class comment and
  `Storage::supportsOperation` in `mlir/lib/Compiler/Target.cpp`.
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
- Observation: Placement does not need a coupling graph. Evidence: the merged
  placement pass replaces dynamic allocations with target sites without using
  routing data, while the mapping pass requires explicit connectivity.
- Observation: A compiler target is only useful when both its connectivity and
  native-operation support are known. Evidence: every planned real target can
  report an explicit set or an unrestricted claim, while deferring missing facts
  to individual passes adds branches without producing a usable target.
- Observation: QDMI 1.3 operation metadata describes one fixed qubit and
  parameter count. Evidence: finite higher-arity operations such as SV1 CCNOT
  are representable, while generic `unitary`, barriers, and control-flow
  constructs do not have a truthful fixed positive compiler-target signature.
- Observation: DDSIM's controlled-gate surface is not limited to `mcx` and
  `mcp`. Evidence: every canonical QCO gate in `GateTable.def`, including gates
  with two or three targets, has generic QIR controlled specializations and the
  DDSIM construction accepts an arbitrary positive control set.
- Observation: The Windows ARM failure was a test-data race, not a compiler
  regression. Evidence: the QIR and invalid-input CTest scripts used the same
  output directory while the QIR script removed that directory at startup.

## Decision Log

- Decision: Describe hardware through facts rather than a modality enum.
  Rationale: sites, connectivity, operations, and optional calibration data are
  useful across hardware modalities, while an enum would force technology
  switches into compiler passes. Date/Author: 2026-08-23, Codex.
- Decision: The initial three-state design was superseded by a complete-target
  contract. Connectivity is all-to-all or explicit; native-operation support is
  unrestricted or explicit. Missing provider metadata is an inference error.
  Rationale: incomplete targets are not meaningful for the supported compiler
  workflows, and rejecting them once removes defensive branches from every pass.
  Date/Author: 2026-09-01, Lukas Burgholzer with Codex assistance.
- Decision: Represent an operation arity as either a fixed width or a variadic
  total width with a minimum. Fixed zero represents `gphase`; variadic `n`
  represents a base operation on `n` targets with any number of additional
  positive controls. Rationale: this one capability describes DDSIM's complete
  controlled standard-gate surface, including controlled multi-target gates,
  without inventing `mc*` aliases or a maximum derived from the device size.
  Date/Author: 2026-09-01, Lukas Burgholzer with Codex assistance.
- Decision: Bridge DDSIM's control capability through one exact versioned
  operation custom-property marker on canonical uncontrolled base gates.
  Rationale: QDMI 1.3 has no standard operation-arity range or modifier
  capability; keeping the workaround narrow and exact makes its QDMI 1.4
  replacement explicit. Date/Author: 2026-09-01, Lukas Burgholzer with Codex
  assistance.
- Decision: Keep this prerequisite free of MLIR target attributes and QDMI
  program features. Rationale: the following target-environment change will
  serialize this validated contract. Date/Author: 2026-08-23, Codex.
- Decision: Use sentinel-free standard containers for site IDs. Rationale: this
  preserves the documented public domain instead of introducing an arbitrary
  range restriction to accommodate an implementation detail. Date/Author:
  2026-09-01, Lukas Burgholzer with Codex assistance.
- Decision: Route explicit connectivity through the mapping pass and all-to-all
  connectivity through the placement pass. Rationale: placement needs only
  target sites, while routing needs a coupling graph. Date/Author: 2026-09-01,
  Lukas Burgholzer with Codex assistance.
- Decision: Supersede the call-site DDSIM workaround with an exact namespaced
  marker in the bundled device's first custom property. Rationale: QDMI 1.3
  cannot enumerate the simulator's all-to-all topology or homogeneous operation
  support compactly, while an exact marker lets only an explicit provider claim
  those facts. Date/Author: 2026-09-01, Lukas Burgholzer with Codex assistance.
- Decision: Build on the independently merged topology-free placement pass.
  Rationale: all-to-all targets still need static site assignment, but no
  routing graph. The separate prerequisite kept each change reviewable.
  Date/Author: 2026-09-01, Lukas Burgholzer with Codex assistance.

## Outcomes & Retrospective

The context-free target contract is implemented in pull request #2218. DDSIM now
exposes enough exact metadata for `CompilerTarget.from_device` under QDMI 1.3,
and the direct path compiles and executes QIR. The compiler and focused Python
tests pass. Target compilation uses the merged placement pass for all-to-all
targets, while explicit coupling graphs use the mapping pass. Incomplete target
metadata fails at construction or QDMI inference instead of being carried into
passes. Historical pre-rescope heads remain available in the existing one-off
archive branches; no new archive branches are part of this or future branch
updates.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` defines the public immutable
`CompilerTarget`. `mlir/lib/Compiler/Target.cpp` validates it and caches routing
and synthesis facts. Mapping and synthesis passes under
`mlir/lib/Dialect/QCO/Transforms/` consume those facts. The QDMI adapter in
`mlir/lib/Compiler/QDMIAdapter.cpp` constructs a target from device metadata.
Tests live in `mlir/unittests/Compiler/test_compiler_target.cpp` and adjacent
mapping and synthesis test directories.

All-to-all means every distinct site pair is connected. Unrestricted means every
representable operation is accepted. Explicit means the target lists the
accepted couplings or operations. QDMI inference rejects absent connectivity or
operation applicability instead of constructing a partial target. Fixed arity
accepts one exact total width. Variadic arity accepts every total width from its
minimum through the target's site count; for the DDSIM compatibility marker, the
additional sites are positive controls around the named base gate.

## Plan of Work

Add small value types to `CompilerTarget` for connectivity and native-operation
support. Connectivity is either all-to-all or carries the existing explicit
coupling vector. Native-operation support is either unrestricted or carries the
existing explicit operation vector. Require both at target construction. Rename
target `numQubits()` to `numSites()` and operation `numQubits()` to `arity()`.
Update mapping, synthesis, the QDMI adapter, bindings, and tests to use the new
vocabulary. Reject missing QDMI facts while constructing the target.

Use deterministic placement for all-to-all connectivity. Use the mapping pass
only for an explicit coupling graph.

Keep site identifiers, ordered operation site tuples, timing units, T1/T2 data,
and fidelity values unchanged. Represent fixed and variadic operation widths
directly; do not add a technology enum, generic property container, or
gate-family-specific controlled aliases. Add the pull request reference to the
existing general Compiler Collection changelog entry.

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

Target tests must cover all-to-all and explicit connectivity, plus unrestricted,
explicit-empty, and explicit native-operation support. C++ and Python target
construction must require both facts. Existing explicit target mapping and
synthesis tests must still pass, and placement must place all-to-all programs
compactly. The build must contain no old public `numQubits()` or operation
qubit-count references. `uvx nox -s lint` and `git diff --check` must pass.

## Idempotence and Recovery

Builds and tests are safe to repeat. Preserve unrelated worktree changes. Use an
exact force-with-lease and verify every signed commit before pushing. Do not
create archive branches for branch rewrites.

## Artifacts and Notes

The original behavior replaced by this work was summarized by the class comment:

    An absent topology means all-to-all connectivity. An absent operation set
    means that every operation is native.

## Interfaces and Dependencies

Use LLVM containers already linked by the compiler. Do not add dependencies. The
public target keeps shared immutable storage. Connectivity and native operation
state are context-free C++ values so the later MLIR attribute layer can
materialize them without making `CompilerTarget` depend on an MLIR context.

Plan revision note (2026-09-01): Updated the plan after the independent
placement pass merged and after the complete-target contract replaced the
intermediate three-state design. All-to-all placement remains separate from
explicit-topology routing; incomplete metadata now fails at inference.

Plan revision note (2026-09-01): Added zero-arity global phase and variadic
controlled standard-gate capabilities for DDSIM. The temporary QDMI v1.3 bridge
is operation-local and does not enumerate `mcx`, `mcp`, or other aliases.
