# Generalize compiler target facts

Status: historical implementation record.

## Goal and scope

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

## Constraints

- The existing `std::optional<std::vector<...>>` parameters use absence to mean
  unrestricted support, so omission cannot safely represent a complete target
  fact. Evidence: the class comment and `Storage::supportsOperation` in
  `mlir/lib/Compiler/Target.cpp`.

- QDMI operation site applicability is optional. Treating an unavailable site
  list as global support promoted missing metadata to a native operation claim.
  Evidence: `QDMI_OPERATION_PROPERTY_SITES` defines the valid site tuples, while
  `Operation::getSites()` returns `std::nullopt` when the provider does not
  report the property.

- QDMI v1.3 has no compact representation for all-to-all connectivity or
  unrestricted operations. Evidence: the DDSIM device omits both optional lists
  because enumerating all pairs of 65,535 sites is not practical.

- LLVM dense containers reserve sentinel values inside the key domain. Evidence:
  `DenseMapInfo<int64_t>` reserves the two largest signed values, while
  `CompilerTarget::SiteId` intentionally accepts every nonnegative `int64_t`
  value.

- Placement does not need a coupling graph. Evidence: the merged placement pass
  replaces dynamic allocations with target sites without using routing data,
  while the mapping pass requires explicit connectivity.

- A compiler target is only useful when both its connectivity and
  native-operation support are known. Evidence: every planned real target can
  report an explicit set or an unrestricted claim, while deferring missing facts
  to individual passes adds branches without producing a usable target.

- QDMI 1.3 operation metadata describes one fixed qubit and parameter count.
  Evidence: finite higher-arity operations such as SV1 CCNOT are representable,
  while generic `unitary`, barriers, and control-flow constructs do not have a
  truthful fixed positive compiler-target signature.

- DDSIM's controlled-gate surface is not limited to `mcx` and `mcp`. Evidence:
  every canonical QCO gate in `GateTable.def`, including gates with two or three
  targets, has generic QIR controlled specializations and the DDSIM construction
  accepts an arbitrary positive control set.

- The Windows ARM failure was a test-data race, not a compiler regression.
  Evidence: the QIR and invalid-input CTest scripts used the same output
  directory while the QIR script removed that directory at startup.

## Decisions

- Describe hardware through facts rather than a modality enum. Rationale: sites,
  connectivity, operations, and optional calibration data are useful across
  hardware modalities, while an enum would force technology switches into
  compiler passes.

- The initial three-state design was superseded by a complete-target contract.
  Connectivity is all-to-all or explicit; native-operation support is
  unrestricted or explicit. Missing provider metadata is an inference error.
  Rationale: incomplete targets are not meaningful for the supported compiler
  workflows, and rejecting them once removes defensive branches from every pass.

- Represent an operation arity as either a fixed width or a variadic total width
  with a minimum. Fixed zero represents `gphase`; variadic `n` represents a base
  operation on `n` targets with any number of additional positive controls.
  Rationale: this one capability describes DDSIM's complete controlled
  standard-gate surface, including controlled multi-target gates, without
  inventing `mc*` aliases or a maximum derived from the device size.

- Bridge DDSIM's control capability through one exact versioned operation
  custom-property marker on canonical uncontrolled base gates. Rationale: QDMI
  1.3 has no standard operation-arity range or modifier capability; keeping the
  workaround narrow and exact makes its QDMI 1.4 replacement explicit.

- Keep this prerequisite free of MLIR target attributes and QDMI program
  features. Rationale: the following target-environment change will serialize
  this validated contract.

- Use sentinel-free standard containers for site IDs. Rationale: this preserves
  the documented public domain instead of introducing an arbitrary range
  restriction to accommodate an implementation detail.

- Route explicit connectivity through the mapping pass and all-to-all
  connectivity through the placement pass. Rationale: placement needs only
  target sites, while routing needs a coupling graph.

- Supersede the call-site DDSIM workaround with an exact namespaced marker in
  the bundled device's first custom property. Rationale: QDMI 1.3 cannot
  enumerate the simulator's all-to-all topology or homogeneous operation support
  compactly, while an exact marker lets only an explicit provider claim those
  facts.

- Build on the independently merged topology-free placement pass. Rationale:
  all-to-all targets still need static site assignment, but no routing graph.
  The separate prerequisite kept each change reviewable.

## Outcome and validation

PR `#2218` implements complete context-free target facts. Exact DDSIM metadata
permits target inference under QDMI 1.3 and direct QIR compilation/execution.
All-to-all targets use placement; explicit coupling graphs use mapping.
Incomplete metadata fails at construction or inference. Compiler and focused
Python tests passed.

## Code and ownership

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

## Acceptance

Target tests must cover all-to-all and explicit connectivity, plus unrestricted,
explicit-empty, and explicit native-operation support. C++ and Python target
construction must require both facts. Existing explicit target mapping and
synthesis tests must still pass, and placement must place all-to-all programs
compactly. The build must contain no old public `numQubits()` or operation
qubit-count references. `uvx nox -s lint` and `git diff --check` must pass.

## Interfaces

Use LLVM containers already linked by the compiler. Do not add dependencies. The
public target keeps shared immutable storage. Connectivity and native operation
state are context-free C++ values so the later MLIR attribute layer can
materialize them without making `CompilerTarget` depend on an MLIR context.
