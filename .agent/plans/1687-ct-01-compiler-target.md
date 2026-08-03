# Add the MLIR-owned compiler target foundation

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core needs one immutable description of the hardware facts that its MLIR
compiler can rely on without linking a live QDMI device, the FoMaC wrapper, or
the legacy Core IR. After this change, C++ compiler code can construct a
`mlir::CompilerTarget` either from a qubit count or from detailed site,
topology, and operation data. It can cheaply copy that target, translate between
provider site identifiers and dense compiler vertices, ask whether an MLIR QCO
operation is native at an ordered hardware locus, and inspect one typed native
synthesis basis that is usable across the complete target.

The focused compiler unit test demonstrates the result. It constructs targets
with sparse signed site identifiers, provider gate aliases, calibrated operation
loci, and one-way topology input; verifies canonicalized topology and
capabilities; creates real QCO operations; and observes correct support answers.
Invalid metadata is rejected once at construction.

## Progress

- [x] (2026-08-03 11:01Z) Verified the assigned repository root, exact starting
  revision, branch, and clean status; read `AGENTS.md`, `docs/ai_usage.md`,
  `.agent/PLANS.md`, and the MQT remediation protocol completely.
- [x] (2026-08-03 11:01Z) Inspected the existing compiler, QCO operation
  interfaces, MLIR library conventions, the prior target experiment, and the
  useful native-gate derivation concepts from pull request #1969.
- [x] (2026-08-03 11:29Z) Added the immutable `mlir::CompilerTarget` public
  interface, shared validated storage, dense mapping and topology caches,
  operation capabilities, typed synthesis basis, and real-QCO-operation
  support.
- [x] (2026-08-03 11:32Z) Added the standalone `MQTCompilerTarget` library with
      only `MLIRIR` and `MLIRQCODialect` as public link dependencies, assigned
      `Target.h` to its build-tree public header file set, and excluded that
      header from `MQTCompilerPipeline`.
- [x] (2026-08-03 11:52Z) Added eight focused tests covering all four
  constructor forms, validation, cheap copies, topology, every required
  symmetric and directional entangler, synthesis-basis resolution, and real
  QCO operations.
- [x] (2026-08-03 11:56Z) Built the target and interface-header verification
  target, passed 8 focused and all 218 compiler tests, ran changed-file
  clang-tidy without local diagnostics, passed the full repository lint, and
  passed `git diff --check`.
- [x] (2026-08-03 11:56Z) Completed an independent pre-commit review and
  resolved all four blocking findings: intrinsic locus validation, complete
  entangler-table coverage, C++/policy diagnostics, and the stale plan. The
  final tree is prepared for one signed atomic commit without a push.
- [x] (2026-08-03 12:35Z) Diagnosed the exact-head CI failures after
      publication. Moved five file-local helpers out of the anonymous namespace
      so LLVM 22's preferred static-linkage check passes. Rebuilt the target,
      passed the 8 focused and all 218 compiler tests, reran changed-file
      clang-tidy, and passed the complete repository lint. The macOS failure was
      an unrelated wall-clock assertion in an unchanged global-phase test and
      will rerun on the replacement head.
- [x] (2026-08-03 13:10Z) Replaced two pointer declarations for `std::ranges`
      results with portable iterator declarations after the Windows ARM build
      exposed MSVC's non-pointer `std::array` iterator type. Rebuilt the target
      and interface header and passed the 8 focused and all 218 compiler tests
      plus the exact LLVM static-linkage check.

## Milestones

### Milestone 1: Establish one immutable target contract

The goal was a compiler-owned model that can be constructed without QDMI, FoMaC,
CoreIR, or a live device. The work added value types for timing units, sites,
operation loci, and operation capabilities plus a shared immutable target
storage object. The result preserves provider metadata while validating the
cross-object contract once, including timing units, ordered loci, and explicit
operation absence versus emptiness. The focused construction and rejection tests
prove the observable contract.

### Milestone 2: Centralize topology and synthesis facts

The goal was to prevent later mapping and synthesis stages from rebuilding
provider-specific representations. The work canonicalized explicit undirected
couplings, cached dense adjacency and all-pairs distances, indexed operation
capabilities, and derived a typed target-wide synthesis basis. The result gives
MAP-01 a reusable distance query and gives SYN-01 a stable gate enum without a
dependency on transformation code. Sparse-site, distance, ordered-locus, and
complete entangler-orientation tests prove these paths.

### Milestone 3: Isolate and verify the foundation

The goal was a dependency-pure MLIR library with independently compilable public
headers. The work created `MQTCompilerTarget`, linked it publicly only to
`MLIRIR` and `MLIRQCODialect`, and assigned `Target.h` to that target's header
file set. The result builds without QDMI, FoMaC, CoreIR, `MQT::MLIRSupport`, or
transformation libraries. The interface-header target, 8 focused tests, 218-test
compiler suite, changed-file clang-tidy, repository lint, diff checks, and
independent review provide the proof. Installed CMake consumer support remains
an explicit series-level item described below.

## Surprises & Discoveries

- Observation: The exact starting revision does not contain the unrelated
  `set_property(TARGET qdmi PROPERTY SYSTEM ON)` setting mentioned by the
  review. Evidence: `cmake/ExternalDependencies.cmake` proceeds directly from
  `FetchContent_MakeAvailable(${FETCH_PACKAGES})` to the JSON installation
  logic. No removal is therefore necessary in this slice.
- Observation: QCO already exposes `UnitaryOpInterface::getBaseSymbol()`,
  `getNumQubits()`, and `getNumParams()`. Evidence:
  `mlir/include/mlir/Dialect/QCO/IR/QCOInterfaces.td` defines all three
  operations. `CompilerTarget::supports(Operation*, locus)` can use that
  interface rather than enumerate every primitive operation.
- Observation: The synthesis implementation currently owns a separate
  `NativeGateKind` plus Euler and Weyl machinery. Pulling that header into the
  target would make the foundation depend on synthesis details. The target can
  instead expose a small stable `GateKind`, `SingleQubitBasis`, and
  `SynthesisBasis` view that a later pipeline change can adapt.
- Observation: The initial configure could not reach upstream dependency hosts
  from the sandbox. Reconfiguring with the repository's already-fetched,
  revision-compatible dependency source trees produced a local release build; no
  source or lockfile was changed for this environment workaround.
- Observation: `add_mlir_library` in this out-of-tree project installs only
  component archives. Component-install probes for `MQTCompilerTarget`,
  `MQTCompilerPipeline`, and `MLIRSupportMQT` installed their static archives
  but no headers or CMake export, and the project emits no `MLIRTargets.cmake`.
  The `Target.h` file set therefore establishes build-tree ownership and
  interface-header verification, not an installed consumer SDK.
- Observation: Independent review caught inconsistent validation in the
  standalone `Operation::supports` query and the all-native fast path, missing
  table-driven coverage for six entanglers, changed-file clang-tidy findings,
  and stale plan sections. The final implementation validates those query
  boundaries, covers all eight required entanglers, and passes the repeated
  static checks.
- Observation: CI's LLVM 22 lint enables
  `llvm-prefer-static-over-anonymous-namespace`, which the local repository lint
  does not surface. Five file-local helpers therefore needed explicit `static`
  linkage outside the anonymous namespace. The simultaneous macOS debug failure
  came from the unchanged
  `GlobalPhaseNormalizationTest.ScalesLinearlyAcrossLargePhaseScopes` timing
  assertion: 84.6 ms exceeded a 79.8 ms threshold.
- Observation: MSVC models `std::array` range results as class iterators rather
  than raw pointers. Declaring the results of `std::ranges::find` and
  `std::ranges::find_if` as `const auto` is portable across MSVC and libc++
  while preserving the existing comparisons and dereferences.

## Decision Log

- Decision: Put all new public types under `mlir` in
  `mlir/include/mlir/Compiler/Target.h` and implementation in
  `mlir/lib/Compiler/Target.cpp`. Rationale: The data structures expose
  compiler-specific dense vertices, QCO operation semantics, and synthesis
  capabilities, so the MLIR compiler owns them. Date/Author: 2026-08-03, GPT-5.6
  via Codex.
- Decision: Store target state behind `std::shared_ptr<const Storage>` and
  expose only const views and queries. Rationale: Default copies share all
  validated topology and capability caches, while construction remains
  value-oriented and no mutable alias can invalidate a cache. Date/Author:
  2026-08-03, GPT-5.6 via Codex.
- Decision: Declare only copy construction and copy assignment for the shared
  target handle. Rvalues therefore use the same cheap shared-pointer copy and do
  not leave a null moved-from handle whose queries would dereference empty
  storage. Rationale: An immutable handle has no expensive ownership to
  transfer, and retaining a valid source removes an otherwise undocumented
  invalid state. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Make the target name optional. Provide unnamed constructors for
  synthetic targets and named constructors for device-derived metadata.
  Rationale: Callers constructing a target from a count should not invent a
  device name, while real devices should retain useful identity metadata.
  Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Represent every hardware site identifier as signed `int64_t`, reject
  negative and duplicate identifiers, and generate `0..N-1` for a count-based
  target after overflow validation. Rationale: `qco.static` carries a
  nonnegative i64 identifier, so the target must guarantee representability
  before any IR mutation. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Treat an absent topology as all-to-all. For an explicit topology,
  normalize each edge to the smaller site identifier first, remove duplicate and
  reversed duplicate edges, build dense adjacency, and reject self-edges,
  unknown sites, or disconnected graphs. Rationale: Mapping needs an undirected
  connected routing graph, and checking that contract once keeps downstream
  passes simple. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Treat an absent operation collection as every operation being
  native; distinguish it from a present empty collection, which supports no
  hardware operation. Preserve provider names while caching a lower-case
  canonical name and the aliases `prx` to `r`, `u3` to `u`, and `cnot` to `cx`.
  Rationale: This retains provider metadata and adopts the useful, narrowly
  scoped alias insight from Simon Hofmann's #1969 work without importing its
  targeting, CLI, Python, or synthesis APIs. The final commit will preserve
  Simon Hofmann's authorship because this semantic source is materially reused.
  Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Require every operation capability to have a positive fixed arity.
  Rationale: Final conformance cannot soundly validate an operation whose arity
  is unknown; the later device adapter must reject a provider operation that
  omits it rather than transferring ambiguity into the compiler model.
  Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Store an optional target-wide duration unit and positive finite
  scale factor, and require it whenever a site T1/T2, operation duration, or
  locus duration is present. T1/T2 must be positive; operation and locus
  durations may be zero for virtual gates. Rationale: Raw `uint64_t` timing
  metadata is not self-describing without the QDMI unit/scale contract, while
  the existing SC device schema intentionally permits nonnegative operation
  durations. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: A gate is globally usable when its matching canonical capability
  covers every site for a one-qubit gate or every routing edge for a two-qubit
  gate. Operand-symmetric entanglers (`cz`, `rxx`, `ryy`, `rzz`, and `iswap`)
  may cover an undirected edge in either orientation. Directional entanglers
  (`cx`, `ecr`, and `rzx`) must cover both ordered orientations. An all-to-all
  target applies the same rule to every distinct pair. `supports(gate, locus)`
  remains strictly ordered. Rationale: A single synthesis basis advertised for
  the whole target must not be inferred from a provider operation that is
  unavailable on some sites or from an unimplemented operand-reversal
  assumption. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Accept a target whose capabilities do not form a complete synthesis
  basis and expose `std::nullopt` from `synthesisBasis()`. Rationale: Such a
  target remains useful for support checking and future diagnostics; SYN-01 can
  decide how to report or augment an incomplete basis without burdening this
  foundation. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Keep full Euler/Weyl decomposition and menu-string construction out
  of `MQTCompilerTarget`. Rationale: CT-01 is the cycle-free target foundation;
  pipeline integration can map the typed basis to the synthesis library later.
  Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Cache flattened all-pairs shortest-path distances for an explicit
  topology after connectivity validation, and answer all-to-all distances as
  zero or one without a matrix. Rationale: MAP-01 must reuse the immutable
  compiler target instead of rebuilding a graph, distances, or an
  `AugmentedDevice`. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Do not add a changelog, upgrade-guide entry, bindings, CLI options,
  compatibility aliases, FoMaC adapter, or QDMI integration. Rationale: Those
  belong to later compiler-target workstreams, and this approved slice
  explicitly requires no compatibility shim or `UPGRADING.md` entry.
  Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Do not add a one-off `Target.h` install rule or claim installed
  CMake-export acceptance. Rationale: The repository currently packages no
  usable MQT MLIR C++ dependency closure; installing this header alone would
  create a misleading partial SDK while QCO and the other MQT MLIR targets
  remain unexported. The coordinator explicitly assigned coherent installed
  header and CMake-target export work to the PIPE/INT series after its
  dependency closure is known. Date/Author: 2026-08-03, GPT-5.6 via Codex.

## Outcomes & Retrospective

CT-01 now provides an immutable, cheaply copied `mlir::CompilerTarget` with
named and unnamed count/detailed constructors; validated site, timing, topology,
and operation metadata; dense site mapping; cached routing distances; ordered
capability checks; real QCO-operation checks; and a typed global synthesis
basis. `MQTCompilerTarget` is a separate build-tree library whose public link
interface is limited to `MLIRIR` and `MLIRQCODialect`.

Validation completed successfully: the public-header verification target
compiled; all 8 focused target tests and all 218 compiler tests passed;
changed-file clang-tidy emitted no local diagnostics; the repository's complete
lint session passed; and `git diff --check` passed. An independent read-only
review's four blocking findings were all remediated before the final commit.

One explicit series-level acceptance item remains unsatisfied in CT-01:
installed headers and a consumable CMake export. The existing AddMLIR
integration installs only archives for all probed MQT MLIR libraries and emits
no target export. PIPE/INT must install and export the complete MQT MLIR
dependency closure coherently, then validate a clean external consumer with
`find_package` and a link against the exported target. The build-tree `FILE_SET`
and interface-header verification in CT-01 are not a substitute for that work.

## Context and Orientation

The repository's MLIR compiler API lives in `mlir/include/mlir/Compiler/` and
its implementation in `mlir/lib/Compiler/`. At the starting revision, that
directory contains only the typed program API in `Programs.h` and
`Programs.cpp`; `mlir/lib/Compiler/CMakeLists.txt` builds them as the large
`MQTCompilerPipeline` library.

The QCO dialect is the compiler's quantum-operation intermediate representation.
Its operation declarations are generated from
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td`, and
`mlir/include/mlir/Dialect/QCO/IR/QCOInterfaces.td` defines
`UnitaryOpInterface`. A dense compiler vertex is a zero-based position used by
routing algorithms. A hardware site identifier is the provider-visible signed
integer that later appears in `qco.static`; the two are not interchangeable when
providers use sparse identifiers.

A capability states that an operation with a canonical name, arity, and
parameter count is allowed at an ordered locus. A locus is an ordered list of
hardware site identifiers. An absent locus collection means the capability is
global for every valid tuple of its arity; a present empty collection means it
supports no tuple. A synthesis basis is one recognized single-qubit Euler basis
plus one recognized two-qubit entangling gate that is usable over the entire
target. CT-01 only describes that basis; the existing native-synthesis
implementation remains under `mlir/lib/Dialect/QCO/Transforms/Decomposition/`.

The focused compiler unit test executable is configured by
`mlir/unittests/Compiler/CMakeLists.txt` and produced at
`build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler`.
`add_mlir_library` builds component libraries in this out-of-tree project, but
its generated component install rule currently installs only archives. Public
header file sets still provide build-tree ownership and CMake's interface-header
verification target.

## Plan of Work

Add `mlir/include/mlir/Compiler/Target.h`. Define
`CompilerTarget::DurationUnit`, `Site`, `OperationLocus`, and `Operation` as
immutable public value types with constructors that validate their local scalar
data. Operation arity is mandatory. Define `GateKind`, `SingleQubitBasis`, and
`SynthesisBasis` as typed compiler capabilities. Define named and unnamed
`CompilerTarget` overloads for both a qubit count and detailed sites. The
constructors accept optional topology and operation collections so absence
remains semantically distinct from an explicit empty collection.

Add `mlir/lib/Compiler/Target.cpp`. Build a private shared `Storage` object,
validate cross-references and the timing-unit invariant, cache site identifiers
and the site-to-vertex map, canonicalize and validate explicit topology, build
its dense adjacency and all-pairs distance matrix, check connectivity once,
group operations by canonical name, determine globally supported typed gates,
and resolve the preferred synthesis basis. Implement queries for metadata, dense
mapping, adjacency and distance, canonical operation support, typed gate
support, and QCO `Operation*` support. Structural QCO barrier and global-phase
operations require no hardware capability; controlled single-X and single-Z
bodies map to `cx` and `cz`; measure and reset map to arity-one,
zero-real-parameter capabilities; other unitary operations use
`UnitaryOpInterface`.

Update `mlir/lib/Compiler/CMakeLists.txt` to build `Target.cpp` as the separate
`MQTCompilerTarget` library linked publicly only against the MLIR IR and QCO
dialect libraries. Give `Target.h` to this target's public header file set and
exclude it from the existing pipeline's recursive header collection. This keeps
build-tree ownership unambiguous, enables interface-header verification, and
prevents the foundational library from linking QDMI or CoreIR. Do not claim this
creates an installed CMake consumer until the complete MQT MLIR dependency
closure is exported by the PIPE/INT series.

Add `mlir/unittests/Compiler/test_compiler_target.cpp` and include it in the
existing compiler test executable. Cover both constructor families, optional
metadata, cheap shared copies, sparse site mappings, topology normalization,
every validation boundary, absent versus empty operation semantics, provider
aliases, ordered loci, global gate coverage, basis preference, and real QCO
operation support. Link the test executable to `MQTCompilerTarget`.

Do not edit `cmake/ExternalDependencies.cmake` unless the scoped QDMI `SYSTEM`
setting appears through an in-scope integration update. It is absent at the
starting revision.

## Concrete Steps

From the repository root, create and inspect the implementation:

    git status --short
    git diff -- mlir/include/mlir/Compiler/Target.h \
      mlir/lib/Compiler/Target.cpp mlir/lib/Compiler/CMakeLists.txt \
      mlir/unittests/Compiler/test_compiler_target.cpp \
      mlir/unittests/Compiler/CMakeLists.txt

Configure and build through the worktree-local wrapper:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release \
      --target mqt-core-mlir-unittests-compiler

Run only the new focused tests first, then the complete compiler test binary:

    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerTarget*'
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler

Run the repository-required lint and final source checks:

    ./.agent/run.sh uvx nox -s lint
    git diff --check
    git status --short

When validation and independent review pass, create one signed commit:

    git add .agent/plans/1687-ct-01-compiler-target.md \
      mlir/include/mlir/Compiler/Target.h \
      mlir/lib/Compiler/Target.cpp mlir/lib/Compiler/CMakeLists.txt \
      mlir/unittests/Compiler/test_compiler_target.cpp \
      mlir/unittests/Compiler/CMakeLists.txt
    git commit -S

The commit subject uses a fitting gitmoji. Its body contains
`Co-authored-by: Simon Hofmann <simon.t.hofmann@tum.de>` for the materially
reused alias and synthesis-basis derivation semantics and
`Assisted-by: GPT-5.6 via Codex` for AI assistance. No push or other remote
mutation is authorized.

## Validation and Acceptance

Construction from a count produces sites `0..N-1`; construction from detailed
sites preserves order, optional target and site names, T1/T2, operation
duration/fidelity, and ordered loci. A copied target returns views backed by the
same immutable storage.

Negative or duplicate site identifiers, an empty target, count overflow, zero
operation arity, zero coherence times, invalid fidelity values, timing data
without a valid duration unit/scale, malformed/duplicate loci, unknown site
references, self-couplings, and disconnected explicit topologies throw
`std::invalid_argument`. Zero operation and locus durations remain valid.
Reversed and duplicate topology input is normalized to one sorted undirected
edge.

An absent topology reports all distinct site pairs as adjacent and distance one.
An explicit topology reports only its validated edges and returns cached
shortest-path distances. Dense vertex/site translations are stable for sparse,
unsorted provider identifiers.

An absent operation collection supports every well-formed hardware operation. A
present empty collection supports none except structural barrier and
global-phase operations. Provider spelling is retained while canonical lookup
recognizes case, whitespace, `prx`, `u3`, and `cnot`. Ordered loci remain
directional.

The typed global gate view excludes a capability that is missing on one site or
one routing edge. Symmetric entanglers may use either locus orientation;
directional entanglers require both. `synthesisBasis()` exists only when one
complete supported single-qubit basis and one globally supported entangler
exist, with documented deterministic preference; an incomplete basis does not
invalidate the target. `supports(Operation*, locus)` agrees with typed
capability queries for actual QCO primitives and controlled X/Z operations.

`MQTCompilerTarget` builds as its own MLIR library. Its direct link interface
contains `MLIRIR` and `MLIRQCODialect` but neither QDMI, FoMaC, CoreIR,
`MQT::MLIRSupport`, nor transformation libraries. `Target.h` compiles through
the generated interface-header verification target. The focused and complete
compiler unit test runs pass, lint passes, and the final diff contains only this
ExecPlan, compiler-target source/header, and compiler CMake/test updates.

Installed headers and a CMake-exported target are not accepted by this slice:
the repository's existing AddMLIR integration installs only archives for the
whole MQT MLIR family. PIPE/INT must complete this series-level acceptance
criterion for the complete dependency closure and prove it with an external
consumer.

## Idempotence and Recovery

All inspection, configuration, build, test, lint, and diff commands are safe to
repeat. CMake and compiler caches remain inside the worktree through
`.agent/run.sh`. If compilation fails, edit only the files named in this plan
and rerun the focused target. Never reset or clean the worktree to recover;
inspect the diff and preserve any unexpected user changes. No generated file is
edited manually.

## Artifacts and Notes

Starting evidence:

    HEAD fe1935473e940f44ac312376934aabcbfb4a0e8c
    branch agent/1687-ct-01-compiler-target
    status clean

The checked-out `cmake/ExternalDependencies.cmake` has no QDMI `SYSTEM` property
after `FetchContent_MakeAvailable`, so the review cleanup is already present in
the foundation base.

Final validation evidence:

    MQTCompilerTarget_verify_interface_header_sets: built successfully
    CompilerTarget*: 8 tests passed
    complete compiler executable: 218 tests passed
    changed-file clang-tidy: no local diagnostics
    ./.agent/run.sh uvx nox -s lint: passed
    git diff --check: passed

Component-install evidence:

    MQTCompilerTarget: archive only
    MQTCompilerPipeline: archive only
    MLIRSupportMQT: archive only
    installed CMake target export: absent

This packaging evidence is an explicit unresolved PIPE/INT acceptance item; it
is not counted as successful CT-01 export validation.

## Interfaces and Dependencies

At completion, `mlir/Compiler/Target.h` defines:

    namespace mlir {
    class CompilerTarget {
    public:
      using SiteId = int64_t;
      using Coupling = std::pair<SiteId, SiteId>;
      class DurationUnit;
      class Site;
      class OperationLocus;
      class Operation;
      enum class GateKind : uint8_t;
      enum class SingleQubitBasis : uint8_t;
      struct SynthesisBasis;

      explicit CompilerTarget(size_t numQubits, ...);
      CompilerTarget(std::string name, size_t numQubits, ...);
      explicit CompilerTarget(std::vector<Site> sites, ...);
      CompilerTarget(std::string name, std::vector<Site> sites, ...);

      std::optional<StringRef> name() const;
      size_t numQubits() const;
      ArrayRef<Site> sites() const;
      ArrayRef<SiteId> siteIds() const;
      std::optional<size_t> vertexForSite(SiteId site) const;
      SiteId siteForVertex(size_t vertex) const;
      bool hasExplicitTopology() const;
      ArrayRef<Coupling> couplings() const;
      bool areAdjacent(size_t sourceVertex, size_t targetVertex) const;
      size_t distanceBetween(size_t sourceVertex, size_t targetVertex) const;
      void forEachNeighbour(size_t vertex,
                            llvm::function_ref<void(size_t)> callback) const;
      const std::optional<DurationUnit>& durationUnit() const;
      bool hasExplicitOperations() const;
      ArrayRef<Operation> operations() const;
      bool supportsOperation(StringRef name, ArrayRef<SiteId> locus,
                             std::optional<size_t> numParameters) const;
      bool supports(Operation* operation, ArrayRef<SiteId> locus) const;
      bool supports(GateKind gate, ArrayRef<SiteId> locus) const;
      ArrayRef<GateKind> globallySupportedGates() const;
      std::optional<SynthesisBasis> synthesisBasis() const;
    };
    }

Every constructor also accepts an optional target-wide `DurationUnit` after the
operation collection. Exact parameter spellings may be refined to satisfy
repository style, but the semantics above are fixed. `MQTCompilerTarget` links
`MLIRIR` and `MLIRQCODialect`; it must not link `MQT::CoreFoMaC`, QDMI
libraries, `MQT::CoreIR`, `MQT::MLIRSupport`, or transformation libraries.

Revision note (2026-08-03): Created the initial self-contained plan after exact
checkout verification and source/provenance research. The implementation and
validation sections will be updated as evidence replaces planned behavior.

Revision note (2026-08-03): Refined global entangler coverage before
implementation: symmetric operations may cover an undirected edge in one
orientation, while directional operations require both ordered orientations.
Also recorded that incomplete synthesis capability is a valid target state.

Revision note (2026-08-03): Refined the foundation while the first
implementation draft was still uncompiled: operation arity is mandatory, timing
data requires a target-wide unit/scale, and explicit topology owns the all-pairs
distance cache consumed by later mapping work. Confirmed measure and reset are
explicit arity-one operation semantics.

Revision note (2026-08-03): Recorded completed implementation and validation,
the independent-review remediations, and the repository-wide installed-export
boundary. Per series coordination, CT-01 remains the dependency-pure build-tree
foundation, while PIPE/INT owns coherent installed CMake consumer support for
the complete MQT MLIR dependency closure.
