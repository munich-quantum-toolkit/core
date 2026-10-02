# Add the MLIR-owned compiler target foundation

Status: historical implementation record.

Later target facts and inference:
[complete compiler targets](generalize-compiler-target.md).

## Goal and scope

MQT Core needs one immutable description of the hardware facts that its MLIR
compiler can rely on without linking a live QDMI device, the FoMaC wrapper, or
the legacy Core IR. After this change, C++ compiler code can construct a
`mlir::CompilerTarget` either from a qubit count or from detailed site,
topology, and operation data. It can cheaply copy that target, translate between
target site identifiers and dense compiler vertices, ask whether an MLIR QCO
operation is native, and inspect one typed synthesis basis. Operation
capabilities are homogeneous across the target; ordered site tuples retain only
site-specific calibration data.

The focused compiler unit test demonstrates the result. It constructs targets
with sparse signed site identifiers, device gate aliases, calibrated operation
site tuples, and one-way topology input; verifies canonicalized topology and
homogeneous capabilities; creates real QCO operations; and observes correct
support answers. Invalid metadata is rejected once at construction.

## Constraints

- QCO already exposes `UnitaryOpInterface::getBaseSymbol()`, `getNumQubits()`,
  and `getNumParams()`. Evidence:
  `mlir/include/mlir/Dialect/QCO/IR/QCOInterfaces.td` defines all three
  operations. `CompilerTarget::supports(Operation*)` can use that interface
  rather than enumerate every primitive operation.

- The synthesis implementation currently owns a separate `NativeGateKind` plus
  Euler and Weyl machinery. Pulling that header into the target would make the
  foundation depend on synthesis details. The target can instead expose a small
  stable `GateKind`, `SingleQubitBasis`, and `SynthesisBasis` view that a later
  pipeline change can adapt.

- `add_mlir_library` in this out-of-tree project installs only component
  archives. Component-install probes for `MQTCompilerTarget`,
  `MQTCompilerPipeline`, and `MLIRSupportMQT` installed their static archives
  but no headers or CMake export, and the project emits no `MLIRTargets.cmake`.
  The `Target.h` file set therefore establishes build-tree ownership and
  interface-header verification, not an installed consumer SDK.

- Independent review caught inconsistent validation in the all-native fast path,
  missing table-driven coverage for six entanglers, changed-file clang-tidy
  findings, and stale plan sections. The final implementation validates those
  query boundaries, covers all eight required entanglers, and passes the
  repeated static checks.

- MSVC models `std::array` range results as class iterators rather than raw
  pointers, while LLVM's qualified-auto check sees libc++'s iterator as a
  pointer and modernize-use-auto rejects an explicit iterator type. Deducing the
  `std::ranges::find` and `std::ranges::find_if` results with `const auto` is
  portable across both implementations; two targeted qualified-auto suppressions
  document the platform-specific false positive without weakening either check
  globally.

## Decisions

- Put all new public types under `mlir` in `mlir/include/mlir/Compiler/Target.h`
  and implementation in `mlir/lib/Compiler/Target.cpp`. Rationale: The data
  structures expose compiler-specific dense vertices, QCO operation semantics,
  and synthesis capabilities, so the MLIR compiler owns them.

- Store target state behind `std::shared_ptr<const Storage>` and expose only
  const views and queries. Rationale: Default copies share all validated
  topology and capability caches, while construction remains value-oriented and
  no mutable alias can invalidate a cache.

- Declare only copy construction and copy assignment for the shared target
  handle. Rvalues therefore use the same cheap shared-pointer copy and do not
  leave a null moved-from handle whose queries would dereference empty storage.
  Rationale: An immutable handle has no expensive ownership to transfer, and
  retaining a valid source removes an otherwise undocumented invalid state.

- Make the target name optional. Provide unnamed constructors for synthetic
  targets and named constructors for device-derived metadata. Rationale: Callers
  constructing a target from a count should not invent a device name, while real
  devices should retain useful identity metadata.

- Represent every hardware site identifier as signed `int64_t`, reject negative
  and duplicate identifiers, and generate `0..N-1` for a count-based target
  after overflow validation. Rationale: `qco.static` carries a nonnegative i64
  identifier, so the target must guarantee representability before any IR
  mutation.

- Treat an absent topology as all-to-all. For an explicit topology, normalize
  each edge to the smaller site identifier first, remove duplicate and reversed
  duplicate edges, build dense adjacency, and reject self-edges, unknown sites,
  or disconnected graphs. Rationale: Mapping needs an undirected connected
  routing graph, and checking that contract once keeps downstream passes simple.

- Preserve reported operation names while caching canonical aliases (`prx` to
  `r`, `u3` to `u`, and `cnot` to `cx`). Absent capabilities mean unrestricted
  support; a present empty list supports no operation. This keeps metadata and
  support queries distinct.

- Require every operation capability to have a positive fixed arity. Rationale:
  Final conformance cannot soundly validate an operation whose arity is unknown;
  the later device adapter must reject a device operation that omits it rather
  than transferring ambiguity into the compiler model.

- Store an optional target-wide duration unit and positive finite scale factor,
  and require it whenever a site T1/T2, operation duration, or site-tuple
  duration is present. T1/T2 must be positive; operation and site tuple
  durations may be zero for virtual gates. Rationale: Raw `uint64_t` timing
  metadata is not self-describing without the QDMI unit/scale contract, while
  the existing SC device schema intentionally permits nonnegative operation
  durations.

- Treat each operation capability as homogeneous across the target. Site tuples
  retain ordered duration and fidelity overrides but do not alter support. A
  recognized gate is available when its canonical name, arity, and parameter
  count match one capability; two-qubit gates consequently work in either
  operand orientation. Rationale: Current compilation targets expose one gate
  set, while site variation is calibration metadata. This removes directional
  probing and per-site capability reconstruction from every compiler pass.

- Accept a target whose capabilities do not form a complete synthesis basis and
  expose `std::nullopt` from `synthesisBasis()`. Rationale: Such a target
  remains useful for support checking and future diagnostics; SYN-01 can decide
  how to report or augment an incomplete basis without burdening this
  foundation.

- Keep full Euler/Weyl decomposition and menu-string construction out of
  `MQTCompilerTarget`. Rationale: CT-01 is the cycle-free target foundation;
  pipeline integration can map the typed basis to the synthesis library later.

- Cache flattened all-pairs shortest-path distances for an explicit topology
  after connectivity validation, and answer all-to-all distances as zero or one
  without a matrix. Rationale: MAP-01 must reuse the immutable compiler target
  instead of rebuilding a graph, distances, or an `AugmentedDevice`.

- Do not add a one-off `Target.h` install rule or claim installed CMake-export
  acceptance. Rationale: The repository currently packages no usable MQT MLIR
  C++ dependency closure; installing this header alone would create a misleading
  partial SDK while QCO and the other MQT MLIR targets remain unexported. The
  coordinator explicitly assigned coherent installed header and CMake-target
  export work to the PIPE/INT series after its dependency closure is known.

## Outcome and validation

The immutable target stores validated hardware facts, dense site mappings,
cached distances, and a synthesis basis. Its build-tree library links only
MLIRIR and the QCO dialect. Public-header compilation, focused target/compiler
tests, clang-tidy, lint, and diff checks passed.

Installed headers and CMake exports were not delivered by this slice. A usable
installed SDK requires the complete dialect, generated-header, and pipeline
dependency closure; exporting this header alone is insufficient.

## Code and ownership

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
routing algorithms. A hardware site identifier is the device-visible signed
integer that later appears in `qco.static`; the two are not interchangeable when
devices use sparse identifiers.

A capability states that an operation with a canonical name, arity, and
parameter count is available throughout the target. A site tuple is an ordered
list of hardware site identifiers with optional duration and fidelity overrides;
it does not restrict support. A synthesis basis is one recognized single-qubit
Euler basis plus one recognized two-qubit entangling gate. CT-01 only describes
that basis; the existing native-synthesis implementation remains under
`mlir/lib/Dialect/QCO/Transforms/Decomposition/`.

The focused compiler unit test executable is configured by
`mlir/unittests/Compiler/CMakeLists.txt` and produced at
`build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler`.
`add_mlir_library` builds component libraries in this out-of-tree project, but
its generated component install rule currently installs only archives. Public
header file sets still provide build-tree ownership and CMake's interface-header
verification target.

## Acceptance

Construction from a count produces sites `0..N-1`; construction from detailed
sites preserves order, optional target and site names, T1/T2, operation
duration/fidelity, and ordered site tuples. A copied target returns views backed
by the same immutable storage.

Negative or duplicate site identifiers, an empty target, count overflow, zero
operation arity, zero coherence times, invalid fidelity values, timing data
without a valid duration unit/scale, malformed/duplicate site tuples, unknown
site references, self-couplings, and disconnected explicit topologies throw
`std::invalid_argument`. Zero operation and site-tuple durations remain valid.
Reversed and duplicate topology input is normalized to one sorted undirected
edge.

An absent topology reports all distinct site pairs as adjacent and distance one.
An explicit topology reports only its validated edges and returns cached
shortest-path distances. Dense vertex/site translations are stable for sparse,
unsorted target identifiers.

An absent operation collection supports every well-formed hardware operation. A
present empty collection supports none except structural barrier and
global-phase operations. Reported operation spelling is retained while canonical
lookup recognizes case, whitespace, `prx`, `u3`, and `cnot`. Site tuples
preserve ordered calibration data without restricting operation support.

The typed gate view follows the homogeneous operation set. `synthesisBasis()`
exists only when one complete supported single-qubit basis and one supported
entangler exist, with documented deterministic preference; an incomplete basis
does not invalidate the target. `supports(Operation*)` agrees with typed
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

## Interfaces

At completion, `mlir/Compiler/Target.h` defines:

    namespace mlir {
    class CompilerTarget {
    public:
      using SiteId = int64_t;
      using Coupling = std::pair<SiteId, SiteId>;
      class DurationUnit;
      class Site;
      class SiteTuple;
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
      bool supportsOperation(StringRef name, size_t numQubits,
                             std::optional<size_t> numParameters) const;
      bool supports(Operation* operation) const;
      bool supports(GateKind gate) const;
      ArrayRef<GateKind> supportedGates() const;
      std::optional<SynthesisBasis> synthesisBasis() const;
    };
    }

Every constructor also accepts an optional target-wide `DurationUnit` after the
operation collection. Exact parameter spellings may be refined to satisfy
repository style, but the semantics above are fixed. `MQTCompilerTarget` links
`MLIRIR` and `MLIRQCODialect`; it must not link `MQT::CoreFoMaC`, QDMI
libraries, `MQT::CoreIR`, `MQT::MLIRSupport`, or transformation libraries.
