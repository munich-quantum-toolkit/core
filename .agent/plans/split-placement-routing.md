# Split deterministic placement from topology routing

Status: historical implementation record.

## Goal and scope

MQT Core currently runs its complete place-and-route pass even when a compiler
target has all-to-all connectivity and no routing is necessary. After this
change, target compilation uses a small deterministic placement pass for such
targets. The placement pass converts dynamic scalar and tensor-backed qubits to
the first available static target sites without inserting routing qubits or
changing structured control flow. The existing mapping pass becomes responsible
only for targets with an explicit coupling graph and reuses the same allocation
rewrite.

A user can observe the split by compiling a two-qubit program for an all-to-all
target with more sites than the program uses. The result contains one
`qco.static` operation per program qubit and no routing `qco.swap` operations.
An explicit-topology target still runs the existing router and preserves its
current behavior.

## Constraints

- The original placement split used absent topology to mean all-to-all. The
  later complete-target contract makes this fact explicit and rejects incomplete
  provider metadata; it does not add an unknown-connectivity placement mode.

- The existing allocation rewrite also creates vacant static qubits because the
  router represents every hardware site as a program token. A compact
  deterministic layout can reuse the rewrite unchanged by containing only the
  program qubits. Evidence: `place` iterates `layout.nHardwareQubits()`, while
  `Layout::fromMapping` sizes the layout from the supplied mapping.

- A changed-file lint command that selects no sources does not validate an
  uncommitted diff. This task used direct checks from the lint compilation
  database; those were focused checks rather than a completed changed-file
  session.

- A TableGen pass declaration necessarily generates a targetless factory and
  command-line registration, but placement cannot run without a
  `CompilerTarget`. Evidence: the generated default constructor left the target
  empty and could only terminate through `reportFatalUsageError`.

## Decisions

- Add a dedicated target-bound placement pass and keep `place-and-route` as the
  routing pass. Rationale: Placement and routing are established compiler stages
  with different information requirements. A separate pass removes the
  topology-dependent router from all-to-all compilation.

- Implement placement directly as a `PassWrapper` and expose only
  `createPlacementPass(const CompilerTarget&)`. Rationale: placement has no
  valid targetless form, so TableGen would create an unusable public API and
  command-line registration.

- Share internal allocation discovery and rewriting rather than invoking one
  pass from another. Rationale: The router must retain the wire and layout state
  returned by placement, which a nested pass invocation cannot expose safely.

- Assign program qubit `i` to target vertex `i` in the standalone placement
  pass. Rationale: The established discovery order is deterministic, compact,
  and does not invent a topology-dependent optimization.

- Keep target inference separate from placement. Complete all-to-all targets use
  placement; explicit topology uses routing. Incomplete provider facts fail
  during inference, as described in the successor target record.

## Outcome and validation

Target compilation now runs compact deterministic placement for the current main
branch's all-to-all target representation and retains topology-aware mapping
only for explicit coupling graphs. Both passes use the same allocation
discovery, capacity validation, and rewrite. The public surface grows by one
target-aware pass factory; no new dependency, option, or target-model concept
was introduced.

The direct mapping binary passed all 86 tests, including scalar placement,
tensor placement with classical outputs, compact noncontiguous site selection,
and both placement and direct-mapping failure atomicity. The compiler binary
passed all 139 tests, including compact all-to-all target compilation and the
existing explicit-topology pipeline. Generated MLIR documentation, repository
format/lint, focused direct `clang-tidy`, and `git diff --check` all pass.

The intermediate unknown-connectivity proposal was superseded by
[the complete-target contract](generalize-compiler-target.md), which rejects
incomplete provider metadata during inference. Placement remains deterministic
for complete all-to-all targets.

## Code and ownership

`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` contains the current
`MappingPass`. The pass first discovers top-level `qco.alloc` and
`qtensor.alloc` roots, replaces those dynamic roots with `qco.static` target
sites, and then routes two-qubit operations by inserting `qco.swap` operations.
The discovery code enforces the tensor lifetime shape supported by mapping: all
tensor qubits are extracted before any are inserted, and allocations occur in
the entry function body.

Placement means assigning each program qubit to one hardware site. Routing means
changing that assignment during the program so that every multi-qubit operation
acts on connected sites. A target with all-to-all connectivity needs placement
but not routing. An explicit topology is a listed set of connected site pairs
and requires the router.

`mlir/include/mlir/Dialect/QCO/Transforms/Passes.td` declares the generated
mapping pass and its documentation. The target-bound placement pass is private
to `Mapping.cpp`. `mlir/include/mlir/Dialect/QCO/Transforms/Mapping/Mapping.h`
declares the target-aware pass factories.
`mlir/lib/Compiler/TargetCompilation.cpp` builds the target compilation
pipeline. Direct mapping tests live in
`mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp`, and end-to-end
target pipeline tests live in
`mlir/unittests/Compiler/test_compiler_pipeline.cpp`.

## Acceptance

The direct placement tests must verify the input module before the pass and the
output module after success. For a target whose site identifiers are not dense,
program qubit zero must use the first listed target site and program qubit one
must use the second. A target with extra sites must not materialize unused
`qco.static` operations. Scalar `qco.alloc`, `qtensor.alloc`, `qtensor.extract`,
`qtensor.insert`, and `qtensor.dealloc` operations must be removed according to
the existing supported tensor contract.

Running `MappingPass` directly with a non-explicit target must fail before
changing the printed module. Running it with an explicit topology must retain
the existing executable routing behavior. End-to-end target compilation for an
all-to-all target must succeed, use only the static sites needed by the program,
and contain no router-inserted swap.

Target inference owns rejection of incomplete connectivity metadata; placement
accepts the complete target contract described in
[the target record](generalize-compiler-target.md).

## Interfaces

The final public C++ factory is:

    std::unique_ptr<Pass>
    createPlacementPass(const CompilerTarget& target);

The placement implementation is an internal `PassWrapper` with no targetless
factory or command-line registration. `MappingPass` keeps its existing factory
and options. Both implementations use the existing `CompilerTarget`, `Layout`,
`WireIterator`, `TensorIterator`, QCO, QTensor, and MLIR rewrite APIs. No new
dependency is added.
