# Quantum ThinLTO feasibility and prototype

Status: in progress. The source audit confirms a target-compilation boundary
for `qco.call`; small analytical balloons establish the required summary
properties. A compatible MLIR 23.1-or-newer build is still required for runtime
baseline measurements.

## Goal and scope

Establish whether target-specific, layout-transforming summaries can compile
reusable QCO unitary functions without mandatory whole-program flattening.

The first result must distinguish three claims:

1. Current target compilation cannot preserve a quantum-call boundary.
2. Forced flattening can make planning inputs materially larger than the
   modular program.
3. A useful quantum summary must describe its output physical layout and may
   need more than one physical variant.

The prototype supports only private, nonrecursive `mqt.unitary` functions with
scalar qubit arguments, no allocation, measurement, reset, QTensor, or
adaptive control inside a summarized callee. It uses an explicit undirected
target and one- or two-qubit call interfaces. It does not change the default
compiler pipeline, public benchmark registry, or target model.

The eventual target output may be flattened. The claim is reduced planning,
routing, and optimization work before late materialization, not asymptotically
smaller native code.

## Evidence and decisions

`qco.call` already has the required logical ABI: it returns one continuation
for each qubit operand. `mlir/include/mlir/Dialect/QCO/IR/QCOOps.td` and
`mlir/lib/Dialect/QCO/IR/Operations/CallOp.cpp` define that contract.

The QCO function model intentionally provides local wire traversal and no
callee-body mapping or cache. See `.agent/plans/qco-function-model.md` and PR
#2336. ThinLTO must build a separate, explicit summary index; it must not
reintroduce speculative callee-body inference into `WireIterator`.

`mlir/include/mlir/Dialect/QCO/Transforms/Passes.td` requires qubit-carrying
calls to be inlined before `place-and-route`. The target pipeline in
`mlir/lib/Compiler/TargetCompilation.cpp` does not add an inliner. A call has
no compile-time unitary matrix, so target-native synthesis cannot lower it, and
target conformance rejects private function inputs that remain quantum values.

The first summary index is compiler-private. It has two layers:

```text
LogicalFunctionSummary
  transitive body hash, ordered interface, interaction trace, eligibility,
  semantic witness

PhysicalVariant
  logical summary hash, target-capability fingerprint, input-site guard,
  recipe, output layout delta, cost vector, semantic and conformance witness
```

Calibration only ranks valid physical variants. It does not alter their
semantic identity. A changed body, target capability, or compiler policy
invalidates the applicable cache key.

## First balloons

The balloons are deliberately small and do not claim mapper wall time or native
hardware fidelity.

- A hierarchy with a six-gate leaf, eight leaf calls per tile, and repeated
  tile calls has 142 modular planning inputs but 6,144 gate operations after
  forced inlining at 128 repetitions: a 43.27x expansion. Final native output
  still contains the 6,144 gates.
- On a three-site line, a two-qubit endpoint call requires one initial routing
  swap. A fixed-layout procedure ABI restores that swap after every call, while
  an output-layout summary needs one swap for repeated calls. At 64 calls the
  model gives 128 versus 1 routing swaps.
- The same endpoint call has left and right one-swap output layouts. Alternating
  continuation contexts cost 48 swaps with one fixed physical variant and 32
  with context-sensitive selection in a 32-call model.

These examples determine the required experiment suite: hierarchy scaling,
layout propagation, and context-sensitive variant selection.

## Work remaining

- [x] Audit the QCO call ABI, mapper contract, target-synthesis behavior, and
  related function-model work.
- [x] Run analytical hierarchy and layout balloons.
- [ ] Make a compatible MLIR 23.1-or-newer build available and reproduce the
  baseline target-compilation rejection for a modular QCO program.
- [ ] Add an experiment-only `mlir/bench/ThinLTOBalloon.cpp` driver. It will
  generate equivalent modular and transitively flattened QCO modules and emit
  one JSON result per configuration.
- [ ] Measure operation counts, mapping and synthesis time, peak RSS, swaps,
  two-qubit count, final size, and modular-compilation rejection on fixed
  topology and mapper settings.
- [ ] Implement a two-qubit layout-summary spike with late materialization and
  a fallback to current inlining for unsupported calls.
- [ ] Verify small unitary cases with the existing DD functionality and every
  materialized result with target conformance.
- [ ] Decide go or no-go from measured scaling and layout-quality evidence.

## Go or no-go criteria

Proceed to the full prototype only if runtime experiments show both of the
following:

1. Flattening causes meaningful pre-materialization time or peak-memory growth
   on the hierarchy sweep.
2. A bounded set of output-layout variants reaches or approaches eager global
   mapping quality and outperforms a fixed-layout outlined baseline on the
   layout balloons.

The paper claim must be limited to target-specific, layout-transforming
procedure summaries with late materialization. Do not claim the first quantum
ThinLTO system until a dedicated prior-art review supports that statement.

## Validation

The current source audit used the QCO call definition, mapper pass contract,
target pipeline, target synthesis, target conformance, and the completed QCO
function-model plan. The analytical balloons were executed with `uv run` and
produce the values recorded above.

The local MLIR dependency available during this investigation is version 22.0.0.
`cmake/SetupMLIR.cmake` requires 23.1 or newer, so CMake configuration correctly
stopped before compiling the experiment driver. No runtime MQT benchmark result
is claimed until that dependency is available.
