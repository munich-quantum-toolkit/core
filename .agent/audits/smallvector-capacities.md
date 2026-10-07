# Contract audit: SmallVector capacities

Status: guidance and all accepted findings applied and validated. Baseline:
`220422824ba6df9983c63ce3d766303f2984c631`, initially clean. Date: 2026-10-07.
Dependencies: LLVM/MLIR 23.1.0; clang-tidy 23.1.2.

## Result

The code now follows the clarified guidance in the audited scope. Sixteen
production helper or callback contracts and two test helpers use
capacity-independent views or output parameters. Unexplained scratch capacities
use LLVM defaults, while retained storage choices have short reasons in the
source. An inline capacity is not a maximum size; these changes preserve
sequence contents, order, and ownership.

Bounded synthesis vectors retain their capacities. Large element types retain
explicit capacities required by LLVM. Storage choices trade object size against
heap allocation; no speed claim or timing benchmark accompanies these changes.

## Scope and contract

The owning guidance is
[Data structures and performance](../../docs/development.md#data-structures-and-performance).
The [MLIR landing page](../../docs/mlir/index.md) already routes developers to
that policy. The addition sits after the general container-selection paragraph,
before namespace/include rules and performance evidence.

[Issue #2675](https://github.com/munich-quantum-toolkit/core/issues/2675) and
its
[linked review](https://github.com/munich-quantum-toolkit/core/pull/2578#discussion_r4179545951)
ask for the distinction in
[LLVM's guidance](https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h):
omit `N` without a motivated choice, retain justified inline storage, and avoid
capacity-dependent borrowed interfaces. Returning or accepting an owning
container is a different contract from borrowing its storage.

Searched all tracked C++ headers, sources, and TableGen files, including
bindings, benchmarks, and tests; excluded vendored and generated files. A
balanced-template scan found 1,428 `SmallVector<...>` spellings, including
nested templates. Of these, 95 spell out a capacity in 37 files: 73 in MLIR
production code, one in bindings, one in benchmarks, and 20 in tests. Also
checked all three explicit `to_vector<2>` uses, concrete-vector
reference/pointer parameters, by-value parameters, and all seven direct vector
aliases and their callers. These are source occurrence counts, not counts of
distinct containers or runtime allocations.

Open changes overlap with `Target.cpp` in PR #2689 and with the QCO builder,
mapping, and some tests in PR #2696. The affected helpers and callers were
rechecked before applying the findings. PR #2280 adds a separate
constant-propagation subsystem; it is outside this checkout's inventory. The
layout semantics in issue `#1867` do not concern inline capacity policy.

## Findings

### 1. Capacity-independent borrowed interfaces (applied)

These helpers read, replace, or append elements in caller-owned storage. None
uses the container's inline capacity. The default `SmallVector<T>` still names a
concrete capacity and therefore does not solve this interface problem.

| Source                                                                                                                                                                                                        | Contract and applied parameter                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mlir/include/mqt/Dialect/QIR/Builder/QIRProgramBuilder.h`, `mlir/lib/Dialect/QIR/Builder/QIRProgramBuilder.cpp`: `createCallOp`                                                                              | Read parameters through `ArrayRef<mqt::FloatParameter>` and targets through `ValueRange`. Both baseline parameters were concrete-vector const references; gate-builder callers supply short initializer lists.                                                                                                                |
| `mlir/lib/Conversion/QCOToJeff/QCOToJeff.cpp`: `createPPROp`                                                                                                                                                  | Read `pauliGates` through `ArrayRef<int32_t>`; it only constructs a `DenseI32ArrayAttr`. The caller supplies two Pauli codes.                                                                                                                                                                                                 |
| `mlir/include/mqt/Dialect/QCO/Transforms/Decomposition/Weyl.h`, `mlir/lib/Dialect/QCO/Transforms/Decomposition/BasisDecomposer.cpp`: `decomp0`, `decomp1`, `decomp2Supercontrolled`, `decomp3Supercontrolled` | Append factors through `SmallVectorImpl<Matrix2x2>&`. Keep the owning vector in `TwoQubitNativeDecomposition`.                                                                                                                                                                                                                |
| `mlir/lib/Dialect/QCO/Utils/Matrix.cpp`: `assignFixedImpl`                                                                                                                                                    | Resize/assign caller-owned matrix storage through `SmallVectorImpl<Complex>&`.                                                                                                                                                                                                                                                |
| `mlir/lib/Dialect/QCO/Builder/QCOProgramBuilder.cpp`: `qcoIndexSwitch`'s `buildRegion` lambda                                                                                                                 | Replace the preceding branch's values through `SmallVectorImpl<Value>&`. The callback's returned vector remains an owning result.                                                                                                                                                                                             |
| `mlir/lib/Dialect/QCO/Transforms/Decomposition/DecomposeMultiControlled.cpp`: `GateEmitter`                                                                                                                   | Borrow `MutableArrayRef<Value>` for the constructor and stored view. `wire` and `setWire` only access existing elements; callers finish constructing storage first.                                                                                                                                                           |
| `mlir/include/mqt/Dialect/QCO/Utils/Drivers.h`: `WalkProgramGraphFn` / `ReleasedOps`                                                                                                                          | Accept `SmallVectorImpl<Operation*>&` in the callback. The callback appends operations, and `walkProgramGraph` owns and clears the storage. Update three production callbacks in `Mapping.cpp` and six callbacks in `test_drivers.cpp` together. Preserve the public owning alias unless its removal is separately justified. |
| `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`: `RoutingState::fromLayout`, `generateLayout`, `permuteWires`                                                                                           | The `Wires` alias hides another concrete capacity. Read roots and layout inputs through `ArrayRef<WireIterator>`; replace permutation storage through `SmallVectorImpl<WireIterator>&` to preserve moving the reordered buffer. Keep explicit owning copies for destructive traversals.                                       |
| `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`: `Node::initializeChild`, `Node::h`, `search`                                                                                                           | The `Window` alias also hides a concrete capacity. These three read-only parameters can use `ArrayRef<QubitIndexPair>`; nodes do not retain the view.                                                                                                                                                                         |
| `mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp`: `flatGHZ`, `cxcz`                                                                                                                           | Borrow `MutableArrayRef<Value>`; both replace elements without resizing.                                                                                                                                                                                                                                                      |

The benefit is a contract that accepts different inline capacities and, for
views, other contiguous storage. Preserve element order, mutation of the
caller's values, and view lifetimes. Most helpers are private. The exposed
driver callback now accepts `SmallVectorImpl<Operation*>&`, and every in-tree
callback uses that parameter. The `ReleasedOps` alias remains available for
owning storage. Clients with an explicit `ReleasedOps&` callback parameter must
use `SmallVectorImpl<Operation*>&` instead. Owning traversal copies remain
explicit in `generateLayout`; permutation storage still moves into the caller's
vector.

### 2. Unexplained scratch capacities use defaults (applied)

The following old capacities did not match a bound or concrete storage need in
the inspected contracts. Each group now uses the default. Return types and their
owning declarations were changed together. The choices below refer to the
audited baseline.

| Source and storage                                                                                                                                                                                                 | Evidence                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bindings/mlir/qiskit/QiskitExport.cpp`: `validateClassicalSnapshot` worklist, 16                                                                                                                                  | The expression budget is 16,384 visited nodes; operands and branch yields grow the worklist independently of 16.                                                                                      |
| `mlir/lib/Dialect/MQT/Transforms/NormalizeGlobalPhases.cpp`: stack 4, instructions 4, leaves 2, hoisting order 8, contributions 4, direct phases 4                                                                 | Expressions, dependency slices, and block phases grow with the input. Binary accumulation does not impose these element counts.                                                                       |
| `mlir/include/mqt/Dialect/QTensor/IR/QTensorOps.td`: `FromElementsOp` type vector, 2                                                                                                                               | Length comes from the result tensor's number of elements; the operation's own example has three qubits.                                                                                               |
| `mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp`: `AffineForm::coefficients` and `affineConstraint`, 4                                                                                                             | Coefficients grow with the affine domain's dimension variables; constraints append another constant. Change all four matching type spellings together.                                                |
| `mlir/lib/Dialect/QCO/Transforms/Decomposition/DecomposeMultiControlled.cpp`: plan operations 32, incrementer wire lists 16, controlled-SWAP controls 4                                                            | Plan length and wire counts grow with the input. Existing reserve estimates are the sizing mechanism, not these capacities. Keep `PlanOp::wires` capacity 4, which covers the largest plan primitive. |
| `mlir/lib/Dialect/QCO/Transforms/Decomposition/Pauli.cpp`: `mergeDiagonalRotations` groups, 4                                                                                                                      | The map can collect distinct pairs throughout a block, so this is not the bounded commuting group in `fusePauliRotationRun`.                                                                          |
| `mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp`: `FusableTwoQubitRun::ops`, 8                                                                                                                | Two wires bound the run's arity, not its number of operations.                                                                                                                                        |
| `mlir/lib/Dialect/QIR/Execution/JIT/IRRewriter.cpp`: pending blocks/functions and irreversible calls, 8; `mlir/lib/Dialect/QIR/Transforms/AttachQIRAttributes.cpp`: worklist, 8                                    | The input CFG, call graph, and call count determine the lengths.                                                                                                                                      |
| `mlir/lib/Dialect/QIR/Execution/Runtime/QIR.cpp`: control vectors, 4; `mlir/include/mqt/Dialect/QIR/Execution/Runtime/Runtime.h` and `mlir/lib/Dialect/QIR/Execution/Runtime/Runtime.cpp`: translated addresses, 5 | QIR control arrays and the controls-plus-targets spans have variable length. Coordinate return types with their owning declarations.                                                                  |
| `mlir/include/mqt/Dialect/QCO/Utils/Matrix.h`: `EigenDecomposition::eigenvalues`, 8                                                                                                                                | Dynamic matrix order sets the eigenvalue count. The fixed 2x2 and 4x4 results already use arrays.                                                                                                     |
| `mlir/include/mqt/Dialect/QCO/Utils/Drivers.h`: owning `ReleasedOps` storage, 8                                                                                                                                    | A graph traversal can release an input-dependent number of operations. Keep storage separate from the callback interface in finding 1.                                                                |

### 3. Retained storage choices have reasons (applied)

These choices retain their capacities with short source comments. The evidence
below explains why default replacement can change their allocation or storage
tradeoff.

- `verifyDenseUnitaryMatrix` in `mlir/lib/Dialect/MQT/Utils/DenseUnitary.cpp`
  stores 16 complex entries inline, covering 4x4 matrices. Valid input spans one
  through eight qubits, so 16 is not the overall bound. The LLVM default stores
  only three complex entries and therefore allocates for every valid matrix,
  including 2x2 input. The source comment records the small-matrix intent.
- `printBoxLine` in `mlir/lib/Support/PrettyPrinting.cpp` uses four inline
  `SmallString<128>` objects. They occupy 624 bytes versus 168 with the default
  on the audited host. The existing fast path avoids constructing this vector
  for fitting text. A change to the default's one inline element introduces heap
  allocation for text wrapping to two through four lines. The source comment
  records the intent to keep these lines inline.
- `CompilerTarget::Storage::adjacency` (4) and `capabilities` (1) in
  `mlir/lib/Compiler/Target.cpp`, and the `unmatched` map's index vectors (1) in
  `mlir/lib/Dialect/QCO/Transforms/Decomposition/Euler.cpp`, and
  `Dependencies::successors` (2) in `mlir/lib/Dialect/QCO/Utils/Sorting.cpp`,
  embed vectors in other storage. Small explicit capacities can reduce per-entry
  footprint. Hardware degree, duplicate capabilities, repeated angle terms, and
  dependency fan-out are not bounded by those numbers. The source comments
  record the footprint intent.
- `ZFrame::sums` in
  `mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/ZFramePropagation.cpp` keeps
  four accumulation levels inline in each per-wire map entry. On the audited
  host, capacity 4 uses 48 bytes versus 64 for the default. Binary accumulation
  can need more levels; four is a footprint choice, not a maximum. The final
  inventory recheck confirmed this additional retained choice.
- `getOperationSites` and `NativeCostTracker::append` in
  `mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp`, and
  `vertices` in `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`, use
  capacity 2. Their broader callers include variable-arity operations, so the
  synthesis-output bound cannot justify them. Capacity 2 can still target the
  common one/two-qubit case. The source comments record that intent.

## Choices to retain

- `emitPauliRotations`, `emitPauliSequence`, and `emitCliffordSandwich` return
  one or two owned values; `emitCartan` returns two. Keep their capacity 2 and
  matching `to_vector<2>` storage. They are not borrowed-vector parameters.
  `mergeDiagonalRotations` also accepts at most two qubits per operation, so its
  operand scratch vector has a capacity bound of 2.
- `PauliRotationSequence::rotations` has at most three terms; native Weyl
  synthesis has at most three entangler parameters. Euler emission has at most
  five steps. Their capacities 3, 3, and 5 match these bounds.
- `fusePauliRotationRun` admits only distinct pairwise-commuting two-qubit Pauli
  groups. There are at most three nonidentity members. Its capacity 4 and the
  entangling subset's capacity 3 cover that bound. Exhaustive enumeration of the
  15 nonidentity two-qubit Paulis found no commuting subset of size 4.
- Scalarization stores one branch for `scf.for`, two for `qco.if` and
  `scf.while`. Keep `TensorSlots::branches` capacity 2. `TensorSlots` exceeds
  LLVM's 256-byte cutoff for default-capacity element types, so its explicit
  outer capacity 1 cannot simply be removed.
- Keep zero-capacity vectors for the large native-cost runs and routing states.
  A native run contains a 256-byte matrix plus metadata, and routing states
  contain the native-cost tracker. The default is unsuitable for these large
  elements. Zero capacity for trial and trailing-matrix storage avoids embedding
  a layout or matrix. Trial count defaults to the logical CPU count, but a
  single-trial run could use inline storage. These two choices are not required
  to compile; their source comments record the footprint tradeoff.
- QC measurement/reset temporaries hold exactly one qubit. Supported scalar
  folding expects one result, and the direct floating-point evaluator handles
  unary/binary operations. Standard gates have at most three parameters. Keep
  the corresponding capacities 1, 2, and 3.
- Benchmark controls and test fixtures have explicit cardinalities: two controls
  or targets, two/three matrix cases, 9/11/13 qubits, and at most three rotation
  angles. The Pauli-twirling helper's current callers produce zero or four Pauli
  operations. Keep those capacities. Fixed arrays are an optional local
  simplification for exact-size fixtures, not required API work.
- `CompilerTarget::Storage` takes vectors by value and moves them into owned
  storage. `joinBits` takes and reduces its own vector. Control-flow callbacks
  return owned vectors. `Wires` parameters passed by value to
  `generateGreedyLayout`, `getWindow`, and the `RoutingState` constructor own
  their traversal state. DD tensor and memref aliases are shared owning storage;
  `SlotLayout` stays local, and `MultiFn` returns an owned vector. Keep these
  ownership contracts.

## Validation and limits

- `uvx nox -s lint`: passed after inspecting and applying Markdown formatting.
- `uvx nox --non-interactive -s docs`: passed, including generated references,
  executable examples, and generated HTML link checks. Inspected the rendered
  policy subsection and its LLVM link. Verified the external LLVM target and its
  capacity guidance directly.
- Read the installed LLVM 23.1.0 `SmallVector.h`: the default aims for 64-byte
  objects and requires `sizeof(T) <= 256`.
- A temporary C++ sizeof probe on macOS arm64, using Apple Clang and the
  installed LLVM 23.1.0 headers/support library, measured the actual worklist
  and wrapped-line types: 144 versus 64 bytes and 624 versus 168 bytes. Mirrored
  private layouts measured a 2,064-byte capacity-32 plan vector versus 80 with
  the default, 368-byte tensor slots, and 272-byte native runs. Those private
  layout figures are diagnostic evidence, not ABI promises or speed results.
- Repeated the balanced-template inventory after the fixes: 1,409 spellings, of
  which 66 specify capacities in 26 files. All remaining capacities have a
  bound, a concrete storage reason, or LLVM's large-element requirement. No
  concrete-vector borrowed reference/pointer parameters remain in this scope;
  owning return types and callback results remain concrete containers.
- `cmake --preset release` and `cmake --build --preset release --parallel 8`:
  passed using Apple Clang and the installed LLVM/MLIR 23.1.0. The final rebuild
  passed after source comments were added.
- `ctest --preset release --parallel 4`: 3,948 passed, zero failed, and one
  existing QDMI job-ID test skipped. This includes all 2,883 MLIR tests.
- Final executable documentation rebuild passed with rebuilt Python bindings.
- `.nox/docs/bin/python -m pytest -q -o addopts= test/python/test_mlir_qiskit_translation.py`:
  all 441 passed against the rebuilt bindings. Disabled only the configured
  xdist option for this serial run.
- `uvx nox -s cpp-lint` with Homebrew LLVM 23.1.2 on `PATH`: passed with zero
  findings. Verified that all 25 changed C++ implementation/test files were
  checked on every line against `origin/main`; the header-verification target
  also passed. Staged the task diff so cpp-linter included uncommitted changes.
- No timing benchmark was performed.
