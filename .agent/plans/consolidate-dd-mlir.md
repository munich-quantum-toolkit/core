# Consolidate DD gate semantics with QCO

Status: historical implementation record.

## Goal and scope

MQT Core currently implements every supported named quantum gate twice: once as
a QCO operation matrix and once in the low-level decision-diagram (DD) package.
After this change, QCO is the sole source of named-gate semantics. The QCO
interpreter and QIR runtime obtain canonical QCO matrices through one small
adapter and turn those matrices into DD operations. The exported `MQT::CoreDD`
library remains an MLIR-independent collection of backend-neutral DD data
structures and matrix-to-DD primitives, including in builds configured with
`BUILD_MQT_CORE_MLIR=OFF`.

Users can observe the result by running the existing QCO DD simulation and QIR
runtime tests: fixed, parameterized, controlled, and multi-qubit gates retain
their behavior. A DD-only build still configures and tests successfully, but the
obsolete public `dd::GateType`, `dd::opTo*GateMatrix`, and `dd::getGateDD`
convenience API no longer exists in the v4 C++ surface.

## Constraints

- The QCO DD interpreter cannot rely solely on
  `UnitaryOpInterface::getUnitaryMatrix()`. That interface only sees constants
  present in IR, while `DDArgumentBindings` also supplies concrete values for
  runtime SSA parameters. The standard-gate path must continue resolving those
  bindings before calling each operation's static QCO matrix factory.

- Controlled QCO operations must keep a base matrix plus sparse `dd::Controls`.
  Materializing `CtrlOp`'s full dense matrix would grow exponentially with the
  number of controls and lose the existing DD fast path.

- QIR Runtime and JIT are internal build-tree targets, whereas CoreDD is
  installed and exported. Making CoreDD depend on a QCO target would give the
  installed static library an unexported dependency and reverse the intended
  dependency direction.

- DD-only builds and tests are intentional and documented. Moving CoreDD under
  `mlir/` would also make it unavailable when bindings and the current DD test
  subtree are configured earlier in the top-level build.

- `dd::applyGlobalPhase` does not duplicate named-gate semantics; it is a
  backend-neutral operation that mutates and returns a `VectorDD`. Keeping it
  also avoids an unrelated v4 API removal.

- `llvm::DenseMap<dd::Qubit, ...>` cannot key the two largest valid qubit
  indices because LLVM reserves those values as sentinels. The arbitrary-target
  adapter instead sorts its operands once; its recursion follows target levels
  only and adds intervening identity levels iteratively.

- Integration tests that build their expected DD through the new adapter can
  hide conversion and operand-order defects. Direct adapter tests therefore
  compare the specialized paths with raw DD constructors and the
  arbitrary-target path with an independently embedded dense matrix.

- QCO's fixed one- and two-qubit matrices already use the row-major layout
  consumed by the DD package. Read-only entry views let the adapter call the
  package's specialized builders without allocating and copying an intermediate
  DD matrix.

- The actual #2334 merge adds compiler-input Python helpers, not another DD
  construction primitive. Those helpers already route through the QCO
  interpreter and therefore inherit this work without another adapter.

## Decisions

- Keep `MQT::CoreDD` unconditional, MLIR-independent, and exported. Rationale:
  Matrix-to-DD construction is backend-neutral and useful without a program
  dialect; QCO should depend on this primitive layer, never the reverse.

- Add one internal `MLIRQCODDAdapter` shared by the QCO interpreter and QIR
  runtime. Rationale: These are the two real consumers that translate canonical
  QCO matrices to DD nodes; sharing only this narrow conversion avoids linking
  QIR to the complete interpreter.

- Remove the DD-specific operation column from `GateTable.def` and generate QIR
  dispatch directly from the existing QCO operation key. Rationale: A second
  named-gate enum is redundant once QCO owns the matrices.

- Keep measurement projector matrices private to `Package.cpp` and retain raw DD
  matrix aliases and constructors. Rationale: Projectors are DD implementation
  details, while raw matrices are the backend-neutral boundary.

- Retain `dd::applyGlobalPhase` in `dd/Operations.hpp` while removing
  `dd::getGateDD`. Rationale: Applying a scalar phase is a raw DD primitive,
  unlike the removed enum-to-named-matrix dispatch.

- Treat qubit-range, uniqueness, and control/target disjointness as verified-IR
  invariants at the internal adapter boundary. Rationale: QCO execution and
  generated QIR may assume valid IR; duplicating their verifiers in the matrix
  adapter would add guardrails for unreachable inputs. The adapter retains the
  existing matrix-arity check needed by dynamic custom unitaries.

- Expose row-major span overloads on the raw DD builders rather than replace
  their one-, two-, and three-qubit algorithms with the arbitrary-size adapter
  recursion. Rationale: The specialized builders are the package's idiomatic
  fast paths; a view changes only storage access and keeps CoreDD independent of
  QCO.

- Keep the QIR control-array copy. Rationale: The control-array ABI requires
  aligned pointer storage with an owning lifetime. Copy its contiguous storage
  once rather than invoking the element accessor for each pointer.

- Replace the earlier dynamic-RCCX decision with `Matrix8x8` following explicit
  user approval. Rationale: RCCX has a fixed three-qubit matrix and can use the
  same allocation-free path as smaller named gates. Add only the access,
  adjoint, comparison, and assignment operations actually needed here, not a
  speculative general-purpose 8x8 numerical API.

- Store a named gate's resolved parameters and DD-builder pointer, rather than a
  matrix variant. Rationale: No gate needs to carry storage sized for the
  largest matrix; the selected builder constructs its native matrix immediately
  before the shared DD adapter consumes it.

## Outcome and validation

QCO now owns every named-gate matrix used by MQT Core. One internal adapter
turns those matrices into DDs for both direct QCO interpretation and QIR
execution. CoreDD remains an installed, MLIR-independent primitive library; its
raw matrix constructors and global-phase operation remain available, while the
duplicate `dd::GateType`, named matrix formulas, and dispatch are gone.

Historical validation passed the focused QCO, QIR, standalone DD, and Python
DD-helper suites and repository lint. Full C++ lint did not complete because of
a QTensor link failure; direct checks were supplemental, not a substitute for
that gate.

The fixed RCCX matrix now follows the same storage-view path as smaller gates.
Unitary-interface extraction uses one shared generated body for all five matrix
types, and named-gate dispatch holds only resolved parameters and a builder
pointer rather than a temporary dynamic matrix. No changes to #2334's public
Python contracts were needed.

The QIR runtime now consumes QCO matrix views directly, translates controls and
targets in one allocation, and avoids exception-driven static-address detection.
Its internal template utilities were reduced to one fold-expression helper. The
deliberately retained boundary is raw matrix-to-DD construction; moving CoreDD
under MLIR or removing backend-neutral primitives would make standalone DD
builds impossible and was therefore not part of this consolidation.

## Code and ownership

The exported low-level library is built in `src/dd` and exposed as
`MQT::CoreDD`. `include/mqt-core/dd/GateMatrixDefinitions.hpp` and
`src/dd/GateMatrixDefinitions.cpp` currently define a DD-specific gate enum and
29 named-gate formulas. `include/mqt-core/dd/Operations.hpp` and
`src/dd/Operations.cpp` dispatch those formulas into raw constructors on
`dd::Package`.

The canonical formulas already exist on QCO standard operations declared in
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td` and implemented below
`mlir/lib/Dialect/QCO/IR/Operations/StandardGates`. Fixed gates expose static
`getUnitaryMatrix()` factories; parameterized gates expose static
`unitaryMatrix(...)` factories. Their matrix storage types live in
`mlir/Dialect/QCO/Utils/Matrix.h`.

`mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp` interprets QCO directly into a
DD. `mlir/lib/Dialect/QIR/Execution/Runtime` implements QIR quantum instruction
entry points on the same package. Both currently route named operations through
`dd::GateType`. `mlir/include/mlir/Conversion/GateTable.def` contains an `OP`
column solely to spell those DD enumerators; all other conversion consumers
ignore it.

The change is limited to the gate registry, the QCO-to-DD adapter and
interpreter, QIR runtime, DD implementation details, their direct tests, and v4
documentation. It must not expose the internal MLIR adapter in the installed
CoreDD target or change the behavior and ordering of existing gate matrices.

## Acceptance

All QCO DD functionality tests must pass, including fixed, parameterized,
runtime-bound, controlled, RCCX, custom-unitary, reset, and sampling cases. All
QIR runtime tests must pass, including direct, fixed-control, generic-control,
parameterized, SWAP, global-phase, reset, and state ownership cases. DDSIM QDMI
tests must continue accepting and executing supported OpenQASM and QIR jobs.

The DD-only configuration must build `MQT::CoreDD` and pass its package tests
without LLVM or MLIR targets. The generated installed CoreDD interface must not
mention an MLIR target. No production source may define a second named-gate
formula or `dd::GateType`. QCO matrices must remain the only formula source and
the central gate table must no longer carry a DD operation spelling.

`git diff --check`, C++ lint, and `uvx nox -s lint` must pass, unless an
external tool failure is recorded with its exact diagnostic and does not
originate in the change.

## Interfaces

`MQT::CoreDD` remains the installed backend-neutral library and publishes raw
matrix aliases plus `dd::Package` constructors. It has no MLIR dependency.

`MLIRQCODDAdapter` is an internal MLIR library with public build dependencies on
`MLIRQCODialect`, `MLIRQCOMatrix`, and `MQT::CoreDD`. It is consumed by
`MLIRQCODDFunctionality` and `MQT::CoreQIRRuntime`; it is not appended to
`MQT_CORE_TARGETS` or exported as part of the installed CMake package.

The QIR C ABI remains unchanged. The C++ DD convenience surface removes
`dd::GateType`, `dd::opToSingleQubitGateMatrix`, `dd::opToTwoQubitGateMatrix`,
`dd::opToThreeQubitGateMatrix`, and `dd::getGateDD` for v4. Existing callers
construct DDs from raw matrices or use the QCO-owned adapter when compiling
inside the MLIR stack.
