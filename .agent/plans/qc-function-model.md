# Add reusable QC functions and unitary calls

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently represents an imported quantum program as one QC function.
After this change, a frontend can preserve a reusable helper as a private
`func.func`, call a generic helper with `func.call`, and call a gate definition
with `qc.call`. A `qc.call` is a unitary operation, so the existing QC modifier
and analysis code can handle a custom gate without expanding its body.

The focused QC IR unit tests demonstrate the feature by building a generic
helper and a marked unitary helper with `QCProgramBuilder`, verifying the
module, and checking that the builder selected the correct call operation.

## Progress

- [x] (2026-09-02 21:17Z) Inspected the MQT metadata dialect, QC operation
  interfaces, QC builder state, and current QC unit tests.
- [x] (2026-09-02 22:05Z) Added the frontend-neutral unitary function marker and
      its QC verifier.
- [x] (2026-09-02 22:05Z) Added `qc.call` as a unitary, symbol-using call
  operation.
- [x] (2026-09-02 22:05Z) Added callback-complete QC builder APIs for generic
  and unitary functions.
- [x] (2026-09-02 22:08Z) Added and passed focused marker, call, modifier, and
  builder tests; all 341 QC IR unit tests pass.
- [ ] Update the general compiler launch changelog entry.
- [x] (2026-09-02) Ran all focused QC tests, a fresh release build, all 3,618
  runnable repository tests, and lint.

## Surprises & Discoveries

- Observation: The MQT dialect already links the QC and QCO dialect libraries
  and already verifies operation, function-argument, and function-result
  metadata. Evidence: `mlir/lib/Dialect/MQT/IR/CMakeLists.txt` and
  `MQTDialect::verifyOperationAttribute` provide the required ownership point
  without a new dialect or library.
- Observation: The existing `build/release` cache still referenced MLIR 22 and
  failed in unrelated current-main APIs. Evidence: configuring the same preset
  with `--fresh` selected the repository-configured MLIR 23.1.0 installation,
  after which the QC target built successfully.

## Decision Log

- Decision: Keep all definitions as private `func.func` operations and mark
  unitary definitions with the discardable `mqt.unitary` unit attribute.
  Rationale: `func.func` already supplies MLIR symbol and callable behavior;
  another function operation would duplicate it. Date/Author: 2026-09-02, Codex.
- Decision: Implement unitary behavior on `qc.call`, not on `func.func`.
  Rationale: MLIR operation interfaces are fixed by operation class, while one
  `func.func` class must represent both generic and unitary functions.
  Date/Author: 2026-09-02, Codex.
- Decision: A QC unitary function accepts zero or more `f64` parameters followed
  by one or more scalar `!qc.qubit` arguments and returns no values. Its body
  contains only pure scalar computations, QC unitary operations, and an empty
  `func.return`. Rationale: This is the common subset required by OpenQASM gate
  definitions and Qiskit `Gate` objects. Date/Author: 2026-09-02, Codex.
- Decision: Use complete builder callbacks and a `func::FuncOp` handle at call
  sites. Rationale: This restores insertion state automatically, infers result
  types from the completed body, and avoids paired start/end calls and
  string-only symbol references. Date/Author: 2026-09-02, Codex.

## Outcomes & Retrospective

QC now represents reusable generic functions with `func.call` and unitary gate
definitions with the small `mqt.unitary` plus `qc.call` contract. The builder
uses complete callbacks, so helper construction cannot leak insertion or
allocation state into the entry point. No new function operation, symbol
abstraction, or dependency was needed. The implementation passes the complete
test suite in a fresh release build.

## Context and Orientation

`mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td` declares frontend-neutral
discardable attributes. `mlir/lib/Dialect/MQT/IR/MQTDialect.cpp` verifies those
attributes. `mlir/include/mlir/Dialect/QC/IR/QCInterfaces.td` defines
`qc::UnitaryOpInterface`; modifier operations and frontend exporters use this
interface to recognize unitary operations.
`mlir/include/mlir/Dialect/QC/IR/QCOps.td` defines QC operations.
`mlir/include/mlir/Dialect/QC/Builder/QCProgramBuilder.h` and its implementation
build complete QC modules and track allocations in the current function.

The new `mqt.unitary` marker classifies a function definition. The new `qc.call`
operation refers to such a definition and implements both MLIR's
`CallOpInterface` and QC's `UnitaryOpInterface`. A generic function remains a
normal `func.func` called with `func.call`.

## Plan of Work

Extend the MQT dialect's discardable attributes with `mqt.unitary`. Add inline
query support and an out-of-line setter beside the existing entry-point helpers.
The MQT verifier must accept the attribute only as a unit attribute on a
private, defined, non-entry `func.func`. For a QC signature, require `f64`
parameters before scalar qubits and no results. Walk the body and accept only
regionless, memory-effect-free scalar operations, QC unitary operations, and an
empty function return.

Define `qc.call` in `QCOps.td` with a symbol reference and a variadic operand
list. Implement `CallOpInterface`, `SymbolUserOpInterface`, and
`qc::UnitaryOpInterface`. The symbol-use verifier resolves a private marked
function and checks the exact operand signature. The unitary interface treats
all trailing qubit operands as targets and all leading operands as parameters.

Add `createFunction`, `createUnitaryFunction`, and `call` to `QCProgramBuilder`.
Each creation method inserts one complete private helper before the entry
function under an insertion guard. It resets allocation state for the callback,
emits deallocations for local values that are not returned, sets the inferred
function result types, emits `func.return`, and restores the entry-function
state. `call` emits `qc.call` for a marked function and `func.call` otherwise.

Add focused tests to `mlir/unittests/Dialect/QC/IR/test_qc_ir.cpp`. The tests
must verify valid and invalid unitary markers, symbol/signature checking,
modifier nesting, insertion restoration, generic result inference, and call
selection. Append the eventual pull request reference and contributors to the
existing general compiler launch changelog entry. Do not create another
changelog bullet.

## Concrete Steps

Run all commands from the repository root.

Build and run the focused test while iterating:

    cmake --build --preset release --target mqt-core-mlir-unittest-qc-ir
    ./build/release/mlir/unittests/Dialect/QC/IR/mqt-core-mlir-unittest-qc-ir

Run final validation:

    cmake --preset release
    cmake --build --preset release
    ctest --preset release
    uvx nox -s lint

The focused binary and CTest must report no failures. Lint must finish without
modifying tracked files; if it formats the implementation, inspect the edits and
rerun the affected checks.

## Validation and Acceptance

A parsed private marked QC helper with only unitary operations verifies. A
marked entry point, declaration, result-bearing function, non-`f64` parameter,
non-qubit trailing argument, allocation, measurement, reset, or recursive call
does not verify.

`QCProgramBuilder::createFunction` returns a private function whose result types
match its callback results. `createUnitaryFunction` returns a marked resultless
function. `call` emits `func.call` for the first and `qc.call` for the second.
After either creation callback, subsequent entry operations remain in `main`. A
`qc.call` can appear in a QC inverse or control modifier because it implements
`qc::UnitaryOpInterface`.

## Idempotence and Recovery

All builds and tests are repeatable. The implementation is isolated on a fresh
branch from `origin/main`. If an edit fails, inspect `git diff` and use another
focused patch; do not reset or discard unrelated work. No remote action is part
of this ExecPlan until the complete branch passes validation.

## Artifacts and Notes

The repository was clean before the branch was created. The branch starts at the
current `origin/main` commit.

## Interfaces and Dependencies

The public C++ builder interface will contain:

    func::FuncOp createFunction(
        StringRef name, TypeRange argumentTypes,
        function_ref<SmallVector<Value>(ValueRange)> body);
    func::FuncOp createUnitaryFunction(
        StringRef name, TypeRange argumentTypes,
        function_ref<void(ValueRange)> body);
    SmallVector<Value> call(func::FuncOp callee, ValueRange operands);

No new dependency is required. The implementation uses the existing Func, MQT,
QC, and MLIR call/symbol interfaces.
