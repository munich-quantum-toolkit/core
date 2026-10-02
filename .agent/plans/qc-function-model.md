# Reusable QC functions and unitary calls

Status: complete in PR #2336 (`f57b6c341`). Historical implementation record;
see the [QCO companion](qco-function-model.md) for conversion and wire
semantics.

## Outcome and scope

QC represents reusable helpers as private `func.func` definitions. Generic
helpers use `func.call`; unitary gate definitions carry `mqt.unitary` and use
`qc.call`. The unitary call supports existing modifiers and analyses without
expanding the callee body. This is a compiler foundation for frontend gate
definitions, not a claim that every frontend already preserves functions.

## Decisions

- Reuse `func.func` for symbol and callable behavior. The discardable
  `mqt.unitary` unit attribute classifies a definition; `qc.call` implements
  `CallOpInterface`, `SymbolUserOpInterface`, and `qc::UnitaryOpInterface`. A
  separate function operation would duplicate existing MLIR facilities, and one
  `func.func` class cannot conditionally implement the unitary interface.
- A marked QC function is private, defined, and not an entry point. Its
  signature has zero or more `f64` parameters followed by one or more scalar
  `!qc.qubit` arguments and no results. Its body permits regionless,
  memory-effect-free scalar computations, QC unitary operations, and an empty
  `func.return`. Allocation, measurement, reset, and recursion are excluded.
  This supports the common gate-definition subset of OpenQASM and Qiskit.
- `QCProgramBuilder::createFunction` and `createUnitaryFunction` take complete
  callbacks and return function handles. They insert helpers before the entry
  function, infer generic result types, emit returns and local deallocations,
  and restore insertion state. Complete callbacks avoid paired start/end state
  and partially constructed callees.
- Allocation caches are function-local; allocation mode is module-wide because
  both QC/QCO conversions choose static or dynamic allocation for the module.
  Restoring the mode after each callback would incorrectly allow mixed modes.
- `QCProgramBuilder::call` validates same-module ownership and exact operand
  types before constructing IR, then selects `qc.call` or `func.call` from the
  marker. A borrowed scalar qubit is updated in place and cannot also be
  returned explicitly: conversion appends its latest value to the QCO ABI, so an
  explicit return would duplicate a linear value.

## Source and validation

- `mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td` and
  `mlir/lib/Dialect/MQT/IR/MQTDialect.cpp` own the marker and its verification.
- `mlir/include/mlir/Dialect/QC/IR/QCOps.td` and
  `mlir/lib/Dialect/QC/IR/Operations/CallOp.cpp` own the call contract.
- `mlir/include/mlir/Dialect/QC/Builder/QCProgramBuilder.h` and its
  implementation own complete helper construction and allocation state.
- `mlir/unittests/Dialect/QC/IR/test_qc_ir.cpp` covers markers, symbol and
  signature checks, modifier nesting, insertion restoration, generic results,
  and call selection.

Focused validation from the repository root:

```sh
cmake --build --preset release --target mqt-core-mlir-unittest-qc-ir
./build/release/mlir/unittests/Dialect/QC/IR/mqt-core-mlir-unittest-qc-ir
```

The implementation record reports a successful release build, 3,805 runnable
CTest cases with one additional expected skip, and full lint on 2026-09-03.
These are historical local results shared with the QCO companion, not a fresh
runtime validation performed during document cleanup.
