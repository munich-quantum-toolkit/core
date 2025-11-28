/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/Builder/FluxProgramBuilder.h"

#include "mlir/Dialect/Flux/IR/FluxDialect.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <utility>
#include <variant>

namespace mlir::flux {

FluxProgramBuilder::FluxProgramBuilder(MLIRContext* context)
    : OpBuilder(context), ctx(context), loc(getUnknownLoc()),
      module(create<ModuleOp>(loc)) {}

void FluxProgramBuilder::initialize() {
  // Ensure the Flux dialect is loaded
  ctx->loadDialect<FluxDialect>();

  // Set insertion point to the module body
  setInsertionPointToStart(module.getBody());

  // Create main function as entry point
  auto funcType = getFunctionType({}, {getI64Type()});
  auto mainFunc = create<func::FuncOp>(loc, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = getStringAttr("entry_point");
  mainFunc->setAttr("passthrough", getArrayAttr({entryPointAttr}));

  // Create entry block and set insertion point
  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  setInsertionPointToStart(&entryBlock);
}

Value FluxProgramBuilder::allocQubit() {
  auto allocOp = create<AllocOp>(loc);
  const auto qubit = allocOp.getResult();

  // Track the allocated qubit as valid
  validQubits.insert(qubit);

  return qubit;
}

Value FluxProgramBuilder::staticQubit(const int64_t index) {
  auto indexAttr = getI64IntegerAttr(index);
  auto staticOp = create<StaticOp>(loc, indexAttr);
  const auto qubit = staticOp.getQubit();

  // Track the static qubit as valid
  validQubits.insert(qubit);

  return qubit;
}

llvm::SmallVector<Value>
FluxProgramBuilder::allocQubitRegister(const int64_t size,
                                       const StringRef name) {
  llvm::SmallVector<Value> qubits;
  qubits.reserve(static_cast<size_t>(size));

  auto nameAttr = getStringAttr(name);
  auto sizeAttr = getI64IntegerAttr(size);

  for (int64_t i = 0; i < size; ++i) {
    auto indexAttr = getI64IntegerAttr(i);
    auto allocOp = create<AllocOp>(loc, nameAttr, sizeAttr, indexAttr);
    const auto& qubit = qubits.emplace_back(allocOp.getResult());
    // Track the allocated qubit as valid
    validQubits.insert(qubit);
  }

  return qubits;
}

FluxProgramBuilder::ClassicalRegister&
FluxProgramBuilder::allocClassicalBitRegister(int64_t size, StringRef name) {
  return allocatedClassicalRegisters.emplace_back(name, size);
}

//===----------------------------------------------------------------------===//
// Linear Type Tracking Helpers
//===----------------------------------------------------------------------===//

void FluxProgramBuilder::validateQubitValue(const Value qubit) const {
  if (!validQubits.contains(qubit)) {
    llvm::errs() << "Error: Attempting to use an invalid qubit SSA value. "
                 << "The value may have been consumed by a previous operation "
                 << "or was never created through this \n";
    llvm::reportFatalUsageError(
        "Invalid qubit value used (either consumed or not tracked)");
  }
}

void FluxProgramBuilder::updateQubitTracking(const Value inputQubit,
                                             const Value outputQubit) {
  // Validate the input qubit
  validateQubitValue(inputQubit);

  // Remove the input (consumed) value from tracking
  validQubits.erase(inputQubit);

  // Add the output (new) value to tracking
  validQubits.insert(outputQubit);
}

//===----------------------------------------------------------------------===//
// Measurement and Reset
//===----------------------------------------------------------------------===//

std::pair<Value, Value> FluxProgramBuilder::measure(Value qubit) {
  auto measureOp = create<MeasureOp>(loc, qubit);
  auto qubitOut = measureOp.getQubitOut();
  auto result = measureOp.getResult();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return {qubitOut, result};
}

Value FluxProgramBuilder::measure(Value qubit, const Bit& bit) {
  auto nameAttr = getStringAttr(bit.registerName);
  auto sizeAttr = getI64IntegerAttr(bit.registerSize);
  auto indexAttr = getI64IntegerAttr(bit.registerIndex);
  auto measureOp = create<MeasureOp>(loc, qubit, nameAttr, sizeAttr, indexAttr);
  const auto qubitOut = measureOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

Value FluxProgramBuilder::reset(Value qubit) {
  auto resetOp = create<ResetOp>(loc, qubit);
  const auto qubitOut = resetOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// OneTargetZeroParameter helpers

template <typename OpType>
Value FluxProgramBuilder::createOneTargetZeroParameter(const Value qubit) {
  auto op = create<OpType>(loc, qubit);
  const auto& qubitOut = op.getQubitOut();
  updateQubitTracking(qubit, qubitOut);
  return qubitOut;
}

template <typename OpType>
std::pair<Value, Value>
FluxProgramBuilder::createControlledOneTargetZeroParameter(const Value control,
                                                           const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0]);
             return op->getResults();
           });
  return {controlsOut[0], targetsOut[0]};
}

template <typename OpType>
std::pair<ValueRange, Value>
FluxProgramBuilder::createMultiControlledOneTargetZeroParameter(
    const ValueRange controls, const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0]);
             return op->getResults();
           });
  return {controlsOut, targetsOut[0]};
}

// OneTargetOneParameter helpers

template <typename OpType>
Value FluxProgramBuilder::createOneTargetOneParameter(
    const std::variant<double, Value>& parameter, const Value qubit) {
  auto op = create<OpType>(loc, qubit, parameter);
  const auto& qubitOut = op.getQubitOut();
  updateQubitTracking(qubit, qubitOut);
  return qubitOut;
}

template <typename OpType>
std::pair<Value, Value>
FluxProgramBuilder::createControlledOneTargetOneParameter(
    const std::variant<double, Value>& parameter, const Value control,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], parameter);
             return op->getResults();
           });
  return {controlsOut[0], targetsOut[0]};
}

template <typename OpType>
std::pair<ValueRange, Value>
FluxProgramBuilder::createMultiControlledOneTargetOneParameter(
    const std::variant<double, Value>& parameter, const ValueRange controls,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], parameter);
             return op->getResults();
           });
  return {controlsOut, targetsOut[0]};
}

// OneTargetTwoParameter helpers

template <typename OpType>
Value FluxProgramBuilder::createOneTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const Value qubit) {
  auto op = create<OpType>(loc, qubit, parameter1, parameter2);
  const auto& qubitOut = op.getQubitOut();
  updateQubitTracking(qubit, qubitOut);
  return qubitOut;
}

template <typename OpType>
std::pair<Value, Value>
FluxProgramBuilder::createControlledOneTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const Value control,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op =
                 b.create<OpType>(loc, targets[0], parameter1, parameter2);
             return op->getResults();
           });
  return {controlsOut[0], targetsOut[0]};
}

template <typename OpType>
std::pair<ValueRange, Value>
FluxProgramBuilder::createMultiControlledOneTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const ValueRange controls,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op =
                 b.create<OpType>(loc, targets[0], parameter1, parameter2);
             return op->getResults();
           });
  return {controlsOut, targetsOut[0]};
}

// OneTargetThreeParameter helpers

template <typename OpType>
Value FluxProgramBuilder::createOneTargetThreeParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2,
    const std::variant<double, Value>& parameter3, const Value qubit) {
  auto op = create<OpType>(loc, qubit, parameter1, parameter2, parameter3);
  const auto& qubitOut = op.getQubitOut();
  updateQubitTracking(qubit, qubitOut);
  return qubitOut;
}

template <typename OpType>
std::pair<Value, Value>
FluxProgramBuilder::createControlledOneTargetThreeParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2,
    const std::variant<double, Value>& parameter3, const Value control,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], parameter1,
                                              parameter2, parameter3);
             return op->getResults();
           });
  return {controlsOut[0], targetsOut[0]};
}

template <typename OpType>
std::pair<ValueRange, Value>
FluxProgramBuilder::createMultiControlledOneTargetThreeParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2,
    const std::variant<double, Value>& parameter3, const ValueRange controls,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], parameter1,
                                              parameter2, parameter3);
             return op->getResults();
           });
  return {controlsOut, targetsOut[0]};
}

// TwoTargetZeroParameter helpers

template <typename OpType>
std::pair<Value, Value>
FluxProgramBuilder::createTwoTargetZeroParameter(const Value qubit0,
                                                 const Value qubit1) {
  auto op = create<OpType>(loc, qubit0, qubit1);
  const auto& qubit0Out = op.getQubit0Out();
  const auto& qubit1Out = op.getQubit1Out();
  updateQubitTracking(qubit0, qubit0Out);
  updateQubitTracking(qubit1, qubit1Out);
  return {qubit0Out, qubit1Out};
}

template <typename OpType>
std::pair<Value, std::pair<Value, Value>>
FluxProgramBuilder::createControlledTwoTargetZeroParameter(const Value control,
                                                           const Value qubit0,
                                                           const Value qubit1) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, {qubit0, qubit1},
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], targets[1]);
             return op->getResults();
           });
  return {controlsOut[0], {targetsOut[0], targetsOut[1]}};
}

template <typename OpType>
std::pair<ValueRange, std::pair<Value, Value>>
FluxProgramBuilder::createMultiControlledTwoTargetZeroParameter(
    const ValueRange controls, const Value qubit0, const Value qubit1) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, {qubit0, qubit1},
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], targets[1]);
             return op->getResults();
           });
  return {controlsOut, {targetsOut[0], targetsOut[1]}};
}

// TwoTargetOneParameter helpers

template <typename OpType>
std::pair<Value, Value> FluxProgramBuilder::createTwoTargetOneParameter(
    const std::variant<double, Value>& parameter, const Value qubit0,
    const Value qubit1) {
  auto op = create<OpType>(loc, qubit0, qubit1, parameter);
  const auto& qubit0Out = op.getQubit0Out();
  const auto& qubit1Out = op.getQubit1Out();
  updateQubitTracking(qubit0, qubit0Out);
  updateQubitTracking(qubit1, qubit1Out);
  return {qubit0Out, qubit1Out};
}

template <typename OpType>
std::pair<Value, std::pair<Value, Value>>
FluxProgramBuilder::createControlledTwoTargetOneParameter(
    const std::variant<double, Value>& parameter, const Value control,
    const Value qubit0, const Value qubit1) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, {qubit0, qubit1},
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op =
                 b.create<OpType>(loc, targets[0], targets[1], parameter);
             return op->getResults();
           });
  return {controlsOut[0], {targetsOut[0], targetsOut[1]}};
}

template <typename OpType>
std::pair<ValueRange, std::pair<Value, Value>>
FluxProgramBuilder::createMultiControlledTwoTargetOneParameter(
    const std::variant<double, Value>& parameter, const ValueRange controls,
    const Value qubit0, const Value qubit1) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, {qubit0, qubit1},
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op =
                 b.create<OpType>(loc, targets[0], targets[1], parameter);
             return op->getResults();
           });
  return {controlsOut, {targetsOut[0], targetsOut[1]}};
}

// TwoTargetTwoParameter helpers

template <typename OpType>
std::pair<Value, Value> FluxProgramBuilder::createTwoTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const Value qubit0,
    const Value qubit1) {
  auto op = create<OpType>(loc, qubit0, qubit1, parameter1, parameter2);
  const auto& qubit0Out = op.getQubit0Out();
  const auto& qubit1Out = op.getQubit1Out();
  updateQubitTracking(qubit0, qubit0Out);
  updateQubitTracking(qubit1, qubit1Out);
  return {qubit0Out, qubit1Out};
}

template <typename OpType>
std::pair<Value, std::pair<Value, Value>>
FluxProgramBuilder::createControlledTwoTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const Value control,
    const Value qubit0, const Value qubit1) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, {qubit0, qubit1},
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], targets[1],
                                              parameter1, parameter2);
             return op->getResults();
           });
  return {controlsOut[0], {targetsOut[0], targetsOut[1]}};
}

template <typename OpType>
std::pair<ValueRange, std::pair<Value, Value>>
FluxProgramBuilder::createMultiControlledTwoTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const ValueRange controls,
    const Value qubit0, const Value qubit1) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, {qubit0, qubit1},
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], targets[1],
                                              parameter1, parameter2);
             return op->getResults();
           });
  return {controlsOut, {targetsOut[0], targetsOut[1]}};
}

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  Value FluxProgramBuilder::OP_NAME(const Value qubit) {                       \
    return createOneTargetZeroParameter<OP_CLASS>(qubit);                      \
  }                                                                            \
  std::pair<Value, Value> FluxProgramBuilder::c##OP_NAME(const Value control,  \
                                                         const Value target) { \
    return createControlledOneTargetZeroParameter<OP_CLASS>(control, target);  \
  }                                                                            \
  std::pair<ValueRange, Value> FluxProgramBuilder::mc##OP_NAME(                \
      const ValueRange controls, const Value target) {                         \
    return createMultiControlledOneTargetZeroParameter<OP_CLASS>(controls,     \
                                                                 target);      \
  }

DEFINE_ONE_TARGET_ZERO_PARAMETER(IdOp, id)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, x)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, y)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, z)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, h)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, s)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SdgOp, sdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, t)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TdgOp, tdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXOp, sx)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXdgOp, sxdg)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  Value FluxProgramBuilder::OP_NAME(const std::variant<double, Value>&(PARAM), \
                                    const Value qubit) {                       \
    return createOneTargetOneParameter<OP_CLASS>(PARAM, qubit);                \
  }                                                                            \
  std::pair<Value, Value> FluxProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM), const Value control,          \
      const Value target) {                                                    \
    return createControlledOneTargetOneParameter<OP_CLASS>(PARAM, control,     \
                                                           target);            \
  }                                                                            \
  std::pair<ValueRange, Value> FluxProgramBuilder::mc##OP_NAME(                \
      const std::variant<double, Value>&(PARAM), const ValueRange controls,    \
      const Value target) {                                                    \
    return createMultiControlledOneTargetOneParameter<OP_CLASS>(               \
        PARAM, controls, target);                                              \
  }

DEFINE_ONE_TARGET_ONE_PARAMETER(RXOp, rx, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RYOp, ry, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZOp, rz, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(POp, p, phi)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  Value FluxProgramBuilder::OP_NAME(                                           \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), const Value qubit) {         \
    return createOneTargetTwoParameter<OP_CLASS>(PARAM1, PARAM2, qubit);       \
  }                                                                            \
  std::pair<Value, Value> FluxProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), const Value control,         \
      const Value target) {                                                    \
    return createControlledOneTargetTwoParameter<OP_CLASS>(PARAM1, PARAM2,     \
                                                           control, target);   \
  }                                                                            \
  std::pair<ValueRange, Value> FluxProgramBuilder::mc##OP_NAME(                \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), const ValueRange controls,   \
      const Value target) {                                                    \
    return createMultiControlledOneTargetTwoParameter<OP_CLASS>(               \
        PARAM1, PARAM2, controls, target);                                     \
  }

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,   \
                                          PARAM3)                              \
  Value FluxProgramBuilder::OP_NAME(                                           \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), const Value qubit) {         \
    return createOneTargetThreeParameter<OP_CLASS>(PARAM1, PARAM2, PARAM3,     \
                                                   qubit);                     \
  }                                                                            \
  std::pair<Value, Value> FluxProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), const Value control,         \
      const Value target) {                                                    \
    return createControlledOneTargetThreeParameter<OP_CLASS>(                  \
        PARAM1, PARAM2, PARAM3, control, target);                              \
  }                                                                            \
  std::pair<ValueRange, Value> FluxProgramBuilder::mc##OP_NAME(                \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), const ValueRange controls,   \
      const Value target) {                                                    \
    return createMultiControlledOneTargetThreeParameter<OP_CLASS>(             \
        PARAM1, PARAM2, PARAM3, controls, target);                             \
  }

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  std::pair<Value, Value> FluxProgramBuilder::OP_NAME(Value qubit0,            \
                                                      Value qubit1) {          \
    return createTwoTargetZeroParameter<OP_CLASS>(qubit0, qubit1);             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> FluxProgramBuilder::c##OP_NAME(    \
      const Value control, Value qubit0, Value qubit1) {                       \
    return createControlledTwoTargetZeroParameter<OP_CLASS>(control, qubit0,   \
                                                            qubit1);           \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      FluxProgramBuilder::mc##OP_NAME(const ValueRange controls, Value qubit0, \
                                      Value qubit1) {                          \
    return createMultiControlledTwoTargetZeroParameter<OP_CLASS>(              \
        controls, qubit0, qubit1);                                             \
  }

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCXOp, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECROp, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  std::pair<Value, Value> FluxProgramBuilder::OP_NAME(                         \
      const std::variant<double, Value>&(PARAM), Value qubit0, Value qubit1) { \
    return createTwoTargetOneParameter<OP_CLASS>(PARAM, qubit0, qubit1);       \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> FluxProgramBuilder::c##OP_NAME(    \
      const std::variant<double, Value>&(PARAM), const Value control,          \
      Value qubit0, Value qubit1) {                                            \
    return createControlledTwoTargetOneParameter<OP_CLASS>(PARAM, control,     \
                                                           qubit0, qubit1);    \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      FluxProgramBuilder::mc##OP_NAME(                                         \
          const std::variant<double, Value>&(PARAM),                           \
          const ValueRange controls, Value qubit0, Value qubit1) {             \
    return createMultiControlledTwoTargetOneParameter<OP_CLASS>(               \
        PARAM, controls, qubit0, qubit1);                                      \
  }

DEFINE_TWO_TARGET_ONE_PARAMETER(RXXOp, rxx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RYYOp, ryy, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZXOp, rzx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZZOp, rzz, theta)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

// TwoTargetTwoParameter

#define DEFINE_TWO_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  std::pair<Value, Value> FluxProgramBuilder::OP_NAME(                         \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value qubit0,                \
      Value qubit1) {                                                          \
    return createTwoTargetTwoParameter<OP_CLASS>(PARAM1, PARAM2, qubit0,       \
                                                 qubit1);                      \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> FluxProgramBuilder::c##OP_NAME(    \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), const Value control,         \
      Value qubit0, Value qubit1) {                                            \
    return createControlledTwoTargetTwoParameter<OP_CLASS>(                    \
        PARAM1, PARAM2, control, qubit0, qubit1);                              \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      FluxProgramBuilder::mc##OP_NAME(                                         \
          const std::variant<double, Value>&(PARAM1),                          \
          const std::variant<double, Value>&(PARAM2),                          \
          const ValueRange controls, Value qubit0, Value qubit1) {             \
    return createMultiControlledTwoTargetTwoParameter<OP_CLASS>(               \
        PARAM1, PARAM2, controls, qubit0, qubit1);                             \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

std::pair<ValueRange, ValueRange> FluxProgramBuilder::ctrl(
    const ValueRange controls, const ValueRange targets,
    const std::function<ValueRange(OpBuilder&, ValueRange)>& body) {
  auto ctrlOp = create<CtrlOp>(loc, controls, targets, body);

  // Update tracking
  const auto& controlsOut = ctrlOp.getControlsOut();
  for (const auto& [control, controlOut] : llvm::zip(controls, controlsOut)) {
    updateQubitTracking(control, controlOut);
  }
  const auto& targetsOut = ctrlOp.getTargetsOut();
  for (const auto& [target, targetOut] : llvm::zip(targets, targetsOut)) {
    updateQubitTracking(target, targetOut);
  }

  return {controlsOut, targetsOut};
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

FluxProgramBuilder& FluxProgramBuilder::dealloc(Value qubit) {
  validateQubitValue(qubit);
  validQubits.erase(qubit);

  create<DeallocOp>(loc, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> FluxProgramBuilder::finalize() {
  // Automatically deallocate all remaining valid qubits
  for (const auto qubit : validQubits) {
    create<DeallocOp>(loc, qubit);
  }

  validQubits.clear();

  // Create constant 0 for successful exit code
  auto exitCode = create<arith::ConstantOp>(loc, getI64IntegerAttr(0));

  // Add return statement with exit code 0 to the main function
  create<func::ReturnOp>(loc, ValueRange{exitCode});

  return module;
}

} // namespace mlir::flux
