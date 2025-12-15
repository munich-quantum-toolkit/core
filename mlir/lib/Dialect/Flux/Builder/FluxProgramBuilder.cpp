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
#include <mlir/Dialect/SCF/IR/SCF.h>
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
  funcRegion = &mainFunc->getRegion(0);
  // Create entry block and set insertion point
  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  setInsertionPointToStart(&entryBlock);
}

Value FluxProgramBuilder::allocQubit() {
  auto allocOp = create<AllocOp>(loc);
  const auto qubit = allocOp.getResult();

  // Track the allocated qubit as valid
  validQubits[allocOp->getParentRegion()].insert(qubit);

  return qubit;
}

Value FluxProgramBuilder::staticQubit(const int64_t index) {
  if (index < 0) {
    llvm::reportFatalUsageError("Index must be non-negative");
  }

  auto indexAttr = getI64IntegerAttr(index);
  auto staticOp = create<StaticOp>(loc, indexAttr);
  const auto qubit = staticOp.getQubit();

  // Track the static qubit as valid
  validQubits[staticOp->getParentRegion()].insert(qubit);

  return qubit;
}

llvm::SmallVector<Value>
FluxProgramBuilder::allocQubitRegister(const int64_t size,
                                       const StringRef name) {
  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  llvm::SmallVector<Value> qubits;
  qubits.reserve(static_cast<size_t>(size));

  auto nameAttr = getStringAttr(name);
  auto sizeAttr = getI64IntegerAttr(size);

  for (int64_t i = 0; i < size; ++i) {
    auto indexAttr = getI64IntegerAttr(i);
    auto allocOp = create<AllocOp>(loc, nameAttr, sizeAttr, indexAttr);
    const auto& qubit = qubits.emplace_back(allocOp.getResult());
    // Track the allocated qubit as valid
    validQubits[allocOp->getParentRegion()].insert(qubit);
  }

  return qubits;
}

FluxProgramBuilder::ClassicalRegister&
FluxProgramBuilder::allocClassicalBitRegister(int64_t size, StringRef name) {
  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  return allocatedClassicalRegisters.emplace_back(name, size);
}

//===----------------------------------------------------------------------===//
// Linear Type Tracking Helpers
//===----------------------------------------------------------------------===//

void FluxProgramBuilder::validateQubitValue(Value qubit) {
  if (!validQubits[qubit.getParentRegion()].contains(qubit)) {
    llvm::errs() << "Attempting to use an invalid qubit SSA value. "
                 << "The value may have been consumed by a previous operation "
                 << "or was never created through this builder.\n";
    llvm::reportFatalUsageError(
        "Invalid qubit value used (either consumed or not tracked)");
  }
}

void FluxProgramBuilder::updateQubitTracking(Value inputQubit,
                                             Value outputQubit,
                                             Region* region) {
  // Validate the input qubit
  validateQubitValue(inputQubit);

  // Remove the input (consumed) value from tracking
  validQubits[region].erase(inputQubit);

  // Add the output (new) value to tracking
  validQubits[region].insert(outputQubit);
}

//===----------------------------------------------------------------------===//
// Measurement and Reset
//===----------------------------------------------------------------------===//

std::pair<Value, Value> FluxProgramBuilder::measure(Value qubit) {
  auto measureOp = create<MeasureOp>(loc, qubit);
  auto qubitOut = measureOp.getQubitOut();
  auto result = measureOp.getResult();

  // Update tracking
  updateQubitTracking(qubit, qubitOut, measureOp->getParentRegion());

  return {qubitOut, result};
}

Value FluxProgramBuilder::measure(Value qubit, const Bit& bit) {
  auto nameAttr = getStringAttr(bit.registerName);
  auto sizeAttr = getI64IntegerAttr(bit.registerSize);
  auto indexAttr = getI64IntegerAttr(bit.registerIndex);
  auto measureOp = create<MeasureOp>(loc, qubit, nameAttr, sizeAttr, indexAttr);
  const auto qubitOut = measureOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut, measureOp->getParentRegion());

  return qubitOut;
}

Value FluxProgramBuilder::reset(Value qubit) {
  auto resetOp = create<ResetOp>(loc, qubit);
  const auto qubitOut = resetOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut, resetOp->getParentRegion());

  return qubitOut;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// ZeroTargetOneParameter

#define DEFINE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  void FluxProgramBuilder::OP_NAME(                                            \
      const std::variant<double, Value>&(PARAM)) {                             \
    create<OP_CLASS>(loc, PARAM);                                              \
  }                                                                            \
  Value FluxProgramBuilder::c##OP_NAME(                                        \
      const std::variant<double, Value>&(PARAM), Value control) {              \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, {}, [&](OpBuilder& b, ValueRange /*targets*/) -> ValueRange { \
          b.create<OP_CLASS>(loc, PARAM);                                      \
          return {};                                                           \
        });                                                                    \
    return controlsOut[0];                                                     \
  }                                                                            \
  ValueRange FluxProgramBuilder::mc##OP_NAME(                                  \
      const std::variant<double, Value>&(PARAM), ValueRange controls) {        \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {},                                                     \
             [&](OpBuilder& b, ValueRange /*targets*/) -> ValueRange {         \
               b.create<OP_CLASS>(loc, PARAM);                                 \
               return {};                                                      \
             });                                                               \
    return controlsOut;                                                        \
  }

DEFINE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DEFINE_ZERO_TARGET_ONE_PARAMETER

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  Value FluxProgramBuilder::OP_NAME(Value qubit) {                             \
    auto op = create<OP_CLASS>(loc, qubit);                                    \
    const auto& qubitOut = op.getQubitOut();                                   \
    updateQubitTracking(qubit, qubitOut, op->getParentRegion());               \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> FluxProgramBuilder::c##OP_NAME(Value control,        \
                                                         Value target) {       \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](OpBuilder& b, ValueRange targets) -> ValueRange { \
          const auto op = b.create<OP_CLASS>(loc, targets[0]);                 \
          return op->getResults();                                             \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> FluxProgramBuilder::mc##OP_NAME(                \
      ValueRange controls, Value target) {                                     \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op = b.create<OP_CLASS>(loc, targets[0]);            \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
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
                                    Value qubit) {                             \
    auto op = create<OP_CLASS>(loc, qubit, PARAM);                             \
    const auto& qubitOut = op.getQubitOut();                                   \
    updateQubitTracking(qubit, qubitOut, op->getParentRegion());               \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> FluxProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM), Value control,                \
      Value target) {                                                          \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](OpBuilder& b, ValueRange targets) -> ValueRange { \
          const auto op = b.create<OP_CLASS>(loc, targets[0], PARAM);          \
          return op->getResults();                                             \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> FluxProgramBuilder::mc##OP_NAME(                \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value target) {                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op = b.create<OP_CLASS>(loc, targets[0], PARAM);     \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
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
      const std::variant<double, Value>&(PARAM2), Value qubit) {               \
    auto op = create<OP_CLASS>(loc, qubit, PARAM1, PARAM2);                    \
    const auto& qubitOut = op.getQubitOut();                                   \
    updateQubitTracking(qubit, qubitOut, op->getParentRegion());               \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> FluxProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control,               \
      Value target) {                                                          \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](OpBuilder& b, ValueRange targets) -> ValueRange { \
          const auto op = b.create<OP_CLASS>(loc, targets[0], PARAM1, PARAM2); \
          return op->getResults();                                             \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> FluxProgramBuilder::mc##OP_NAME(                \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target) {                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op =                                                 \
                   b.create<OP_CLASS>(loc, targets[0], PARAM1, PARAM2);        \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
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
      const std::variant<double, Value>&(PARAM3), Value qubit) {               \
    auto op = create<OP_CLASS>(loc, qubit, PARAM1, PARAM2, PARAM3);            \
    const auto& qubitOut = op.getQubitOut();                                   \
    updateQubitTracking(qubit, qubitOut, op->getParentRegion());               \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> FluxProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value control,               \
      Value target) {                                                          \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](OpBuilder& b, ValueRange targets) -> ValueRange { \
          const auto op =                                                      \
              b.create<OP_CLASS>(loc, targets[0], PARAM1, PARAM2, PARAM3);     \
          return op->getResults();                                             \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> FluxProgramBuilder::mc##OP_NAME(                \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), ValueRange controls,         \
      Value target) {                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op = b.create<OP_CLASS>(loc, targets[0], PARAM1,     \
                                                  PARAM2, PARAM3);             \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
  }

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  std::pair<Value, Value> FluxProgramBuilder::OP_NAME(Value qubit0,            \
                                                      Value qubit1) {          \
    auto op = create<OP_CLASS>(loc, qubit0, qubit1);                           \
    const auto& qubit0Out = op.getQubit0Out();                                 \
    const auto& qubit1Out = op.getQubit1Out();                                 \
    updateQubitTracking(qubit0, qubit0Out, op->getParentRegion());             \
    updateQubitTracking(qubit1, qubit1Out, op->getParentRegion());             \
    return {qubit0Out, qubit1Out};                                             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> FluxProgramBuilder::c##OP_NAME(    \
      Value control, Value qubit0, Value qubit1) {                             \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(control, {qubit0, qubit1},                                        \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op =                                                 \
                   b.create<OP_CLASS>(loc, targets[0], targets[1]);            \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut[0], {targetsOut[0], targetsOut[1]}};                   \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      FluxProgramBuilder::mc##OP_NAME(ValueRange controls, Value qubit0,       \
                                      Value qubit1) {                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {qubit0, qubit1},                                       \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op =                                                 \
                   b.create<OP_CLASS>(loc, targets[0], targets[1]);            \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut, {targetsOut[0], targetsOut[1]}};                      \
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
    auto op = create<OP_CLASS>(loc, qubit0, qubit1, PARAM);                    \
    const auto& qubit0Out = op.getQubit0Out();                                 \
    const auto& qubit1Out = op.getQubit1Out();                                 \
    updateQubitTracking(qubit0, qubit0Out, op->getParentRegion());             \
    updateQubitTracking(qubit1, qubit1Out, op->getParentRegion());             \
    return {qubit0Out, qubit1Out};                                             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> FluxProgramBuilder::c##OP_NAME(    \
      const std::variant<double, Value>&(PARAM), Value control, Value qubit0,  \
      Value qubit1) {                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(control, {qubit0, qubit1},                                        \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op =                                                 \
                   b.create<OP_CLASS>(loc, targets[0], targets[1], PARAM);     \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut[0], {targetsOut[0], targetsOut[1]}};                   \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      FluxProgramBuilder::mc##OP_NAME(                                         \
          const std::variant<double, Value>&(PARAM), ValueRange controls,      \
          Value qubit0, Value qubit1) {                                        \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {qubit0, qubit1},                                       \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op =                                                 \
                   b.create<OP_CLASS>(loc, targets[0], targets[1], PARAM);     \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut, {targetsOut[0], targetsOut[1]}};                      \
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
    auto op = create<OP_CLASS>(loc, qubit0, qubit1, PARAM1, PARAM2);           \
    const auto& qubit0Out = op.getQubit0Out();                                 \
    const auto& qubit1Out = op.getQubit1Out();                                 \
    updateQubitTracking(qubit0, qubit0Out, op->getParentRegion());             \
    updateQubitTracking(qubit1, qubit1Out, op->getParentRegion());             \
    return {qubit0Out, qubit1Out};                                             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> FluxProgramBuilder::c##OP_NAME(    \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control, Value qubit0, \
      Value qubit1) {                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(control, {qubit0, qubit1},                                        \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op = b.create<OP_CLASS>(loc, targets[0], targets[1], \
                                                  PARAM1, PARAM2);             \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut[0], {targetsOut[0], targetsOut[1]}};                   \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      FluxProgramBuilder::mc##OP_NAME(                                         \
          const std::variant<double, Value>&(PARAM1),                          \
          const std::variant<double, Value>&(PARAM2), ValueRange controls,     \
          Value qubit0, Value qubit1) {                                        \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {qubit0, qubit1},                                       \
             [&](OpBuilder& b, ValueRange targets) -> ValueRange {             \
               const auto op = b.create<OP_CLASS>(loc, targets[0], targets[1], \
                                                  PARAM1, PARAM2);             \
               return op->getResults();                                        \
             });                                                               \
    return {controlsOut, {targetsOut[0], targetsOut[1]}};                      \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

ValueRange FluxProgramBuilder::barrier(ValueRange qubits) {
  auto op = create<BarrierOp>(loc, qubits);
  const auto& qubitsOut = op.getQubitsOut();
  for (const auto& [inputQubit, outputQubit] : llvm::zip(qubits, qubitsOut)) {
    updateQubitTracking(inputQubit, outputQubit, op->getParentRegion());
  }
  return qubitsOut;
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

std::pair<ValueRange, ValueRange> FluxProgramBuilder::ctrl(
    ValueRange controls, ValueRange targets,
    const std::function<ValueRange(OpBuilder&, ValueRange)>& body) {
  auto ctrlOp = create<CtrlOp>(loc, controls, targets, body);

  // Update tracking
  const auto& controlsOut = ctrlOp.getControlsOut();
  for (const auto& [control, controlOut] : llvm::zip(controls, controlsOut)) {
    updateQubitTracking(control, controlOut, ctrlOp->getParentRegion());
  }
  const auto& targetsOut = ctrlOp.getTargetsOut();
  for (const auto& [target, targetOut] : llvm::zip(targets, targetsOut)) {
    updateQubitTracking(target, targetOut, ctrlOp->getParentRegion());
  }

  return {controlsOut, targetsOut};
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

FluxProgramBuilder& FluxProgramBuilder::dealloc(Value qubit) {
  validateQubitValue(qubit);
  validQubits[qubit.getParentRegion()].erase(qubit);

  create<DeallocOp>(loc, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> FluxProgramBuilder::finalize() {
  // Automatically deallocate all remaining valid qubits
  for (const auto qubit : validQubits[funcRegion]) {
    create<DeallocOp>(loc, qubit);
  }

  validQubits.clear();

  // Create constant 0 for successful exit code
  auto exitCode = create<arith::ConstantOp>(loc, getI64IntegerAttr(0));

  // Add return statement with exit code 0 to the main function
  create<func::ReturnOp>(loc, ValueRange{exitCode});

  return module;
}
Value FluxProgramBuilder::arithConstantIndex(int i) {

  const auto op =
      create<arith::ConstantOp>(loc, getIndexType(), getIndexAttr(i));
  return op->getResult(0);
}

Value FluxProgramBuilder::arithConstantBool(bool b) {
  const auto i1Type = getI1Type();
  const auto op =
      b ? create<arith::ConstantOp>(loc, i1Type, getIntegerAttr(i1Type, 1))
        : create<arith::ConstantOp>(loc, i1Type, getIntegerAttr(i1Type, 0));
  return op->getResult(0);
}

ValueRange
FluxProgramBuilder::scfFor(Value lowerbound, Value upperbound, Value step,
                           const std::function<void(OpBuilder&)>& body) {
  auto op = create<scf::ForOp>(loc, lowerbound, upperbound, step, ValueRange{},
                               [&](OpBuilder& b, Location, Value, ValueRange) {
                                 body(b); // adapt
                                 b.create<scf::YieldOp>(loc);
                               });

  return op->getResults();
}
} // namespace mlir::flux
