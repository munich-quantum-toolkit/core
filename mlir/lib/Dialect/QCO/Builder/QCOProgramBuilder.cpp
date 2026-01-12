/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
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
#include <mlir/Support/LLVM.h>
#include <string>
#include <utility>
#include <variant>

namespace mlir::qco {

QCOProgramBuilder::QCOProgramBuilder(MLIRContext* context)
    : OpBuilder(context), ctx(context), loc(getUnknownLoc()),
      module(ModuleOp::create(loc)) {
  ctx->loadDialect<QCODialect>();
}

void QCOProgramBuilder::initialize() {
  // Set insertion point to the module body
  setInsertionPointToStart(module.getBody());

  // Create main function as entry point
  auto funcType = getFunctionType({}, {getI64Type()});
  auto mainFunc = func::FuncOp::create(*this, loc, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = getStringAttr("entry_point");
  mainFunc->setAttr("passthrough", getArrayAttr({entryPointAttr}));
  // Create entry block and set insertion point
  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  setInsertionPointToStart(&entryBlock);
}

Value QCOProgramBuilder::allocQubit() {
  checkFinalized();

  auto allocOp = AllocOp::create(*this, loc);
  const auto qubit = allocOp.getResult();

  // Track the allocated qubit as valid
  validQubits[allocOp->getParentRegion()].insert(qubit);

  return qubit;
}

Value QCOProgramBuilder::staticQubit(const int64_t index) {
  checkFinalized();

  if (index < 0) {
    llvm::reportFatalUsageError("Index must be non-negative");
  }

  auto indexAttr = getI64IntegerAttr(index);
  auto staticOp = StaticOp::create(*this, loc, indexAttr);
  const auto qubit = staticOp.getQubit();

  // Track the static qubit as valid
  validQubits[staticOp->getParentRegion()].insert(qubit);

  return qubit;
}

llvm::SmallVector<Value>
QCOProgramBuilder::allocQubitRegister(const int64_t size,
                                      const std::string& name) {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  llvm::SmallVector<Value> qubits;
  qubits.reserve(static_cast<size_t>(size));

  auto nameAttr = getStringAttr(name);
  auto sizeAttr = getI64IntegerAttr(size);

  for (int64_t i = 0; i < size; ++i) {
    const auto indexAttr = getI64IntegerAttr(i);
    auto allocOp = AllocOp::create(*this, loc, nameAttr, sizeAttr, indexAttr);
    const auto& qubit = qubits.emplace_back(allocOp.getResult());
    // Track the allocated qubit as valid
    validQubits[allocOp->getParentRegion()].insert(qubit);
  }

  return qubits;
}

QCOProgramBuilder::ClassicalRegister
QCOProgramBuilder::allocClassicalBitRegister(const int64_t size,
                                             std::string name) const {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  return {.name = std::move(name), .size = size};
}

//===----------------------------------------------------------------------===//
// Linear Type Tracking Helpers
//===----------------------------------------------------------------------===//

void QCOProgramBuilder::validateQubitValue(Value qubit, Region* region) const {
  auto qubits = validQubits.lookup(region);

  if (qubits.empty() || !qubits.contains(qubit)) {
    llvm::errs() << "Attempting to use an invalid qubit SSA value. "
                 << "The value may have been consumed by a previous operation "
                 << "or was never created through this builder.\n";
    llvm::reportFatalUsageError(
        "Invalid qubit value used (either consumed or not tracked)");
  }
}

void QCOProgramBuilder::updateQubitTracking(Value inputQubit, Value outputQubit,
                                            Region* region) {
  // Validate the input qubit
  validateQubitValue(inputQubit, region);
  // Remove the input (consumed) value from tracking
  validQubits[region].erase(inputQubit);
  // Add the output (new) value to tracking
  validQubits[region].insert(outputQubit);
}

//===----------------------------------------------------------------------===//
// Measurement and Reset
//===----------------------------------------------------------------------===//

std::pair<Value, Value> QCOProgramBuilder::measure(Value qubit) {
  checkFinalized();

  auto measureOp = MeasureOp::create(*this, loc, qubit);
  auto qubitOut = measureOp.getQubitOut();
  auto result = measureOp.getResult();

  // Update tracking
  updateQubitTracking(qubit, qubitOut, measureOp->getParentRegion());

  return {qubitOut, result};
}

Value QCOProgramBuilder::measure(Value qubit, const Bit& bit) {
  checkFinalized();

  auto nameAttr = getStringAttr(bit.registerName);
  auto sizeAttr = getI64IntegerAttr(bit.registerSize);
  auto indexAttr = getI64IntegerAttr(bit.registerIndex);
  auto measureOp =
      MeasureOp::create(*this, loc, qubit, nameAttr, sizeAttr, indexAttr);
  const auto qubitOut = measureOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut, measureOp->getParentRegion());

  return qubitOut;
}

Value QCOProgramBuilder::reset(Value qubit) {
  checkFinalized();

  auto resetOp = ResetOp::create(*this, loc, qubit);
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
  void QCOProgramBuilder::OP_NAME(const std::variant<double, Value>&(PARAM)) { \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, loc, PARAM);                                       \
  }                                                                            \
  Value QCOProgramBuilder::c##OP_NAME(                                         \
      const std::variant<double, Value>&(PARAM), Value control) {              \
    checkFinalized();                                                          \
    const auto controlsOut =                                                   \
        ctrl(control, {},                                                      \
             [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {         \
               OP_NAME(PARAM);                                                 \
               return {};                                                      \
             })                                                                \
            .first;                                                            \
    return controlsOut[0];                                                     \
  }                                                                            \
  ValueRange QCOProgramBuilder::mc##OP_NAME(                                   \
      const std::variant<double, Value>&(PARAM), ValueRange controls) {        \
    checkFinalized();                                                          \
    const auto controlsOut =                                                   \
        ctrl(controls, {},                                                     \
             [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {         \
               OP_NAME(PARAM);                                                 \
               return {};                                                      \
             })                                                                \
            .first;                                                            \
    return controlsOut;                                                        \
  }

DEFINE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DEFINE_ZERO_TARGET_ONE_PARAMETER

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  Value QCOProgramBuilder::OP_NAME(Value qubit) {                              \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, loc, qubit);                             \
    const auto& qubitOut = op.getQubitOut();                                   \
    updateQubitTracking(qubit, qubitOut, op->getParentRegion());               \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> QCOProgramBuilder::c##OP_NAME(Value control,         \
                                                        Value target) {        \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](ValueRange targets) -> llvm::SmallVector<Value> { \
          return {OP_NAME(targets[0])};                                        \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> QCOProgramBuilder::mc##OP_NAME(                 \
      ValueRange controls, Value target) {                                     \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               return {OP_NAME(targets[0])};                                   \
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
  Value QCOProgramBuilder::OP_NAME(const std::variant<double, Value>&(PARAM),  \
                                   Value qubit) {                              \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, loc, qubit, PARAM);                      \
    const auto& qubitOut = op.getQubitOut();                                   \
    updateQubitTracking(qubit, qubitOut, op->getParentRegion());               \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> QCOProgramBuilder::c##OP_NAME(                       \
      const std::variant<double, Value>&(PARAM), Value control,                \
      Value target) {                                                          \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](ValueRange targets) -> llvm::SmallVector<Value> { \
          return {OP_NAME(PARAM, targets[0])};                                 \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> QCOProgramBuilder::mc##OP_NAME(                 \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value target) {                                                          \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               return {OP_NAME(PARAM, targets[0])};                            \
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
  Value QCOProgramBuilder::OP_NAME(const std::variant<double, Value>&(PARAM1), \
                                   const std::variant<double, Value>&(PARAM2), \
                                   Value qubit) {                              \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, loc, qubit, PARAM1, PARAM2);             \
    const auto& qubitOut = op.getQubitOut();                                   \
    updateQubitTracking(qubit, qubitOut, op->getParentRegion());               \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> QCOProgramBuilder::c##OP_NAME(                       \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control,               \
      Value target) {                                                          \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](ValueRange targets) -> llvm::SmallVector<Value> { \
          return {OP_NAME(PARAM1, PARAM2, targets[0])};                        \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> QCOProgramBuilder::mc##OP_NAME(                 \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target) {                                                          \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               return {OP_NAME(PARAM1, PARAM2, targets[0])};                   \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
  }

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,   \
                                          PARAM3)                              \
  Value QCOProgramBuilder::OP_NAME(const std::variant<double, Value>&(PARAM1), \
                                   const std::variant<double, Value>&(PARAM2), \
                                   const std::variant<double, Value>&(PARAM3), \
                                   Value qubit) {                              \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, loc, qubit, PARAM1, PARAM2, PARAM3);     \
    const auto& qubitOut = op.getQubitOut();                                   \
    updateQubitTracking(qubit, qubitOut, op->getParentRegion());               \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> QCOProgramBuilder::c##OP_NAME(                       \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value control,               \
      Value target) {                                                          \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](ValueRange targets) -> llvm::SmallVector<Value> { \
          return {OP_NAME(PARAM1, PARAM2, PARAM3, targets[0])};                \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> QCOProgramBuilder::mc##OP_NAME(                 \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), ValueRange controls,         \
      Value target) {                                                          \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               return {OP_NAME(PARAM1, PARAM2, PARAM3, targets[0])};           \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
  }

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  std::pair<Value, Value> QCOProgramBuilder::OP_NAME(Value qubit0,             \
                                                     Value qubit1) {           \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, loc, qubit0, qubit1);                    \
    const auto& qubit0Out = op.getQubit0Out();                                 \
    const auto& qubit1Out = op.getQubit1Out();                                 \
    updateQubitTracking(qubit0, qubit0Out, op->getParentRegion());             \
    updateQubitTracking(qubit1, qubit1Out, op->getParentRegion());             \
    return {qubit0Out, qubit1Out};                                             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> QCOProgramBuilder::c##OP_NAME(     \
      Value control, Value qubit0, Value qubit1) {                             \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(control, {qubit0, qubit1},                                        \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] = OP_NAME(targets[0], targets[1]);                \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut[0], {targetsOut[0], targetsOut[1]}};                   \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      QCOProgramBuilder::mc##OP_NAME(ValueRange controls, Value qubit0,        \
                                     Value qubit1) {                           \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {qubit0, qubit1},                                       \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] = OP_NAME(targets[0], targets[1]);                \
               return {q0, q1};                                                \
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
  std::pair<Value, Value> QCOProgramBuilder::OP_NAME(                          \
      const std::variant<double, Value>&(PARAM), Value qubit0, Value qubit1) { \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, loc, qubit0, qubit1, PARAM);             \
    const auto& qubit0Out = op.getQubit0Out();                                 \
    const auto& qubit1Out = op.getQubit1Out();                                 \
    updateQubitTracking(qubit0, qubit0Out, op->getParentRegion());             \
    updateQubitTracking(qubit1, qubit1Out, op->getParentRegion());             \
    return {qubit0Out, qubit1Out};                                             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> QCOProgramBuilder::c##OP_NAME(     \
      const std::variant<double, Value>&(PARAM), Value control, Value qubit0,  \
      Value qubit1) {                                                          \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(control, {qubit0, qubit1},                                        \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] = OP_NAME(PARAM, targets[0], targets[1]);         \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut[0], {targetsOut[0], targetsOut[1]}};                   \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      QCOProgramBuilder::mc##OP_NAME(                                          \
          const std::variant<double, Value>&(PARAM), ValueRange controls,      \
          Value qubit0, Value qubit1) {                                        \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {qubit0, qubit1},                                       \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] = OP_NAME(PARAM, targets[0], targets[1]);         \
               return {q0, q1};                                                \
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
  std::pair<Value, Value> QCOProgramBuilder::OP_NAME(                          \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value qubit0,                \
      Value qubit1) {                                                          \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, loc, qubit0, qubit1, PARAM1, PARAM2);    \
    const auto& qubit0Out = op.getQubit0Out();                                 \
    const auto& qubit1Out = op.getQubit1Out();                                 \
    updateQubitTracking(qubit0, qubit0Out, op->getParentRegion());             \
    updateQubitTracking(qubit1, qubit1Out, op->getParentRegion());             \
    return {qubit0Out, qubit1Out};                                             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> QCOProgramBuilder::c##OP_NAME(     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control, Value qubit0, \
      Value qubit1) {                                                          \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(control, {qubit0, qubit1},                                        \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] =                                                 \
                   OP_NAME(PARAM1, PARAM2, targets[0], targets[1]);            \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut[0], {targetsOut[0], targetsOut[1]}};                   \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      QCOProgramBuilder::mc##OP_NAME(                                          \
          const std::variant<double, Value>&(PARAM1),                          \
          const std::variant<double, Value>&(PARAM2), ValueRange controls,     \
          Value qubit0, Value qubit1) {                                        \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {qubit0, qubit1},                                       \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] =                                                 \
                   OP_NAME(PARAM1, PARAM2, targets[0], targets[1]);            \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut, {targetsOut[0], targetsOut[1]}};                      \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

ValueRange QCOProgramBuilder::barrier(ValueRange qubits) {
  checkFinalized();

  auto op = BarrierOp::create(*this, loc, qubits);
  const auto& qubitsOut = op.getQubitsOut();
  for (const auto& [inputQubit, outputQubit] : llvm::zip(qubits, qubitsOut)) {
    updateQubitTracking(inputQubit, outputQubit, op->getParentRegion());
  }
  return qubitsOut;
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

std::pair<ValueRange, ValueRange> QCOProgramBuilder::ctrl(
    ValueRange controls, ValueRange targets,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> body) {
  checkFinalized();

  auto ctrlOp = CtrlOp::create(*this, loc, controls, targets);
  auto& block = ctrlOp.getBodyRegion().emplaceBlock();
  const auto qubitType = QubitType::get(getContext());
  auto* region = block.getParent();
  for (size_t i = 0; i < targets.size(); i++) {
    const auto arg = block.addArgument(qubitType, loc);
    validQubits[region].insert(arg);
  }
  const InsertionGuard guard(*this);
  setInsertionPointToStart(&block);
  const auto innerTargetsOut = body(block.getArguments());
  YieldOp::create(*this, loc, innerTargetsOut);

  if (innerTargetsOut.size() != targets.size()) {
    llvm::reportFatalUsageError(
        "Ctrl body must return exactly one output qubit per target");
  }

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

QCOProgramBuilder& QCOProgramBuilder::dealloc(Value qubit) {
  checkFinalized();

  validateQubitValue(qubit, qubit.getParentRegion());
  validQubits[qubit.getParentRegion()].erase(qubit);

  DeallocOp::create(*this, loc, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// SCF operations
//===----------------------------------------------------------------------===//

ValueRange QCOProgramBuilder::scfFor(
    Value lowerbound, Value upperbound, Value step, ValueRange initArgs,
    llvm::function_ref<llvm::SmallVector<Value>(Value, ValueRange)> body) {
  checkFinalized();

  // Create the empty for operation
  auto forOp = create<scf::ForOp>(loc, lowerbound, upperbound, step, initArgs);
  auto* forBody = forOp.getBody();
  const auto iv = forBody->getArgument(0);
  const auto loopArgs = forBody->getArguments().drop_front();

  // Set the insertionpoint
  const OpBuilder::InsertionGuard guard(*this);
  setInsertionPointToStart(forBody);

  // Add the iterArgs to the validQubits
  auto* bodyRegion = forBody->getParent();
  for (const auto& arg : loopArgs) {
    validQubits[bodyRegion].insert(arg);
  }
  // Build the body
  body(iv, loopArgs);

  // Update the qubit tracking
  for (const auto& [initArg, result] :
       llvm::zip_equal(initArgs, forOp.getResults())) {
    updateQubitTracking(initArg, result, forOp->getParentRegion());
  }

  return forOp->getResults();
}

ValueRange QCOProgramBuilder::scfWhile(
    ValueRange initArgs,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> beforeBody,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> afterBody) {
  checkFinalized();

  // Create the empty while operation
  auto whileOp = create<scf::WhileOp>(loc, initArgs.getTypes(), initArgs);
  const SmallVector<Location> locs(initArgs.size(), loc);

  const OpBuilder::InsertionGuard guard(*this);

  // Construct the before block
  auto* beforeBlock =
      createBlock(&whileOp.getBefore(), {}, initArgs.getTypes(), locs);
  auto beforeArgs = beforeBlock->getArguments();
  auto* beforeRegion = beforeBlock->getParent();

  // Set the insertionpoint
  setInsertionPointToStart(beforeBlock);

  // Add the beforeArgs to the validQubits
  for (const auto& arg : beforeArgs) {
    validQubits[beforeRegion].insert(arg);
  }

  beforeBody(beforeArgs);

  // Construct the after block
  auto* afterBlock =
      createBlock(&whileOp.getAfter(), {}, initArgs.getTypes(), locs);
  auto afterArgs = afterBlock->getArguments();
  auto* afterRegion = afterBlock->getParent();

  // Set the insertionpoint
  setInsertionPointToStart(afterBlock);

  // Add the afterArgs to the validQubits
  for (const auto& arg : afterArgs) {
    validQubits[afterRegion].insert(arg);
  }

  afterBody(afterArgs);

  // Update the qubit tracking
  for (const auto& [arg, result] :
       llvm::zip_equal(initArgs, whileOp.getResults())) {
    updateQubitTracking(arg, result, whileOp->getParentRegion());
  }

  return whileOp->getResults();
}

ValueRange QCOProgramBuilder::scfIf(
    Value condition, ValueRange qubits,
    llvm::function_ref<llvm::SmallVector<Value>()> thenBody,
    llvm::function_ref<llvm::SmallVector<Value>()> elseBody) {
  checkFinalized();

  // Create the empty while operation
  auto ifOp = create<scf::IfOp>(loc, qubits.getTypes(), condition,
                                /*withElseRegion=*/true);
  auto& thenBlock = ifOp.getThenRegion().front();
  auto& elseBlock = ifOp.getElseRegion().front();
  auto* thenRegion = thenBlock.getParent();
  auto* elseRegion = elseBlock.getParent();

  // Set the insertionpoint
  const OpBuilder::InsertionGuard guard(*this);
  setInsertionPointToStart(&thenBlock);

  // Add the qubits to the validQubits of the then and else region
  for (const auto& arg : qubits) {
    validQubits[thenRegion].insert(arg);
    validQubits[elseRegion].insert(arg);
  }

  // Build the then body
  thenBody();

  // Set the insertionpoint
  setInsertionPointToStart(&elseBlock);

  // Build the else body
  elseBody();

  // Update the qubit tracking
  for (const auto& [arg, result] : llvm::zip_equal(qubits, ifOp.getResults())) {
    updateQubitTracking(arg, result, ifOp->getParentRegion());
  }

  return ifOp->getResults();
}

QCOProgramBuilder& QCOProgramBuilder::scfCondition(Value condition,
                                                   ValueRange yieldedValues) {
  checkFinalized();

  create<scf::ConditionOp>(loc, condition, yieldedValues);
  return *this;
}

QCOProgramBuilder& QCOProgramBuilder::scfYield(ValueRange yieldedValues) {
  checkFinalized();

  create<scf::YieldOp>(loc, yieldedValues);
  return *this;
}

//===----------------------------------------------------------------------===//
// Func operations
//===----------------------------------------------------------------------===//

ValueRange QCOProgramBuilder::funcCall(StringRef name, ValueRange operands) {
  checkFinalized();

  const auto callOp =
      create<func::CallOp>(loc, name, operands.getTypes(), operands);
  for (auto [arg, result] : llvm::zip_equal(operands, callOp->getResults())) {
    updateQubitTracking(arg, result, callOp->getParentRegion());
  }
  return callOp->getResults();
}

QCOProgramBuilder& QCOProgramBuilder::funcReturn(ValueRange returnValues) {
  checkFinalized();

  create<func::ReturnOp>(loc, returnValues);
  return *this;
}
QCOProgramBuilder&
QCOProgramBuilder::funcFunc(StringRef name, TypeRange argTypes,
                            TypeRange resultTypes,
                            llvm::function_ref<void(ValueRange)> body) {
  checkFinalized();

  // Set the insertionPoint
  const OpBuilder::InsertionGuard guard(*this);
  setInsertionPointToEnd(module.getBody());

  // Create the empty func operation
  const auto funcType = getFunctionType(argTypes, resultTypes);
  auto funcOp = create<func::FuncOp>(loc, name, funcType);
  auto* entryBlock = funcOp.addEntryBlock();
  auto* region = entryBlock->getParent();
  // Add the arguments to the validQubits
  for (const auto& arg : entryBlock->getArguments()) {
    validQubits[region].insert(arg);
  }

  setInsertionPointToStart(entryBlock);

  // Build function body
  body(entryBlock->getArguments());

  return *this;
}

//===----------------------------------------------------------------------===//
// Arith operations
//===----------------------------------------------------------------------===//

Value QCOProgramBuilder::arithConstantIndex(int64_t i) {
  checkFinalized();

  const auto op =
      create<arith::ConstantOp>(loc, getIndexType(), getIndexAttr(i));
  return op->getResult(0);
}

Value QCOProgramBuilder::arithConstantBool(bool b) {
  checkFinalized();

  const auto i1Type = getI1Type();
  const auto op =
      create<arith::ConstantOp>(loc, i1Type, getIntegerAttr(i1Type, b ? 1 : 0));
  return op->getResult(0);
}
//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

void QCOProgramBuilder::checkFinalized() const {
  if (ctx == nullptr) {
    llvm::reportFatalUsageError(
        "QCOProgramBuilder instance has been finalized");
  }
}

OwningOpRef<ModuleOp> QCOProgramBuilder::finalize() {
  checkFinalized();
  // Ensure that main function exists and insertion point is valid
  auto* insertionBlock = getInsertionBlock();
  func::FuncOp mainFunc = nullptr;
  for (auto op : module.getOps<func::FuncOp>()) {
    if (op.getName() == "main") {
      mainFunc = op;
      break;
    }
  }
  if (!mainFunc) {
    llvm::reportFatalUsageError("Could not find main function");
  }
  if ((insertionBlock == nullptr) ||
      insertionBlock != &mainFunc.getBody().front()) {
    llvm::reportFatalUsageError(
        "Insertion point is not in entry block of main function");
  }

  // Automatically deallocate all still-allocated qubits
  // Sort qubits for deterministic output
  llvm::SmallVector<Value> sortedQubits(
      validQubits[&mainFunc->getRegion(0)].begin(),
      validQubits[&mainFunc->getRegion(0)].end());
  llvm::sort(sortedQubits, [](Value a, Value b) {
    auto* opA = a.getDefiningOp();
    auto* opB = b.getDefiningOp();
    if (!opA || !opB || opA->getBlock() != opB->getBlock()) {
      return a.getAsOpaquePointer() < b.getAsOpaquePointer();
    }
    return opA->isBeforeInBlock(opB);
  });
  for (auto qubit : sortedQubits) {
    DeallocOp::create(*this, loc, qubit);
  }

  validQubits.clear();

  // Create constant 0 for successful exit code
  auto exitCode = arith::ConstantOp::create(*this, loc, getI64IntegerAttr(0));

  // Add return statement with exit code 0 to the main function
  func::ReturnOp::create(*this, loc, ValueRange{exitCode});

  // Invalidate context to prevent use-after-finalize
  ctx = nullptr;

  return module;
}

} // namespace mlir::qco
