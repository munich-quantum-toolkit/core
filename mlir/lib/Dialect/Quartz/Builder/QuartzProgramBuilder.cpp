/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/Builder/QuartzProgramBuilder.h"

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <cstdint>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <variant>

namespace mlir::quartz {

QuartzProgramBuilder::QuartzProgramBuilder(MLIRContext* context)
    : OpBuilder(context), ctx(context), loc(getUnknownLoc()),
      module(ModuleOp::create(loc)) {
  ctx->loadDialect<QuartzDialect>();
}

void QuartzProgramBuilder::initialize() {
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

Value QuartzProgramBuilder::allocQubit() {
  // Create the AllocOp without register metadata
  auto allocOp = create<AllocOp>(loc);
  const auto qubit = allocOp.getResult();

  // Track the allocated qubit for automatic deallocation
  allocatedQubits.insert(qubit);

  return qubit;
}

Value QuartzProgramBuilder::staticQubit(const int64_t index) {
  if (index < 0) {
    llvm::reportFatalUsageError("Index must be non-negative");
  }

  // Create the StaticOp with the given index
  auto indexAttr = getI64IntegerAttr(index);
  auto staticOp = create<StaticOp>(loc, indexAttr);
  return staticOp.getQubit();
}

llvm::SmallVector<Value>
QuartzProgramBuilder::allocQubitRegister(const int64_t size,
                                         const StringRef name) {
  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  // Allocate a sequence of qubits with register metadata
  llvm::SmallVector<Value> qubits;
  qubits.reserve(size);

  auto nameAttr = getStringAttr(name);
  auto sizeAttr = getI64IntegerAttr(size);

  for (int64_t i = 0; i < size; ++i) {
    auto indexAttr = getI64IntegerAttr(i);
    auto allocOp = create<AllocOp>(loc, nameAttr, sizeAttr, indexAttr);
    const auto& qubit = qubits.emplace_back(allocOp.getResult());
    // Track the allocated qubit for automatic deallocation
    allocatedQubits.insert(qubit);
  }

  return qubits;
}

QuartzProgramBuilder::ClassicalRegister&
QuartzProgramBuilder::allocClassicalBitRegister(int64_t size, StringRef name) {
  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  return allocatedClassicalRegisters.emplace_back(name, size);
}

//===----------------------------------------------------------------------===//
// Measurement and Reset
//===----------------------------------------------------------------------===//

Value QuartzProgramBuilder::measure(Value qubit) {
  auto measureOp = create<MeasureOp>(loc, qubit);
  return measureOp.getResult();
}

QuartzProgramBuilder& QuartzProgramBuilder::measure(Value qubit,
                                                    const Bit& bit) {
  auto nameAttr = getStringAttr(bit.registerName);
  auto sizeAttr = getI64IntegerAttr(bit.registerSize);
  auto indexAttr = getI64IntegerAttr(bit.registerIndex);
  create<MeasureOp>(loc, qubit, nameAttr, sizeAttr, indexAttr);
  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::reset(Value qubit) {
  create<ResetOp>(loc, qubit);
  return *this;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// ZeroTargetOneParameter

#define DEFINE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  QuartzProgramBuilder& QuartzProgramBuilder::OP_NAME(                         \
      const std::variant<double, Value>&(PARAM)) {                             \
    create<OP_CLASS>(loc, PARAM);                                              \
    return *this;                                                              \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM), Value control) {              \
    return mc##OP_NAME(PARAM, {control});                                      \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::mc##OP_NAME(                     \
      const std::variant<double, Value>&(PARAM), ValueRange controls) {        \
    create<CtrlOp>(loc, controls,                                              \
                   [&](OpBuilder& b) { b.create<OP_CLASS>(loc, PARAM); });     \
    return *this;                                                              \
  }

DEFINE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DEFINE_ZERO_TARGET_ONE_PARAMETER

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  QuartzProgramBuilder& QuartzProgramBuilder::OP_NAME(Value qubit) {           \
    create<OP_CLASS>(loc, qubit);                                              \
    return *this;                                                              \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::c##OP_NAME(Value control,        \
                                                         Value target) {       \
    return mc##OP_NAME({control}, target);                                     \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::mc##OP_NAME(ValueRange controls, \
                                                          Value target) {      \
    create<CtrlOp>(loc, controls,                                              \
                   [&](OpBuilder& b) { b.create<OP_CLASS>(loc, target); });    \
    return *this;                                                              \
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
  QuartzProgramBuilder& QuartzProgramBuilder::OP_NAME(                         \
      const std::variant<double, Value>&(PARAM), Value qubit) {                \
    create<OP_CLASS>(loc, qubit, PARAM);                                       \
    return *this;                                                              \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM), Value control,                \
      Value target) {                                                          \
    return mc##OP_NAME(PARAM, {control}, target);                              \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::mc##OP_NAME(                     \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value target) {                                                          \
    create<CtrlOp>(loc, controls, [&](OpBuilder& b) {                          \
      b.create<OP_CLASS>(loc, target, PARAM);                                  \
    });                                                                        \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_ONE_PARAMETER(RXOp, rx, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RYOp, ry, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZOp, rz, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(POp, p, theta)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  QuartzProgramBuilder& QuartzProgramBuilder::OP_NAME(                         \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value qubit) {               \
    create<OP_CLASS>(loc, qubit, PARAM1, PARAM2);                              \
    return *this;                                                              \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control,               \
      Value target) {                                                          \
    return mc##OP_NAME(PARAM1, PARAM2, {control}, target);                     \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::mc##OP_NAME(                     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target) {                                                          \
    create<CtrlOp>(loc, controls, [&](OpBuilder& b) {                          \
      b.create<OP_CLASS>(loc, target, PARAM1, PARAM2);                         \
    });                                                                        \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,   \
                                          PARAM3)                              \
  QuartzProgramBuilder& QuartzProgramBuilder::OP_NAME(                         \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value qubit) {               \
    create<OP_CLASS>(loc, qubit, PARAM1, PARAM2, PARAM3);                      \
    return *this;                                                              \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value control,               \
      Value target) {                                                          \
    return mc##OP_NAME(PARAM1, PARAM2, PARAM3, {control}, target);             \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::mc##OP_NAME(                     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), ValueRange controls,         \
      Value target) {                                                          \
    create<CtrlOp>(loc, controls, [&](OpBuilder& b) {                          \
      b.create<OP_CLASS>(loc, target, PARAM1, PARAM2, PARAM3);                 \
    });                                                                        \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  QuartzProgramBuilder& QuartzProgramBuilder::OP_NAME(Value qubit0,            \
                                                      Value qubit1) {          \
    create<OP_CLASS>(loc, qubit0, qubit1);                                     \
    return *this;                                                              \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::c##OP_NAME(                      \
      Value control, Value qubit0, Value qubit1) {                             \
    return mc##OP_NAME({control}, qubit0, qubit1);                             \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::mc##OP_NAME(                     \
      ValueRange controls, Value qubit0, Value qubit1) {                       \
    create<CtrlOp>(loc, controls, [&](OpBuilder& b) {                          \
      b.create<OP_CLASS>(loc, qubit0, qubit1);                                 \
    });                                                                        \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCXOp, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECROp, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  QuartzProgramBuilder& QuartzProgramBuilder::OP_NAME(                         \
      const std::variant<double, Value>&(PARAM), Value qubit0, Value qubit1) { \
    create<OP_CLASS>(loc, qubit0, qubit1, PARAM);                              \
    return *this;                                                              \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM), Value control, Value qubit0,  \
      Value qubit1) {                                                          \
    return mc##OP_NAME(PARAM, {control}, qubit0, qubit1);                      \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::mc##OP_NAME(                     \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value qubit0, Value qubit1) {                                            \
    create<CtrlOp>(loc, controls, [&](OpBuilder& b) {                          \
      b.create<OP_CLASS>(loc, qubit0, qubit1, PARAM);                          \
    });                                                                        \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_ONE_PARAMETER(RXXOp, rxx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RYYOp, ryy, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZXOp, rzx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZZOp, rzz, theta)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

// TwoTargetTwoParameter

#define DEFINE_TWO_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  QuartzProgramBuilder& QuartzProgramBuilder::OP_NAME(                         \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value qubit0,                \
      Value qubit1) {                                                          \
    create<OP_CLASS>(loc, qubit0, qubit1, PARAM1, PARAM2);                     \
    return *this;                                                              \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::c##OP_NAME(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control, Value qubit0, \
      Value qubit1) {                                                          \
    return mc##OP_NAME(PARAM1, PARAM2, {control}, qubit0, qubit1);             \
  }                                                                            \
  QuartzProgramBuilder& QuartzProgramBuilder::mc##OP_NAME(                     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value qubit0, Value qubit1) {                                            \
    create<CtrlOp>(loc, controls, [&](OpBuilder& b) {                          \
      b.create<OP_CLASS>(loc, qubit0, qubit1, PARAM1, PARAM2);                 \
    });                                                                        \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

QuartzProgramBuilder& QuartzProgramBuilder::barrier(ValueRange qubits) {
  create<BarrierOp>(loc, qubits);
  return *this;
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

QuartzProgramBuilder&
QuartzProgramBuilder::ctrl(ValueRange controls,
                           const std::function<void(OpBuilder&)>& body) {
  create<CtrlOp>(loc, controls, body);
  return *this;
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

QuartzProgramBuilder& QuartzProgramBuilder::dealloc(Value qubit) {
  // Check if the qubit is in the tracking set
  if (!allocatedQubits.erase(qubit)) {
    // Qubit was not found in the set - either never allocated or already
    // deallocated
    llvm::reportFatalUsageError(
        "Double deallocation or invalid qubit deallocation");
  }

  // Create the DeallocOp
  create<DeallocOp>(loc, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> QuartzProgramBuilder::finalize() {
  // Automatically deallocate all remaining allocated qubits
  for (Value qubit : allocatedQubits) {
    create<DeallocOp>(loc, qubit);
  }

  // Clear the tracking set
  allocatedQubits.clear();

  // Create constant 0 for successful exit code
  auto exitCode = create<arith::ConstantOp>(loc, getI64IntegerAttr(0));

  // Add return statement with exit code 0 to the main function
  create<func::ReturnOp>(loc, ValueRange{exitCode});

  // Transfer ownership to the caller
  return module;
}

} // namespace mlir::quartz
