/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

using namespace mlir::utils;

namespace mlir::qc {

QCProgramBuilder::QCProgramBuilder(MLIRContext* context)
    : ImplicitLocOpBuilder(
          FileLineColLoc::get(context, "<qc-program-builder>", 1, 1), context),
      ctx(context), module(ModuleOp::create(*this)) {
  ctx->loadDialect<QCDialect>();
}

void QCProgramBuilder::initialize() {
  // Set insertion point to the module body
  setInsertionPointToStart(module.getBody());

  // Create main function as entry point
  auto funcType = getFunctionType({}, {getI64Type()});
  auto mainFunc = func::FuncOp::create(*this, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = getStringAttr("entry_point");
  mainFunc->setAttr("passthrough", getArrayAttr({entryPointAttr}));

  // Create entry block and set insertion point
  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  setInsertionPointToStart(&entryBlock);
}

Value QCProgramBuilder::intConstant(const int64_t value) {
  checkFinalized();
  return arith::ConstantOp::create(*this, getI64IntegerAttr(value)).getResult();
}

Value QCProgramBuilder::allocQubit() {
  checkFinalized();

  // Create the AllocOp without register metadata
  auto allocOp = AllocOp::create(*this);
  const auto qubit = allocOp.getResult();

  // Track the allocated qubit for automatic deallocation
  allocatedQubits.insert(qubit);

  return qubit;
}

Value QCProgramBuilder::staticQubit(const uint64_t index) {
  checkFinalized();

  auto staticOp = StaticOp::create(*this, index);
  return staticOp.getQubit();
}

llvm::SmallVector<Value>
QCProgramBuilder::allocQubitRegister(const int64_t size) {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  auto memrefType = MemRefType::get({size}, QubitType::get(ctx));
  auto memref = memref::AllocOp::create(*this, memrefType);
  allocatedMemrefs.insert(memref);

  llvm::SmallVector<Value> qubits;
  qubits.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    auto index = arith::ConstantIndexOp::create(*this, i);
    auto load = memref::LoadOp::create(*this, memref, index.getResult());
    const auto& qubit = qubits.emplace_back(load.getResult());
    allocatedQubits.insert(qubit);
  }
  return qubits;
}

QCProgramBuilder::ClassicalRegister
QCProgramBuilder::allocClassicalBitRegister(const int64_t size,
                                            std::string name) const {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  return {.name = std::move(name), .size = size};
}

//===----------------------------------------------------------------------===//
// Measurement and Reset
//===----------------------------------------------------------------------===//

Value QCProgramBuilder::measure(Value qubit) {
  checkFinalized();
  auto measureOp = MeasureOp::create(*this, qubit);
  return measureOp.getResult();
}

QCProgramBuilder& QCProgramBuilder::measure(Value qubit, const Bit& bit) {
  checkFinalized();
  auto nameAttr = getStringAttr(bit.registerName);
  auto sizeAttr = getI64IntegerAttr(bit.registerSize);
  auto indexAttr = getI64IntegerAttr(bit.registerIndex);
  MeasureOp::create(*this, qubit, nameAttr, sizeAttr, indexAttr);
  return *this;
}

QCProgramBuilder& QCProgramBuilder::reset(Value qubit) {
  checkFinalized();
  ResetOp::create(*this, qubit);
  return *this;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// ZeroTargetOneParameter

#define DEFINE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  QCProgramBuilder& QCProgramBuilder::OP_NAME(                                 \
      const std::variant<double, Value>&(PARAM)) {                             \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, PARAM);                                            \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(                              \
      const std::variant<double, Value>&(PARAM), Value control) {              \
    checkFinalized();                                                          \
    return mc##OP_NAME(PARAM, {control});                                      \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM), ValueRange controls) {        \
    checkFinalized();                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    CtrlOp::create(*this, controls, [&] { OP_CLASS::create(*this, param); });  \
    return *this;                                                              \
  }

DEFINE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DEFINE_ZERO_TARGET_ONE_PARAMETER

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  QCProgramBuilder& QCProgramBuilder::OP_NAME(Value qubit) {                   \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, qubit);                                            \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(Value control,                \
                                                 Value target) {               \
    checkFinalized();                                                          \
    return mc##OP_NAME({control}, target);                                     \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(ValueRange controls,         \
                                                  Value target) {              \
    checkFinalized();                                                          \
    CtrlOp::create(*this, controls, [&] { OP_CLASS::create(*this, target); }); \
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
  QCProgramBuilder& QCProgramBuilder::OP_NAME(                                 \
      const std::variant<double, Value>&(PARAM), Value qubit) {                \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, qubit, PARAM);                                     \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(                              \
      const std::variant<double, Value>&(PARAM), Value control,                \
      Value target) {                                                          \
    checkFinalized();                                                          \
    return mc##OP_NAME(PARAM, {control}, target);                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value target) {                                                          \
    checkFinalized();                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    CtrlOp::create(*this, controls,                                            \
                   [&] { OP_CLASS::create(*this, target, param); });           \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_ONE_PARAMETER(RXOp, rx, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RYOp, ry, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZOp, rz, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(POp, p, theta)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  QCProgramBuilder& QCProgramBuilder::OP_NAME(                                 \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value qubit) {               \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, qubit, PARAM1, PARAM2);                            \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(                              \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control,               \
      Value target) {                                                          \
    checkFinalized();                                                          \
    return mc##OP_NAME(PARAM1, PARAM2, {control}, target);                     \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target) {                                                          \
    checkFinalized();                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    CtrlOp::create(*this, controls,                                            \
                   [&] { OP_CLASS::create(*this, target, param1, param2); });  \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,   \
                                          PARAM3)                              \
  QCProgramBuilder& QCProgramBuilder::OP_NAME(                                 \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value qubit) {               \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, qubit, PARAM1, PARAM2, PARAM3);                    \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(                              \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value control,               \
      Value target) {                                                          \
    checkFinalized();                                                          \
    return mc##OP_NAME(PARAM1, PARAM2, PARAM3, {control}, target);             \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), ValueRange controls,         \
      Value target) {                                                          \
    checkFinalized();                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    auto param3 = variantToValue(*this, getLoc(), PARAM3);                     \
    CtrlOp::create(*this, controls, [&] {                                      \
      OP_CLASS::create(*this, target, param1, param2, param3);                 \
    });                                                                        \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  QCProgramBuilder& QCProgramBuilder::OP_NAME(Value qubit0, Value qubit1) {    \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, qubit0, qubit1);                                   \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(Value control, Value qubit0,  \
                                                 Value qubit1) {               \
    checkFinalized();                                                          \
    return mc##OP_NAME({control}, qubit0, qubit1);                             \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      ValueRange controls, Value qubit0, Value qubit1) {                       \
    checkFinalized();                                                          \
    CtrlOp::create(*this, controls,                                            \
                   [&] { OP_CLASS::create(*this, qubit0, qubit1); });          \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCXOp, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECROp, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  QCProgramBuilder& QCProgramBuilder::OP_NAME(                                 \
      const std::variant<double, Value>&(PARAM), Value qubit0, Value qubit1) { \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, qubit0, qubit1, PARAM);                            \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(                              \
      const std::variant<double, Value>&(PARAM), Value control, Value qubit0,  \
      Value qubit1) {                                                          \
    checkFinalized();                                                          \
    return mc##OP_NAME(PARAM, {control}, qubit0, qubit1);                      \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value qubit0, Value qubit1) {                                            \
    checkFinalized();                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    CtrlOp::create(*this, controls,                                            \
                   [&] { OP_CLASS::create(*this, qubit0, qubit1, param); });   \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_ONE_PARAMETER(RXXOp, rxx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RYYOp, ryy, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZXOp, rzx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZZOp, rzz, theta)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

// TwoTargetTwoParameter

#define DEFINE_TWO_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  QCProgramBuilder& QCProgramBuilder::OP_NAME(                                 \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value qubit0,                \
      Value qubit1) {                                                          \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, qubit0, qubit1, PARAM1, PARAM2);                   \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(                              \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control, Value qubit0, \
      Value qubit1) {                                                          \
    checkFinalized();                                                          \
    return mc##OP_NAME(PARAM1, PARAM2, {control}, qubit0, qubit1);             \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value qubit0, Value qubit1) {                                            \
    checkFinalized();                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    CtrlOp::create(*this, controls, [&] {                                      \
      OP_CLASS::create(*this, qubit0, qubit1, param1, param2);                 \
    });                                                                        \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

QCProgramBuilder& QCProgramBuilder::barrier(ValueRange qubits) {
  checkFinalized();
  BarrierOp::create(*this, qubits);
  return *this;
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

QCProgramBuilder&
QCProgramBuilder::ctrl(ValueRange controls,
                       const llvm::function_ref<void()>& body) {
  checkFinalized();
  CtrlOp::create(*this, controls, body);
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::inv(const llvm::function_ref<void()>& body) {
  checkFinalized();
  InvOp::create(*this, body);
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::pow(double exponent, const llvm::function_ref<void()>& body) {
  checkFinalized();
  PowOp::create(*this, exponent, body);
  return *this;
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

QCProgramBuilder& QCProgramBuilder::dealloc(Value qubit) {
  checkFinalized();

  if (llvm::isa_and_nonnull<memref::LoadOp>(qubit.getDefiningOp())) {
    llvm::reportFatalUsageError(
        "Register-backed qubits cannot be deallocated manually");
  }

  // Check if the qubit is in the tracking set
  if (!allocatedQubits.erase(qubit)) {
    llvm::reportFatalUsageError("Invalid qubit deallocation");
  }

  // Create the DeallocOp
  DeallocOp::create(*this, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

void QCProgramBuilder::checkFinalized() const {
  if (ctx == nullptr) {
    llvm::reportFatalUsageError("QCProgramBuilder instance has been finalized");
  }
}

OwningOpRef<ModuleOp> QCProgramBuilder::finalize() {
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

  for (auto qubit : allocatedQubits) {
    if (!llvm::isa<memref::LoadOp>(qubit.getDefiningOp())) {
      DeallocOp::create(*this, qubit);
    }
  }
  allocatedQubits.clear();

  for (auto memref : allocatedMemrefs) {
    memref::DeallocOp::create(*this, memref);
  }
  allocatedMemrefs.clear();

  // Create constant 0 for successful exit code
  auto exitCode = intConstant(0);

  // Add return statement with exit code 0 to the main function
  func::ReturnOp::create(*this, exitCode);

  // Invalidate context to prevent use-after-finalize
  ctx = nullptr;

  // Transfer ownership to the caller
  return module;
}

OwningOpRef<ModuleOp> QCProgramBuilder::build(
    MLIRContext* context,
    const llvm::function_ref<void(QCProgramBuilder&)>& buildFunc) {
  QCProgramBuilder builder(context);
  builder.initialize();
  buildFunc(builder);
  return builder.finalize();
}

} // namespace mlir::qc
