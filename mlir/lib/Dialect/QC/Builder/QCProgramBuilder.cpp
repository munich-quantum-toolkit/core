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

#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>

using namespace mlir::mqt;

namespace mlir::qc {
QCProgramBuilder::QCProgramBuilder(MLIRContext* context)
    : ImplicitLocOpBuilder(
          FileLineColLoc::get(context, "<qc-program-builder>", 1, 1), context),
      ctx(context), module(ModuleOp::create(*this)) {
  ctx->loadDialect<cbit::CBitDialect, mqt::MQTDialect, QCDialect>();
}

void QCProgramBuilder::initialize() { initialize({getI64Type()}); }

void QCProgramBuilder::initialize(TypeRange returnTypes) {
  // Set insertion point to the module body
  setInsertionPointToStart(cast<ModuleOp>(module).getBody());

  // Create main function as entry point
  auto funcType = getFunctionType({}, returnTypes);
  auto mainFunc = func::FuncOp::create(*this, "main", funcType);

  mqt::setEntryPoint(mainFunc);

  // Create entry block and set insertion point
  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  setInsertionPointToStart(&entryBlock);
}

void QCProgramBuilder::retype(TypeRange returnTypes) {
  auto mainFunc = mqt::getEntryPoint(cast<ModuleOp>(module));
  if (!mainFunc) {
    llvm::reportFatalUsageError("Main function not found for retyping");
  }
  auto funcType =
      getFunctionType(mainFunc.getFunctionType().getInputs(), returnTypes);
  mainFunc.setType(funcType);
}

Value QCProgramBuilder::boolConstant(const bool value) {
  checkFinalized();
  return arith::ConstantOp::create(*this, getBoolAttr(value)).getResult();
}

Value QCProgramBuilder::intConstant(const int64_t value) {
  checkFinalized();
  return arith::ConstantOp::create(*this, getI64IntegerAttr(value)).getResult();
}

Value QCProgramBuilder::QubitRegister::operator[](const size_t index) const {
  if (index >= qubits.size()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return qubits[index];
}

Value QCProgramBuilder::allocQubit() {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Dynamic);

  // Create the AllocOp without register metadata
  auto allocOp = AllocOp::create(*this);
  auto qubit = allocOp.getResult();

  // Track the allocated qubit for automatic deallocation
  allocatedQubits.insert(qubit);

  return qubit;
}

Value QCProgramBuilder::staticQubit(const uint64_t index) {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Static);

  if (const auto it = staticQubits.find(index); it != staticQubits.end()) {
    return it->second;
  }

  OpBuilder::InsertionGuard guard(*this);
  auto mainFunc = mqt::getEntryPoint(cast<ModuleOp>(module));
  setInsertionPointToStart(&mainFunc.getBody().front());
  auto qubit = StaticOp::create(*this, index).getQubit();
  staticQubits.try_emplace(index, qubit);
  return qubit;
}

QCProgramBuilder::QubitRegister
QCProgramBuilder::allocQubitRegister(const int64_t size, const StringRef name) {
  auto memref = allocQubitRegisterStorage(size, name);

  SmallVector<Value> qubits;
  qubits.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    auto index = arith::ConstantIndexOp::create(*this, i);
    qubits.emplace_back(loadQubit(memref, index));
  }

  return {.value = memref, .qubits = std::move(qubits)};
}

Value QCProgramBuilder::allocQubitRegisterStorage(const int64_t size,
                                                  const StringRef name) {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Dynamic);

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }
  auto memrefType = MemRefType::get({size}, QubitType::get(ctx));
  auto alloc = memref::AllocOp::create(*this, memrefType);
  if (!name.empty()) {
    ctx->getLoadedDialect<mqt::MQTDialect>()
        ->getRegisterNameAttrHelper()
        .setAttr(alloc, getStringAttr(name));
  }
  auto memref = alloc.getResult();
  allocatedQregs.insert(memref);
  return memref;
}

Value QCProgramBuilder::loadQubit(Value memref, Value index) {
  checkFinalized();
  return memref::LoadOp::create(*this, memref, index).getResult();
}

Value QCProgramBuilder::allocClassicalBitRegister(
    const int64_t size, const StringRef name,
    const cbit::Initialization initialization) {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  const auto type = cbit::RegisterType::get(ctx, size);
  auto alloc = cbit::AllocOp::create(*this, type, initialization);
  if (!name.empty()) {
    ctx->getLoadedDialect<mqt::MQTDialect>()
        ->getRegisterNameAttrHelper()
        .setAttr(alloc, getStringAttr(name));
  }
  return alloc.getResult();
}

Value QCProgramBuilder::loadClassicalBit(
    Value reg, const std::variant<int64_t, Value>& index) {
  checkFinalized();
  cbit::validateStaticRegisterIndex(reg, index);
  auto indexValue = variantToValue(*this, getLoc(), index);
  return cbit::LoadOp::create(*this, getI1Type(), reg, indexValue).getResult();
}

void QCProgramBuilder::storeClassicalBit(
    Value value, Value reg, const std::variant<int64_t, Value>& index) {
  checkFinalized();
  cbit::validateStaticRegisterIndex(reg, index);
  auto indexValue = variantToValue(*this, getLoc(), index);
  cbit::StoreOp::create(*this, value, reg, indexValue);
}

//===----------------------------------------------------------------------===//
// Measurement and Reset
//===----------------------------------------------------------------------===//

Value QCProgramBuilder::measure(Value qubit) {
  checkFinalized();
  auto measureOp = MeasureOp::create(*this, qubit);
  return measureOp.getResult();
}

Value QCProgramBuilder::measure(Value qubit, Value reg,
                                const std::variant<int64_t, Value>& index) {
  checkFinalized();
  auto measureOp = MeasureOp::create(*this, qubit);
  auto result = measureOp.getResult();
  storeClassicalBit(result, reg, index);
  return result;
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
    return mc##OP_NAME(PARAM, {control});                                      \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM), ValueRange controls) {        \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    ctrl(controls, ValueRange{},                                               \
         [&](ValueRange /*targets*/) { OP_CLASS::create(*this, param); });     \
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
    return mc##OP_NAME({control}, target);                                     \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(ValueRange controls,         \
                                                  Value target) {              \
    ctrl(controls, target,                                                     \
         [&](Value targetArg) { OP_CLASS::create(*this, targetArg); });        \
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
    return mc##OP_NAME(PARAM, {control}, target);                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value target) {                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    ctrl(controls, target,                                                     \
         [&](Value targetArg) { OP_CLASS::create(*this, targetArg, param); }); \
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
    return mc##OP_NAME(PARAM1, PARAM2, {control}, target);                     \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target) {                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    ctrl(controls, target, [&](Value targetArg) {                              \
      OP_CLASS::create(*this, targetArg, param1, param2);                      \
    });                                                                        \
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
    return mc##OP_NAME(PARAM1, PARAM2, PARAM3, {control}, target);             \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), ValueRange controls,         \
      Value target) {                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    auto param3 = variantToValue(*this, getLoc(), PARAM3);                     \
    ctrl(controls, target, [&](Value targetArg) {                              \
      OP_CLASS::create(*this, targetArg, param1, param2, param3);              \
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
    return mc##OP_NAME({control}, qubit0, qubit1);                             \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      ValueRange controls, Value qubit0, Value qubit1) {                       \
    ctrl(controls, ValueRange{qubit0, qubit1}, [&](ValueRange targets) {       \
      OP_CLASS::create(*this, targets[0], targets[1]);                         \
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
  QCProgramBuilder& QCProgramBuilder::OP_NAME(                                 \
      const std::variant<double, Value>&(PARAM), Value qubit0, Value qubit1) { \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, qubit0, qubit1, PARAM);                            \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(                              \
      const std::variant<double, Value>&(PARAM), Value control, Value qubit0,  \
      Value qubit1) {                                                          \
    return mc##OP_NAME(PARAM, {control}, qubit0, qubit1);                      \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value qubit0, Value qubit1) {                                            \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    ctrl(controls, ValueRange{qubit0, qubit1}, [&](ValueRange targets) {       \
      OP_CLASS::create(*this, targets[0], targets[1], param);                  \
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
    return mc##OP_NAME(PARAM1, PARAM2, {control}, qubit0, qubit1);             \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value qubit0, Value qubit1) {                                            \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    ctrl(controls, ValueRange{qubit0, qubit1}, [&](ValueRange targets) {       \
      OP_CLASS::create(*this, targets[0], targets[1], param1, param2);         \
    });                                                                        \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// ThreeTargetZeroParameter

#define DEFINE_THREE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                  \
  QCProgramBuilder& QCProgramBuilder::OP_NAME(Value qubit0, Value qubit1,      \
                                              Value qubit2) {                  \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, qubit0, qubit1, qubit2);                           \
    return *this;                                                              \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::c##OP_NAME(Value control, Value qubit0,  \
                                                 Value qubit1, Value qubit2) { \
    return mc##OP_NAME({control}, qubit0, qubit1, qubit2);                     \
  }                                                                            \
  QCProgramBuilder& QCProgramBuilder::mc##OP_NAME(                             \
      ValueRange controls, Value qubit0, Value qubit1, Value qubit2) {         \
    ctrl(controls, ValueRange{qubit0, qubit1, qubit2},                         \
         [&](ValueRange targets) {                                             \
           OP_CLASS::create(*this, targets[0], targets[1], targets[2]);        \
         });                                                                   \
    return *this;                                                              \
  }

DEFINE_THREE_TARGET_ZERO_PARAMETER(RCCXOp, rccx)

#undef DEFINE_THREE_TARGET_ZERO_PARAMETER

// BarrierOp

QCProgramBuilder& QCProgramBuilder::barrier(ValueRange qubits) {
  checkFinalized();
  BarrierOp::create(*this, qubits);
  return *this;
}

QCProgramBuilder& QCProgramBuilder::unitary(ValueRange qubits,
                                            DenseElementsAttr matrix) {
  checkFinalized();
  UnitaryOp::create(*this, matrix, qubits);
  return *this;
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

QCProgramBuilder&
QCProgramBuilder::ctrl(ValueRange controls, ValueRange targets,
                       const function_ref<void(ValueRange)>& body) {
  checkFinalized();
  CtrlOp::create(*this, controls, targets, body);
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::ctrl(ValueRange controls, Value target,
                       const function_ref<void(Value)>& body) {
  checkFinalized();
  CtrlOp::create(*this, controls, target, body);
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::ctrl(Value control, Value target,
                       const function_ref<void(Value)>& body) {
  checkFinalized();
  CtrlOp::create(*this, control, target, body);
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::inv(ValueRange qubits,
                      const function_ref<void(ValueRange)>& body) {
  checkFinalized();
  InvOp::create(*this, qubits, body);
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::pow(const std::variant<double, Value>& exponent,
                      ValueRange qubits,
                      const function_ref<void(ValueRange)>& body) {
  checkFinalized();
  PowOp::create(*this, exponent, qubits, body);
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::pow(const std::variant<double, Value>& exponent, Value qubit,
                      const function_ref<void(Value)>& body) {
  checkFinalized();
  PowOp::create(*this, exponent, qubit, body);
  return *this;
}

QCProgramBuilder& QCProgramBuilder::inv(Value qubit,
                                        const function_ref<void(Value)>& body) {
  checkFinalized();
  InvOp::create(*this, qubit, body);
  return *this;
}

//===----------------------------------------------------------------------===//
// SCF operations
//===----------------------------------------------------------------------===//

QCProgramBuilder&
QCProgramBuilder::scfFor(const std::variant<int64_t, Value>& lowerbound,
                         const std::variant<int64_t, Value>& upperbound,
                         const std::variant<int64_t, Value>& step,
                         const function_ref<void(Value)>& body) {
  checkFinalized();

  auto loc = getLoc();
  auto lb = variantToValue(*this, loc, lowerbound);
  auto ub = variantToValue(*this, loc, upperbound);
  auto stepSize = variantToValue(*this, loc, step);

  scf::ForOp::create(*this, lb, ub, stepSize, ValueRange{},
                     [&](OpBuilder& b, Location l, Value iv, ValueRange) {
                       body(iv);
                       scf::YieldOp::create(b, l);
                     });
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::scfWhile(const function_ref<void()>& beforeBody,
                           const function_ref<void()>& afterBody) {
  checkFinalized();

  scf::WhileOp::create(
      *this, TypeRange{}, ValueRange{},
      [&](OpBuilder& b, Location, ValueRange) {
        auto* insertionBlock = b.getInsertionBlock();
        beforeBody();
        if (!isa_and_nonnull<scf::ConditionOp>(
                insertionBlock->getTerminator())) {
          llvm::reportFatalUsageError(
              "scf.while beforeBody must terminate with scf.condition");
        }
      },
      [&](OpBuilder& b, Location loc, ValueRange) {
        afterBody();
        scf::YieldOp::create(b, loc);
      });

  return *this;
}

QCProgramBuilder&
QCProgramBuilder::scfIf(const std::variant<bool, Value>& cond,
                        const function_ref<void()>& thenBody,
                        const function_ref<void()>& elseBody) {
  checkFinalized();

  auto condition = variantToValue(*this, getLoc(), cond);

  auto buildRegion = [&](const function_ref<void()>& body) {
    return [&, body](OpBuilder& b, Location loc) {
      body();
      scf::YieldOp::create(b, loc);
    };
  };

  if (!elseBody) {
    scf::IfOp::create(*this, condition, buildRegion(thenBody));
  } else {
    scf::IfOp::create(*this, condition, buildRegion(thenBody),
                      buildRegion(elseBody));
  }
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::scfIf(Value reg, const std::variant<int64_t, Value>& index,
                        const function_ref<void()>& thenBody,
                        const function_ref<void()>& elseBody) {
  checkFinalized();
  auto condition = loadClassicalBit(reg, index);
  return scfIf(condition, thenBody, elseBody);
}

QCProgramBuilder&
QCProgramBuilder::scfIndexSwitch(const std::variant<int64_t, Value>& arg,
                                 ArrayRef<int64_t> cases,
                                 ArrayRef<function_ref<void()>> caseBodies,
                                 const function_ref<void()>& defaultBody) {
  checkFinalized();

  if (cases.size() != caseBodies.size()) {
    const char* msg = "Each case must have a corresponding case body function";
    llvm::reportFatalUsageError(msg);
    llvm_unreachable(msg);
  }

  auto argValue = variantToValue(*this, getLoc(), arg);
  auto switchOp =
      scf::IndexSwitchOp::create(*this, {}, argValue, cases, cases.size());

  const InsertionGuard guard(*this);
  const auto buildRegion = [&](Region& region, const function_ref<void()>& f) {
    Block* block = createBlock(&region); // Implicitly sets the insertion point.
    f();
    scf::YieldOp::create(*this, getLoc());
  };

  for (auto [region, f] :
       llvm::zip_equal(switchOp.getCaseRegions(), caseBodies)) {
    buildRegion(region, f);
  }

  buildRegion(switchOp.getDefaultRegion(), defaultBody);

  return *this;
}

QCProgramBuilder& QCProgramBuilder::scfCondition(Value condition) {
  checkFinalized();
  scf::ConditionOp::create(*this, condition, ValueRange{});
  return *this;
}

QCProgramBuilder&
QCProgramBuilder::scfCondition(Value reg,
                               const std::variant<int64_t, Value>& index) {
  checkFinalized();
  auto condition = loadClassicalBit(reg, index);
  return scfCondition(condition);
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

QCProgramBuilder& QCProgramBuilder::dealloc(Value qubit) {
  checkFinalized();

  if (isa_and_nonnull<memref::LoadOp>(qubit.getDefiningOp())) {
    llvm::reportFatalUsageError(
        "Register-backed qubits cannot be deallocated manually");
  }

  // Check if the qubit is in the tracking set
  if (!allocatedQubits.remove(qubit)) {
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

void QCProgramBuilder::ensureAllocationMode(
    const AllocationMode requestedMode) {
  if (allocationMode == AllocationMode::Unset) {
    allocationMode = requestedMode;
    return;
  }
  if (allocationMode == requestedMode) {
    return;
  }

  const char* const existingName =
      allocationMode == AllocationMode::Static ? "static" : "dynamic";
  const char* const requestedName =
      requestedMode == AllocationMode::Static ? "static" : "dynamic";

  const std::string message =
      llvm::formatv("Cannot mix {0} and {1} qubit allocation modes in "
                    "QCProgramBuilder",
                    existingName, requestedName)
          .str();
  llvm::reportFatalUsageError(message.c_str());
}

OwningOpRef<ModuleOp> QCProgramBuilder::finalize() {
  checkFinalized();

  auto exitCode = intConstant(0);
  return finalize({exitCode});
}

OwningOpRef<ModuleOp> QCProgramBuilder::finalize(ValueRange returnValues) {
  checkFinalized();

  // Ensure that main function exists and insertion point is valid
  auto* insertionBlock = getInsertionBlock();
  func::FuncOp mainFunc = nullptr;
  for (auto op : cast<ModuleOp>(module).getOps<func::FuncOp>()) {
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
    DeallocOp::create(*this, qubit);
  }
  allocatedQubits.clear();

  for (auto memref : allocatedQregs) {
    memref::DeallocOp::create(*this, memref);
  }
  allocatedQregs.clear();

  // Add return statement with the given return values to the main function
  func::ReturnOp::create(*this, returnValues);

  // Invalidate context to prevent use-after-finalize
  ctx = nullptr;

  // Transfer ownership to the caller
  return cast<ModuleOp>(module);
}

OwningOpRef<ModuleOp> QCProgramBuilder::build(
    MLIRContext* context,
    const function_ref<SmallVector<Value>(QCProgramBuilder&)>& buildFunc) {
  QCProgramBuilder builder(context);
  builder.initialize();
  auto result = buildFunc(builder);
  builder.retype(ValueRange(result).getTypes());
  return builder.finalize(result);
}

OwningOpRef<ModuleOp> QCProgramBuilder::build(
    MLIRContext* context,
    const function_ref<Value(QCProgramBuilder&)>& buildFunc) {
  QCProgramBuilder builder(context);
  builder.initialize();
  auto result = buildFunc(builder);
  builder.retype(result.getType());
  return builder.finalize(result);
}

} // namespace mlir::qc
