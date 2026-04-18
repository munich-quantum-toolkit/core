/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"

#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace mlir::qir {

QIRProgramBuilder::QIRProgramBuilder(MLIRContext* context)
    : ImplicitLocOpBuilder(
          FileLineColLoc::get(context, "<qir-program-builder>", 1, 1), context),
      module(ModuleOp::create(*this)),
      ptrType(LLVM::LLVMPointerType::get(context)),
      voidType(LLVM::LLVMVoidType::get(context)) {
  getContext()->loadDialect<LLVM::LLVMDialect>();
}

void QIRProgramBuilder::initialize() {
  // Set insertion point to the module body
  setInsertionPointToStart(module.getBody());

  // Create main function: () -> i64
  auto funcType = LLVM::LLVMFunctionType::get(getI64Type(), {});
  auto mainFuncOp = LLVM::LLVMFuncOp::create(*this, "main", funcType);
  mainFunc = mainFuncOp.getOperation();

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = StringAttr::get(getContext(), "entry_point");
  mainFuncOp->setAttr("passthrough",
                      ArrayAttr::get(getContext(), {entryPointAttr}));

  // Create the 4-block structure for QIR Base Profile
  entryBlock = mainFuncOp.addEntryBlock(*this);
  bodyBlock = mainFuncOp.addBlock();
  measurementsBlock = mainFuncOp.addBlock();
  outputBlock = mainFuncOp.addBlock();

  // Create exit code constant in entry block
  setInsertionPointToStart(entryBlock);
  exitCode = intConstant(0);

  auto initSig = LLVM::LLVMFunctionType::get(voidType, ptrType);
  auto initDec =
      getOrCreateFunctionDeclaration(*this, module, QIR_INITIALIZE, initSig);
  auto zero = LLVM::ZeroOp::create(*this, ptrType);
  LLVM::CallOp::create(*this, initDec, zero.getResult());

  // Add unconditional branches between blocks
  setInsertionPointToEnd(entryBlock);
  LLVM::BrOp::create(*this, bodyBlock);

  setInsertionPointToEnd(bodyBlock);
  LLVM::BrOp::create(*this, measurementsBlock);

  setInsertionPointToEnd(measurementsBlock);
  LLVM::BrOp::create(*this, outputBlock);

  // Return the exit code (success) in output block
  setInsertionPointToEnd(outputBlock);
  LLVM::ReturnOp::create(*this, exitCode);

  // Set insertion point to body block for user operations
  setInsertionPointToStart(bodyBlock);
}

Value QIRProgramBuilder::intConstant(const int64_t value) {
  checkFinalized();
  return LLVM::ConstantOp::create(*this, getI64IntegerAttr(value)).getResult();
}

Value QIRProgramBuilder::doubleConstant(double value) {
  checkFinalized();
  return LLVM::ConstantOp::create(*this, getF64FloatAttr(value)).getResult();
}

Value QIRProgramBuilder::allocQubit() {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Dynamic);

  metadata_.useDynamicQubit = true;

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert allocations and constants in entry block
  setInsertionPoint(entryBlock->getTerminator());

  auto fnSig = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
  auto fnDec =
      getOrCreateFunctionDeclaration(*this, module, QIR_QUBIT_ALLOC, fnSig);

  auto zero = LLVM::ZeroOp::create(*this, ptrType);
  auto qubit = LLVM::CallOp::create(*this, fnDec, zero.getResult()).getResult();

  qubits.insert(qubit);

  return qubit;
}

Value QIRProgramBuilder::staticQubit(const int64_t index) {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Static);

  if (index < 0) {
    llvm::reportFatalUsageError("Index must be non-negative");
  }

  Value qubit;
  if (const auto it = staticQubits.find(index); it != staticQubits.end()) {
    qubit = it->second;
  } else {
    qubit = createPointerFromIndex(*this, getLoc(), index);
    // Cache for reuse
    staticQubits[index] = qubit;
  }

  // Update qubit count
  if (std::cmp_greater_equal(index, metadata_.numQubits)) {
    metadata_.numQubits = static_cast<size_t>(index) + 1;
  }

  return qubit;
}

SmallVector<Value> QIRProgramBuilder::allocQubitRegister(const int64_t size) {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Dynamic);

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  metadata_.useDynamicQubit = true;

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert allocations and constants in entry block
  setInsertionPoint(entryBlock->getTerminator());

  SmallVector<Value> qubits;
  qubits.reserve(size);

  auto allocFnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(getContext()), {getI64Type(), ptrType, ptrType});
  auto allocFnDecl = getOrCreateFunctionDeclaration(
      *this, module, QIR_QUBIT_ARRAY_ALLOC, allocFnSignature);

  auto array =
      LLVM::AllocaOp::create(*this, ptrType, ptrType, intConstant(size));
  auto zero = LLVM::ZeroOp::create(*this, ptrType);
  LLVM::CallOp::create(
      *this, allocFnDecl,
      ValueRange{intConstant(size), array.getResult(), zero.getResult()});

  qubitArrays.insert(array.getResult());

  for (int64_t i = 0; i < size; ++i) {
    auto gep = LLVM::GEPOp::create(*this, ptrType, ptrType, array.getResult(),
                                   ValueRange{intConstant(i)});
    auto load = LLVM::LoadOp::create(*this, ptrType, gep.getResult());
    qubits.push_back(load.getResult());
  }

  return qubits;
}

QIRProgramBuilder::ClassicalRegister
QIRProgramBuilder::allocClassicalBitRegister(const int64_t size,
                                             const std::string& name) {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  if (name.starts_with("__unnamed__")) {
    llvm::reportFatalUsageError(
        "Classical register names starting with '__unnamed__' are reserved");
  }
  if (resultArrays.contains(name)) {
    llvm::reportFatalUsageError("Classical register already exists");
  }

  metadata_.useDynamicResult = true;

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert allocations and constants in entry block
  setInsertionPoint(entryBlock->getTerminator());

  auto fnSig =
      LLVM::LLVMFunctionType::get(voidType, {getI64Type(), ptrType, ptrType});
  auto fnDec = getOrCreateFunctionDeclaration(*this, module,
                                              QIR_RESULT_ARRAY_ALLOC, fnSig);

  auto array =
      LLVM::AllocaOp::create(*this, ptrType, ptrType, intConstant(size));
  auto zero = LLVM::ZeroOp::create(*this, ptrType);
  LLVM::CallOp::create(
      *this, fnDec,
      ValueRange{intConstant(size), array.getResult(), zero.getResult()});

  resultArrays.try_emplace(name, array.getResult());

  for (int64_t i = 0; i < size; ++i) {
    auto gep = LLVM::GEPOp::create(*this, ptrType, ptrType, array.getResult(),
                                   ValueRange{intConstant(i)});
    auto load = LLVM::LoadOp::create(*this, ptrType, gep.getResult());
    loadedResults.try_emplace({stringSaver.save(name), i}, load.getResult());
  }

  return {.name = name, .size = size};
}

Value QIRProgramBuilder::measure(Value qubit, const int64_t resultIndex) {
  checkFinalized();

  if (resultIndex < 0) {
    llvm::reportFatalUsageError("Result index must be non-negative");
  }

  metadata_.useDynamicResult = true;

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert allocations and constants in entry block
  setInsertionPoint(entryBlock->getTerminator());

  // Get or create result pointer
  Value result;
  if (const auto it = resultPtrs.find(resultIndex); it != resultPtrs.end()) {
    result = it->second;
  } else {
    auto fnSig = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(*this, module, QIR_RESULT_ALLOC, fnSig);
    auto zero = LLVM::ZeroOp::create(*this, ptrType);
    result = LLVM::CallOp::create(*this, fnDec, zero.getResult()).getResult();
    resultPtrs.try_emplace(resultIndex, result);
  }

  // Switch to measurements block
  setInsertionPoint(measurementsBlock->getTerminator());

  // Create measure call
  const auto mzSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  auto mzDec =
      getOrCreateFunctionDeclaration(*this, module, QIR_MEASURE, mzSig);
  LLVM::CallOp::create(*this, mzDec, ValueRange{qubit, result});

  return result;
}

QIRProgramBuilder& QIRProgramBuilder::measure(Value qubit, const Bit& bit) {
  checkFinalized();

  auto it = loadedResults.find({bit.registerName, bit.registerIndex});
  if (it == loadedResults.end()) {
    llvm::reportFatalUsageError(
        "Bit does not belong to an allocated classical register");
  }
  auto result = it->second;

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Switch to measurements block
  setInsertionPoint(measurementsBlock->getTerminator());

  // Create measure call
  const auto fnSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  auto fnDec =
      getOrCreateFunctionDeclaration(*this, module, QIR_MEASURE, fnSig);
  LLVM::CallOp::create(*this, fnDec, ValueRange{qubit, result});

  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::reset(Value qubit) {
  checkFinalized();

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Switch to measurements block
  setInsertionPoint(measurementsBlock->getTerminator());

  // Create reset call
  const auto qirSignature = LLVM::LLVMFunctionType::get(voidType, ptrType);
  auto fnDecl =
      getOrCreateFunctionDeclaration(*this, module, QIR_RESET, qirSignature);
  LLVM::CallOp::create(*this, fnDecl, ValueRange{qubit});

  return *this;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

void QIRProgramBuilder::createCallOp(
    const SmallVector<std::variant<double, Value>>& parameters,
    ValueRange controls, const SmallVector<Value>& targets, StringRef fnName) {
  checkFinalized();

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert constants in entry block
  setInsertionPoint(entryBlock->getTerminator());

  SmallVector<Value> parameterOperands;
  parameterOperands.reserve(parameters.size());
  for (const auto& parameter : parameters) {
    Value parameterOperand;
    if (std::holds_alternative<double>(parameter)) {
      parameterOperand = doubleConstant(std::get<double>(parameter));
    } else {
      parameterOperand = std::get<Value>(parameter);
    }
    parameterOperands.push_back(parameterOperand);
  }

  // Insert in body block (before branch)
  setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(parameters.size() + controls.size() + targets.size());
  const auto floatType = Float64Type::get(getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointers
  for (size_t i = 0; i < targets.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add parameter types
  for (size_t i = 0; i < parameters.size(); ++i) {
    argumentTypes.push_back(floatType);
  }

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(voidType, argumentTypes);

  // Declare QIR function
  auto fnDecl =
      getOrCreateFunctionDeclaration(*this, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(parameters.size() + controls.size() + targets.size());
  operands.append(controls.begin(), controls.end());
  operands.append(targets.begin(), targets.end());
  operands.append(parameterOperands.begin(), parameterOperands.end());

  LLVM::CallOp::create(*this, fnDecl, operands);
}

// GPhaseOp

QIRProgramBuilder&
QIRProgramBuilder::gphase(const std::variant<double, Value>& theta) {
  createCallOp({theta}, {}, {}, QIR_GPHASE);
  return *this;
}

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_NAME_BIG, OP_NAME_SMALL)           \
  QIRProgramBuilder& QIRProgramBuilder::OP_NAME_SMALL(Value qubit) {           \
    createCallOp({}, {}, {qubit}, QIR_##OP_NAME_BIG);                          \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(Value control,        \
                                                         Value target) {       \
    createCallOp({}, {control}, {target}, QIR_C##OP_NAME_BIG);                 \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(ValueRange controls, \
                                                          Value target) {      \
    createCallOp({}, controls, {target},                                       \
                 getFnName##OP_NAME_BIG(controls.size()));                     \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_ZERO_PARAMETER(I, id)
DEFINE_ONE_TARGET_ZERO_PARAMETER(X, x)
DEFINE_ONE_TARGET_ZERO_PARAMETER(Y, y)
DEFINE_ONE_TARGET_ZERO_PARAMETER(Z, z)
DEFINE_ONE_TARGET_ZERO_PARAMETER(H, h)
DEFINE_ONE_TARGET_ZERO_PARAMETER(S, s)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SDG, sdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(T, t)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TDG, tdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SX, sx)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXDG, sxdg)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_NAME_BIG, OP_NAME_SMALL, PARAM)     \
  QIRProgramBuilder& QIRProgramBuilder::OP_NAME_SMALL(                         \
      const std::variant<double, Value>&(PARAM), Value qubit) {                \
    createCallOp({PARAM}, {}, {qubit}, QIR_##OP_NAME_BIG);                     \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM), Value control,                \
      Value target) {                                                          \
    createCallOp({PARAM}, {control}, {target}, QIR_C##OP_NAME_BIG);            \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value target) {                                                          \
    createCallOp({PARAM}, controls, {target},                                  \
                 getFnName##OP_NAME_BIG(controls.size()));                     \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_ONE_PARAMETER(RX, rx, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RY, ry, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZ, rz, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(P, p, theta)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_NAME_BIG, OP_NAME_SMALL, PARAM1,    \
                                        PARAM2)                                \
  QIRProgramBuilder& QIRProgramBuilder::OP_NAME_SMALL(                         \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value qubit) {               \
    createCallOp({PARAM1, PARAM2}, {}, {qubit}, QIR_##OP_NAME_BIG);            \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control,               \
      Value target) {                                                          \
    createCallOp({PARAM1, PARAM2}, {control}, {target}, QIR_C##OP_NAME_BIG);   \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target) {                                                          \
    createCallOp({PARAM1, PARAM2}, controls, {target},                         \
                 getFnName##OP_NAME_BIG(controls.size()));                     \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_TWO_PARAMETER(R, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_NAME_BIG, OP_NAME_SMALL, PARAM1,  \
                                          PARAM2, PARAM3)                      \
  QIRProgramBuilder& QIRProgramBuilder::OP_NAME_SMALL(                         \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value qubit) {               \
    createCallOp({PARAM1, PARAM2, PARAM3}, {}, {qubit}, QIR_##OP_NAME_BIG);    \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value control,               \
      Value target) {                                                          \
    createCallOp({PARAM1, PARAM2, PARAM3}, {control}, {target},                \
                 QIR_C##OP_NAME_BIG);                                          \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), ValueRange controls,         \
      Value target) {                                                          \
    createCallOp({PARAM1, PARAM2, PARAM3}, controls, {target},                 \
                 getFnName##OP_NAME_BIG(controls.size()));                     \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_THREE_PARAMETER(U, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_NAME_BIG, OP_NAME_SMALL)           \
  QIRProgramBuilder& QIRProgramBuilder::OP_NAME_SMALL(Value target0,           \
                                                      Value target1) {         \
    createCallOp({}, {}, {target0, target1}, QIR_##OP_NAME_BIG);               \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      Value control, Value target0, Value target1) {                           \
    createCallOp({}, {control}, {target0, target1}, QIR_C##OP_NAME_BIG);       \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      ValueRange controls, Value target0, Value target1) {                     \
    createCallOp({}, controls, {target0, target1},                             \
                 getFnName##OP_NAME_BIG(controls.size()));                     \
    return *this;                                                              \
  }

// NOLINTNEXTLINE(bugprone-exception-escape)
DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAP, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ISWAP, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCX, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECR, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_NAME_BIG, OP_NAME_SMALL, PARAM)     \
  QIRProgramBuilder& QIRProgramBuilder::OP_NAME_SMALL(                         \
      const std::variant<double, Value>&(PARAM), Value target0,                \
      Value target1) {                                                         \
    createCallOp({PARAM}, {}, {target0, target1}, QIR_##OP_NAME_BIG);          \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM), Value control, Value target0, \
      Value target1) {                                                         \
    createCallOp({PARAM}, {control}, {target0, target1}, QIR_C##OP_NAME_BIG);  \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value target0, Value target1) {                                          \
    createCallOp({PARAM}, controls, {target0, target1},                        \
                 getFnName##OP_NAME_BIG(controls.size()));                     \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_ONE_PARAMETER(RXX, rxx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RYY, ryy, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZX, rzx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZZ, rzz, theta)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

// TwoTargetTwoParameter

#define DEFINE_TWO_TARGET_TWO_PARAMETER(OP_NAME_BIG, OP_NAME_SMALL, PARAM1,    \
                                        PARAM2)                                \
  QIRProgramBuilder& QIRProgramBuilder::OP_NAME_SMALL(                         \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value target0,               \
      Value target1) {                                                         \
    createCallOp({PARAM1, PARAM2}, {}, {target0, target1}, QIR_##OP_NAME_BIG); \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control,               \
      Value target0, Value target1) {                                          \
    createCallOp({PARAM1, PARAM2}, {control}, {target0, target1},              \
                 QIR_C##OP_NAME_BIG);                                          \
    return *this;                                                              \
  }                                                                            \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target0, Value target1) {                                          \
    createCallOp({PARAM1, PARAM2}, controls, {target0, target1},               \
                 getFnName##OP_NAME_BIG(controls.size()));                     \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPLUSYY, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMINUSYY, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

void QIRProgramBuilder::checkFinalized() const {
  if (isFinalized) {
    llvm::reportFatalUsageError(
        "QIRProgramBuilder instance has been finalized");
  }
}

void QIRProgramBuilder::ensureAllocationMode(
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
                    "QIRProgramBuilder",
                    existingName, requestedName)
          .str();
  llvm::reportFatalUsageError(message.c_str());
}

void QIRProgramBuilder::generateOutputRecording() {
  if (resultArrays.empty() && resultPtrs.empty()) {
    return; // No measurements to record
  }

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert in output block (before return)
  setInsertionPoint(outputBlock->getTerminator());

  if (!resultPtrs.empty()) {
    auto fnSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(*this, module, QIR_RECORD_OUTPUT, fnSig);
    // Create output recording for each result pointer
    for (const auto& [index, ptr] : resultPtrs) {
      auto label = createResultLabel(*this, module,
                                     "__unnamed__" + std::to_string(index))
                       .getResult();
      LLVM::CallOp::create(*this, fnDec, ValueRange{ptr, label});
    }
  }

  if (!resultArrays.empty()) {
    auto fnSig =
        LLVM::LLVMFunctionType::get(voidType, {getI64Type(), ptrType, ptrType});
    auto fnDec = getOrCreateFunctionDeclaration(*this, module,
                                                QIR_ARRAY_RECORD_OUTPUT, fnSig);
    // Create output recording for each register
    for (const auto& [name, results] : resultArrays) {
      auto size = results.getDefiningOp<LLVM::AllocaOp>().getArraySize();
      auto label = createResultLabel(*this, module, name).getResult();
      LLVM::CallOp::create(*this, fnDec, ValueRange{size, results, label});
    }
  }
}

OwningOpRef<ModuleOp> QIRProgramBuilder::finalize() {
  checkFinalized();

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Release resources in output block
  setInsertionPoint(outputBlock->getTerminator());

  for (auto qubit : qubits) {
    auto sig = LLVM::LLVMFunctionType::get(voidType, {ptrType});
    auto dec =
        getOrCreateFunctionDeclaration(*this, module, QIR_QUBIT_RELEASE, sig);
    LLVM::CallOp::create(*this, dec, ValueRange{qubit});
  }

  for (auto array : qubitArrays) {
    auto sig = LLVM::LLVMFunctionType::get(voidType, {getI64Type(), ptrType});
    auto dec = getOrCreateFunctionDeclaration(*this, module,
                                              QIR_QUBIT_ARRAY_RELEASE, sig);
    auto size = array.getDefiningOp<LLVM::AllocaOp>().getArraySize();
    LLVM::CallOp::create(*this, dec, ValueRange{size, array});
  }

  // Generate output recording in output block
  generateOutputRecording();

  for (auto& [_, ptr] : resultPtrs) {
    auto sig = LLVM::LLVMFunctionType::get(voidType, {ptrType});
    auto dec =
        getOrCreateFunctionDeclaration(*this, module, QIR_RESULT_RELEASE, sig);
    LLVM::CallOp::create(*this, dec, ptr);
  }

  for (auto& [_, array] : resultArrays) {
    auto sig = LLVM::LLVMFunctionType::get(voidType, {getI64Type(), ptrType});
    auto dec = getOrCreateFunctionDeclaration(*this, module,
                                              QIR_RESULT_ARRAY_RELEASE, sig);
    auto size = array.getDefiningOp<LLVM::AllocaOp>().getArraySize();
    LLVM::CallOp::create(*this, dec, ValueRange{size, array});
  }

  auto mainFuncOp = llvm::cast<LLVM::LLVMFuncOp>(mainFunc);
  setQIRAttributes(mainFuncOp, metadata_);

  isFinalized = true;

  return module;
}

OwningOpRef<ModuleOp> QIRProgramBuilder::build(
    MLIRContext* context,
    const llvm::function_ref<void(QIRProgramBuilder&)>& buildFunc) {
  QIRProgramBuilder builder(context);
  builder.initialize();
  buildFunc(builder);
  return builder.finalize();
}

} // namespace mlir::qir
