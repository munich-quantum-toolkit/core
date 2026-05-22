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

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
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
#include <iterator>
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
  setInsertionPointToStart(cast<ModuleOp>(module).getBody());

  // Create main function: () -> i64
  auto funcType = LLVM::LLVMFunctionType::get(getI64Type(), {});
  auto mainFuncOp = LLVM::LLVMFuncOp::create(*this, "main", funcType);
  mainFunc = mainFuncOp.getOperation();

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = StringAttr::get(getContext(), "entry_point");
  mainFuncOp->setAttr("passthrough",
                      ArrayAttr::get(getContext(), {entryPointAttr}));

  // Create the base block structure for the QIR Profiles
  entryBlock = mainFuncOp.addEntryBlock(*this);
  bodyBlock = mainFuncOp.addBlock();
  outputBlock = mainFuncOp.addBlock();

  // Only create the measurement block if the Base Profile is used
  if (profile == Profile::Base) {
    measurementsBlock = createBlock(&mainFuncOp.getBody());
    mainFuncOp.getBlocks().splice(Region::iterator(outputBlock),
                                  mainFuncOp.getBlocks(), measurementsBlock);
    setInsertionPointToEnd(bodyBlock);
    LLVM::BrOp::create(*this, measurementsBlock);

    setInsertionPointToEnd(measurementsBlock);
    LLVM::BrOp::create(*this, outputBlock);
  } else {
    setInsertionPointToEnd(bodyBlock);
    LLVM::BrOp::create(*this, outputBlock);
  }

  // Create exit code constant in entry block
  setInsertionPointToStart(entryBlock);
  exitCode = intConstant(0);

  // Add initialize call
  auto initSig = LLVM::LLVMFunctionType::get(voidType, ptrType);
  auto initDec =
      getOrCreateFunctionDeclaration(*this, module, QIR_INITIALIZE, initSig);
  auto zero = LLVM::ZeroOp::create(*this, ptrType);
  LLVM::CallOp::create(*this, initDec, zero.getResult());

  setInsertionPointToEnd(entryBlock);
  LLVM::BrOp::create(*this, bodyBlock);

  // Return the exit code (success) in output block
  setInsertionPointToEnd(outputBlock);
  LLVM::ReturnOp::create(*this, exitCode);

  // Set insertion point to body block for user operations
  setInsertionPointToStart(bodyBlock);
}

Value QIRProgramBuilder::resolveIntVariant(
    const std::variant<int64_t, Value>& variant) {
  if (std::holds_alternative<int64_t>(variant)) {
    return LLVM::ConstantOp::create(*this, IntegerType::get(context, 64),
                                    getIndexAttr(std::get<int64_t>(variant)))
        .getResult();
  }
  return std::get<Value>(variant);
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

  Value qubit;
  if (profile == Profile::Adaptive) {
    ensureAllocationMode(AllocationMode::Dynamic);
    metadata_.useDynamicQubit = true;

    auto fnSig = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(*this, module, QIR_QUBIT_ALLOC, fnSig);

    auto zero = LLVM::ZeroOp::create(*this, ptrType);
    qubit = LLVM::CallOp::create(*this, fnDec, zero.getResult()).getResult();
  } else {
    qubit = staticQubit(static_cast<int64_t>(metadata_.numQubits));
  }

  qubits.insert(qubit);

  return qubit;
}

Value QIRProgramBuilder::staticQubit(const int64_t index) {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Static);
  const InsertionGuard guard(*this);

  // Insert allocations and constants in entry block
  setInsertionPoint(entryBlock->getTerminator());

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

Value QIRProgramBuilder::staticResult(const int64_t index) {
  checkFinalized();

  const InsertionGuard guard(*this);

  // Insert allocations and constants in entry block
  setInsertionPoint(entryBlock->getTerminator());

  if (index < 0) {
    llvm::reportFatalUsageError("Index must be non-negative");
  }

  Value result;
  if (const auto it = resultPtrs.find(index); it != resultPtrs.end()) {
    result = it->second;
  } else {
    result = createPointerFromIndex(*this, getLoc(), index);
    // Cache for reuse
    resultPtrs[index] = result;
  }

  // Update result count
  if (std::cmp_greater_equal(index, metadata_.numResults)) {
    metadata_.numResults = static_cast<size_t>(index) + 1;
  }

  return result;
}

Value QIRProgramBuilder::QubitRegister::operator[](const size_t index) const {
  if (index >= qubits.size()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return qubits[index];
}

QIRProgramBuilder::QubitRegister
QIRProgramBuilder::allocQubitRegister(const int64_t size) {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  Value array;
  SmallVector<Value> qubits;

  qubits.reserve(size);

  if (profile == Profile::Adaptive) {
    // Create a dynamic qubit array and load the qubits in the Adaptive Profile
    ensureAllocationMode(AllocationMode::Dynamic);
    metadata_.useArrays = true;
    metadata_.useDynamicQubit = true;

    auto allocFnSignature =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(getContext()),
                                    {getI64Type(), ptrType, ptrType});
    auto allocFnDecl = getOrCreateFunctionDeclaration(
        *this, module, QIR_QUBIT_ARRAY_ALLOC, allocFnSignature);

    array = LLVM::AllocaOp::create(*this, ptrType, ptrType, intConstant(size))
                .getResult();
    auto zero = LLVM::ZeroOp::create(*this, ptrType);
    LLVM::CallOp::create(
        *this, allocFnDecl,
        ValueRange{intConstant(size), array, zero.getResult()});

    qubitArrays.insert(array);

    for (int64_t i = 0; i < size; ++i) {
      auto index = intConstant(i);
      auto gep = LLVM::GEPOp::create(*this, ptrType, ptrType, array,
                                     ValueRange{index});
      auto load = LLVM::LoadOp::create(*this, ptrType, gep.getResult());
      qubits.push_back(load.getResult());
      loadedQubits[array].insert(index);
    }
  } else {
    // Create static qubits in the Base Profile
    for (int64_t i = 0; i < size; ++i) {
      auto qubit = staticQubit(static_cast<int64_t>(metadata_.numQubits));
      qubits.push_back(qubit);
    }
  }

  return {.value = array, .qubits = std::move(qubits)};
}

Value QIRProgramBuilder::load(Value reg, Value index) {
  if (profile == Profile::Base) {
    llvm::reportFatalUsageError("Arrays can only be used if the "
                                "Adaptive Profile is selected.");
  }
  if (loadedQubits[reg].contains(index)) {
    llvm::reportFatalUsageError(
        "Qubit was already extracted from the register at this index");
  }

  auto gep = LLVM::GEPOp::create(*this, ptrType, ptrType, reg, index);
  auto load = LLVM::LoadOp::create(*this, ptrType, gep.getResult());
  loadedQubits[reg].insert(index);

  return load.getResult();
}

QIRProgramBuilder::Bit
QIRProgramBuilder::ClassicalRegister::operator[](const int64_t index) const {
  if (index < 0 || index >= size) {
    const std::string msg = "Bit index " + std::to_string(index) +
                            " out of bounds for register '" + name +
                            "' of size " + std::to_string(size);
    llvm::reportFatalUsageError(msg.c_str());
  }
  return {.registerName = name, .registerSize = size, .registerIndex = index};
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
  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert allocations and constants in entry block
  setInsertionPoint(entryBlock->getTerminator());

  if (profile == Profile::Adaptive) {
    // Create a dynamic result array for the Adaptive Profile
    metadata_.useDynamicResult = true;
    metadata_.useArrays = true;

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
  } else {
    // Use static results in the Base Profile
    for (int64_t i = 0; i < size; ++i) {
      auto result = staticResult(static_cast<int64_t>(metadata_.numResults));
      loadedResults.try_emplace({stringSaver.save(name), i}, result);
    }
  }

  return {.name = name, .size = size};
}

Value QIRProgramBuilder::measure(Value qubit, const int64_t resultIndex) {
  checkFinalized();

  if (resultIndex < 0) {
    llvm::reportFatalUsageError("Result index must be non-negative");
  }

  // Save current insertion point
  const InsertionGuard guard(*this);
  auto insertionPoint = saveInsertionPoint();
  // Insert allocations and constants in entry block
  setInsertionPoint(entryBlock->getTerminator());

  // Get or create result pointer
  auto result = staticResult(resultIndex);

  restoreInsertionPoint(insertionPoint);

  // Only set the insertionpoint if the Base Profile is used
  if (profile == Profile::Base) {
    setInsertionPoint(measurementsBlock->getTerminator());
  }

  // Create measure call
  const auto mzSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  auto mzDec =
      getOrCreateFunctionDeclaration(*this, module, QIR_MEASURE, mzSig);
  LLVM::CallOp::create(*this, mzDec, ValueRange{qubit, result});

  return result;
}

Value QIRProgramBuilder::measure(Value qubit, const Bit& bit) {
  checkFinalized();
  const InsertionGuard guard(*this);

  auto it = loadedResults.find({bit.registerName, bit.registerIndex});
  if (it == loadedResults.end()) {
    llvm::reportFatalUsageError("Bit does not belong to a result pointer");
  }
  auto result = it->second;
  if (profile == Profile::Adaptive) {
    metadata_.useDynamicResult = true;
  } else {
    setInsertionPoint(measurementsBlock->getTerminator());
  }

  // Create measure call
  const auto fnSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  auto fnDec =
      getOrCreateFunctionDeclaration(*this, module, QIR_MEASURE, fnSig);
  LLVM::CallOp::create(*this, fnDec, ValueRange{qubit, result});

  return result;
}

QIRProgramBuilder& QIRProgramBuilder::reset(Value qubit) {
  checkFinalized();
  if (profile == Profile::Base) {
    llvm::reportFatalUsageError("Reset operation can only be used if the "
                                "Adaptive Profile is selected.");
  }

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
  auto insertionPoint = saveInsertionPoint();

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
  // Restore insertion point
  restoreInsertionPoint(insertionPoint);

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
// SCF Operations
//===----------------------------------------------------------------------===//
QIRProgramBuilder&
QIRProgramBuilder::scfFor(const std::variant<int64_t, Value>& lowerbound,
                          const std::variant<int64_t, Value>& upperbound,
                          const std::variant<int64_t, Value>& step,
                          const function_ref<void(Value)>& body) {
  checkFinalized();
  if (profile == Profile::Base) {
    llvm::reportFatalUsageError("For operation can only be used if the "
                                "Adaptive Profile is selected.");
  }

  int& backwardsBranchingFlag = metadata_.backwardsBranching;
  if (backwardsBranchingFlag != 1 && backwardsBranchingFlag != 3) {
    backwardsBranchingFlag += 1;
  }

  auto loc = getLoc();
  auto lb = resolveIntVariant(lowerbound);
  auto ub = resolveIntVariant(upperbound);
  auto stepSize = resolveIntVariant(step);
  auto i64Type = getI64Type();
  auto* currentBlock = getInsertionBlock();

  // Create the blocks
  auto* conditionBlock =
      createBlock(currentBlock->getParent(),
                  std::next(Region::iterator(currentBlock)), {i64Type}, {loc});
  auto* loopBlock = createBlock(conditionBlock->getParent(),
                                std::next(Region::iterator(conditionBlock)));
  auto* nextBlock = createBlock(loopBlock->getParent(),
                                std::next(Region::iterator(loopBlock)));

  // Move the current terminator to the next block if it exists
  if (currentBlock->mightHaveTerminator()) {
    auto* currentTerminator = currentBlock->getTerminator();
    currentTerminator->moveBefore(nextBlock, nextBlock->end());
  }

  // Add jump to condition block
  setInsertionPointToEnd(currentBlock);
  LLVM::BrOp::create(*this, lb, conditionBlock);

  // Add conditional jump to loop block or next block
  setInsertionPointToEnd(conditionBlock);
  auto cmp = LLVM::ICmpOp::create(*this, LLVM::ICmpPredicate::slt,
                                  conditionBlock->getArgument(0), ub);
  LLVM::CondBrOp::create(*this, cmp.getResult(), loopBlock, nextBlock);

  // Build loop body
  setInsertionPointToStart(loopBlock);
  body(conditionBlock->getArgument(0));

  // Update loop condition and jump back to the condition block
  auto addOp = LLVM::AddOp::create(*this, i64Type,
                                   conditionBlock->getArgument(0), stepSize);
  LLVM::BrOp::create(*this, addOp.getResult(), conditionBlock);

  // Set insertionpoint to next block
  setInsertionPointToStart(nextBlock);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::scfIf(const std::variant<bool, Value>& cond,
                         const function_ref<void()>& thenBody,
                         const function_ref<void()>& elseBody) {
  checkFinalized();
  if (profile == Profile::Base) {
    llvm::reportFatalUsageError("If operation can only be used if the "
                                "Adaptive Profile is selected.");
  }

  auto* currentBlock = getInsertionBlock();

  // Create the blocks
  auto* thenBlock = createBlock(currentBlock->getParent(),
                                std::next(Region::iterator(currentBlock)));
  auto* nextBlock = createBlock(thenBlock->getParent(),
                                std::next(Region::iterator(thenBlock)));

  // Move the current terminator to the next block if it exists
  if (currentBlock->mightHaveTerminator()) {
    auto* currentTerminator = currentBlock->getTerminator();
    currentTerminator->moveBefore(nextBlock, nextBlock->end());
  }
  // Build the then body
  setInsertionPointToStart(thenBlock);
  thenBody();
  LLVM::BrOp::create(*this, nextBlock);

  // Optionally build the else body
  Block* elseBlock = nullptr;
  if (elseBody) {
    elseBlock = createBlock(thenBlock->getParent(),
                            std::next(Region::iterator(thenBlock)));
    setInsertionPointToStart(elseBlock);
    elseBody();
    LLVM::BrOp::create(*this, nextBlock);
  }

  // Add read result operation to the current block and add conditional jump
  setInsertionPointToEnd(currentBlock);

  Value branchCondition;
  if (std::holds_alternative<bool>(cond)) {
    branchCondition =
        LLVM::ConstantOp::create(
            *this, getI1Type(),
            getIntegerAttr(getI1Type(), std::get<bool>(cond) ? 1 : 0))
            .getResult();
  } else {
    auto conditionValue = std::get<Value>(cond);
    if (conditionValue.getType() != ptrType) {
      llvm::reportFatalUsageError("Condition value must be llvm.ptr type");
    }

    const auto fnSig = LLVM::LLVMFunctionType::get(getI1Type(), {ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(*this, module, QIR_READ_RESULT, fnSig);
    auto readOp = LLVM::CallOp::create(*this, fnDec, std::get<Value>(cond));
    branchCondition = readOp.getResult();
  }

  LLVM::CondBrOp::create(*this, branchCondition, thenBlock,
                         elseBody ? elseBlock : nextBlock);

  // Set insertionpoint to next block
  setInsertionPointToStart(nextBlock);

  return *this;
}
QIRProgramBuilder&
QIRProgramBuilder::scfWhile(const function_ref<Value()>& beforeBody,
                            const function_ref<void()>& afterBody) {
  checkFinalized();
  if (profile == Profile::Base) {
    llvm::reportFatalUsageError("While operation can only be used if the "
                                "Adaptive Profile is selected.");
  }

  int& backwardsBranchingFlag = metadata_.backwardsBranching;
  if (backwardsBranchingFlag != 2 && backwardsBranchingFlag != 3) {
    backwardsBranchingFlag += 2;
  }

  auto* currentBlock = getInsertionBlock();
  // Build the blocks
  auto* beforeBlock = createBlock(currentBlock->getParent(),
                                  std::next(Region::iterator(currentBlock)));
  // Only create afterBlock if afterBody is provided
  Block* afterBlock =
      afterBody ? createBlock(beforeBlock->getParent(),
                              std::next(Region::iterator(beforeBlock)))
                : nullptr;
  auto* nextBlock = (afterBlock != nullptr)
                        ? createBlock(afterBlock->getParent(),
                                      std::next(Region::iterator(afterBlock)))
                        : createBlock(beforeBlock->getParent(),
                                      std::next(Region::iterator(beforeBlock)));
  // Move the current terminator to the next block if it exists
  if (currentBlock->mightHaveTerminator()) {
    auto* currentTerminator = currentBlock->getTerminator();
    currentTerminator->moveBefore(nextBlock, nextBlock->end());
  }

  setInsertionPointToEnd(currentBlock);
  LLVM::BrOp::create(*this, beforeBlock);

  // Build the before body and the conditional jump
  setInsertionPointToStart(beforeBlock);
  auto condition = beforeBody();

  if (condition.getType() != ptrType) {
    llvm::reportFatalUsageError("Before region must return a llvm.ptr type");
  }
  const auto fnSig = LLVM::LLVMFunctionType::get(getI1Type(), {ptrType});
  auto fnDec =
      getOrCreateFunctionDeclaration(*this, module, QIR_READ_RESULT, fnSig);
  auto readOp = LLVM::CallOp::create(*this, fnDec, condition);

  // Build the after body if it exists
  if (afterBody) {
    LLVM::CondBrOp::create(*this, readOp.getResult(), afterBlock, nextBlock);
    setInsertionPointToStart(afterBlock);
    afterBody();
    LLVM::BrOp::create(*this, beforeBlock);
  } else {
    LLVM::CondBrOp::create(*this, readOp.getResult(), beforeBlock, nextBlock);
  }

  // Set the insertion point to the next block
  setInsertionPointToStart(nextBlock);
  return *this;
}

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

  if (profile == Profile::Adaptive) {
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
  }

  // Generate output recording in output block
  generateOutputRecording();

  if (profile == Profile::Adaptive) {
    for (auto& [_, ptr] : resultPtrs) {
      auto sig = LLVM::LLVMFunctionType::get(voidType, {ptrType});
      auto dec = getOrCreateFunctionDeclaration(*this, module,
                                                QIR_RESULT_RELEASE, sig);
      LLVM::CallOp::create(*this, dec, ptr);
    }

    for (auto& [_, array] : resultArrays) {
      auto sig = LLVM::LLVMFunctionType::get(voidType, {getI64Type(), ptrType});
      auto dec = getOrCreateFunctionDeclaration(*this, module,
                                                QIR_RESULT_ARRAY_RELEASE, sig);
      auto size = array.getDefiningOp<LLVM::AllocaOp>().getArraySize();
      LLVM::CallOp::create(*this, dec, ValueRange{size, array});
    }
  }

  auto mainFuncOp = cast<LLVM::LLVMFuncOp>(mainFunc);
  metadata_.useAdaptive = profile == Profile::Adaptive;
  setQIRAttributes(mainFuncOp, metadata_);

  isFinalized = true;

  return cast<ModuleOp>(module);
}

OwningOpRef<ModuleOp> QIRProgramBuilder::build(
    MLIRContext* context,
    const function_ref<void(QIRProgramBuilder&)>& buildFunc, Profile profile) {
  QIRProgramBuilder builder(context);
  builder.profile = profile;
  builder.initialize();
  buildFunc(builder);
  return builder.finalize();
}

} // namespace mlir::qir
