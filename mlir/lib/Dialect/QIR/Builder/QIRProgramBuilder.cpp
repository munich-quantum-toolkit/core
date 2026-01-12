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

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
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
  mainFunc = LLVM::LLVMFuncOp::create(*this, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = StringAttr::get(getContext(), "entry_point");
  mainFunc->setAttr("passthrough",
                    ArrayAttr::get(getContext(), {entryPointAttr}));

  // Create the 4-block structure for QIR Base Profile
  entryBlock = mainFunc.addEntryBlock(*this);
  bodyBlock = mainFunc.addBlock();
  measurementsBlock = mainFunc.addBlock();
  outputBlock = mainFunc.addBlock();

  // Create exit code constant in entry block (where constants belong) and add
  // QIR initialization call in entry block (after exit code constant)
  setInsertionPointToStart(entryBlock);
  auto zeroOp = LLVM::ZeroOp::create(*this, ptrType);
  exitCode = LLVM::ConstantOp::create(*this, getI64IntegerAttr(0));
  const auto initType = LLVM::LLVMFunctionType::get(voidType, ptrType);
  auto initFunc =
      getOrCreateFunctionDeclaration(*this, module, QIR_INITIALIZE, initType);
  LLVM::CallOp::create(*this, initFunc, ValueRange{zeroOp.getResult()});

  // Add unconditional branches between blocks
  setInsertionPointToEnd(entryBlock);
  LLVM::BrOp::create(*this, bodyBlock);

  setInsertionPointToEnd(bodyBlock);
  LLVM::BrOp::create(*this, measurementsBlock);

  setInsertionPointToEnd(measurementsBlock);
  LLVM::BrOp::create(*this, outputBlock);

  // Return the exit code (success) in output block
  setInsertionPointToEnd(outputBlock);
  LLVM::ReturnOp::create(*this, ValueRange{exitCode.getResult()});

  // Set insertion point to body block for user operations
  setInsertionPointToStart(bodyBlock);
}

Value QIRProgramBuilder::staticQubit(const int64_t index) {
  checkFinalized();

  if (index < 0) {
    llvm::reportFatalUsageError("Index must be non-negative");
  }

  // Check cache
  Value val{};
  if (const auto it = ptrCache.find(index); it != ptrCache.end()) {
    val = it->second;
  } else {
    val = createPointerFromIndex(*this, getLoc(), index);
    // Cache for reuse
    ptrCache[index] = val;
  }

  // Update qubit count
  if (std::cmp_greater_equal(index, metadata_.numQubits)) {
    metadata_.numQubits = static_cast<size_t>(index) + 1;
  }

  return val;
}

SmallVector<Value> QIRProgramBuilder::allocQubitRegister(const int64_t size) {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  SmallVector<Value> qubits;
  qubits.reserve(size);

  for (int64_t i = 0; i < size; ++i) {
    qubits.push_back(staticQubit(static_cast<int64_t>(metadata_.numQubits)));
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

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert in measurements block (before branch)
  setInsertionPoint(measurementsBlock->getTerminator());

  const auto numResults = static_cast<int64_t>(metadata_.numResults);
  for (int64_t i = 0; i < size; ++i) {
    Value val{};
    if (const auto it = ptrCache.find(numResults + i); it != ptrCache.end()) {
      val = it->second;
    } else {
      val = createPointerFromIndex(*this, getLoc(), numResults + i);
      // Cache for reuse
      ptrCache[numResults + i] = val;
    }
    registerResultMap.insert({{stringSaver.save(name), i}, val});
  }
  metadata_.numResults += size;
  return {.name = name, .size = size};
}

Value QIRProgramBuilder::measure(Value qubit, const int64_t resultIndex) {
  checkFinalized();

  if (resultIndex < 0) {
    llvm::reportFatalUsageError("Result index must be non-negative");
  }

  // Choose a safe default register name
  static constexpr auto DEFAULT_REG_NAME = "c";
  StringRef regName{DEFAULT_REG_NAME};
  if (llvm::any_of(registerResultMap, [](const auto& entry) {
        return entry.first.first == DEFAULT_REG_NAME;
      })) {
    static constexpr auto FALLBACK_REG_NAME = "__unnamed__";
    regName = FALLBACK_REG_NAME;
  }

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert in measurements block (before branch)
  setInsertionPoint(measurementsBlock->getTerminator());

  const auto key = std::make_pair(regName, resultIndex);
  if (const auto it = registerResultMap.find(key);
      it != registerResultMap.end()) {
    return it->second;
  }

  Value resultValue{};
  if (const auto it = ptrCache.find(resultIndex); it != ptrCache.end()) {
    resultValue = it->second;
  } else {
    resultValue = createPointerFromIndex(*this, getLoc(), resultIndex);
    ptrCache[resultIndex] = resultValue;
    registerResultMap.try_emplace(key, resultValue);
  }

  // Update result count
  if (std::cmp_greater_equal(resultIndex, metadata_.numResults)) {
    metadata_.numResults = static_cast<size_t>(resultIndex) + 1;
  }

  // Create mz call
  const auto mzSignature =
      LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  auto mzDecl =
      getOrCreateFunctionDeclaration(*this, module, QIR_MEASURE, mzSignature);
  LLVM::CallOp::create(*this, mzDecl, ValueRange{qubit, resultValue});

  return resultValue;
}

QIRProgramBuilder& QIRProgramBuilder::measure(Value qubit, const Bit& bit) {
  checkFinalized();

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert in measurements block (before branch)
  setInsertionPoint(measurementsBlock->getTerminator());

  // Check if we already have a result pointer for this register slot
  const auto& registerName = bit.registerName;
  const auto registerIndex = bit.registerIndex;
  const auto key = std::make_pair(registerName, registerIndex);
  if (!registerResultMap.contains(key)) {
    llvm::reportFatalInternalError("Result pointer not found");
  }
  const auto resultValue = registerResultMap.at(key);

  // Create mz call
  const auto mzSignature =
      LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  auto mzDecl =
      getOrCreateFunctionDeclaration(*this, module, QIR_MEASURE, mzSignature);
  LLVM::CallOp::create(*this, mzDecl, ValueRange{qubit, resultValue});

  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::reset(Value qubit) {
  checkFinalized();

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert in measurements block (before branch)
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
      parameterOperand =
          LLVM::ConstantOp::create(*this,
                                   getF64FloatAttr(std::get<double>(parameter)))
              .getResult();
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

void QIRProgramBuilder::generateOutputRecording() {
  if (registerResultMap.empty()) {
    return; // No measurements to record
  }

  // Save current insertion point
  const InsertionGuard guard(*this);

  // Insert in output block (before return)
  setInsertionPoint(outputBlock->getTerminator());

  // Group measurements by register
  llvm::StringMap<SmallVector<std::pair<int64_t, Value>>> registerGroups;
  for (const auto& [key, resultPtr] : registerResultMap) {
    const auto& [regName, regIdx] = key;
    registerGroups[regName].emplace_back(regIdx, resultPtr);
  }

  // Sort registers by name for deterministic output
  SmallVector<std::pair<std::string, SmallVector<std::pair<int64_t, Value>>>>
      sortedRegisters;
  for (auto& [name, measurements] : registerGroups) {
    sortedRegisters.emplace_back(name, std::move(measurements));
  }
  sort(sortedRegisters,
       [](const auto& a, const auto& b) { return a.first < b.first; });

  // Create array_record_output call
  const auto arrayRecordSig =
      LLVM::LLVMFunctionType::get(voidType, {getI64Type(), ptrType});
  const auto arrayRecordDecl = getOrCreateFunctionDeclaration(
      *this, module, QIR_ARRAY_RECORD_OUTPUT, arrayRecordSig);

  // Create result_record_output calls for each measurement
  const auto resultRecordSig =
      LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  const auto resultRecordDecl = getOrCreateFunctionDeclaration(
      *this, module, QIR_RECORD_OUTPUT, resultRecordSig);

  // Generate output recording for each register
  for (auto& [registerName, measurements] : sortedRegisters) {
    // Sort measurements by register index
    sort(measurements,
         [](const auto& a, const auto& b) { return a.first < b.first; });

    const auto arraySize = measurements.size();
    auto arrayLabelOp = createResultLabel(*this, module, registerName);
    auto arraySizeConst = create<LLVM::ConstantOp>(
        getI64IntegerAttr(static_cast<int64_t>(arraySize)));

    LLVM::CallOp::create(
        *this, arrayRecordDecl,
        ValueRange{arraySizeConst.getResult(), arrayLabelOp.getResult()});

    for (const auto& [regIdx, resultPtr] : measurements) {
      // Create label for result: "{registerName}{regIdx}r"
      const std::string resultLabel =
          registerName + std::to_string(regIdx) + "r";
      auto resultLabelOp = createResultLabel(*this, module, resultLabel);

      LLVM::CallOp::create(*this, resultRecordDecl,
                           ValueRange{resultPtr, resultLabelOp.getResult()});
    }
  }
}

OwningOpRef<ModuleOp> QIRProgramBuilder::finalize() {
  checkFinalized();

  // Generate output recording in the output block
  generateOutputRecording();

  setQIRAttributes(mainFunc, metadata_);

  isFinalized = true;

  return module;
}

} // namespace mlir::qir
