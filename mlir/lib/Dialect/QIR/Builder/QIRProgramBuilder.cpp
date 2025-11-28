/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"

#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <cassert>
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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <string>
#include <utility>
#include <variant>

namespace mlir::qir {

QIRProgramBuilder::QIRProgramBuilder(MLIRContext* context)
    : builder(context),
      module(builder.create<ModuleOp>(UnknownLoc::get(context))),
      loc(UnknownLoc::get(context)) {}

void QIRProgramBuilder::initialize() {
  // Ensure LLVM dialect is loaded
  builder.getContext()->loadDialect<LLVM::LLVMDialect>();

  // Set insertion point to the module body
  builder.setInsertionPointToStart(module.getBody());

  // Create main function: () -> i64
  auto funcType = LLVM::LLVMFunctionType::get(builder.getI64Type(), {});
  mainFunc = builder.create<LLVM::LLVMFuncOp>(loc, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = StringAttr::get(builder.getContext(), "entry_point");
  mainFunc->setAttr("passthrough",
                    ArrayAttr::get(builder.getContext(), {entryPointAttr}));

  // Create the 4-block structure for QIR Base Profile
  entryBlock = mainFunc.addEntryBlock(builder);
  bodyBlock = mainFunc.addBlock();
  measurementsBlock = mainFunc.addBlock();
  outputBlock = mainFunc.addBlock();

  // Create exit code constant in entry block (where constants belong) and add
  // QIR initialization call in entry block (after exit code constant)
  builder.setInsertionPointToStart(entryBlock);
  auto zeroOp = builder.create<LLVM::ZeroOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()));
  exitCode =
      builder.create<LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  const auto initType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto initFunc =
      getOrCreateFunctionDeclaration(builder, module, QIR_INITIALIZE, initType);
  builder.create<LLVM::CallOp>(loc, initFunc, ValueRange{zeroOp.getResult()});

  // Add unconditional branches between blocks
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<LLVM::BrOp>(loc, bodyBlock);

  builder.setInsertionPointToEnd(bodyBlock);
  builder.create<LLVM::BrOp>(loc, measurementsBlock);

  builder.setInsertionPointToEnd(measurementsBlock);
  builder.create<LLVM::BrOp>(loc, outputBlock);

  // Return the exit code (success) in output block
  builder.setInsertionPointToEnd(outputBlock);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{exitCode.getResult()});

  // Set insertion point to body block for user operations
  builder.setInsertionPointToStart(bodyBlock);
}

Value QIRProgramBuilder::staticQubit(const int64_t index) {
  // Check cache
  Value val{};
  if (const auto it = ptrCache.find(index); it != ptrCache.end()) {
    val = it->second;
  } else {
    val = createPointerFromIndex(builder, loc, index);
    // Cache for reuse
    ptrCache[index] = val;
  }

  // Update qubit count
  if (std::cmp_greater_equal(index, metadata_.numQubits)) {
    metadata_.numQubits = static_cast<size_t>(index) + 1;
  }

  return val;
}

llvm::SmallVector<Value>
QIRProgramBuilder::allocQubitRegister(const int64_t size) {
  llvm::SmallVector<Value> qubits;
  qubits.reserve(size);

  for (int64_t i = 0; i < size; ++i) {
    qubits.push_back(staticQubit(static_cast<int64_t>(metadata_.numQubits)));
  }

  return qubits;
}

QIRProgramBuilder::ClassicalRegister&
QIRProgramBuilder::allocClassicalBitRegister(const int64_t size,
                                             StringRef name) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  const auto numResults = static_cast<int64_t>(metadata_.numResults);
  auto& reg = allocatedClassicalRegisters.emplace_back(name, size);
  for (int64_t i = 0; i < size; ++i) {
    Value val{};
    if (const auto it = ptrCache.find(numResults + i); it != ptrCache.end()) {
      val = it->second;
    } else {
      val = createPointerFromIndex(builder, loc, numResults + i);
      // Cache for reuse
      ptrCache[numResults + i] = val;
    }
    registerResultMap.insert({{name, i}, val});
  }
  metadata_.numResults += size;
  return reg;
}

Value QIRProgramBuilder::measure(const Value qubit, const int64_t resultIndex) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  const auto key = std::make_pair("c", resultIndex);
  if (const auto it = registerResultMap.find(key);
      it != registerResultMap.end()) {
    return it->second;
  }

  Value resultValue{};
  if (const auto it = ptrCache.find(resultIndex); it != ptrCache.end()) {
    resultValue = it->second;
  } else {
    resultValue = createPointerFromIndex(builder, loc, resultIndex);
    ptrCache[resultIndex] = resultValue;
    registerResultMap.insert({key, resultValue});
  }

  // Update result count
  if (std::cmp_greater_equal(resultIndex, metadata_.numResults)) {
    metadata_.numResults = static_cast<size_t>(resultIndex) + 1;
  }

  // Create mz call
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  const auto mzSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
  auto mzDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_MEASURE, mzSignature);
  builder.create<LLVM::CallOp>(loc, mzDecl, ValueRange{qubit, resultValue});

  return resultValue;
}

QIRProgramBuilder& QIRProgramBuilder::measure(const Value qubit,
                                              const Bit& bit) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  // Check if we already have a result pointer for this register slot
  const auto& registerName = bit.registerName;
  const auto registerIndex = bit.registerIndex;
  const auto key = std::make_pair(registerName, registerIndex);
  assert(registerResultMap.contains(key));
  const auto resultValue = registerResultMap.at(key);

  // Create mz call
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  const auto mzSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
  auto mzDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_MEASURE, mzSignature);
  builder.create<LLVM::CallOp>(loc, mzDecl, ValueRange{qubit, resultValue});

  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::reset(const Value qubit) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  // Create reset call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_RESET, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{qubit});

  return *this;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// Helper methods

void QIRProgramBuilder::createOneTargetZeroParameter(const ValueRange controls,
                                                     const Value target,
                                                     StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 1);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 1);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

void QIRProgramBuilder::createOneTargetOneParameter(
    const std::variant<double, Value>& parameter, const ValueRange controls,
    const Value target, StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value parameterOperand;
  if (std::holds_alternative<double>(parameter)) {
    parameterOperand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter)))
            .getResult();
  } else {
    parameterOperand = std::get<Value>(parameter);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 2);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);
  // Add parameter type
  argumentTypes.push_back(Float64Type::get(builder.getContext()));

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 2);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target);
  operands.push_back(parameterOperand);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

void QIRProgramBuilder::createOneTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const ValueRange controls,
    const Value target, StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value parameter1Operand;
  if (std::holds_alternative<double>(parameter1)) {
    parameter1Operand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter1)))
            .getResult();
  } else {
    parameter1Operand = std::get<Value>(parameter1);
  }

  Value parameter2Operand;
  if (std::holds_alternative<double>(parameter2)) {
    parameter2Operand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter2)))
            .getResult();
  } else {
    parameter2Operand = std::get<Value>(parameter2);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 3);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  const auto floatType = Float64Type::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);
  // Add parameter types
  argumentTypes.push_back(floatType);
  argumentTypes.push_back(floatType);

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 3);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target);
  operands.push_back(parameter1Operand);
  operands.push_back(parameter2Operand);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

void QIRProgramBuilder::createOneTargetThreeParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2,
    const std::variant<double, Value>& parameter3, const ValueRange controls,
    const Value target, StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value parameter1Operand;
  if (std::holds_alternative<double>(parameter1)) {
    parameter1Operand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter1)))
            .getResult();
  } else {
    parameter1Operand = std::get<Value>(parameter1);
  }

  Value parameter2Operand;
  if (std::holds_alternative<double>(parameter2)) {
    parameter2Operand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter2)))
            .getResult();
  } else {
    parameter2Operand = std::get<Value>(parameter2);
  }

  Value parameter3Operand;
  if (std::holds_alternative<double>(parameter3)) {
    parameter3Operand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter3)))
            .getResult();
  } else {
    parameter3Operand = std::get<Value>(parameter3);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 4);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  const auto floatType = Float64Type::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);
  // Add parameter types
  argumentTypes.push_back(floatType);
  argumentTypes.push_back(floatType);
  argumentTypes.push_back(floatType);

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 4);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target);
  operands.push_back(parameter1Operand);
  operands.push_back(parameter2Operand);
  operands.push_back(parameter3Operand);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

void QIRProgramBuilder::createTwoTargetZeroParameter(const ValueRange controls,
                                                     const Value target0,
                                                     const Value target1,
                                                     StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 2);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointers
  argumentTypes.push_back(ptrType);
  argumentTypes.push_back(ptrType);

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 2);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target0);
  operands.push_back(target1);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

void QIRProgramBuilder::createTwoTargetOneParameter(
    const std::variant<double, Value>& parameter, const ValueRange controls,
    const Value target0, const Value target1, StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value parameterOperand;
  if (std::holds_alternative<double>(parameter)) {
    parameterOperand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter)))
            .getResult();
  } else {
    parameterOperand = std::get<Value>(parameter);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 3);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointers
  argumentTypes.push_back(ptrType);
  argumentTypes.push_back(ptrType);
  // Add parameter type
  argumentTypes.push_back(Float64Type::get(builder.getContext()));

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 3);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target0);
  operands.push_back(target1);
  operands.push_back(parameterOperand);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

void QIRProgramBuilder::createTwoTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const ValueRange controls,
    const Value target0, const Value target1, StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value parameter1Operand;
  if (std::holds_alternative<double>(parameter1)) {
    parameter1Operand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter1)))
            .getResult();
  } else {
    parameter1Operand = std::get<Value>(parameter1);
  }

  Value parameter2Operand;
  if (std::holds_alternative<double>(parameter2)) {
    parameter2Operand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter2)))
            .getResult();
  } else {
    parameter2Operand = std::get<Value>(parameter2);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 4);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  const auto floatType = Float64Type::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointers
  argumentTypes.push_back(ptrType);
  argumentTypes.push_back(ptrType);
  // Add parameter types
  argumentTypes.push_back(floatType);
  argumentTypes.push_back(floatType);

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 4);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target0);
  operands.push_back(target1);
  operands.push_back(parameter1Operand);
  operands.push_back(parameter2Operand);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_NAME_BIG, OP_NAME_SMALL)           \
  QIRProgramBuilder& QIRProgramBuilder::OP_NAME_SMALL(const Value qubit) {     \
    createOneTargetZeroParameter({}, qubit, QIR_##OP_NAME_BIG);                \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(const Value control,  \
                                                         const Value target) { \
    createOneTargetZeroParameter({control}, target, QIR_C##OP_NAME_BIG);       \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const ValueRange controls, const Value target) {                         \
    StringRef fnName;                                                          \
    if (controls.size() == 1) {                                                \
      fnName = QIR_C##OP_NAME_BIG;                                             \
    } else if (controls.size() == 2) {                                         \
      fnName = QIR_CC##OP_NAME_BIG;                                            \
    } else if (controls.size() == 3) {                                         \
      fnName = QIR_CCC##OP_NAME_BIG;                                           \
    } else {                                                                   \
      llvm::report_fatal_error(                                                \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
    createOneTargetZeroParameter(controls, target, fnName);                    \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_ZERO_PARAMETER(ID, id)
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
      const std::variant<double, Value>&(PARAM), const Value qubit) {          \
    createOneTargetOneParameter(PARAM, {}, qubit, QIR_##OP_NAME_BIG);          \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM), const Value control,          \
      const Value target) {                                                    \
    createOneTargetOneParameter(PARAM, {control}, target, QIR_C##OP_NAME_BIG); \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM), const ValueRange controls,    \
      const Value target) {                                                    \
    StringRef fnName;                                                          \
    if (controls.size() == 1) {                                                \
      fnName = QIR_C##OP_NAME_BIG;                                             \
    } else if (controls.size() == 2) {                                         \
      fnName = QIR_CC##OP_NAME_BIG;                                            \
    } else if (controls.size() == 3) {                                         \
      fnName = QIR_CCC##OP_NAME_BIG;                                           \
    } else {                                                                   \
      llvm::report_fatal_error(                                                \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
    createOneTargetOneParameter(PARAM, controls, target, fnName);              \
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
      const std::variant<double, Value>&(PARAM2), const Value qubit) {         \
    createOneTargetTwoParameter(PARAM1, PARAM2, {}, qubit, QIR_##OP_NAME_BIG); \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), const Value control,         \
      const Value target) {                                                    \
    createOneTargetTwoParameter(PARAM1, PARAM2, {control}, target,             \
                                QIR_C##OP_NAME_BIG);                           \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), const ValueRange controls,   \
      const Value target) {                                                    \
    StringRef fnName;                                                          \
    if (controls.size() == 1) {                                                \
      fnName = QIR_C##OP_NAME_BIG;                                             \
    } else if (controls.size() == 2) {                                         \
      fnName = QIR_CC##OP_NAME_BIG;                                            \
    } else if (controls.size() == 3) {                                         \
      fnName = QIR_CCC##OP_NAME_BIG;                                           \
    } else {                                                                   \
      llvm::report_fatal_error(                                                \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
    createOneTargetTwoParameter(PARAM1, PARAM2, controls, target, fnName);     \
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
      const std::variant<double, Value>&(PARAM3), const Value qubit) {         \
    createOneTargetThreeParameter(PARAM1, PARAM2, PARAM3, {}, qubit,           \
                                  QIR_##OP_NAME_BIG);                          \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), const Value control,         \
      const Value target) {                                                    \
    createOneTargetThreeParameter(PARAM1, PARAM2, PARAM3, {control}, target,   \
                                  QIR_C##OP_NAME_BIG);                         \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), const ValueRange controls,   \
      const Value target) {                                                    \
    StringRef fnName;                                                          \
    if (controls.size() == 1) {                                                \
      fnName = QIR_C##OP_NAME_BIG;                                             \
    } else if (controls.size() == 2) {                                         \
      fnName = QIR_CC##OP_NAME_BIG;                                            \
    } else if (controls.size() == 3) {                                         \
      fnName = QIR_CCC##OP_NAME_BIG;                                           \
    } else {                                                                   \
      llvm::report_fatal_error(                                                \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
    createOneTargetThreeParameter(PARAM1, PARAM2, PARAM3, controls, target,    \
                                  fnName);                                     \
    return *this;                                                              \
  }

DEFINE_ONE_TARGET_THREE_PARAMETER(U, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_NAME_BIG, OP_NAME_SMALL)           \
  QIRProgramBuilder& QIRProgramBuilder::OP_NAME_SMALL(const Value target0,     \
                                                      const Value target1) {   \
    createTwoTargetZeroParameter({}, target0, target1, QIR_##OP_NAME_BIG);     \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const Value control, const Value target0, const Value target1) {         \
    createTwoTargetZeroParameter({control}, target0, target1,                  \
                                 QIR_C##OP_NAME_BIG);                          \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const ValueRange controls, const Value target0, const Value target1) {   \
    StringRef fnName;                                                          \
    if (controls.size() == 1) {                                                \
      fnName = QIR_C##OP_NAME_BIG;                                             \
    } else if (controls.size() == 2) {                                         \
      fnName = QIR_CC##OP_NAME_BIG;                                            \
    } else if (controls.size() == 3) {                                         \
      fnName = QIR_CCC##OP_NAME_BIG;                                           \
    } else {                                                                   \
      llvm::report_fatal_error(                                                \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
    createTwoTargetZeroParameter(controls, target0, target1, fnName);          \
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
      const std::variant<double, Value>&(PARAM), const Value target0,          \
      const Value target1) {                                                   \
    createTwoTargetOneParameter(PARAM, {}, target0, target1,                   \
                                QIR_##OP_NAME_BIG);                            \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM), const Value control,          \
      const Value target0, const Value target1) {                              \
    createTwoTargetOneParameter(PARAM, {control}, target0, target1,            \
                                QIR_C##OP_NAME_BIG);                           \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM), const ValueRange controls,    \
      const Value target0, const Value target1) {                              \
    StringRef fnName;                                                          \
    if (controls.size() == 1) {                                                \
      fnName = QIR_C##OP_NAME_BIG;                                             \
    } else if (controls.size() == 2) {                                         \
      fnName = QIR_CC##OP_NAME_BIG;                                            \
    } else if (controls.size() == 3) {                                         \
      fnName = QIR_CCC##OP_NAME_BIG;                                           \
    } else {                                                                   \
      llvm::report_fatal_error(                                                \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
    createTwoTargetOneParameter(PARAM, controls, target0, target1, fnName);    \
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
      const std::variant<double, Value>&(PARAM2), const Value target0,         \
      const Value target1) {                                                   \
    createTwoTargetTwoParameter(PARAM1, PARAM2, {}, target0, target1,          \
                                QIR_##OP_NAME_BIG);                            \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::c##OP_NAME_SMALL(                      \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), const Value control,         \
      const Value target0, const Value target1) {                              \
    createTwoTargetTwoParameter(PARAM1, PARAM2, {control}, target0, target1,   \
                                QIR_C##OP_NAME_BIG);                           \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  QIRProgramBuilder& QIRProgramBuilder::mc##OP_NAME_SMALL(                     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), const ValueRange controls,   \
      const Value target0, const Value target1) {                              \
    StringRef fnName;                                                          \
    if (controls.size() == 1) {                                                \
      fnName = QIR_C##OP_NAME_BIG;                                             \
    } else if (controls.size() == 2) {                                         \
      fnName = QIR_CC##OP_NAME_BIG;                                            \
    } else if (controls.size() == 3) {                                         \
      fnName = QIR_CCC##OP_NAME_BIG;                                           \
    } else {                                                                   \
      llvm::report_fatal_error(                                                \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
    createTwoTargetTwoParameter(PARAM1, PARAM2, controls, target0, target1,    \
                                fnName);                                       \
    return *this;                                                              \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPLUSYY, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMINUSYY, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

void QIRProgramBuilder::generateOutputRecording() {
  if (registerResultMap.empty()) {
    return; // No measurements to record
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in output block (before return)
  builder.setInsertionPoint(outputBlock->getTerminator());

  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  // Group measurements by register
  llvm::StringMap<llvm::SmallVector<std::pair<int64_t, Value>>> registerGroups;
  for (const auto& [key, resultPtr] : registerResultMap) {
    const auto& [regName, regIdx] = key;
    registerGroups[regName].emplace_back(regIdx, resultPtr);
  }

  // Sort registers by name for deterministic output
  llvm::SmallVector<
      std::pair<StringRef, llvm::SmallVector<std::pair<int64_t, Value>>>>
      sortedRegisters;
  for (auto& [name, measurements] : registerGroups) {
    sortedRegisters.emplace_back(name, std::move(measurements));
  }
  sort(sortedRegisters,
       [](const auto& a, const auto& b) { return a.first < b.first; });

  // Create array_record_output call
  const auto arrayRecordSig =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(builder.getContext()),
                                  {builder.getI64Type(), ptrType});
  const auto arrayRecordDecl = getOrCreateFunctionDeclaration(
      builder, module, QIR_ARRAY_RECORD_OUTPUT, arrayRecordSig);

  // Create result_record_output calls for each measurement
  const auto resultRecordSig = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
  const auto resultRecordDecl = getOrCreateFunctionDeclaration(
      builder, module, QIR_RECORD_OUTPUT, resultRecordSig);

  // Generate output recording for each register
  for (auto& [registerName, measurements] : sortedRegisters) {
    // Sort measurements by register index
    sort(measurements,
         [](const auto& a, const auto& b) { return a.first < b.first; });

    const auto arraySize = measurements.size();
    auto arrayLabelOp = createResultLabel(builder, module, registerName);
    auto arraySizeConst = builder.create<LLVM::ConstantOp>(
        loc, builder.getI64IntegerAttr(static_cast<int64_t>(arraySize)));

    builder.create<LLVM::CallOp>(
        loc, arrayRecordDecl,
        ValueRange{arraySizeConst.getResult(), arrayLabelOp.getResult()});

    for (const auto [regIdx, resultPtr] : measurements) {
      // Create label for result: "{registerName}{regIdx}r"
      const std::string resultLabel =
          registerName.str() + std::to_string(regIdx) + "r";
      auto resultLabelOp = createResultLabel(builder, module, resultLabel);

      builder.create<LLVM::CallOp>(
          loc, resultRecordDecl,
          ValueRange{resultPtr, resultLabelOp.getResult()});
    }
  }
}

OwningOpRef<ModuleOp> QIRProgramBuilder::finalize() {
  // Generate output recording in the output block
  generateOutputRecording();

  setQIRAttributes(mainFunc, metadata_);

  return module;
}

} // namespace mlir::qir
