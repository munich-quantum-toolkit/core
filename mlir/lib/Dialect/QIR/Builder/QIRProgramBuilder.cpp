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

void QIRProgramBuilder::createOneTargetZeroParameter(const Value qubit,
                                                     StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Create call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{qubit});
}

void QIRProgramBuilder::createControlledOneTargetZeroParameter(
    const Value control, const Value target, StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Create call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      {LLVM::LLVMPointerType::get(builder.getContext()),
       LLVM::LLVMPointerType::get(builder.getContext())});
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{control, target});
}

// XOp

QIRProgramBuilder& QIRProgramBuilder::x(const Value qubit) {
  createOneTargetZeroParameter(qubit, QIR_X);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::cx(const Value control,
                                         const Value target) {
  createControlledOneTargetZeroParameter(control, target, QIR_CX);
  return *this;
}

// SOp

QIRProgramBuilder& QIRProgramBuilder::s(const Value qubit) {
  createOneTargetZeroParameter(qubit, QIR_S);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::cs(const Value control,
                                         const Value target) {
  createControlledOneTargetZeroParameter(control, target, QIR_CS);
  return *this;
}

// SdgOp

QIRProgramBuilder& QIRProgramBuilder::sdg(const Value qubit) {
  createOneTargetZeroParameter(qubit, QIR_SDG);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::csdg(const Value control,
                                           const Value target) {
  createControlledOneTargetZeroParameter(control, target, QIR_CSDG);
  return *this;
}

// RXOp

QIRProgramBuilder&
QIRProgramBuilder::rx(const std::variant<double, Value>& theta,
                      const Value qubit) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value thetaOperand;
  if (std::holds_alternative<double>(theta)) {
    thetaOperand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(theta)))
            .getResult();
  } else {
    thetaOperand = std::get<Value>(theta);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Create rx call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      {LLVM::LLVMPointerType::get(builder.getContext()),
       Float64Type::get(builder.getContext())});
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_RX, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{qubit, thetaOperand});

  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::crx(const std::variant<double, Value>& theta,
                       const Value control, const Value target) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value thetaOperand;
  if (std::holds_alternative<double>(theta)) {
    thetaOperand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(theta)))
            .getResult();
  } else {
    thetaOperand = std::get<Value>(theta);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Create crx call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      {LLVM::LLVMPointerType::get(builder.getContext()),
       LLVM::LLVMPointerType::get(builder.getContext()),
       Float64Type::get(builder.getContext())});
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_CRX, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl,
                               ValueRange{control, target, thetaOperand});

  return *this;
}

// U2Op

QIRProgramBuilder&
QIRProgramBuilder::u2(const std::variant<double, Value>& phi,
                      const std::variant<double, Value>& lambda,
                      const Value qubit) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value phiOperand;
  if (std::holds_alternative<double>(phi)) {
    phiOperand = builder
                     .create<LLVM::ConstantOp>(
                         loc, builder.getF64FloatAttr(std::get<double>(phi)))
                     .getResult();
  } else {
    phiOperand = std::get<Value>(phi);
  }

  Value lambdaOperand;
  if (std::holds_alternative<double>(lambda)) {
    lambdaOperand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(lambda)))
            .getResult();
  } else {
    lambdaOperand = std::get<Value>(lambda);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Create u2 call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      {LLVM::LLVMPointerType::get(builder.getContext()),
       Float64Type::get(builder.getContext()),
       Float64Type::get(builder.getContext())});
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_U2, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl,
                               ValueRange{qubit, phiOperand, lambdaOperand});

  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::cu2(const std::variant<double, Value>& phi,
                       const std::variant<double, Value>& lambda,
                       const Value control, const Value target) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value phiOperand;
  if (std::holds_alternative<double>(phi)) {
    phiOperand = builder
                     .create<LLVM::ConstantOp>(
                         loc, builder.getF64FloatAttr(std::get<double>(phi)))
                     .getResult();
  } else {
    phiOperand = std::get<Value>(phi);
  }

  Value lambdaOperand;
  if (std::holds_alternative<double>(lambda)) {
    lambdaOperand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(lambda)))
            .getResult();
  } else {
    lambdaOperand = std::get<Value>(lambda);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Create cu2 call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      {LLVM::LLVMPointerType::get(builder.getContext()),
       LLVM::LLVMPointerType::get(builder.getContext()),
       Float64Type::get(builder.getContext()),
       Float64Type::get(builder.getContext())});
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_CU2, qirSignature);
  builder.create<LLVM::CallOp>(
      loc, fnDecl, ValueRange{control, target, phiOperand, lambdaOperand});

  return *this;
}

// SWAPOp

QIRProgramBuilder& QIRProgramBuilder::swap(const Value qubit0,
                                           const Value qubit1) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Create swap call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      {LLVM::LLVMPointerType::get(builder.getContext()),
       LLVM::LLVMPointerType::get(builder.getContext())});
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_SWAP, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{qubit0, qubit1});

  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::cswap(const Value control,
                                            const Value target0,
                                            const Value target1) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Create cswap call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      {LLVM::LLVMPointerType::get(builder.getContext()),
       LLVM::LLVMPointerType::get(builder.getContext()),
       LLVM::LLVMPointerType::get(builder.getContext())});
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_CSWAP, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl,
                               ValueRange{control, target0, target1});

  return *this;
}

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
