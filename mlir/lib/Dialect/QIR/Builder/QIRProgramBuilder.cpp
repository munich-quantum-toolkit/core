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

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <string>
#include <utility>

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

Value QIRProgramBuilder::allocQubit() {
  // Create function signature: () -> ptr
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMPointerType::get(builder.getContext()), {});

  auto fnDecl = getOrCreateFunctionDeclaration(
      builder, module, QIR_QUBIT_ALLOCATE, qirSignature);

  // Call qubit_allocate
  auto callOp = builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{});
  const auto qubit = callOp.getResult();

  // Track for automatic deallocation
  allocatedQubits.insert(qubit);

  // Update counts
  metadata_.numQubits++;
  metadata_.useDynamicQubit = true;

  return qubit;
}

Value QIRProgramBuilder::staticQubit(const int64_t index) {
  // Check cache
  if (staticQubitCache.contains(index)) {
    return staticQubitCache.at(index);
  }

  // Use common utility function to create pointer from index
  const auto qubit = createPointerFromIndex(builder, loc, index);

  // Cache for reuse
  staticQubitCache[index] = qubit;

  // Update qubit count
  if (std::cmp_greater_equal(index, metadata_.numQubits)) {
    metadata_.numQubits = static_cast<size_t>(index) + 1;
  }

  return qubit;
}

llvm::SmallVector<Value>
QIRProgramBuilder::allocQubitRegister(const int64_t size) {
  llvm::SmallVector<Value> qubits;
  qubits.reserve(size);

  for (int64_t i = 0; i < size; ++i) {
    qubits.push_back(allocQubit());
  }

  return qubits;
}

Value QIRProgramBuilder::measure(const Value qubit, const int64_t resultIndex) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  // Get or create result pointer (separate from qubit pointers)
  Value resultValue = nullptr;
  if (resultPointerCache.contains(resultIndex)) {
    resultValue = resultPointerCache.at(resultIndex);
  } else {
    resultValue = createPointerFromIndex(builder, loc, resultIndex);
    resultPointerCache[resultIndex] = resultValue;
    registerResultMap.insert({{"c", resultIndex}, resultValue});
    metadata_.numResults++;
  }

  // Create mz call
  const auto mzSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
  auto mzDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_MEASURE, mzSignature);
  builder.create<LLVM::CallOp>(loc, mzDecl, ValueRange{qubit, resultValue});

  return resultValue;
}

QIRProgramBuilder& QIRProgramBuilder::measure(const Value qubit,
                                              const StringRef registerName,
                                              const int64_t registerIndex) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  // Check if we already have a result pointer for this register slot
  const auto key = std::make_pair(registerName, registerIndex);

  Value resultValue = nullptr;
  if (const auto it = registerResultMap.find(key);
      it != registerResultMap.end()) {
    resultValue = it->second;
  } else {
    resultValue = createPointerFromIndex(
        builder, loc, static_cast<int64_t>(metadata_.numResults));
    // Cache for reuse
    resultPointerCache[metadata_.numResults] = resultValue;
    registerResultMap.insert({key, resultValue});
    metadata_.numResults++;
  }

  // Create mz call
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

QIRProgramBuilder& QIRProgramBuilder::x(const Value qubit) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Create x call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_X, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{qubit});

  return *this;
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

QIRProgramBuilder& QIRProgramBuilder::dealloc(const Value qubit) {
  allocatedQubits.erase(qubit);

  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  // Create release call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto fnDecl = getOrCreateFunctionDeclaration(builder, module,
                                               QIR_QUBIT_RELEASE, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{qubit});

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
      std::pair<llvm::StringRef, llvm::SmallVector<std::pair<int64_t, Value>>>>
      sortedRegisters;
  for (auto& [name, measurements] : registerGroups) {
    sortedRegisters.emplace_back(name, std::move(measurements));
  }
  llvm::sort(sortedRegisters,
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
    llvm::sort(measurements,
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
  for (const Value qubit : allocatedQubits) {
    dealloc(qubit);
  }
  allocatedQubits.clear();

  // Generate output recording in the output block
  generateOutputRecording();

  setQIRAttributes(mainFunc, metadata_);

  return module;
}

} // namespace mlir::qir
