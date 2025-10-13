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
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <string>

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

  // Create main function: () -> void
  auto funcType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {});
  mainFunc = builder.create<LLVM::LLVMFuncOp>(loc, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = StringAttr::get(builder.getContext(), "entry_point");
  mainFunc->setAttr("passthrough",
                    ArrayAttr::get(builder.getContext(), {entryPointAttr}));

  // Create the 4-block structure for QIR base profile
  entryBlock = mainFunc.addEntryBlock(builder);
  mainBlock = mainFunc.addBlock();
  irreversibleBlock = mainFunc.addBlock();
  endBlock = mainFunc.addBlock();

  // Add unconditional branches between blocks
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<LLVM::BrOp>(loc, mainBlock);

  builder.setInsertionPointToEnd(mainBlock);
  builder.create<LLVM::BrOp>(loc, irreversibleBlock);

  builder.setInsertionPointToEnd(irreversibleBlock);
  builder.create<LLVM::BrOp>(loc, endBlock);

  builder.setInsertionPointToEnd(endBlock);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{});

  // Add QIR initialization call in entry block
  builder.setInsertionPointToStart(entryBlock);
  auto zeroOp = builder.create<LLVM::ZeroOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()));

  // Move to before the branch
  builder.setInsertionPoint(entryBlock->getTerminator());

  const auto initType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto initFunc =
      getOrCreateFunctionDeclaration(builder, module, QIR_INITIALIZE, initType);
  builder.create<LLVM::CallOp>(loc, initFunc, ValueRange{zeroOp.getResult()});

  // Set insertion point to main block for user operations
  builder.setInsertionPointToStart(mainBlock);
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
  if (staticQubitCache.contains(static_cast<size_t>(index))) {
    return staticQubitCache.at(static_cast<size_t>(index));
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert at start of entry block (after initialize but before branch)
  builder.setInsertionPoint(entryBlock->getTerminator());

  // Use common utility function to create pointer from index
  const auto qubit = createPointerFromIndex(builder, loc, index);

  // Cache for reuse
  staticQubitCache[static_cast<size_t>(index)] = qubit;

  // Update qubit count
  if (static_cast<size_t>(index) >= metadata_.numQubits) {
    metadata_.numQubits = static_cast<size_t>(index) + 1;
  }

  return qubit;
}

SmallVector<Value> QIRProgramBuilder::allocQubitRegister(const int64_t size) {
  SmallVector<Value> qubits;
  qubits.reserve(static_cast<size_t>(size));

  for (int64_t i = 0; i < size; ++i) {
    qubits.push_back(allocQubit());
  }

  return qubits;
}

Value QIRProgramBuilder::measure(const Value qubit, const size_t resultIndex) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in irreversible block (before branch)
  builder.setInsertionPoint(irreversibleBlock->getTerminator());

  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  // Get or create result pointer (separate from qubit pointers)
  Value resultValue = nullptr;
  if (resultPointerCache.contains(resultIndex)) {
    resultValue = resultPointerCache.at(resultIndex);
  } else {
    // Create at start of entry block using common utility
    builder.setInsertionPoint(entryBlock->getTerminator());
    resultValue =
        createPointerFromIndex(builder, loc, static_cast<int64_t>(resultIndex));
    resultPointerCache[resultIndex] = resultValue;

    // Restore to irreversible block
    builder.setInsertionPoint(irreversibleBlock->getTerminator());
  }

  // Create mz call
  const auto mzSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
  auto mzDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_MEASURE, mzSignature);
  builder.create<LLVM::CallOp>(loc, mzDecl, ValueRange{qubit, resultValue});

  // Get or create result label
  LLVM::AddressOfOp labelOp;
  if (resultLabelCache.contains(resultIndex)) {
    labelOp = resultLabelCache.at(resultIndex);
  } else {
    // Use common utility function to create result label
    labelOp =
        createResultLabel(builder, module, "r" + std::to_string(resultIndex));
    resultLabelCache.try_emplace(resultIndex, labelOp);
    metadata_.numResults++;
  }

  // Create record_output call
  const auto recordSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
  auto recordDecl = getOrCreateFunctionDeclaration(
      builder, module, QIR_RECORD_OUTPUT, recordSignature);
  builder.create<LLVM::CallOp>(loc, recordDecl,
                               ValueRange{resultValue, labelOp.getResult()});

  return resultValue;
}

QIRProgramBuilder& QIRProgramBuilder::reset(const Value qubit) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in irreversible block (before branch)
  builder.setInsertionPoint(irreversibleBlock->getTerminator());

  // Create reset call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_RESET, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{qubit});

  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::dealloc(const Value qubit) {
  allocatedQubits.erase(qubit);

  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in irreversible block (before branch)
  builder.setInsertionPoint(irreversibleBlock->getTerminator());

  // Create release call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto fnDecl = getOrCreateFunctionDeclaration(builder, module,
                                               QIR_QUBIT_RELEASE, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{qubit});

  return *this;
}

OwningOpRef<ModuleOp> QIRProgramBuilder::finalize() {
  for (const Value qubit : allocatedQubits) {
    dealloc(qubit);
  }
  allocatedQubits.clear();

  setQIRAttributes(mainFunc, metadata_);

  return module;
}

} // namespace mlir::qir
