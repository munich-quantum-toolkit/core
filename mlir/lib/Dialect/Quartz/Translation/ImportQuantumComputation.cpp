/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/Translation/ImportQuantumComputation.h"

#include "ir/QuantumComputation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <ranges>
#include <utility>

namespace mlir {
using namespace mlir::quartz;

namespace {

struct QregInfo {
  const qc::QuantumRegister* qregPtr;
  mlir::Value qreg;
  llvm::SmallVector<mlir::Value> qubits;
};

using BitMemInfo = std::pair<mlir::Value, std::size_t>; // (memref, localIdx)
using BitIndexVec = llvm::SmallVector<BitMemInfo>;

/**
 * @brief Allocates a quantum register in the MLIR module.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param context The MLIR context in which types are created
 * @param numQubits The number of qubits to allocate in the register
 * @return mlir::Value The allocated quantum register value
 */
mlir::Value allocateQreg(mlir::OpBuilder& builder, mlir::MLIRContext* context,
                         const std::size_t numQubits) {
  const auto& qubitType = QubitType::get(context);
  auto memRefType =
      mlir::MemRefType::get({static_cast<int64_t>(numQubits)}, qubitType);
  auto memref = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(),
                                                      memRefType);
  return memref.getResult();
}

/**
 * @brief Extracts all qubits from a quantum register.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param qreg The quantum register from which to extract qubits
 * @param numQubits The number of qubits to extract
 * @return llvm::SmallVector<mlir::Value> Vector of extracted qubit values
 */
llvm::SmallVector<mlir::Value> extractQubits(mlir::OpBuilder& builder,
                                             mlir::Value qreg,
                                             const std::size_t numQubits) {
  llvm::SmallVector<mlir::Value> qubits;
  qubits.reserve(numQubits);

  for (std::size_t qubit = 0; qubit < numQubits; ++qubit) {
    auto index = builder.create<mlir::arith::ConstantIndexOp>(
        builder.getUnknownLoc(), qubit);
    qubits.emplace_back(
        builder
            .create<mlir::memref::LoadOp>(builder.getUnknownLoc(), qreg,
                                          mlir::ValueRange{index})
            .getResult());
  }

  return qubits;
}

/**
 * @brief Allocates quantum registers and extracts qubits.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param context The MLIR context in which types are created
 * @param quantumComputation The quantum computation to translate
 * @return llvm::SmallVector<QregInfo> Vector containing information about all
 * quantum registers
 */
llvm::SmallVector<QregInfo>
getQregs(mlir::OpBuilder& builder, mlir::MLIRContext* context,
         const qc::QuantumComputation& quantumComputation) {
  // Build list of pointers for sorting
  llvm::SmallVector<const qc::QuantumRegister*> qregPtrs;
  qregPtrs.reserve(quantumComputation.getQuantumRegisters().size() +
                   quantumComputation.getAncillaRegisters().size());
  for (const auto& qreg :
       quantumComputation.getQuantumRegisters() | std::views::values) {
    qregPtrs.emplace_back(&qreg);
  }
  for (const auto& qreg :
       quantumComputation.getAncillaRegisters() | std::views::values) {
    qregPtrs.emplace_back(&qreg);
  }

  // Sort by start index
  std::ranges::sort(
      qregPtrs, [](const qc::QuantumRegister* a, const qc::QuantumRegister* b) {
        return a->getStartIndex() < b->getStartIndex();
      });

  // Allocate quantum registers and extract qubits
  llvm::SmallVector<QregInfo> qregs;
  for (const auto* qregPtr : qregPtrs) {
    const auto qreg = allocateQreg(builder, context, qregPtr->getSize());
    auto qubits = extractQubits(builder, qreg, qregPtr->getSize());
    qregs.emplace_back(qregPtr, qreg, std::move(qubits));
  }

  return qregs;
}

/**
 * @brief Builds a mapping from global qubit index to extracted qubit value.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param context The MLIR context in which types are created
 * @param qregs Vector containing information about all quantum registers
 * @return llvm::SmallVector<mlir::Value> Sorted vector of qubit values
 */
llvm::SmallVector<mlir::Value>
getQubits(const qc::QuantumComputation& quantumComputation,
          llvm::SmallVector<QregInfo>& qregs) {
  llvm::SmallVector<mlir::Value> flatQubits;
  const auto maxPhys = quantumComputation.getHighestPhysicalQubitIndex();
  flatQubits.resize(static_cast<size_t>(maxPhys) + 1);
  for (const auto& qreg : qregs) {
    for (std::size_t i = 0; i < qreg.qregPtr->getSize(); ++i) {
      const auto globalIdx =
          static_cast<size_t>(qreg.qregPtr->getStartIndex() + i);
      flatQubits[globalIdx] = qreg.qubits[i];
    }
  }

  return flatQubits;
}

/**
 * @brief Deallocates the quantum register in the MLIR module.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param qreg The quantum register to deallocate
 */
void deallocateQreg(mlir::OpBuilder& builder, mlir::Value qreg) {
  builder.create<mlir::memref::DeallocOp>(builder.getUnknownLoc(), qreg);
}

/**
 * @brief Allocates a classical register in the MLIR module.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param numBits The number of bits to allocate in the register
 * @return mlir::Value The allocated classical register value
 */
mlir::Value allocateBits(mlir::OpBuilder& builder, int64_t numBits) {
  auto memRefType = mlir::MemRefType::get({numBits}, builder.getI1Type());
  auto memref = builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(),
                                                       memRefType);
  return memref.getResult();
}

/**
 * @brief Builds a mapping from global bit index to (memref, localIdx).
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param numBits The number of bits to allocate in the register
 * @return mlir::Value The allocated classical register value
 */
BitIndexVec getBitMap(mlir::OpBuilder& builder,
                      const qc::QuantumComputation& quantumComputation) {
  // Build list of pointers for sorting
  llvm::SmallVector<const qc::ClassicalRegister*> cregPtrs;
  cregPtrs.reserve(quantumComputation.getClassicalRegisters().size());
  for (const auto& [_, reg] : quantumComputation.getClassicalRegisters()) {
    cregPtrs.emplace_back(&reg);
  }

  // Sort by start index
  std::ranges::sort(cregPtrs, [](const qc::ClassicalRegister* a,
                                 const qc::ClassicalRegister* b) {
    return a->getStartIndex() < b->getStartIndex();
  });

  // Build mapping
  BitIndexVec bitMap;
  bitMap.resize(quantumComputation.getNcbits());
  for (const auto* reg : cregPtrs) {
    auto mem = allocateBits(builder, static_cast<int64_t>(reg->getSize()));
    for (std::size_t i = 0; i < reg->getSize(); ++i) {
      const auto globalIdx = static_cast<std::size_t>(reg->getStartIndex() + i);
      bitMap[globalIdx] = {mem, i};
    }
  }

  return bitMap;
}
} // namespace

/**
 * @brief Translates a QuantumComputation to an MLIR module with Quartz
 * operations.
 *
 * This function takes a quantum computation and translates it into an MLIR
 * module containing Quartz dialect operations. It creates a main function that
 * contains all quantum operations from the input computation.
 *
 * @param context The MLIR context in which the module will be created
 * @param quantumComputation The quantum computation to translate
 * @return mlir::OwningOpRef<mlir::ModuleOp> The translated MLIR module
 */
mlir::OwningOpRef<mlir::ModuleOp> translateQuantumComputationToMLIR(
    mlir::MLIRContext* context,
    const qc::QuantumComputation& quantumComputation) {
  mlir::OpBuilder builder(context);
  const auto loc = builder.getUnknownLoc();

  // Create module
  auto module = builder.create<mlir::ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  // Create main function as entry point
  auto funcType = builder.getFunctionType({}, {});
  auto mainFunc = builder.create<mlir::func::FuncOp>(loc, "main", funcType);

  // Add entry_point attribute to identify the main function
  const auto entryPointAttr = mlir::StringAttr::get(context, "entry_point");
  mainFunc->setAttr("passthrough",
                    mlir::ArrayAttr::get(context, {entryPointAttr}));

  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Allocate quantum registers and extract qubits
  auto qregs = getQregs(builder, context, quantumComputation);
  auto qubits = getQubits(quantumComputation, qregs);

  // Allocate classical registers
  auto bitMap = getBitMap(builder, quantumComputation);

  // Add operations and handle potential failures
  // if (addOperations(builder, quantumComputation, qubits, bitMap).failed()) {
  //   // Even if operations fail, return the module with what we could
  //   translate emitError(loc) << "Failed to translate some quantum
  //   operations";
  // }

  // Deallocate quantum registers
  for (const auto& qreg : qregs) {
    deallocateQreg(builder, qreg.qreg);
  }

  // Create terminator
  builder.create<mlir::func::ReturnOp>(loc);

  return module;
}
} // namespace mlir
