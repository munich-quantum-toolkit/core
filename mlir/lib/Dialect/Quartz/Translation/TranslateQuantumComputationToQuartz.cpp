/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/Translation/TranslateQuantumComputationToQuartz.h"

#include "ir/QuantumComputation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "mlir/Dialect/Quartz/Builder/QuartzProgramBuilder.h"

#include <algorithm>
#include <cstddef>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <ranges>
#include <utility>

namespace mlir {

using namespace quartz;

namespace {

/**
 * @brief Structure to store information about quantum registers
 *
 * @details
 * Maps quantum registers from the input QuantumComputation to their
 * corresponding MLIR qubit values allocated via the QuartzProgramBuilder.
 */
struct QregInfo {
  const qc::QuantumRegister* qregPtr;
  llvm::SmallVector<Value> qubits;
};

using BitMemInfo = std::pair<QuartzProgramBuilder::ClassicalRegister*,
                             std::size_t>; // (register ref, localIdx)
using BitIndexVec = llvm::SmallVector<BitMemInfo>;

/**
 * @brief Allocates quantum registers using the QuartzProgramBuilder
 *
 * @details
 * Processes all quantum and ancilla registers from the QuantumComputation,
 * sorting them by start index, and allocates them using the builder's
 * allocQubitRegister method which generates quartz.alloc operations with
 * proper register metadata.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @return Vector containing information about all quantum registers
 */
llvm::SmallVector<QregInfo>
allocateQregs(QuartzProgramBuilder& builder,
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

  // Allocate quantum registers using the builder
  llvm::SmallVector<QregInfo> qregs;
  for (const auto* qregPtr : qregPtrs) {
    auto qubits =
        builder.allocQubitRegister(qregPtr->getSize(), qregPtr->getName());
    qregs.emplace_back(qregPtr, std::move(qubits));
  }

  return qregs;
}

/**
 * @brief Builds a flat mapping from global qubit index to qubit value
 *
 * @details
 * Creates a flat array where each index corresponds to a physical qubit
 * index, and the value is the MLIR Value representing that qubit. This
 * simplifies operation translation by providing direct qubit lookup.
 *
 * @param quantumComputation The quantum computation being translated
 * @param qregs Vector containing information about all quantum registers
 * @return Flat vector of qubit values indexed by physical qubit index
 */
llvm::SmallVector<Value>
buildQubitMap(const qc::QuantumComputation& quantumComputation,
              const llvm::SmallVector<QregInfo>& qregs) {
  llvm::SmallVector<Value> flatQubits;
  const auto maxPhys = quantumComputation.getHighestPhysicalQubitIndex();
  flatQubits.resize(static_cast<size_t>(maxPhys) + 1);

  for (const auto& qreg : qregs) {
    for (std::size_t i = 0; i < qreg.qregPtr->getSize(); ++i) {
      const auto globalIdx = qreg.qregPtr->getStartIndex() + i;
      flatQubits[globalIdx] = qreg.qubits[i];
    }
  }

  return flatQubits;
}

/**
 * @brief Allocates classical registers using the QuartzProgramBuilder
 *
 * @details
 * Creates classical bit registers and builds a mapping from global classical
 * bit indices to (register, local_index) pairs. This is used for measurement
 * result storage.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @return Vector mapping global bit indices to register and local indices
 */
BitIndexVec
allocateClassicalRegisters(QuartzProgramBuilder& builder,
                           const qc::QuantumComputation& quantumComputation) {
  // Build list of pointers for sorting
  llvm::SmallVector<const qc::ClassicalRegister*> cregPtrs;
  cregPtrs.reserve(quantumComputation.getClassicalRegisters().size());
  for (const auto& reg :
       quantumComputation.getClassicalRegisters() | std::views::values) {
    cregPtrs.emplace_back(&reg);
  }

  // Sort by start index
  std::ranges::sort(cregPtrs, [](const qc::ClassicalRegister* a,
                                 const qc::ClassicalRegister* b) {
    return a->getStartIndex() < b->getStartIndex();
  });

  // Build mapping using the builder
  BitIndexVec bitMap;
  bitMap.resize(quantumComputation.getNcbits());
  for (const auto* reg : cregPtrs) {
    auto& mem =
        builder.allocClassicalBitRegister(reg->getSize(), reg->getName());
    for (std::size_t i = 0; i < reg->getSize(); ++i) {
      const auto globalIdx = static_cast<std::size_t>(reg->getStartIndex() + i);
      bitMap[globalIdx] = {&mem, i};
    }
  }

  return bitMap;
}

/**
 * @brief Adds measurement operations for a single operation
 *
 * @details
 * Translates measurement operations from the QuantumComputation to
 * quartz.measure operations, storing results in classical registers.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The measurement operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 * @param bitMap Mapping from global bit index to (register, local_index)
 */
void addMeasureOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
                  const llvm::SmallVector<Value>& qubits,
                  const BitIndexVec& bitMap) {
  const auto& measureOp =
      dynamic_cast<const qc::NonUnitaryOperation&>(operation);
  const auto& targets = measureOp.getTargets();
  const auto& classics = measureOp.getClassics();

  for (std::size_t i = 0; i < targets.size(); ++i) {
    const auto& qubit = qubits[targets[i]];
    const auto bitIdx = static_cast<std::size_t>(classics[i]);
    const auto& [mem, localIdx] = bitMap[bitIdx];

    // Use builder's measure method which keeps output record
    builder.measure(qubit, (*mem)[localIdx]);
  }
}

/**
 * @brief Adds reset operations for a single operation
 *
 * @details
 * Translates reset operations from the QuantumComputation to
 * quartz.reset operations.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The reset operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addResetOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
                const llvm::SmallVector<Value>& qubits) {
  for (const auto& target : operation.getTargets()) {
    const Value qubit = qubits[target];
    builder.reset(qubit);
  }
}

// Temporary implementation of XOp translation
void addXOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& target = operation.getTargets()[0];
  const Value qubit = qubits[target];
  builder.x(qubit);
}

// Temporary implementation of RXOp translation
void addRXOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
             const llvm::SmallVector<Value>& qubits) {
  const auto angle = operation.getParameter()[0];
  const auto& target = operation.getTargets()[0];
  const Value qubit = qubits[target];
  builder.rx(angle, qubit);
}

/**
 * @brief Translates operations from QuantumComputation to Quartz dialect
 *
 * @details
 * Iterates through all operations in the QuantumComputation and translates
 * them to Quartz dialect operations. Currently supports:
 * - Measurement operations
 * - Reset operations
 *
 * Unary gates and other operations will be added as the Quartz dialect
 * is expanded.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 * @param bitMap Mapping from global bit index to (register, local_index)
 * @return Success if all supported operations were translated
 */
LogicalResult
translateOperations(QuartzProgramBuilder& builder,
                    const qc::QuantumComputation& quantumComputation,
                    const llvm::SmallVector<Value>& qubits,
                    const BitIndexVec& bitMap) {
  for (const auto& operation : quantumComputation) {
    switch (operation->getType()) {
    case qc::OpType::Measure:
      addMeasureOp(builder, *operation, qubits, bitMap);
      break;
    case qc::OpType::Reset:
      addResetOp(builder, *operation, qubits);
      break;
    case qc::OpType::X:
      addXOp(builder, *operation, qubits);
      break;
    case qc::OpType::RX:
      addRXOp(builder, *operation, qubits);
      break;
    default:
      // Unsupported operation - skip for now
      // As the Quartz dialect is expanded, more operations will be supported
      continue;
    }
  }

  return success();
}

} // namespace

/**
 * @brief Translates a QuantumComputation to an MLIR module with Quartz
 * operations
 *
 * @details
 * This function takes a quantum computation and translates it into an MLIR
 * module containing Quartz dialect operations. It uses the QuartzProgramBuilder
 * to handle module and function creation, resource allocation, and operation
 * translation.
 *
 * The translation process:
 * 1. Creates a QuartzProgramBuilder and initializes it (creates main function
 *    with signature () -> i64)
 * 2. Allocates quantum registers using quartz.alloc with register metadata
 * 3. Tracks classical registers for measurement results
 * 4. Translates operations (currently: measure, reset)
 * 5. Finalizes the module (adds return statement with exit code 0)
 *
 * The generated main function returns exit code 0 to indicate successful
 * execution of the quantum program.
 *
 * Currently supported operations:
 * - Measurement (quartz.measure)
 * - Reset (quartz.reset)
 *
 * Operations not yet supported are silently skipped. As the Quartz dialect
 * is expanded with gate operations, this translation will be enhanced.
 *
 * @param context The MLIR context in which the module will be created
 * @param quantumComputation The quantum computation to translate
 * @return OwningOpRef containing the translated MLIR module
 */
OwningOpRef<ModuleOp> translateQuantumComputationToQuartz(
    MLIRContext* context, const qc::QuantumComputation& quantumComputation) {
  // Create and initialize the builder (creates module and main function)
  QuartzProgramBuilder builder(context);
  builder.initialize();

  // Allocate quantum registers using the builder
  const auto qregs = allocateQregs(builder, quantumComputation);

  // Build flat qubit mapping for easy lookup
  const auto qubits = buildQubitMap(quantumComputation, qregs);

  // Allocate classical registers using the builder
  const auto bitMap = allocateClassicalRegisters(builder, quantumComputation);

  // Translate operations
  if (translateOperations(builder, quantumComputation, qubits, bitMap)
          .failed()) {
    // Note: Currently all operations succeed or are skipped
    // This check is here for future error handling
  }

  // Finalize and return the module (adds return statement and transfers
  // ownership)
  return builder.finalize();
}

} // namespace mlir
