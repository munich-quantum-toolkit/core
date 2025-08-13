/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTRef/Translation/ImportQuantumComputation.h"

#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

namespace {

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
  const auto& qregType = mqt::ir::ref::QubitRegisterType::get(context);
  auto sizeAttr = builder.getI64IntegerAttr(static_cast<int64_t>(numQubits));

  auto allocOp = builder.create<mqt::ir::ref::AllocOp>(
      builder.getUnknownLoc(), qregType, nullptr, sizeAttr);

  return allocOp.getResult();
}

/**
 * @brief Deallocates a quantum register in the MLIR module.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param quantumRegister The quantum register to deallocate
 */
void deallocateQreg(mlir::OpBuilder& builder, mlir::Value qreg) {
  builder.create<mqt::ir::ref::DeallocOp>(builder.getUnknownLoc(), qreg);
}

/**
 * @brief Extracts individual qubits from a quantum register.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param context The MLIR context in which types are created
 * @param qreg The quantum register from which to extract qubits
 * @param numQubits The number of qubits to extract
 * @return llvm::SmallVector<mlir::Value> Vector of extracted qubit values
 */
llvm::SmallVector<mlir::Value> extractQubits(mlir::OpBuilder& builder,
                                             mlir::MLIRContext* context,
                                             mlir::Value qreg,
                                             const std::size_t numQubits) {
  const auto& qubitType = mqt::ir::ref::QubitType::get(context);
  llvm::SmallVector<mlir::Value> qubits;
  qubits.reserve(numQubits);

  const auto loc = builder.getUnknownLoc();
  for (std::size_t qubit = 0; qubit < numQubits; ++qubit) {
    auto indexAttr = builder.getI64IntegerAttr(static_cast<int64_t>(qubit));
    auto extractOp = builder.create<mqt::ir::ref::ExtractOp>(
        loc, qubitType, qreg, nullptr, indexAttr);
    qubits.emplace_back(extractOp.getResult());
  }

  return qubits;
}

/**
 * @brief Adds a single quantum operation to the MLIR module.
 *
 * @tparam OpType The type of the operation to create
 * @param builder The MLIR OpBuilder
 * @param operation The quantum operation to add
 * @param qubits The qubits of the quantum register
 */
template <typename OpType>
void addOperation(mlir::OpBuilder& builder, const qc::Operation& operation,
                  const llvm::SmallVector<mlir::Value>& qubits) {
  // Define operation parameters
  mlir::DenseF64ArrayAttr staticParamsAttr = nullptr;
  if (const auto& parameters = operation.getParameter(); !parameters.empty()) {
    staticParamsAttr = builder.getDenseF64ArrayAttr(parameters);
  }

  // Define input qubits
  const auto& targets = operation.getTargets();
  llvm::SmallVector<mlir::Value> inQubits;
  inQubits.reserve(targets.size());
  for (const auto& t : targets) {
    inQubits.emplace_back(qubits[t]);
  }

  // Define control qubits
  llvm::SmallVector<mlir::Value> posCtrlQubits;
  llvm::SmallVector<mlir::Value> negCtrlQubits;
  if (const auto& controls = operation.getControls(); !controls.empty()) {
    posCtrlQubits.reserve(controls.size());
    negCtrlQubits.reserve(controls.size());
    for (const auto& [qubit, type] : controls) {
      if (type == qc::Control::Type::Pos) {
        posCtrlQubits.emplace_back(qubits[qubit]);
      } else if (type == qc::Control::Type::Neg) {
        negCtrlQubits.emplace_back(qubits[qubit]);
      }
    }
  }

  // Create operation
  builder.create<OpType>(builder.getUnknownLoc(), staticParamsAttr, nullptr,
                         mlir::ValueRange{}, inQubits, posCtrlQubits,
                         negCtrlQubits);
}

/**
 * @brief Adds a measure operation to the MLIR module.
 *
 * @param builder The MLIR OpBuilder
 * @param operation The measure operation to add
 * @param qubits The qubits of the quantum register
 */
void addMeasureOp(mlir::OpBuilder& builder, const qc::Operation& operation,
                  const llvm::SmallVector<mlir::Value>& qubits) {
  const auto& bitType = mlir::IntegerType::get(builder.getContext(), 1);
  const auto& targets = operation.getTargets();
  for (const auto& target : targets) {
    const mlir::Value inQubit = qubits[target];
    builder.create<mqt::ir::ref::MeasureOp>(builder.getUnknownLoc(), bitType,
                                            inQubit);
  }
}

/**
 * @brief Adds a reset operation to the MLIR module.
 *
 * @param builder The MLIR OpBuilder
 * @param operation The reset operation to add
 * @param qubits The qubits of the quantum register
 */
void addResetOp(mlir::OpBuilder& builder, const qc::Operation& operation,
                const llvm::SmallVector<mlir::Value>& qubits) {
  const auto& targets = operation.getTargets();
  for (const auto& target : targets) {
    const mlir::Value inQubit = qubits[target];
    builder.create<mqt::ir::ref::ResetOp>(builder.getUnknownLoc(), inQubit);
  }
}

#define ADD_OP_CASE(op)                                                        \
  case qc::OpType::op:                                                         \
    addOperation<mqt::ir::ref::op##Op>(builder, *operation, qubits);           \
    break;

/**
 * @brief Adds quantum operations to the MLIR module.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @param qubits The qubits of the quantum register
 * @return mlir::LogicalResult Success if all operations were added, failure
 * otherwise
 */
llvm::LogicalResult
addOperations(mlir::OpBuilder& builder,
              const qc::QuantumComputation& quantumComputation,
              const llvm::SmallVector<mlir::Value>& qubits) {
  for (const auto& operation : quantumComputation) {
    switch (operation->getType()) {
      ADD_OP_CASE(I)
      ADD_OP_CASE(H)
      ADD_OP_CASE(X)
      ADD_OP_CASE(Y)
      ADD_OP_CASE(Z)
      ADD_OP_CASE(S)
      ADD_OP_CASE(Sdg)
      ADD_OP_CASE(T)
      ADD_OP_CASE(Tdg)
      ADD_OP_CASE(V)
      ADD_OP_CASE(Vdg)
      ADD_OP_CASE(U)
      ADD_OP_CASE(U2)
      ADD_OP_CASE(P)
      ADD_OP_CASE(SX)
      ADD_OP_CASE(SXdg)
      ADD_OP_CASE(RX)
      ADD_OP_CASE(RY)
      ADD_OP_CASE(RZ)
      ADD_OP_CASE(SWAP)
      ADD_OP_CASE(iSWAP)
      ADD_OP_CASE(iSWAPdg)
      ADD_OP_CASE(Peres)
      ADD_OP_CASE(Peresdg)
      ADD_OP_CASE(DCX)
      ADD_OP_CASE(ECR)
      ADD_OP_CASE(RXX)
      ADD_OP_CASE(RYY)
      ADD_OP_CASE(RZZ)
      ADD_OP_CASE(RZX)
      ADD_OP_CASE(XXminusYY)
      ADD_OP_CASE(XXplusYY)
    case qc::OpType::Measure:
      addMeasureOp(builder, *operation, qubits);
      break;
    case qc::OpType::Reset:
      addResetOp(builder, *operation, qubits);
      break;
    default:
      return llvm::failure();
    }
  }
  return llvm::success();
}
} // namespace

/**
 * @brief Translates a QuantumComputation to an MLIR module with MQTRef
 * operations.
 *
 * This function takes a quantum computation and translates it into an MLIR
 * module containing MQTRef dialect operations. It creates a main function that
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

  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Parse quantum computation
  const auto numQubits = quantumComputation.getNqubits();
  const auto qreg = allocateQreg(builder, context, numQubits);

  // Add operations and handle potential failures
  if (const auto qubits = extractQubits(builder, context, qreg, numQubits);
      failed(addOperations(builder, quantumComputation, qubits))) {
    // Even if operations fail, return the module with what we could translate
    emitError(loc) << "Failed to translate some quantum operations";
  }

  deallocateQreg(builder, qreg);

  // Create terminator
  builder.create<mlir::func::ReturnOp>(loc);

  return module;
}
