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
 * @brief Extracts individual qubits from a quantum register.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param context The MLIR context in which types are created
 * @param quantumRegister The quantum register from which to extract qubits
 * @param numQubits The number of qubits to extract
 * @return llvm::SmallVector<mlir::Value> Vector of extracted qubit values
 */
llvm::SmallVector<mlir::Value> extractQubits(mlir::OpBuilder& builder,
                                             mlir::MLIRContext* context,
                                             mlir::Value quantumRegister,
                                             const std::size_t numQubits) {
  const auto& qubitType = mqt::ir::ref::QubitType::get(context);
  llvm::SmallVector<mlir::Value> qubits;
  qubits.reserve(numQubits);

  const auto loc = builder.getUnknownLoc();
  for (std::size_t qubit = 0; qubit < numQubits; ++qubit) {
    auto indexAttr = builder.getI64IntegerAttr(static_cast<int64_t>(qubit));
    auto extractOp = builder.create<mqt::ir::ref::ExtractOp>(
        loc, qubitType, quantumRegister, nullptr, indexAttr);
    qubits.emplace_back(extractOp.getResult());
  }

  return qubits;
}

/**
 * @brief Adds a single quantum operation to the MLIR module.
 *
 * @tparam OpType The type of the operation to create.
 * @param builder The MLIR OpBuilder.
 * @param operation The quantum operation to add.
 * @param qubits The qubits of the quantum register.
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
 * @param builder The MLIR OpBuilder.
 * @param operation The measure operation to add.
 * @param qubits The qubits of the quantum register.
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
 * @param builder The MLIR OpBuilder.
 * @param operation The reset operation to add.
 * @param qubits The qubits of the quantum register.
 */
void addResetOp(mlir::OpBuilder& builder, const qc::Operation& operation,
                const llvm::SmallVector<mlir::Value>& qubits) {
  const auto& targets = operation.getTargets();
  for (const auto& target : targets) {
    const mlir::Value inQubit = qubits[target];
    builder.create<mqt::ir::ref::ResetOp>(builder.getUnknownLoc(), inQubit);
  }
}

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
    case qc::OpType::I:
      addOperation<mqt::ir::ref::IOp>(builder, *operation, qubits);
      break;
    case qc::OpType::H:
      addOperation<mqt::ir::ref::HOp>(builder, *operation, qubits);
      break;
    case qc::OpType::X:
      addOperation<mqt::ir::ref::XOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Y:
      addOperation<mqt::ir::ref::YOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Z:
      addOperation<mqt::ir::ref::ZOp>(builder, *operation, qubits);
      break;
    case qc::OpType::S:
      addOperation<mqt::ir::ref::SOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Sdg:
      addOperation<mqt::ir::ref::SdgOp>(builder, *operation, qubits);
      break;
    case qc::OpType::T:
      addOperation<mqt::ir::ref::TOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Tdg:
      addOperation<mqt::ir::ref::TdgOp>(builder, *operation, qubits);
      break;
    case qc::OpType::V:
      addOperation<mqt::ir::ref::VOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Vdg:
      addOperation<mqt::ir::ref::VdgOp>(builder, *operation, qubits);
      break;
    case qc::OpType::U:
      addOperation<mqt::ir::ref::UOp>(builder, *operation, qubits);
      break;
    case qc::OpType::U2:
      addOperation<mqt::ir::ref::U2Op>(builder, *operation, qubits);
      break;
    case qc::OpType::P:
      addOperation<mqt::ir::ref::POp>(builder, *operation, qubits);
      break;
    case qc::OpType::SX:
      addOperation<mqt::ir::ref::SXOp>(builder, *operation, qubits);
      break;
    case qc::OpType::SXdg:
      addOperation<mqt::ir::ref::SXdgOp>(builder, *operation, qubits);
      break;
    case qc::OpType::RX:
      addOperation<mqt::ir::ref::RXOp>(builder, *operation, qubits);
      break;
    case qc::OpType::RY:
      addOperation<mqt::ir::ref::RYOp>(builder, *operation, qubits);
      break;
    case qc::OpType::RZ:
      addOperation<mqt::ir::ref::RZOp>(builder, *operation, qubits);
      break;
    case qc::OpType::SWAP:
      addOperation<mqt::ir::ref::SWAPOp>(builder, *operation, qubits);
      break;
    case qc::OpType::iSWAP:
      addOperation<mqt::ir::ref::iSWAPOp>(builder, *operation, qubits);
      break;
    case qc::OpType::iSWAPdg:
      addOperation<mqt::ir::ref::iSWAPdgOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Peres:
      addOperation<mqt::ir::ref::PeresOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Peresdg:
      addOperation<mqt::ir::ref::PeresdgOp>(builder, *operation, qubits);
      break;
    case qc::OpType::DCX:
      addOperation<mqt::ir::ref::DCXOp>(builder, *operation, qubits);
      break;
    case qc::OpType::ECR:
      addOperation<mqt::ir::ref::ECROp>(builder, *operation, qubits);
      break;
    case qc::OpType::RXX:
      addOperation<mqt::ir::ref::RXXOp>(builder, *operation, qubits);
      break;
    case qc::OpType::RYY:
      addOperation<mqt::ir::ref::RYYOp>(builder, *operation, qubits);
      break;
    case qc::OpType::RZZ:
      addOperation<mqt::ir::ref::RZZOp>(builder, *operation, qubits);
      break;
    case qc::OpType::RZX:
      addOperation<mqt::ir::ref::RZXOp>(builder, *operation, qubits);
      break;
    case qc::OpType::XXminusYY:
      addOperation<mqt::ir::ref::XXminusYY>(builder, *operation, qubits);
      break;
    case qc::OpType::XXplusYY:
      addOperation<mqt::ir::ref::XXplusYY>(builder, *operation, qubits);
      break;
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

  // Create terminator
  builder.create<mlir::func::ReturnOp>(loc);

  return module;
}
