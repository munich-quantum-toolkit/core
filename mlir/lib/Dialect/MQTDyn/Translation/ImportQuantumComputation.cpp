/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTDyn/Translation/ImportQuantumComputation.h"

#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <stdexcept>

namespace {

/**
 * @brief Adds a quantum register to the MLIR module.
 *
 * @param builder The MLIR OpBuilder.
 * @param context The MLIR context.
 * @param numQubits The number of qubits in the quantum register.
 * @return The allocated quantum register.
 */
mlir::Value allocateQreg(mlir::OpBuilder& builder, mlir::MLIRContext& context,
                         const std::size_t numQubits) {
  const auto& qregType = mqt::ir::dyn::QubitRegisterType::get(&context);
  auto sizeAttr = builder.getI64IntegerAttr(static_cast<int64_t>(numQubits));

  auto allocOp = builder.create<mqt::ir::dyn::AllocOp>(
      builder.getUnknownLoc(), qregType, nullptr, sizeAttr);

  return allocOp.getResult();
}

/**
 * @brief Adds qubits to the MLIR module.
 *
 * @param builder The MLIR OpBuilder.
 * @param context The MLIR context.
 * @param quantumRegister The quantum register to extract qubits from.
 * @param numQubits The number of qubits to extract.
 * @return The extracted qubits.
 */
llvm::SmallVector<mlir::Value> extractQubits(mlir::OpBuilder& builder,
                                             mlir::MLIRContext& context,
                                             mlir::Value quantumRegister,
                                             const std::size_t numQubits) {
  const auto& qubitType = mqt::ir::dyn::QubitType::get(&context);
  llvm::SmallVector<mlir::Value> qubits;
  qubits.reserve(numQubits);

  for (std::size_t qubit = 0; qubit < numQubits; ++qubit) {
    auto indexAttr = builder.getI64IntegerAttr(static_cast<int64_t>(qubit));
    auto extractOp = builder.create<mqt::ir::dyn::ExtractOp>(
        builder.getUnknownLoc(), qubitType, quantumRegister, nullptr,
        indexAttr);
    qubits.emplace_back(extractOp.getResult());
  }

  return qubits;
}

/**
 * @brief Adds a single QuantumComputation operation to the MLIR module.
 *
 * @tparam OpType The type of the operation to create.
 * @param builder The MLIR OpBuilder.
 * @param operation The QuantumComputation quantum operation to add.
 * @param qubits The qubits of the quantum register.
 */
template <typename OpType>
void addOperation(mlir::OpBuilder& builder, const qc::Operation& operation,
                  const llvm::SmallVector<mlir::Value>& qubits) {
  // Define operation parameters
  llvm::SmallVector<mlir::Value> paramsVec;
  if (const auto& parameters = operation.getParameter(); !parameters.empty()) {
    paramsVec.reserve(parameters.size());
    for (const auto& parameter : parameters) {
      paramsVec.emplace_back(
          builder
              .create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(), builder.getF64FloatAttr(parameter))
              .getResult());
    }
  }
  mlir::ValueRange params(paramsVec);

  // Define input qubits
  const auto& targets = operation.getTargets();
  llvm::SmallVector<mlir::Value> inQubitsVec;
  inQubitsVec.reserve(targets.size());
  for (const auto& t : targets) {
    inQubitsVec.emplace_back(qubits[t]);
  }
  const mlir::ValueRange inQubits(inQubitsVec);

  // Define control qubits
  llvm::SmallVector<mlir::Value> posCtrlVec;
  llvm::SmallVector<mlir::Value> negCtrlVec;
  if (const auto& controls = operation.getControls(); !controls.empty()) {
    posCtrlVec.reserve(controls.size());
    negCtrlVec.reserve(controls.size());
    for (const auto& [qubit, type] : controls) {
      if (type == qc::Control::Type::Pos) {
        posCtrlVec.emplace_back(qubits[qubit]);
      } else if (type == qc::Control::Type::Neg) {
        negCtrlVec.emplace_back(qubits[qubit]);
      }
    }
  }
  mlir::ValueRange posCtrlQubits{posCtrlVec};
  mlir::ValueRange negCtrlQubits{negCtrlVec};

  // Create operation
  builder.create<OpType>(builder.getUnknownLoc(), nullptr, nullptr, params,
                         inQubits, posCtrlQubits, negCtrlQubits);
}

/**
 * @brief Adds QuantumComputation operations to the MLIR module.
 *
 * @param builder The MLIR OpBuilder.
 * @param quantumComputation The QuantumComputation to translate.
 * @param qubits The qubits of the quantum register.
 */
void addOperations(mlir::OpBuilder& builder,
                   const qc::QuantumComputation& quantumComputation,
                   const llvm::SmallVector<mlir::Value>& qubits) {
  for (const auto& operation : quantumComputation) {
    switch (operation->getType()) {
    case qc::OpType::I:
      addOperation<mqt::ir::dyn::IOp>(builder, *operation, qubits);
      break;
    case qc::OpType::H:
      addOperation<mqt::ir::dyn::HOp>(builder, *operation, qubits);
      break;
    case qc::OpType::X:
      addOperation<mqt::ir::dyn::XOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Y:
      addOperation<mqt::ir::dyn::YOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Z:
      addOperation<mqt::ir::dyn::ZOp>(builder, *operation, qubits);
      break;
    case qc::OpType::S:
      addOperation<mqt::ir::dyn::SOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Sdg:
      addOperation<mqt::ir::dyn::SdgOp>(builder, *operation, qubits);
      break;
    case qc::OpType::T:
      addOperation<mqt::ir::dyn::TOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Tdg:
      addOperation<mqt::ir::dyn::TdgOp>(builder, *operation, qubits);
      break;
    case qc::OpType::V:
      addOperation<mqt::ir::dyn::VOp>(builder, *operation, qubits);
      break;
    case qc::OpType::Vdg:
      addOperation<mqt::ir::dyn::VdgOp>(builder, *operation, qubits);
      break;
    case qc::OpType::RX:
      addOperation<mqt::ir::dyn::RXOp>(builder, *operation, qubits);
      break;
    case qc::OpType::SWAP:
      addOperation<mqt::ir::dyn::SWAPOp>(builder, *operation, qubits);
      break;
    default:
      throw std::runtime_error("Unsupported operation type: " +
                               operation->getName());
    }
  }
}

} // namespace

/**
 * @brief Translates a QuantumComputation to MQTDyn.
 *
 * @param context The MLIR context.
 * @param quantumComputation The QuantumComputation to translate.
 * @return The translated MLIR module.
 */
mlir::OwningOpRef<mlir::ModuleOp> translateQuantumComputationToMLIR(
    mlir::MLIRContext& context,
    const qc::QuantumComputation& quantumComputation) {
  mlir::OpBuilder builder(&context);

  // Create module
  auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  // Create main function as entry point
  auto funcType = builder.getFunctionType({}, {});

  auto mainFunc = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                     "main", funcType);

  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Parse quantum computation
  const auto numQubits = quantumComputation.getNqubits();
  const auto qreg = allocateQreg(builder, context, numQubits);
  const auto qubits = extractQubits(builder, context, qreg, numQubits);

  addOperations(builder, quantumComputation, qubits);

  // Create terminator
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

  return module;
}
