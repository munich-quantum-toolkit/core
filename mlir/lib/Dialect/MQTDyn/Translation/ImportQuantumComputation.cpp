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
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <vector>

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
                         std::size_t numQubits) {
  const auto& qregType = mqt::ir::dyn::QubitRegisterType::get(&context);
  auto sizeAttr = builder.getI64IntegerAttr(numQubits);

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
std::vector<mlir::Value> extractQubits(mlir::OpBuilder& builder,
                                       mlir::MLIRContext& context,
                                       mlir::Value quantumRegister,
                                       std::size_t numQubits) {
  const auto& qubitType = mqt::ir::dyn::QubitType::get(&context);
  std::vector<mlir::Value> qubits;
  qubits.reserve(numQubits);

  for (std::size_t qubit = 0; qubit < numQubits; ++qubit) {
    auto indexAttr = builder.getI64IntegerAttr(qubit);
    auto extractOp = builder.create<mqt::ir::dyn::ExtractOp>(
        builder.getUnknownLoc(), qubitType, quantumRegister, nullptr,
        indexAttr);
    qubits.push_back(extractOp.getResult());
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
                  const std::vector<mlir::Value>& qubits) {
  // Define operation parameters
  mlir::ValueRange params;
  auto parameters = operation.getParameter();
  std::vector<mlir::Value> paramsVec;
  if (!parameters.empty()) {
    paramsVec.reserve(parameters.size());
    for (const auto& parameter : parameters) {
      auto param = builder.create<mlir::arith::ConstantOp>(
          builder.getUnknownLoc(), builder.getF64Type(),
          builder.getF64FloatAttr(parameter));
      paramsVec.push_back(param.getResult());
    }
    params = mlir::ValueRange(paramsVec);
  }

  // Define input qubits
  std::vector<mlir::Value> inQubitsVec;
  auto targets = operation.getTargets();
  inQubitsVec.reserve(targets.size());
  for (const auto& t : targets) {
    inQubitsVec.push_back(qubits[t]);
  }
  const mlir::ValueRange inQubits(inQubitsVec);

  // Define control qubits
  mlir::ValueRange posCtrlQubits;
  mlir::ValueRange negCtrlQubits;
  std::vector<mlir::Value> posCtrlVec;
  std::vector<mlir::Value> negCtrlVec;
  auto controls = operation.getControls();
  if (!controls.empty()) {
    posCtrlVec.reserve(controls.size());
    negCtrlVec.reserve(controls.size());
    for (const auto& control : controls) {
      if (control.type == qc::Control::Type::Pos) {
        posCtrlVec.push_back(qubits[control.qubit]);
      } else if (control.type == qc::Control::Type::Neg) {
        negCtrlVec.push_back(qubits[control.qubit]);
      }
    }
    posCtrlQubits = mlir::ValueRange{posCtrlVec};
    negCtrlQubits = mlir::ValueRange{negCtrlVec};
  }

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
                   const std::vector<mlir::Value>& qubits) {
  for (const auto& operation : quantumComputation) {
    if (operation->getType() == qc::OpType::H) {
      addOperation<mqt::ir::dyn::HOp>(builder, *operation, qubits);
    } else if (operation->getType() == qc::OpType::RX) {
      addOperation<mqt::ir::dyn::RXOp>(builder, *operation, qubits);
    } else if (operation->getType() == qc::OpType::X) {
      addOperation<mqt::ir::dyn::XOp>(builder, *operation, qubits);
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
mlir::OwningOpRef<mlir::ModuleOp>
translateQuantumComputationToMLIR(mlir::MLIRContext& context,
                                  qc::QuantumComputation& quantumComputation) {
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
  auto numQubits = quantumComputation.getNqubits();
  auto qreg = allocateQreg(builder, context, numQubits);
  auto qubits = extractQubits(builder, context, qreg, numQubits);

  addOperations(builder, quantumComputation, qubits);

  // Create terminator
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

  return module;
}
