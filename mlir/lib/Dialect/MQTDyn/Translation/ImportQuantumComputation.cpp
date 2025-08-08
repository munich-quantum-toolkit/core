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
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <cstddef>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <vector>

mlir::Value allocateQreg(mlir::OpBuilder& builder, mlir::MLIRContext& context,
                         std::size_t numQubits) {
  const auto& qregType = mqt::ir::dyn::QubitRegisterType::get(&context);
  auto sizeAttr = builder.getI64IntegerAttr(numQubits);

  auto allocOp = builder.create<mqt::ir::dyn::AllocOp>(
      builder.getUnknownLoc(), qregType, nullptr, sizeAttr);

  return allocOp.getResult();
}

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

template <typename OpType>
void addOperation(mlir::OpBuilder& builder, const qc::Operation& operation,
                  const std::vector<mlir::Value>& qubits) {
  // Define operation parameters
  const mlir::ValueRange params;

  // Define input qubits
  auto target = operation.getTargets()[0];
  const mlir::SmallVector<mlir::Value, 1> inQubitsVec = {qubits[target]};
  const mlir::ValueRange inQubits = {inQubitsVec};

  // Define positive control qubits
  mlir::ValueRange posCtrlQubits;
  std::vector<mlir::Value> controlsVec;
  auto controls = operation.getControls();
  if (!controls.empty()) {
    controlsVec.reserve(controls.size());
    for (const auto& control : controls) {
      controlsVec.push_back(qubits[control.qubit]);
    }
    posCtrlQubits = mlir::ValueRange{controlsVec};
  }

  // Define negative control qubits
  const mlir::ValueRange negCtrlQubits;

  // Create operation
  builder.create<OpType>(builder.getUnknownLoc(), nullptr, nullptr, params,
                         inQubits, posCtrlQubits, negCtrlQubits);
}

void addOperations(mlir::OpBuilder& builder,
                   const qc::QuantumComputation& quantumComputation,
                   const std::vector<mlir::Value>& qubits) {
  for (const auto& operation : quantumComputation) {
    if (operation->getType() == qc::OpType::H) {
      addOperation<mqt::ir::dyn::HOp>(builder, *operation, qubits);
    } else if (operation->getType() == qc::OpType::X) {
      addOperation<mqt::ir::dyn::XOp>(builder, *operation, qubits);
    }
  }
}

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
