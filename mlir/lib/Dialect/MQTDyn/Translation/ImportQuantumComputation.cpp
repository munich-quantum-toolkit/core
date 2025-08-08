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
#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

mlir::OwningOpRef<mlir::ModuleOp>
translateQuantumComputationToMLIR(mlir::MLIRContext& context,
                                  qc::QuantumComputation& quantumComputation) {

  mlir::OpBuilder builder(&context);

  auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  // Create function
  auto funcType = builder.getFunctionType({}, {});

  auto mainFunc = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                     "main", funcType);

  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Define register
  const auto& qregType = mqt::ir::dyn::QubitRegisterType::get(&context);
  auto numQubits = quantumComputation.getNqubits();
  auto sizeAttr = builder.getI64IntegerAttr(numQubits);

  auto allocOp = builder.create<mqt::ir::dyn::AllocOp>(
      builder.getUnknownLoc(), qregType, nullptr, sizeAttr);

  // Extract qubits
  const auto& qubitType = mqt::ir::dyn::QubitType::get(&context);

  std::vector<mlir::Value> allQubits = {};
  allQubits.reserve(numQubits);

  for (std::size_t qubit = 0; qubit < numQubits; ++qubit) {
    auto indexAttr = builder.getI64IntegerAttr(qubit);
    auto extractOp = builder.create<mqt::ir::dyn::ExtractOp>(
        builder.getUnknownLoc(), qubitType, allocOp.getResult(), nullptr,
        indexAttr);
    allQubits.push_back(extractOp.getResult());
  }

  // Add gates
  for (const auto& operation : quantumComputation) {
    if (operation->getType() == qc::OpType::H) {
      mlir::ValueRange params;
      mlir::ValueRange posCtrlQubits;
      mlir::ValueRange negCtrlQubits;

      auto target = operation->getTargets()[0];
      mlir::SmallVector<mlir::Value, 1> inQubitsVec = {allQubits[target]};
      mlir::ValueRange inQubits = {inQubitsVec};

      builder.create<mqt::ir::dyn::HOp>(builder.getUnknownLoc(), nullptr,
                                        nullptr, params, inQubits,
                                        posCtrlQubits, negCtrlQubits);
    }
  }

  // Create terminator
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

  return module;
}
