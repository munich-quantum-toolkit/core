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

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

mlir::OwningOpRef<mlir::ModuleOp>
translateQuantumComputationToMLIR(mlir::MLIRContext& context,
                                  qc::QuantumComputation& quantumComputation) {

  mlir::OpBuilder builder(&context);

  auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  const auto& qregType = mqt::ir::dyn::QubitRegisterType::get(&context);
  auto numQubits = quantumComputation.getNqubits();
  auto sizeAttr = builder.getI64IntegerAttr(numQubits);

  builder.create<mqt::ir::dyn::AllocOp>(builder.getUnknownLoc(), qregType,
                                        nullptr, sizeAttr);

  //   for (const auto& operation : quantumComputation) {
  //     if (operation->getType() == qc::OpType::X) {
  //       auto staticParamsAttr = builder.getDenseF64ArrayAttr({});
  //       auto paramsMaskAttr = builder.getDenseBoolArrayAttr({});
  //       mlir::ValueRange params;
  //       mlir::ValueRange posCtrlQubits;
  //       mlir::ValueRange negCtrlQubits;

  //       auto target = operation->getTargets()[0];
  //       mlir::Value inQubit = allQubits[target];
  //       mlir::ValueRange inQubits = {inQubit};

  //       auto xOp = builder.create<mqt::ir::dyn::XOp>(
  //           builder.getUnknownLoc(), staticParamsAttr, paramsMaskAttr,
  //           params, inQubits, posCtrlQubits, negCtrlQubits);
  //       module.push_back(xOp);
  //     }
  //   }

  return module;
}
