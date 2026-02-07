/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

LogicalResult IfOp::verify() {
  for (const auto& type : getInputs().getTypes()) {
    if (llvm::isa<QubitType>(type)) {
      continue;
    }
    auto tensor = dyn_cast<TensorType>(type);
    if (tensor && llvm::isa<QubitType>(tensor.getElementType())) {
      continue;
    }
    return emitOpError(
        "Types of inputs must be qubit type or tensor of qubit type!");
  }

  return success();
}

void IfOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, Value condition,
    ValueRange inputs,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> thenBuilder,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> elseBuilder) {

  build(odsBuilder, odsState, inputs.getTypes(), condition, inputs);

  auto& thenBlock = odsState.regions.front()->emplaceBlock();
  auto& elseBlock = odsState.regions.back()->emplaceBlock();

  thenBlock.addArguments(
      inputs.getTypes(),
      SmallVector<Location>(inputs.size(), odsState.location));
  odsBuilder.setInsertionPointToStart(&thenBlock);

  qco::YieldOp::create(odsBuilder, odsState.location,
                       thenBuilder(thenBlock.getArguments()));
  elseBlock.addArguments(
      inputs.getTypes(),
      SmallVector<Location>(inputs.size(), odsState.location));
  odsBuilder.setInsertionPointToStart(&elseBlock);
  qco::YieldOp::create(odsBuilder, odsState.location,
                       elseBuilder(elseBlock.getArguments()));
}

Block* IfOp::thenBlock() { return &getThenRegion().back(); }
YieldOp IfOp::thenYield() { return cast<YieldOp>(&thenBlock()->back()); }
Block* IfOp::elseBlock() { return &getElseRegion().back(); }
YieldOp IfOp::elseYield() { return cast<YieldOp>(&elseBlock()->back()); }
