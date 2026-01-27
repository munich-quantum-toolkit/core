/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Eigen/Core"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Cancel nested inverse modifiers, i.e., `inv(inv(x)) => x`.
 */
struct CancelNestedInv final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    auto innerUnitary = op.getBodyUnitary();
    auto innerInvOp = llvm::dyn_cast<InvOp>(innerUnitary.getOperation());
    if (!innerInvOp) {
      return failure();
    }

    // Remove both inverse operations
    auto innerInnerUnitary = innerInvOp.getBodyUnitary();
    auto* clonedOp = rewriter.clone(*innerInnerUnitary.getOperation());
    rewriter.replaceOp(op, clonedOp->getResults());

    return success();
  }
};

} // namespace

UnitaryOpInterface InvOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody()->front());
}

size_t InvOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t InvOp::getNumTargets() { return getTargetsIn().size(); }

size_t InvOp::getNumControls() { return getBodyUnitary().getNumControls(); }

Value InvOp::getInputQubit(const size_t i) {
  return getBodyUnitary().getInputQubit(i);
}

Value InvOp::getOutputQubit(const size_t i) {
  return getBodyUnitary().getOutputQubit(i);
}

Value InvOp::getInputTarget(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Target index out of bounds");
  }
  return getTargetsIn()[i];
}

Value InvOp::getOutputTarget(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Target index out of bounds");
  }
  return getTargetsOut()[i];
}

Value InvOp::getInputControl(const size_t i) {
  return getBodyUnitary().getInputControl(i);
}

Value InvOp::getOutputControl(const size_t i) {
  return getBodyUnitary().getOutputControl(i);
}

Value InvOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getTargetsOut()[i]) {
      return getTargetsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value InvOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getTargetsIn()[i]) {
      return getTargetsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

size_t InvOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value InvOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  ValueRange targets, UnitaryOpInterface bodyUnitary) {
  build(odsBuilder, odsState, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  auto* op = odsBuilder.clone(*bodyUnitary.getOperation());
  YieldOp::create(odsBuilder, odsState.location, op->getResults());
}

void InvOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, ValueRange targets,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> bodyBuilder) {
  build(odsBuilder, odsState, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  const auto qubitType = QubitType::get(odsBuilder.getContext());
  for (size_t i = 0; i < targets.size(); ++i) {
    block.addArgument(qubitType, odsState.location);
  }

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  YieldOp::create(odsBuilder, odsState.location,
                  bodyBuilder(block.getArguments()));
}

LogicalResult InvOp::verify() {
  auto& block = *getBody();
  if (block.getOperations().size() != 2) {
    return emitOpError("body region must have exactly two operations");
  }
  const auto numTargets = getNumTargets();
  if (block.getArguments().size() != numTargets) {
    return emitOpError(
        "number of block arguments must match the number of targets");
  }
  const auto qubitType = QubitType::get(getContext());
  for (size_t i = 0; i < numTargets; ++i) {
    if (block.getArgument(i).getType() != qubitType) {
      return emitOpError("block argument type at index ")
             << i << " does not match target type";
    }
  }
  if (!llvm::isa<UnitaryOpInterface>(block.front())) {
    return emitOpError(
        "first operation in body region must be a unitary operation");
  }
  if (!llvm::isa<YieldOp>(block.back())) {
    return emitOpError(
        "second operation in body region must be a yield operation");
  }
  if (const auto numYieldOperands = block.back().getNumOperands();
      numYieldOperands != numTargets) {
    return emitOpError("yield operation must yield ")
           << numTargets << " values, but found " << numYieldOperands;
  }

  SmallPtrSet<Value, 4> uniqueQubitsIn;
  for (const auto& target : getTargetsIn()) {
    if (!uniqueQubitsIn.insert(target).second) {
      return emitOpError("duplicate target qubit found");
    }
  }

  auto bodyUnitary = getBodyUnitary();
  if (bodyUnitary.getNumQubits() != numTargets) {
    return emitOpError("body unitary must operate on exactly ")
           << numTargets << " target qubits, but found "
           << bodyUnitary.getNumQubits();
  }
  const auto numQubits = bodyUnitary.getNumQubits();
  for (size_t i = 0; i < numQubits; i++) {
    if (bodyUnitary.getInputQubit(i) != block.getArgument(i)) {
      return emitOpError("body unitary must use target alias block argument ")
             << i << " (and not the original target operand)";
    }
  }

  // Also require yield to forward the unitary's outputs in-order.
  for (size_t i = 0; i < numTargets; ++i) {
    if (block.back().getOperand(i) != bodyUnitary.getOutputQubit(i)) {
      return emitOpError("yield operand ")
             << i << " must be the body unitary output qubit " << i;
    }
  }

  SmallPtrSet<Value, 4> uniqueQubitsOut;
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubitsOut.insert(bodyUnitary.getOutputQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  if (llvm::isa<BarrierOp>(bodyUnitary.getOperation())) {
    return emitOpError("BarrierOp cannot be controlled");
  }

  return success();
}

void InvOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<CancelNestedInv>(context);
}

std::optional<Eigen::MatrixXcd> InvOp::getUnitaryMatrix() {
  auto&& bodyUnitary = getBodyUnitary();
  if (!bodyUnitary) {
    return std::nullopt;
  }
  auto&& targetMatrix = bodyUnitary.getUnitaryMatrix<Eigen::MatrixXcd>();
  if (!targetMatrix) {
    return std::nullopt;
  }

  targetMatrix->adjointInPlace();

  return targetMatrix;
}
