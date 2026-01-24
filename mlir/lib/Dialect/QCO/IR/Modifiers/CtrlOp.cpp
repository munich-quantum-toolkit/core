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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Merge nested control modifiers into a single one.
 */
struct MergeNestedCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one positive control
    // Trivial case is handled by RemoveTrivialCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    auto bodyCtrlOp =
        llvm::dyn_cast<CtrlOp>(op.getBodyUnitary().getOperation());
    if (!bodyCtrlOp) {
      return failure();
    }

    // Merge controls
    const auto newControls = llvm::to_vector(
        llvm::concat<Value>(op.getControlsIn(), bodyCtrlOp.getControlsIn()));
    rewriter.replaceOpWithNewOp<CtrlOp>(op, newControls, op.getTargetsIn(),
                                        bodyCtrlOp.getBodyUnitary());

    return success();
  }
};

/**
 * @brief Remove control modifiers without controls.
 */
struct RemoveTrivialCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() > 0) {
      return failure();
    }

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto* clonedBody = rewriter.clone(*op.getBodyUnitary().getOperation());
    rewriter.replaceOp(op, clonedBody->getResults());

    return success();
  }
};

/**
 * @brief Inline controlled GPhase operations.
 */
struct CtrlInlineGPhase final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one positive control
    // Trivial case is handled by RemoveTrivialCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    auto gPhaseOp =
        llvm::dyn_cast<GPhaseOp>(op.getBodyUnitary().getOperation());
    if (!gPhaseOp) {
      return failure();
    }

    const auto controls = op.getControlsIn();
    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls.drop_back(), controls.back(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto pOp = POp::create(rewriter, op.getLoc(), targets[0],
                                 gPhaseOp.getTheta());
          return {pOp.getQubitOut()};
        });

    return success();
  }
};

/**
 * @brief Inline controlled identity operations.
 */
struct CtrlInlineId final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one positive control
    // Trivial case is handled by RemoveTrivialCtrl
    if (op.getNumControls() == 0 ||
        !llvm::isa<IdOp>(op.getBodyUnitary().getOperation())) {
      return failure();
    }

    auto idOp = IdOp::create(rewriter, op.getLoc(), op.getTargetsIn().front());

    rewriter.replaceOp(op, llvm::to_vector(llvm::concat<Value>(
                               op.getControlsIn(), idOp->getResults())));

    return success();
  }
};

} // namespace

UnitaryOpInterface CtrlOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody()->front());
}

size_t CtrlOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t CtrlOp::getNumTargets() { return getTargetsIn().size(); }

size_t CtrlOp::getNumControls() { return getControlsIn().size(); }

Value CtrlOp::getInputQubit(const size_t i) {
  const auto numControls = getNumControls();
  if (i < numControls) {
    return getControlsIn()[i];
  }
  if (numControls <= i && i < getNumQubits()) {
    return getTargetsIn()[i - numControls];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

std::invoke_result_t<decltype(llvm::concat<Value, ValueRange, ValueRange>),
                     ValueRange, ValueRange>
CtrlOp::getInputQubits() {
  return llvm::concat<Value>(ValueRange{getControlsIn()},
                             ValueRange{getTargetsIn()});
}

Value CtrlOp::getOutputQubit(const size_t i) {
  const auto numControls = getNumControls();
  if (i < numControls) {
    return getControlsOut()[i];
  }
  if (numControls <= i && i < getNumQubits()) {
    return getTargetsOut()[i - numControls];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

std::invoke_result_t<decltype(llvm::concat<Value, ValueRange, ValueRange>),
                     ValueRange, ValueRange>
CtrlOp::getOutputQubits() {
  return llvm::concat<Value>(ValueRange{getControlsOut()},
                             ValueRange{getTargetsOut()});
}

Value CtrlOp::getInputTarget(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Target index out of bounds");
  }
  return getTargetsIn()[i];
}

Value CtrlOp::getOutputTarget(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Target index out of bounds");
  }
  return getTargetsOut()[i];
}

Value CtrlOp::getInputControl(const size_t i) {
  if (i >= getNumControls()) {
    llvm::reportFatalUsageError("Control index out of bounds");
  }
  return getControlsIn()[i];
}

Value CtrlOp::getOutputControl(const size_t i) {
  if (i >= getNumControls()) {
    llvm::reportFatalUsageError("Control index out of bounds");
  }
  return getControlsOut()[i];
}

Value CtrlOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumControls(); ++i) {
    if (output == getControlsOut()[i]) {
      return getControlsIn()[i];
    }
  }
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getTargetsOut()[i]) {
      return getTargetsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value CtrlOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumControls(); ++i) {
    if (input == getControlsIn()[i]) {
      return getControlsOut()[i];
    }
  }
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getTargetsIn()[i]) {
      return getTargetsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

size_t CtrlOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value CtrlOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, ValueRange targets,
                   UnitaryOpInterface bodyUnitary) {
  build(odsBuilder, odsState, controls, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  // Create block arguments and map targets to them
  IRMapping mapping;
  const auto qubitType = QubitType::get(odsBuilder.getContext());
  for (const auto target : targets) {
    mapping.map(target, block.addArgument(qubitType, odsState.location));
  }

  // Clone the operation using the mapping
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  auto* op = odsBuilder.clone(*bodyUnitary.getOperation(), mapping);
  YieldOp::create(odsBuilder, odsState.location, op->getResults());
}

void CtrlOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, ValueRange controls,
    ValueRange targets,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> bodyBuilder) {
  build(odsBuilder, odsState, controls, targets);
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

LogicalResult CtrlOp::verify() {
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
  for (const auto& control : getControlsIn()) {
    if (!uniqueQubitsIn.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
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
  for (const auto& control : getControlsOut()) {
    if (!uniqueQubitsOut.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
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

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results
      .add<MergeNestedCtrl, RemoveTrivialCtrl, CtrlInlineGPhase, CtrlInlineId>(
          context);
}

std::optional<Eigen::MatrixXcd> CtrlOp::getUnitaryMatrix() {
  auto&& bodyUnitary = getBodyUnitary();
  if (!bodyUnitary) {
    return std::nullopt;
  }
  auto&& targetMatrix = bodyUnitary.getUnitaryMatrix<Eigen::MatrixXcd>();
  if (!targetMatrix) {
    return std::nullopt;
  }

  // get dimensions of target matrix
  const auto targetDim = targetMatrix->cols();
  assert(targetMatrix->cols() == targetMatrix->rows());

  // define dimensions and type of output matrix
  assert(getNumControls() < sizeof(unsigned long long) * 8);
  const auto dim = static_cast<int64_t>((1ULL << getNumControls()) * targetDim);

  // initialize result with identity
  Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Identity(dim, dim);

  // apply target matrix
  matrix.bottomRightCorner(targetDim, targetDim) = *targetMatrix;

  return matrix;
}
