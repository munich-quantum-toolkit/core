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
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

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
    // Require at least one control
    // Trivial case is handled by ReduceCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    // TODO: Relax this condition?
    if (op.getNumBodyUnitaries() != 1) {
      return failure();
    }
    auto innerCtrlOp = dyn_cast<CtrlOp>(op.getBodyUnitary(0).getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    auto outerTargets = op.getTargetsIn();
    auto outerControls = op.getControlsIn();
    auto innerTargets = innerCtrlOp.getTargetsIn();

    SmallVector<Value> controls;
    SmallVector<Value> targets;
    llvm::append_range(controls, outerControls);
    for (auto [arg, qubit] :
         llvm::zip_equal(op.getBody()->getArguments(), outerTargets)) {
      if (llvm::is_contained(innerTargets, arg)) {
        targets.push_back(qubit);
      } else {
        controls.push_back(qubit);
      }
    }

    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls, targets,
        [&](ValueRange targetArgs) -> SmallVector<Value> {
          auto* innerCtrlBody = innerCtrlOp.getBody();
          IRMapping mapping;
          utils::populateMapping(*innerCtrlBody, mapping, innerTargets,
                                 outerTargets, targets, targetArgs);
          for (auto& op : innerCtrlBody->without_terminator()) {
            rewriter.clone(op, mapping);
          }
          SmallVector<Value> yields;
          for (auto value : innerCtrlBody->getTerminator()->getOperands()) {
            yields.push_back(mapping.lookup(value));
          }
          return yields;
        });

    return success();
  }
};

/**
 * @brief Reduce controls for well-known gates.
 * @details Removes empty control ops and handles controlled IdOp, GPhaseOp and
 * BarrierOp.
 */
struct ReduceCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // TODO: Relax this condition?
    if (op.getNumBodyUnitaries() != 1) {
      return failure();
    }
    auto* innerOp = op.getBodyUnitary(0).getOperation();

    // Inline ops from empty control modifiers, IdOp and BarrierOp
    if (op.getNumControls() == 0 || isa<IdOp, BarrierOp>(innerOp)) {
      const auto numTargets = op.getNumTargets();
      auto outerTargets = op.getTargetsIn();
      SmallVector<Value> targets;
      for (auto target : innerOp->getOperands().take_front(numTargets)) {
        targets.push_back(
            utils::getValueFromBlockArgument(target, outerTargets));
      }

      rewriter.moveOpBefore(innerOp, op);
      innerOp->setOperands(0, numTargets, targets);
      rewriter.replaceAllUsesWith(op.getControlsOut(), op.getControlsIn());
      rewriter.replaceAllUsesWith(op.getTargetsOut(), innerOp->getResults());
      rewriter.eraseOp(op);
      return success();
    }

    // The remaining code explicitly handles GPhaseOp and nothing else
    auto gPhaseOp = dyn_cast<GPhaseOp>(innerOp);
    if (!gPhaseOp) {
      return failure();
    }

    // Special case for single control: replace with a single POp
    if (op.getNumControls() == 1) {
      rewriter.replaceOpWithNewOp<POp>(op, op.getInputControl(0),
                                       gPhaseOp.getTheta());
      return success();
    }

    // Adjust the segment sizes of the control and target operands
    const auto opSegmentsAttrName = CtrlOp::getOperandSegmentSizeAttr();
    auto segmentsAttr =
        op->getAttrOfType<DenseI32ArrayAttr>(opSegmentsAttrName);
    auto newSegments = DenseI32ArrayAttr::get(
        rewriter.getContext(), {segmentsAttr[0] - 1, segmentsAttr[1] + 1});
    op->setAttr(opSegmentsAttrName, newSegments);
    const auto opResultSegmentsAttrName = CtrlOp::getResultSegmentSizeAttr();
    op->setAttr(opResultSegmentsAttrName, newSegments);

    // Add a block argument for the target qubit
    auto arg = op.getBody()->addArgument(QubitType::get(rewriter.getContext()),
                                         op.getLoc());

    // Replace the current GPhaseOp with a PhaseOp
    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(gPhaseOp);
    auto pOp =
        POp::create(rewriter, gPhaseOp.getLoc(), arg, gPhaseOp.getTheta());

    // Add the results of the POp to the yield operation
    auto yieldOp = cast<YieldOp>(op.getBody()->back());
    yieldOp->setOperands(pOp->getResults());

    // Erase the GPhaseOp
    rewriter.eraseOp(gPhaseOp);

    return success();
  }
};

/**
 * @brief Erase control modifiers that do not have any body unitaries.
 */
struct EraseEmptyCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumBodyUnitaries() != 0) {
      return failure();
    }

    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};

} // namespace

size_t CtrlOp::getNumBodyUnitaries() {
  size_t count = 0;
  for (auto& op : *getBody()) {
    if (isa<UnitaryOpInterface>(op)) {
      count++;
    }
  }
  return count;
}

UnitaryOpInterface CtrlOp::getBodyUnitary(const size_t i) {
  size_t count = 0;
  for (auto& op : *getBody()) {
    if (isa<UnitaryOpInterface>(op)) {
      if (count == i) {
        return cast<UnitaryOpInterface>(op);
      }
      count++;
    }
  }
  llvm::reportFatalUsageError("Unitary index out of bounds");
}

Value CtrlOp::getInputQubit(const size_t i) {
  const auto numControls = getNumControls();
  if (i < numControls) {
    return getControlsIn()[i];
  }
  if (numControls <= i && i < getNumQubits()) {
    return getTargetsIn()[i - numControls];
  }
  llvm::reportFatalUsageError("Qubit index out of bounds");
}

Value CtrlOp::getOutputQubit(const size_t i) {
  const auto numControls = getNumControls();
  if (i < numControls) {
    return getControlsOut()[i];
  }
  if (numControls <= i && i < getNumQubits()) {
    return getTargetsOut()[i - numControls];
  }
  llvm::reportFatalUsageError("Qubit index out of bounds");
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

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, ValueRange targets,
                   function_ref<SmallVector<Value>(ValueRange)> bodyBuilder) {
  build(odsBuilder, odsState, controls, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  auto qubitType = QubitType::get(odsBuilder.getContext());
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
  if (!isa<YieldOp>(block.back())) {
    return emitOpError(
        "last operation in body region must be a yield operation");
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

  SmallPtrSet<Value, 4> uniqueQubitsOut;
  for (const auto& control : getControlsOut()) {
    if (!uniqueQubitsOut.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
  for (size_t i = 0; i < numTargets; i++) {
    if (!uniqueQubitsOut.insert(block.back().getOperand(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  return success();
}

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<MergeNestedCtrl, ReduceCtrl, EraseEmptyCtrl>(context);
}

std::optional<Eigen::MatrixXcd> CtrlOp::getUnitaryMatrix() {
  // TODO: Relax this condition
  if (getNumBodyUnitaries() != 1) {
    return std::nullopt;
  }

  auto bodyUnitary = getBodyUnitary(0);
  if (!bodyUnitary) {
    return std::nullopt;
  }
  auto targetMatrix = bodyUnitary.getUnitaryMatrix<Eigen::MatrixXcd>();
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
