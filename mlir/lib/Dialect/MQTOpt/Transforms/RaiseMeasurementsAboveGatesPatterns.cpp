/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <unordered_set>
#include <vector>

namespace mqt::ir::opt {

static const std::unordered_set<std::string> INVERTING_GATES = {"x", "y"};
static const std::unordered_set<std::string> DIAGONAL_GATES = {
    "i", "z", "s", "sdg", "t", "tdg", "p", "rz"};

/**
 * @brief This pattern is responsible for raising measurements above any phase
 * gates.
 */
struct RaiseMeasurementsAbovePhaseGatesPattern final
    : mlir::OpRewritePattern<MeasureOp> {

  explicit RaiseMeasurementsAbovePhaseGatesPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    if (op.getInQubits().size() != 1) {
      return mlir::failure(); // only support single-qubit measurements.
    }
    const auto qubitVariable = op.getInQubits().front();
    auto* predecessor = qubitVariable.getDefiningOp();
    const auto name = predecessor->getName().stripDialect().str();

    auto predecessorOp = mlir::dyn_cast<UnitaryInterface>(predecessor);

    if (!predecessorOp) {
      return mlir::failure();
    }

    if (DIAGONAL_GATES.count(name) == 1) {
      rewriter.replaceAllUsesWith(qubitVariable,
                                  predecessorOp.getInQubits().front());
      rewriter.eraseOp(predecessor);
      return mlir::success();
    }

    return mlir::failure();
  }
};

/**
 * @brief This pattern is responsible for raising measurements above any
 * non-phase gates.
 */
struct RaiseMeasurementsAboveOtherGatesPattern final
    : mlir::OpRewritePattern<MeasureOp> {

  explicit RaiseMeasurementsAboveOtherGatesPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    if (op.getInQubits().size() != 1) {
      return mlir::failure(); // only support single-qubit measurements.
    }
    const auto qubitVariable = op.getInQubits().front();
    auto* predecessor = qubitVariable.getDefiningOp();
    const auto name = predecessor->getName().stripDialect().str();

    auto predecessorOp = mlir::dyn_cast<UnitaryInterface>(predecessor);

    if (!predecessorOp) {
      return mlir::failure();
    }

    if (INVERTING_GATES.count(name) == 1 &&
        predecessorOp.getAllInQubits().size() == 1) {
      rewriter.replaceAllUsesWith(qubitVariable,
                                  predecessorOp.getInQubits().front());
      rewriter.eraseOp(predecessor);
      rewriter.setInsertionPointAfter(op);
      const mlir::Value trueConstant = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getBoolAttr(true));
      auto inversion = rewriter.create<mlir::arith::XOrIOp>(
          op.getLoc(), op.getOutBits().front(), trueConstant);
      // We need `replaceUsesWithIf` so that we can replace all uses except for
      // the one use that defines the inverted bit.
      rewriter.replaceUsesWithIf(op.getOutBits().front(), inversion.getResult(),
                                 [&](mlir::OpOperand& operand) {
                                   return operand.getOwner() != inversion;
                                 });
      return mlir::success();
    }

    return mlir::failure();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `RaiseMeasurementsAbovePhaseGatesPattern` and
 * `RaiseMeasurementsAboveOtherGatesPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateRaiseMeasurementsAboveGatesPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<RaiseMeasurementsAbovePhaseGatesPattern>(patterns.getContext());
  patterns.add<RaiseMeasurementsAboveOtherGatesPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
