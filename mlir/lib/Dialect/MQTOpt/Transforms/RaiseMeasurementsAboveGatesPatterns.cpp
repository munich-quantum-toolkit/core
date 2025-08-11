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
    const auto qubitVariable = op.getInQubit();
    auto* predecessor = qubitVariable.getDefiningOp();
    const auto name = predecessor->getName().stripDialect().str();

    auto predecessorUnitary = mlir::dyn_cast<UnitaryInterface>(predecessor);

    if (!predecessorUnitary) {
      return mlir::failure();
    }

    if (DIAGONAL_GATES.count(name) == 1) {
      rewriter.replaceAllUsesWith(qubitVariable,
                                  predecessorUnitary.getInQubits().front());
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
struct RaiseMeasurementsAboveInvertingGatesPattern final
    : mlir::OpRewritePattern<MeasureOp> {

  explicit RaiseMeasurementsAboveInvertingGatesPattern(
      mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Checks if the users of the measured qubit are all resets.
   * @param op The MeasureOp to check.
   * @return True if all users are resets, false otherwise.
   */
  static bool checkUsersAreResets(MeasureOp op) {
    return llvm::all_of(op.getOutQubit().getUsers(), [](mlir::Operation* user) {
      return mlir::isa<ResetOp>(user);
    });
  }

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    if (!checkUsersAreResets(op)) {
      return mlir::failure(); // if the qubit is still used after the
                              // measurement, we cannot raise it above the gate.
    }
    const auto qubitVariable = op.getInQubit();
    auto* predecessor = qubitVariable.getDefiningOp();
    const auto name = predecessor->getName().stripDialect().str();

    auto predecessorUnitary = mlir::dyn_cast<UnitaryInterface>(predecessor);

    if (!predecessorUnitary) {
      return mlir::failure();
    }

    if (INVERTING_GATES.count(name) == 1 &&
        predecessorUnitary.getAllInQubits().size() == 1) {
      rewriter.replaceAllUsesWith(qubitVariable,
                                  predecessorUnitary.getInQubits().front());
      rewriter.eraseOp(predecessor);
      rewriter.setInsertionPointAfter(op);
      const mlir::Value trueConstant = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getBoolAttr(true));
      auto inversion = rewriter.create<mlir::arith::XOrIOp>(
          op.getLoc(), op.getOutBit(), trueConstant);
      // We need `replaceUsesWithIf` so that we can replace all uses except for
      // the one use that defines the inverted bit.
      rewriter.replaceUsesWithIf(op.getOutBit(), inversion.getResult(),
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
 * `RaiseMeasurementsAboveInvertingGatesPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateRaiseMeasurementsAboveGatesPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<RaiseMeasurementsAbovePhaseGatesPattern>(patterns.getContext());
  patterns.add<RaiseMeasurementsAboveInvertingGatesPattern>(
      patterns.getContext());
}

} // namespace mqt::ir::opt
