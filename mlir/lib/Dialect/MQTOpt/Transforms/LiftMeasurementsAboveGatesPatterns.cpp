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
 * @brief This pattern is responsible for lifting measurements above any phase
 * gates.
 */
struct LiftMeasurementsAbovePhaseGatesPattern final
    : mlir::OpRewritePattern<MeasureOp> {

  explicit LiftMeasurementsAbovePhaseGatesPattern(mlir::MLIRContext* context)
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
 * @brief This pattern is responsible for lifting measurements above any
 * non-phase gates.
 */
struct LiftMeasurementsAboveInvertingGatesPattern final
    : mlir::OpRewritePattern<MeasureOp> {

  explicit LiftMeasurementsAboveInvertingGatesPattern(
      mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Checks if the register index remains unused until de-allocation.
   *
   * If the specific index is not known, any access to the register will be
   * checked instead.
   *
   * @param reg The register to check.
   * @param index The index of the qubit in the register to check, if known.
   * @param indexPresent True if the index is known, false otherwise.
   * @return
   */
  static bool registerIndexRemainsUnused(const mlir::Value reg, size_t index,
                                         bool indexPresent) {
    const auto numUsers =
        std::distance(reg.getUsers().begin(), reg.getUsers().end());
    if (numUsers == 0) {
      // No more users, so the qubit is trivially unused.
      return true;
    }
    if (numUsers > 1) {
      llvm::report_fatal_error(
          "Register has more than one user. This should never happen.");
      return false;
    }
    const auto* user = *reg.getUsers().begin();

    if (mlir::isa<DeallocOp>(user)) {
      // De-allocs clear the register, so the qubit is unused.
      return true;
    }

    if (auto insertOp = mlir::dyn_cast<InsertOp>(user)) {
      // The value of the register changes but we still look for the same index.
      return registerIndexRemainsUnused(insertOp.getOutQreg(), index,
                                        indexPresent);
    }

    if (auto extractOp = mlir::dyn_cast<ExtractOp>(user)) {
      // A qubit is extracted again. This can lead to two scenarios:
      // 1) A different index is extracted: This changes the register value but
      // we do not care about it otherwise. 2) The searched index is extracted:
      // If the qubit gets reset, this guarantees it is unused. If it is
      // inserted again, we continue the search. 3) We don't know the searched
      // or extracted index: If the qubit is reset or inserted again, we
      // continue the search.

      if (extractOp.getIndexAttr().has_value() && indexPresent) {
        const size_t extractedIndex = extractOp.getIndexAttr().value();
        if (extractedIndex != index) {
          // Scenario 1
          return registerIndexRemainsUnused(extractOp.getOutQreg(), index,
                                            indexPresent);
        }
        // Scenario 2
        return outputQubitRemainsUnused(extractOp.getOutQubit());
      }
      // Scenario 3
      if (!outputQubitRemainsUnused(extractOp.getOutQubit())) {
        // This check can only give negative results
        return false;
      }
      return registerIndexRemainsUnused(extractOp.getOutQreg(), index,
                                        indexPresent);
    }

    return true;
  }

  /**
   * @brief Checks if the users of the measured qubit are all resets.
   * @param outQubit The output qubit to check.
   * @return True if all users are resets, false otherwise.
   */
  static bool outputQubitRemainsUnused(mlir::Value outQubit) {
    return llvm::all_of(outQubit.getUsers(), [](mlir::Operation* user) {
      if (mlir::isa<ResetOp>(user)) {
        return true;
      }
      auto insertOp = mlir::dyn_cast<InsertOp>(user);
      if (!insertOp) {
        return false;
      }
      const mlir::Value outQreg = insertOp.getOutQreg();
      if (insertOp.getIndexAttr().has_value()) {
        return registerIndexRemainsUnused(
            outQreg, insertOp.getIndexAttr().value(), true);
      }
      return registerIndexRemainsUnused(outQreg, 0, false);
    });
  }

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    if (!outputQubitRemainsUnused(op.getOutQubit())) {
      return mlir::failure(); // if the qubit is still used after the
                              // measurement, we cannot lift it above the gate.
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
 * `LiftMeasurementsAbovePhaseGatesPattern` and
 * `LiftMeasurementsAboveInvertingGatesPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateLiftMeasurementsAboveGatesPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<LiftMeasurementsAbovePhaseGatesPattern>(patterns.getContext());
  patterns.add<LiftMeasurementsAboveInvertingGatesPattern>(
      patterns.getContext());
}

} // namespace mqt::ir::opt
