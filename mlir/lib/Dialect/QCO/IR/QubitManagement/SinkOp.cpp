/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Check if given quantum operation is unused (i.e., only used by
 * sinks and no memory effects).
 *
 * @param op The operation to check.
 * @return bool True if the operation is unused, false otherwise.
 */
inline bool checkDeadGate(Operation* op) {
  if (!isMemoryEffectFree(op)) {
    // This ignores operations and regions that have children with memory
    // effects, which should never be considered dead.
    return false;
  }
  return llvm::all_of(op->getUsers(),
                      [](Operation* user) { return isa<SinkOp>(user); });
}

/**
 * @brief Remove matching alloc/static and sink pairs without operations
 * between them.
 */
struct RemoveAllocSinkPair final : OpRewritePattern<SinkOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SinkOp op,
                                PatternRewriter& rewriter) const override {
    auto* defOp = op.getQubit().getDefiningOp();
    if (!isa_and_nonnull<AllocOp, StaticOp>(defOp)) {
      return failure();
    }

    rewriter.eraseOp(op);
    rewriter.eraseOp(defOp);
    return success();
  }
};

/**
 * @brief Remove dead gates.
 */
struct DeadGateElimination final : public OpRewritePattern<SinkOp> {

  explicit DeadGateElimination(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(SinkOp op,
                                PatternRewriter& rewriter) const override {
    Value currentValue = op.getQubit();
    auto* currentOp = currentValue.getDefiningOp();
    bool success = false;
    while (currentOp != nullptr) {
      if (!checkDeadGate(currentOp)) {
        break;
      }

      currentValue =
          TypeSwitch<Operation*, Value>(currentOp)
              .Case<MeasureOp>([&](auto measureOp) {
                rewriter.replaceAllUsesWith(measureOp.getQubitOut(),
                                            measureOp.getQubitIn());
                rewriter.eraseOp(measureOp);
                return measureOp.getQubitIn();
              })
              .Case<IfOp>([&](auto ifOp) {
                auto newValue = ifOp.getInputForOutput(currentValue);
                rewriter.replaceOp(ifOp, ifOp.getQubits());
                return newValue;
              })
              .Case<ResetOp>([&](auto resetOp) {
                rewriter.replaceOp(resetOp, resetOp->getOperands());
                return resetOp.getQubitIn();
              })
              .Case<UnitaryOpInterface>([&](auto unitaryOp) {
                auto newValue = unitaryOp.getInputForOutput(currentValue);
                rewriter.replaceOp(unitaryOp, unitaryOp.getInputQubits());
                return newValue;
              })
              .Default([&](auto) { return nullptr; });

      if (currentValue == nullptr) {
        break;
      }
      currentOp = currentValue.getDefiningOp();
      success = true;
    }

    return llvm::success(success);
  }
};

} // namespace

void SinkOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<RemoveAllocSinkPair, DeadGateElimination>(context);
}
