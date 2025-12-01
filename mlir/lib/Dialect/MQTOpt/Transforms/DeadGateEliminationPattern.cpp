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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::opt {

/**
 * @brief This pattern eliminates Unitary operations that happen directly before
 * de-allocs.
 */
struct DeadGateEliminationPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit DeadGateEliminationPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern<UnitaryInterface>(context) {}

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    for (auto* user : op->getUsers()) {
      if (!mlir::isa<DeallocQubitOp>(user)) {
        return mlir::failure();
      }
    }

    for (auto outQubit : op.getAllOutQubits()) {
      rewriter.replaceAllUsesWith(outQubit, op.getCorrespondingInput(outQubit));
    }
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `DeadGateEliminationPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateDeadGateEliminationPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<DeadGateEliminationPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
