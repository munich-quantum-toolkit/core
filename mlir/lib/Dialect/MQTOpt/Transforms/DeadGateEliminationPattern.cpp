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

  /**
       * @brief Constructs a DeadGateEliminationPattern associated with the given MLIR context.
       */
      explicit DeadGateEliminationPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern<UnitaryInterface>(context) {}

  /**
   * @brief Eliminates a unitary operation whose outputs are only deallocated by replacing its outputs with the corresponding inputs and erasing the operation.
   *
   * @param op The unitary operation to match and potentially rewrite.
   * @param rewriter Pattern rewriter used to perform replacements and erase the operation.
   * @return mlir::LogicalResult `mlir::success()` if every user of `op` was a `DeallocQubitOp`, the outputs were replaced with their corresponding inputs, and `op` was erased; `mlir::failure()` if any user is not a `DeallocQubitOp`.
   */
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
 * @brief Registers the DeadGateEliminationPattern into a rewrite pattern set.
 *
 * @param patterns Rewrite pattern set to which the pattern will be added.
 */
void populateDeadGateEliminationPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<DeadGateEliminationPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt