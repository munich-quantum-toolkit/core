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

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::opt {

/**
 * @brief This pattern attempts to remove SWAP gates by re-ordering qubits.
 */
struct ElidePermutationsPattern final : mlir::OpRewritePattern<SWAPOp> {

  explicit ElidePermutationsPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(SWAPOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.isControlled()) {
      return mlir::failure();
    }

    auto inQubits = op.getInQubits();
    assert(inQubits.size() == 2);

    rewriter.replaceAllOpUsesWith(op, {inQubits[1], inQubits[0]});

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

/**
 * @brief Populates the given pattern set with the `ElidePermutationsPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateElidePermutationsPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<ElidePermutationsPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
