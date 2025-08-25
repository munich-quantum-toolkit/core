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
#include "mlir/IR/BuiltinAttributes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
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
    bool isControlledSwap = !op.getAllCtrlInQubits().empty();
    if (isControlledSwap) {
      return mlir::failure();
    }

    auto inQubits = op.getInQubits();
    auto outQubits = op.getOutQubits();

    std::vector<mlir::Value> swappedQubits;
    for (auto&& reversed : llvm::reverse(inQubits)) {
      swappedQubits.push_back(reversed);
    }

    rewriter.replaceAllOpUsesWith(op, swappedQubits);

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
