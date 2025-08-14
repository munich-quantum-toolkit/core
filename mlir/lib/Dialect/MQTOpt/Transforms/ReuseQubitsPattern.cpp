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

/**
 * @brief This pattern is responsible applying qubit reuse.
 */
struct ReuseQubitsPattern final : mlir::OpRewritePattern<MeasureOp> {

  explicit ReuseQubitsPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    return mlir::failure();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `ReuseQubitsPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateReuseQubitsPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<ReuseQubitsPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
