/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"
#include "mlir/Dialect/MQTDyn/Transforms/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::dyn {

/**
 * @brief This pattern attempts to fold constants of `mqtdyn.extractQubit`
 * operations.
 */
struct ConstantFoldExtractQubitPattern final
    : mlir::OpRewritePattern<ExtractOp> {

  explicit ConstantFoldExtractQubitPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(ExtractOp op,
                  mlir::PatternRewriter& rewriter) const override {
    const auto index = op.getIndex();
    if (!index) {
      return mlir::failure();
    }

    auto* definition = index.getDefiningOp();
    if (!mlir::isa<mlir::arith::ConstantOp>(definition)) {
      return mlir::failure();
    }

    auto constant = mlir::cast<mlir::arith::ConstantOp>(definition);
    const auto value =
        mlir::cast<mlir::IntegerAttr>(constant.getValue()).getInt();

    rewriter.replaceOpWithNewOp<ExtractOp>(op, op.getOutQubit().getType(),
                                           op.getInQreg(), mlir::Value(),
                                           rewriter.getI64IntegerAttr(value));
    return mlir::success();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `ConstantFoldExtractQubitPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateConstantFoldExtractQubitPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<ConstantFoldExtractQubitPattern>(patterns.getContext());
}

} // namespace mqt::ir::dyn
