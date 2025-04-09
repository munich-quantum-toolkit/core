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

  mlir::LogicalResult match(ExtractOp op) const override {
    auto index = op.getIndex();
    if (!index) {
      return mlir::failure();
    }
    auto* definition = index.getDefiningOp();
    if (!mlir::isa<mlir::arith::ConstantOp>(definition)) {
      return mlir::failure();
    }
    return mlir::success();
  }

  void rewrite(ExtractOp op, mlir::PatternRewriter& rewriter) const override {
    auto index = op.getIndex();
    auto definition =
        mlir::cast<mlir::arith::ConstantOp>(index.getDefiningOp());
    auto value = mlir::cast<mlir::IntegerAttr>(definition.getValue()).getInt();
    rewriter.replaceOpWithNewOp<ExtractOp>(op, op.getOutQubit().getType(),
                                           op.getInQureg(), mlir::Value(),
                                           rewriter.getI64IntegerAttr(value));
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
