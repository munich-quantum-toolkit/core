/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/FluxUtils.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <variant>

using namespace mlir;
using namespace mlir::flux;

namespace {

/**
 * @brief Merge subsequent P operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentP final : OpRewritePattern<POp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(POp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetOneParameter(op, rewriter);
  }
};

} // namespace

void POp::build(OpBuilder& odsBuilder, OperationState& odsState,
                const Value qubitIn, const std::variant<double, Value>& theta) {
  Value thetaOperand = nullptr;
  if (std::holds_alternative<double>(theta)) {
    thetaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(theta)));
  } else {
    thetaOperand = std::get<Value>(theta);
  }
  build(odsBuilder, odsState, qubitIn, thetaOperand);
}

void POp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<MergeSubsequentP>(context);
}
