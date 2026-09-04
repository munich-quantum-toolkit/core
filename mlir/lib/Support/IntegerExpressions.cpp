/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/IntegerExpressions.h"

#include <llvm/ADT/APInt.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/PatternMatch.h>

#include <cstdint>

using namespace mlir;

namespace {
/// Expand the operations absent from jeff using the same exact-width
/// arithmetic.
struct ExpandIntegerOperation final : RewritePattern {
  explicit ExpandIntegerOperation(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (!isa<math::CtPopOp, LLVM::FshlOp, LLVM::FshrOp>(op)) {
      return failure();
    }
    auto type = dyn_cast<IntegerType>(op->getResult(0).getType());
    if (!type || type.getWidth() > 64) {
      return failure();
    }
    const auto width = type.getWidth();
    auto loc = op->getLoc();
    if (width == 1) {
      rewriter.replaceOp(op, op->getOperand(isa<LLVM::FshrOp>(op) ? 1 : 0));
      return success();
    }
    if (isa<math::CtPopOp>(op)) {
      /// Count pairs, nibbles, then bytes. Multiplication sums the byte counts.
      /// This uses eight copies of the input when expanded to a source tree,
      /// rather than one copy per bit.
      unsigned wordWidth = 8;
      while (wordWidth < width) {
        wordWidth *= 2;
      }
      auto wordType = rewriter.getIntegerType(wordWidth);
      const auto constant = [&](uint64_t bits) -> Value {
        return arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getIntegerAttr(wordType,
                                    APInt(64, bits).trunc(wordWidth)));
      };
      const auto shift = [&](Value value, unsigned distance) -> Value {
        return arith::ShRUIOp::create(rewriter, loc, value, constant(distance));
      };
      const auto mask = [&](Value value, uint64_t bits) -> Value {
        return arith::AndIOp::create(rewriter, loc, value, constant(bits));
      };
      Value count = op->getOperand(0);
      if (width != wordWidth) {
        count = arith::ExtUIOp::create(rewriter, loc, wordType, count);
      }
      count = arith::SubIOp::create(
          rewriter, loc, count, mask(shift(count, 1), 0x5555555555555555ULL));
      count = arith::AddIOp::create(
          rewriter, loc, mask(count, 0x3333333333333333ULL),
          mask(shift(count, 2), 0x3333333333333333ULL));
      count = mask(arith::AddIOp::create(rewriter, loc, count, shift(count, 4)),
                   0x0F0F0F0F0F0F0F0FULL);
      if (wordWidth > 8) {
        count = arith::MulIOp::create(rewriter, loc, count,
                                      constant(0x0101010101010101ULL));
        count = shift(count, wordWidth - 8);
      }
      if (width != wordWidth) {
        count = arith::TruncIOp::create(rewriter, loc, type, count);
      }
      rewriter.replaceOp(op, count);
      return success();
    }
    auto size = arith::ConstantIntOp::create(rewriter, loc, type, width);
    auto amount =
        arith::RemUIOp::create(rewriter, loc, op->getOperand(2), size);
    auto inverse = arith::SubIOp::create(rewriter, loc, size, amount);
    const bool left = isa<LLVM::FshlOp>(op);
    auto first = mqt::buildZeroFillingShift(
        rewriter, loc, op->getOperand(0),
        left ? amount.getResult() : inverse.getResult(), true);
    auto second = mqt::buildZeroFillingShift(
        rewriter, loc, op->getOperand(1),
        left ? inverse.getResult() : amount.getResult(), false);
    rewriter.replaceOpWithNewOp<arith::OrIOp>(op, first, second);
    return success();
  }
};

} // namespace

void mlir::mqt::populateIntegerExpansionPatterns(RewritePatternSet& patterns) {
  patterns.add<ExpandIntegerOperation>(patterns.getContext());
}
