/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::mqt {

/// Lower ranked copies to loops; QIR can retain contiguous copies for memcpy.
struct LowerMemRefCopy final : OpRewritePattern<memref::CopyOp> {
  explicit LowerMemRefCopy(MLIRContext* context, bool onlyStrided = false)
      : OpRewritePattern(context, /*benefit=*/2), onlyStrided_(onlyStrided) {}

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter& rewriter) const override {
    auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
    auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
    if (!sourceType || !targetType ||
        (onlyStrided_ &&
         memref::isStaticShapeAndContiguousRowMajor(sourceType) &&
         memref::isStaticShapeAndContiguousRowMajor(targetType))) {
      return failure();
    }
    const auto loc = copy.getLoc();
    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> sizes;
    for (int64_t dimension = 0; dimension < sourceType.getRank(); ++dimension) {
      sizes.push_back(
          memref::DimOp::create(rewriter, loc, copy.getSource(), dimension));
    }
    scf::buildLoopNest(
        rewriter, loc, SmallVector<Value>(sizes.size(), zero), sizes,
        SmallVector<Value>(sizes.size(), one),
        [&](OpBuilder& builder, Location location, ValueRange indices) {
          auto value = memref::LoadOp::create(builder, location,
                                              copy.getSource(), indices);
          memref::StoreOp::create(builder, location, value, copy.getTarget(),
                                  indices);
        });
    rewriter.eraseOp(copy);
    return success();
  }

private:
  bool onlyStrided_;
};

} // namespace mlir::mqt
