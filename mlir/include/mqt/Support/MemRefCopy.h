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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::mqt {

/// Lower ranked copies to loops; QIR can retain contiguous copies for memcpy.
struct LowerMemRefCopy final : OpRewritePattern<memref::CopyOp> {
  explicit LowerMemRefCopy(MLIRContext* context, bool onlyStrided = false)
      : OpRewritePattern(context, /*benefit=*/2), onlyStrided_(onlyStrided) {}

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter& rewriter) const override;

private:
  bool onlyStrided_;
};

} // namespace mlir::mqt
