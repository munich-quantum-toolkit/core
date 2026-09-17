/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QC/IR/QCOps.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstddef>

using namespace mlir;
using namespace mlir::qc;

Value BarrierOp::getTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubits()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

LogicalResult MaskedBarrierOp::verify() {
  if (getQubits().size() != getMasks().size()) {
    return emitOpError("requires one mask for each qubit");
  }
  llvm::SmallDenseSet<Value> seen;
  for (auto qubit : getQubits()) {
    if (!seen.insert(qubit).second) {
      return emitOpError("requires distinct qubit operands");
    }
  }
  return success();
}

namespace {
struct FoldMaskedBarrier final : OpRewritePattern<MaskedBarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaskedBarrierOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<Value> selected;
    SmallVector<Value> masks;
    bool allConstant = true;
    for (auto [qubit, mask] : llvm::zip_equal(op.getQubits(), op.getMasks())) {
      APInt value;
      const bool constant = matchPattern(mask, m_ConstantInt(&value));
      allConstant &= constant;
      if (!constant || !value.isZero()) {
        selected.push_back(qubit);
        masks.push_back(mask);
      }
    }
    if (!allConstant) {
      if (selected.size() == op.getQubits().size()) {
        return failure();
      }
      MaskedBarrierOp::create(rewriter, op.getLoc(), selected, masks);
    } else if (!selected.empty()) {
      BarrierOp::create(rewriter, op.getLoc(), selected);
    }
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void MaskedBarrierOp::getCanonicalizationPatterns(RewritePatternSet& patterns,
                                                  MLIRContext* context) {
  patterns.add<FoldMaskedBarrier>(context);
}
