/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::qtensor;

/// Find the allocation proving that an extracted constant slot is fresh.
static AllocOp findFreshAllocation(ExtractOp extract) {
  auto current = extract.getTensor();
  const auto extractIndex = getConstantIntValue(extract.getIndex());
  if (!extractIndex) {
    return {};
  }

  while (auto* definingOp = current.getDefiningOp()) {
    if (auto alloc = dyn_cast<AllocOp>(definingOp)) {
      return alloc;
    }

    if (auto nestedExtract = dyn_cast<ExtractOp>(definingOp)) {
      const auto index = getConstantIntValue(nestedExtract.getIndex());
      if (!index || *index == *extractIndex) {
        return {};
      }
      current = nestedExtract.getTensor();
      continue;
    }

    if (auto insert = dyn_cast<InsertOp>(definingOp)) {
      const auto index = getConstantIntValue(insert.getIndex());
      if (!index || *index == *extractIndex) {
        return {};
      }
      current = insert.getDest();
      continue;
    }

    return {};
  }

  return {};
}

namespace {
/// Remove a reset after extracting a freshly allocated qubit.
struct RemoveResetAfterExtract final : OpRewritePattern<qco::ResetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(qco::ResetOp reset,
                                PatternRewriter& rewriter) const override {
    auto extract = reset.getQubitIn().getDefiningOp<ExtractOp>();
    if (!extract) {
      return failure();
    }
    auto alloc = findFreshAllocation(extract);
    if (!alloc) {
      return failure();
    }
    if (extract.getTensor() == alloc.getResult()) {
      rewriter.replaceOp(reset, reset.getQubitIn());
      return success();
    }

    /// Reuse the allocation proof for every fresh slot in this linear chain.
    /// Folding them together avoids one backward traversal per reset.
    auto* resetBlock = reset->getBlock();
    llvm::SmallDenseSet<int64_t> accessed;
    auto tensor = alloc.getResult();
    while (true) {
      auto* user = *tensor.user_begin();
      if (auto nextExtract = dyn_cast<ExtractOp>(user)) {
        const auto index = getConstantIntValue(nextExtract.getIndex());
        if (!index) {
          break;
        }
        if (accessed.insert(*index).second) {
          if (auto nextReset =
                  dyn_cast<qco::ResetOp>(*nextExtract.getResult().user_begin());
              nextReset && nextReset->getBlock() == resetBlock) {
            rewriter.replaceOp(nextReset, nextReset.getQubitIn());
          }
        }
        tensor = nextExtract.getOutTensor();
        continue;
      }
      if (auto insert = dyn_cast<InsertOp>(user)) {
        const auto index = getConstantIntValue(insert.getIndex());
        if (!index) {
          break;
        }
        accessed.insert(*index);
        tensor = insert.getResult();
        continue;
      }
      break;
    }
    return success();
  }
};

/// Fold an insert followed immediately by an extract at the same index.
struct FoldExtractAfterInsertPattern final : OpRewritePattern<ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extract,
                                PatternRewriter& rewriter) const override {
    auto insert = extract.getTensor().getDefiningOp<InsertOp>();
    if (!insert ||
        !isEqualConstantIntOrValue(insert.getIndex(), extract.getIndex())) {
      return failure();
    }

    rewriter.replaceOp(extract, {insert.getDest(), insert.getScalar()});
    rewriter.eraseOp(insert);
    return success();
  }
};

} // namespace

LogicalResult ExtractOp::verify() {
  if (getOperation()->getParentOfType<qco::CtrlOp>() ||
      getOperation()->getParentOfType<qco::InvOp>() ||
      getOperation()->getParentOfType<qco::PowOp>()) {
    return emitOpError("cannot access a qubit tensor inside a QCO modifier");
  }

  auto tensorDim = getTensor().getType().getDimSize(0);
  auto index = getConstantIntValue(getIndex());

  if (index) {
    if (*index < 0) {
      return emitOpError("Index must be non-negative");
    }
    if (!ShapedType::isDynamic(tensorDim) && *index >= tensorDim) {
      return emitOpError("Index exceeds tensor dimension");
    }
  }
  return success();
}

void ExtractOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<FoldExtractAfterInsertPattern, RemoveResetAfterExtract>(context);
}
