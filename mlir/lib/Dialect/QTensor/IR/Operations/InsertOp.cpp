/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>

using namespace mlir;
using namespace mlir::qtensor;

namespace {
/// Remove both operations so forwarding the tensor preserves linearity.
struct FoldInsertAfterExtract final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insert,
                                PatternRewriter& rewriter) const override {
    auto extract = insert.getScalar().getDefiningOp<ExtractOp>();
    if (!extract || insert.getDest() != extract.getOutTensor() ||
        !isEqualConstantIntOrValue(insert.getIndex(), extract.getIndex())) {
      return failure();
    }
    rewriter.replaceOp(insert, extract.getTensor());
    rewriter.eraseOp(extract);
    return success();
  }
};

/// Group commuting extracts before inserts in one traversal of the SSA chain.
struct CommuteInsertExtractChains final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insert,
                                PatternRewriter& rewriter) const override {
    auto extract = dyn_cast<ExtractOp>(*insert.getResult().getUsers().begin());
    if (!extract || insert->getBlock() != extract->getBlock() ||
        !insert->isBeforeInBlock(extract)) {
      return failure();
    }

    const auto insertIndex = getConstantIntValue(insert.getIndex());
    const auto extractIndex = getConstantIntValue(extract.getIndex());
    if (!insertIndex || !extractIndex || insertIndex == extractIndex) {
      return failure();
    }

    /// A bottom-up greedy walk may reach the last pair first. Include the
    /// commuting prefix too, rather than normalizing every suffix separately.
    auto firstInsert = insert;
    llvm::SmallDenseSet<int64_t> extractedIndices{*extractIndex};
    auto tensor = insert.getDest();
    while (auto* definingOp = tensor.getDefiningOp()) {
      if (definingOp->getBlock() != insert->getBlock() ||
          !definingOp->isBeforeInBlock(firstInsert)) {
        break;
      }
      if (auto previousExtract = dyn_cast<ExtractOp>(definingOp)) {
        const auto index = getConstantIntValue(previousExtract.getIndex());
        if (!index) {
          break;
        }
        extractedIndices.insert(*index);
        tensor = previousExtract.getTensor();
      } else if (auto previousInsert = dyn_cast<InsertOp>(definingOp)) {
        const auto index = getConstantIntValue(previousInsert.getIndex());
        if (!index || extractedIndices.contains(*index)) {
          break;
        }
        firstInsert = previousInsert;
        tensor = previousInsert.getDest();
      } else {
        break;
      }
    }

    SmallVector<InsertOp> inserts{firstInsert};
    SmallVector<ExtractOp> extracts;
    llvm::SmallDenseSet<int64_t> insertedIndices{
        *getConstantIntValue(firstInsert.getIndex()),
    };
    size_t numInsertsToMove = 0;
    tensor = firstInsert.getResult();
    Operation* previous = firstInsert;
    while (true) {
      auto* user = *tensor.user_begin();
      if (user->getBlock() != insert->getBlock() ||
          !previous->isBeforeInBlock(user)) {
        break;
      }
      if (auto nextInsert = dyn_cast<InsertOp>(user)) {
        const auto index = getConstantIntValue(nextInsert.getIndex());
        if (!index) {
          break;
        }
        insertedIndices.insert(*index);
        inserts.push_back(nextInsert);
        tensor = nextInsert.getResult();
      } else if (auto nextExtract = dyn_cast<ExtractOp>(user)) {
        const auto index = getConstantIntValue(nextExtract.getIndex());
        if (!index || insertedIndices.contains(*index)) {
          break;
        }
        extracts.push_back(nextExtract);
        numInsertsToMove = inserts.size();
        tensor = nextExtract.getOutTensor();
      } else {
        break;
      }
      previous = user;
    }
    if (extracts.empty()) {
      return failure();
    }

    /// Leave trailing inserts in place: their operands may follow the last
    /// extract. Earlier inserts' operands dominate their new positions.
    inserts.resize(numInsertsToMove);
    auto tail = extracts.back().getOutTensor();
    tensor = firstInsert.getDest();
    for (auto nextExtract : extracts) {
      rewriter.modifyOpInPlace(
          nextExtract, [&] { nextExtract.getTensorMutable().assign(tensor); });
      tensor = nextExtract.getOutTensor();
    }
    previous = extracts.back();
    for (auto nextInsert : inserts) {
      rewriter.moveOpAfter(nextInsert, previous);
      rewriter.modifyOpInPlace(
          nextInsert, [&] { nextInsert.getDestMutable().assign(tensor); });
      tensor = nextInsert.getResult();
      previous = nextInsert;
    }
    rewriter.replaceAllUsesExcept(tail, tensor, inserts.front());
    return success();
  }
};
} // namespace

LogicalResult InsertOp::verify() {
  auto dstDim = getDest().getType().getDimSize(0);
  auto index = getConstantIntValue(getIndex());

  if (index) {
    if (*index < 0) {
      return emitOpError("Index must be non-negative");
    }
    if (!ShapedType::isDynamic(dstDim) && *index >= dstDim) {
      return emitOpError("Index exceeds tensor dimension");
    }
  }

  return success();
}

void InsertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<FoldInsertAfterExtract, CommuteInsertExtractChains>(context);
}
