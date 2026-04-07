/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorUtils.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

LogicalResult ExtractOp::verify() {
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

/**
 * @brief Check if a `qtensor.extract` operation reads from a `qtensor.insert`
 * operation.
 */
static InsertOp foldExtractAfterInsert(ExtractOp extractOp) {
  auto insertOp = extractOp.getTensor().getDefiningOp<InsertOp>();
  if (!insertOp) {
    return nullptr;
  }

  if (!areEquivalentIndices(insertOp.getIndex(), extractOp.getIndex())) {
    return nullptr;
  }

  return insertOp;
}

LogicalResult ExtractOp::fold(FoldAdaptor /*adaptor*/,
                              SmallVectorImpl<OpFoldResult>& results) {
  if (auto insertOp = foldExtractAfterInsert(*this)) {
    results.emplace_back(insertOp.getDest());
    results.emplace_back(insertOp.getScalar());
    return success();
  }

  return failure();
}

namespace {

/**
 * @brief Remove matching insert-extract pairs through commuting disjoint
 * tensor-chain operations.
 */
struct RemoveInsertExtractPair final : OpRewritePattern<ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extractOp,
                                PatternRewriter& rewriter) const override {
    llvm::SmallVector<Operation*> traversedOps;
    Value current = extractOp.getTensor();
    InsertOp matchedInsertOp = nullptr;

    auto extractIndex = extractOp.getIndex();
    if (!getConstantIntValue(extractIndex)) {
      return failure();
    }

    while (auto* definingOp = current.getDefiningOp()) {
      if (!isTensorChainOp(definingOp)) {
        break;
      }

      if (auto insertOp = llvm::dyn_cast<InsertOp>(definingOp)) {
        auto insertIndex = insertOp.getIndex();
        if (!getConstantIntValue(insertIndex)) {
          return failure();
        }
        if (areEquivalentIndices(insertIndex, extractIndex)) {
          matchedInsertOp = insertOp;
          break;
        }
      } else if (auto nestedExtractOp = llvm::dyn_cast<ExtractOp>(definingOp)) {
        auto nestedExtractIndex = nestedExtractOp.getIndex();
        if (!getConstantIntValue(nestedExtractIndex)) {
          return failure();
        }
        // Do not reorder reads from the same index
        if (areEquivalentIndices(extractIndex, nestedExtractIndex)) {
          return failure();
        }
      } else {
        return failure();
      }

      traversedOps.push_back(definingOp);
      current = getTensorChainInput(definingOp);
    }

    if (!matchedInsertOp) {
      return failure();
    }

    Value outTensor = matchedInsertOp.getDest();
    if (!traversedOps.empty()) {
      Operation* oldestCommutedOp = traversedOps.back();
      rewriter.modifyOpInPlace(oldestCommutedOp, [&]() {
        setTensorChainInput(oldestCommutedOp, matchedInsertOp.getDest());
      });
      outTensor = getTensorChainOutput(traversedOps.front());
      if (!outTensor) {
        return failure();
      }
    }

    rewriter.replaceOp(extractOp, {outTensor, matchedInsertOp.getScalar()});
    return success();
  }
};

} // namespace

void ExtractOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<RemoveInsertExtractPair>(context);
}
