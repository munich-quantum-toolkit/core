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

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>

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

enum class AccessRelation : std::uint8_t { Disjoint, Overlap, Unknown };

/**
 * @brief Checks whether two index values are equivalent for matching.
 */
static bool areEquivalentIndices(Value lhs, Value rhs) {
  return getAsOpFoldResult(lhs) == getAsOpFoldResult(rhs);
}

/**
 * @brief Classify the relation between a scalar index and a slice range.
 */
static AccessRelation classifyIndexAndRange(Value index, Value offset,
                                            Value size) {
  if (areEquivalentIndices(index, offset)) {
    return AccessRelation::Overlap;
  }

  const auto indexValue = getConstantIntValue(index);
  const auto offsetValue = getConstantIntValue(offset);
  const auto sizeValue = getConstantIntValue(size);
  if (!indexValue || !offsetValue || !sizeValue) {
    return AccessRelation::Unknown;
  }

  if (*indexValue < *offsetValue || *indexValue >= *offsetValue + *sizeValue) {
    return AccessRelation::Disjoint;
  }
  return AccessRelation::Overlap;
}

/**
 * @brief Tensor-transforming ops in a chain that can commute with extracts.
 */
static bool isTensorChainOp(Operation* op) {
  return llvm::isa<InsertOp, ExtractOp, InsertSliceOp, ExtractSliceOp>(op);
}

/**
 * @brief Returns the tensor input of a tensor-transforming op.
 */
static Value getTensorChainInput(Operation* op) {
  if (auto insertOp = llvm::dyn_cast<InsertOp>(op)) {
    return insertOp.getDest();
  }
  if (auto extractOp = llvm::dyn_cast<ExtractOp>(op)) {
    return extractOp.getTensor();
  }
  if (auto insertSliceOp = llvm::dyn_cast<InsertSliceOp>(op)) {
    return insertSliceOp.getDest();
  }
  if (auto extractSliceOp = llvm::dyn_cast<ExtractSliceOp>(op)) {
    return extractSliceOp.getTensor();
  }
  return nullptr;
}

/**
 * @brief Returns the tensor output of a tensor-transforming op.
 */
static Value getTensorChainOutput(Operation* op) {
  if (auto insertOp = llvm::dyn_cast<InsertOp>(op)) {
    return insertOp.getResult();
  }
  if (auto extractOp = llvm::dyn_cast<ExtractOp>(op)) {
    return extractOp.getOutTensor();
  }
  if (auto insertSliceOp = llvm::dyn_cast<InsertSliceOp>(op)) {
    return insertSliceOp.getResult();
  }
  if (auto extractSliceOp = llvm::dyn_cast<ExtractSliceOp>(op)) {
    return extractSliceOp.getOutTensor();
  }
  return nullptr;
}

/**
 * @brief Rewire the tensor input of a tensor-transforming op.
 */
static void setTensorChainInput(Operation* op, Value tensor) {
  if (llvm::isa<InsertOp, InsertSliceOp>(op)) {
    op->setOperand(1, tensor);
    return;
  }
  if (llvm::isa<ExtractOp, ExtractSliceOp>(op)) {
    op->setOperand(0, tensor);
  }
}

/**
 * @brief If an ExtractOp consumes an InsertOp with the same index,
 * return the scalar and the destTensor from the InsertOp directly.
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
    Value currentTensor = extractOp.getTensor();
    InsertOp matchedInsertOp = nullptr;

    while (auto* definingOp = currentTensor.getDefiningOp()) {
      if (!isTensorChainOp(definingOp)) {
        break;
      }

      if (auto insertOp = llvm::dyn_cast<InsertOp>(definingOp)) {
        if (areEquivalentIndices(insertOp.getIndex(), extractOp.getIndex())) {
          matchedInsertOp = insertOp;
          break;
        }
      } else if (auto nestedExtractOp = llvm::dyn_cast<ExtractOp>(definingOp)) {
        if (areEquivalentIndices(nestedExtractOp.getIndex(),
                                 extractOp.getIndex())) {
          // Do not reorder reads from the same index.
          return failure();
        }
      } else if (auto insertSliceOp =
                     llvm::dyn_cast<InsertSliceOp>(definingOp)) {
        if (classifyIndexAndRange(
                extractOp.getIndex(), insertSliceOp.getOffset(),
                insertSliceOp.getSize()) != AccessRelation::Disjoint) {
          return failure();
        }
      } else if (auto extractSliceOp =
                     llvm::dyn_cast<ExtractSliceOp>(definingOp)) {
        if (classifyIndexAndRange(
                extractOp.getIndex(), extractSliceOp.getOffset(),
                extractSliceOp.getSize()) != AccessRelation::Disjoint) {
          return failure();
        }
      }

      traversedOps.push_back(definingOp);
      currentTensor = getTensorChainInput(definingOp);
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
