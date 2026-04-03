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

#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Value.h>

#include <cstdint>

namespace mlir::qtensor {

/**
 * @brief Relation of two tensor accesses.
 */
enum class AccessRelation : std::uint8_t { Disjoint, Overlap, Equal, Unknown };

/**
 * @brief Checks whether two index values are equivalent for matching.
 */
inline bool areEquivalentIndices(Value lhs, Value rhs) {
  return getAsOpFoldResult(lhs) == getAsOpFoldResult(rhs);
}

/**
 * @brief Checks whether two slice ranges are equivalent for matching.
 */
inline bool areEquivalentRanges(Value lhsOffset, Value lhsSize, Value rhsOffset,
                                Value rhsSize) {
  return areEquivalentIndices(lhsOffset, rhsOffset) &&
         areEquivalentIndices(lhsSize, rhsSize);
}

/**
 * @brief Classify the relation between a scalar index and a slice range.
 */
inline AccessRelation classifyIndexAndRange(Value index, Value offset,
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
 * @brief Classify the relation between two slice ranges.
 */
inline AccessRelation classifyRanges(Value lhsOffset, Value lhsSize,
                                     Value rhsOffset, Value rhsSize) {
  if (areEquivalentRanges(lhsOffset, lhsSize, rhsOffset, rhsSize)) {
    return AccessRelation::Equal;
  }

  const auto lhsOffsetValue = getConstantIntValue(lhsOffset);
  const auto lhsSizeValue = getConstantIntValue(lhsSize);
  const auto rhsOffsetValue = getConstantIntValue(rhsOffset);
  const auto rhsSizeValue = getConstantIntValue(rhsSize);
  if (!lhsOffsetValue || !lhsSizeValue || !rhsOffsetValue || !rhsSizeValue) {
    if (areEquivalentIndices(lhsOffset, rhsOffset)) {
      return AccessRelation::Overlap;
    }
    return AccessRelation::Unknown;
  }

  const auto lhsEnd = *lhsOffsetValue + *lhsSizeValue;
  const auto rhsEnd = *rhsOffsetValue + *rhsSizeValue;
  if (lhsEnd <= *rhsOffsetValue || rhsEnd <= *lhsOffsetValue) {
    return AccessRelation::Disjoint;
  }
  return AccessRelation::Overlap;
}

/**
 * @brief Tensor-transforming ops in a chain that can commute by index/range.
 */
inline bool isTensorChainOp(Operation* op) {
  return llvm::isa<InsertOp, ExtractOp, InsertSliceOp, ExtractSliceOp>(op);
}

/**
 * @brief Returns the tensor input of a tensor-transforming op.
 */
inline Value getTensorChainInput(Operation* op) {
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
inline Value getTensorChainOutput(Operation* op) {
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
inline void setTensorChainInput(Operation* op, Value tensor) {
  if (llvm::isa<InsertOp, InsertSliceOp>(op)) {
    op->setOperand(1, tensor);
    return;
  }
  if (llvm::isa<ExtractOp, ExtractSliceOp>(op)) {
    op->setOperand(0, tensor);
  }
}

} // namespace mlir::qtensor
