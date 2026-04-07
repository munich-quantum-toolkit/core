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

/**
 * Determine whether two index Values represent the same constant integer.
 *
 * @param lhs The first index Value to compare.
 * @param rhs The second index Value to compare.
 * @returns `true` if both `lhs` and `rhs` are constant integers and have equal
 *          values, `false` otherwise.
 */
/**
 * Check if an operation is a QTensor scalar insert or extract.
 *
 * @param op The operation to test.
 * @returns `true` if `op` is an `InsertOp` or `ExtractOp`, `false` otherwise.
 */
/**
 * Get the tensor used as the input/source for a tensor-chain operation.
 *
 * @param op The tensor-transforming operation.
 * @returns The tensor input `Value` (`InsertOp::getDest()` or
 *          `ExtractOp::getTensor()`), or `nullptr` if `op` is not a supported
 *          tensor-chain operation.
 */
/**
 * Get the tensor produced as the output of a tensor-chain operation.
 *
 * @param op The tensor-transforming operation.
 * @returns The tensor output `Value` (`InsertOp::getResult()` or
 *          `ExtractOp::getOutTensor()`), or `nullptr` if `op` is not a
 *          supported tensor-chain operation.
 */
/**
 * Set the tensor input operand on a tensor-chain operation.
 *
 * @param op The tensor-transforming operation to modify.
 * @param tensor The tensor `Value` to set as the new chain input.
 *
 * This sets operand index 1 for `InsertOp` and operand index 0 for `ExtractOp`.
 * If `op` is not a supported tensor-chain operation, no change is made.
 */
namespace mlir::qtensor {

/**
 * @brief Checks whether two index values are equivalent.
 *
 * @details This is a conservative check that returns true if both indices are
 * constant integers with the same value. It returns false if either index is
 * non-constant or if they have different constant values. Note that this means
 * that some equivalent indices may be considered non-equivalent by this
 * function, but no non-equivalent indices will be considered equivalent.
 */
inline bool areEquivalentIndices(Value lhs, Value rhs) {
  auto lhsValue = getConstantIntValue(lhs);
  auto rhsValue = getConstantIntValue(rhs);
  if (!lhsValue || !rhsValue) {
    return false;
  }
  return *lhsValue == *rhsValue;
}

/**
 * @brief Tensor-transforming ops in a scalar extract/insert chain.
 */
inline bool isTensorChainOp(Operation* op) {
  return llvm::isa<InsertOp, ExtractOp>(op);
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
  return nullptr;
}

/**
 * @brief Rewire the tensor input of a tensor-transforming op.
 */
inline void setTensorChainInput(Operation* op, Value tensor) {
  if (llvm::isa<InsertOp>(op)) {
    op->setOperand(1, tensor);
    return;
  }
  if (llvm::isa<ExtractOp>(op)) {
    op->setOperand(0, tensor);
  }
}

} // namespace mlir::qtensor
