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

namespace mlir::qtensor {

/**
 * @brief Checks whether two index values are equivalent for matching.
 */
inline bool areEquivalentIndices(Value lhs, Value rhs) {
  return getAsOpFoldResult(lhs) == getAsOpFoldResult(rhs);
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
