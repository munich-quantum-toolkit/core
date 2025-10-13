/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/TestUtils.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>

namespace mlir {

bool modulesAreEquivalent(ModuleOp lhs, ModuleOp rhs) {
  // First verify both modules are valid
  if (failed(verify(lhs)) || failed(verify(rhs))) {
    return false;
  }

  // Compare module attributes
  if (lhs->getAttrs() != rhs->getAttrs()) {
    return false;
  }

  for (auto [lhsOp, rhsOp] : llvm::zip(lhs.getOps(), rhs.getOps())) {
    if (!operationsAreEquivalent(&lhsOp, &rhsOp)) {
      return false;
    }
  }

  return true;
}

bool operationsAreEquivalent(Operation* lhs, Operation* rhs) {
  // Check operation name
  if (lhs->getName() != rhs->getName()) {
    return false;
  }

  // Check attributes (skip for LLVMFuncOp)
  if (!llvm::isa<LLVM::LLVMFuncOp>(lhs) && !llvm::isa<LLVM::LLVMFuncOp>(rhs)) {
    if (lhs->getAttrs() != rhs->getAttrs()) {
      return false;
    }
  }

  // Check number of operands and results
  if (lhs->getNumOperands() != rhs->getNumOperands() ||
      lhs->getNumResults() != rhs->getNumResults()) {
    return false;
  }

  // Check result types
  for (auto [lhsResult, rhsResult] :
       llvm::zip(lhs->getResults(), rhs->getResults())) {
    if (lhsResult.getType() != rhsResult.getType()) {
      return false;
    }
  }

  // Check operand types (not values, as SSA values differ)
  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    if (lhsOperand.getType() != rhsOperand.getType()) {
      return false;
    }
  }

  // Check regions
  if (lhs->getNumRegions() != rhs->getNumRegions()) {
    return false;
  }

  for (auto [lhsRegion, rhsRegion] :
       llvm::zip(lhs->getRegions(), rhs->getRegions())) {
    if (!regionsAreEquivalent(&lhsRegion, &rhsRegion)) {
      return false;
    }
  }

  return true;
}

bool regionsAreEquivalent(Region* lhs, Region* rhs) {
  // Check number of blocks
  if (lhs->getBlocks().size() != rhs->getBlocks().size()) {
    return false;
  }

  auto lhsBlockIt = lhs->begin();
  auto rhsBlockIt = rhs->begin();

  while (lhsBlockIt != lhs->end() && rhsBlockIt != rhs->end()) {
    if (!blocksAreEquivalent(&(*lhsBlockIt), &(*rhsBlockIt))) {
      return false;
    }
    ++lhsBlockIt;
    ++rhsBlockIt;
  }

  return true;
}

bool blocksAreEquivalent(Block* lhs, Block* rhs) {
  // Check number of arguments
  if (lhs->getNumArguments() != rhs->getNumArguments()) {
    return false;
  }

  // Check argument types
  for (auto [lhsArg, rhsArg] :
       llvm::zip(lhs->getArguments(), rhs->getArguments())) {
    if (lhsArg.getType() != rhsArg.getType()) {
      return false;
    }
  }

  // Check operations in the block
  auto lhsOpIt = lhs->begin();
  auto rhsOpIt = rhs->begin();

  while (lhsOpIt != lhs->end() && rhsOpIt != rhs->end()) {
    if (!operationsAreEquivalent(&(*lhsOpIt), &(*rhsOpIt))) {
      return false;
    }
    ++lhsOpIt;
    ++rhsOpIt;
  }

  return lhsOpIt == lhs->end() && rhsOpIt == rhs->end();
}

} // namespace mlir
