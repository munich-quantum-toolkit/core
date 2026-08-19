/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "IPOUtils.h"

#include <llvm/ADT/DenseSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>

#include <utility>

namespace mlir::qco {

func::FuncOp copyFunction(func::FuncOp funcOp, StringRef newName) {
  auto newFunc = funcOp.clone();
  newFunc.setName(newName.str());
  // Cloning carries the original's visibility over. A specialization is
  // internal to the stage that made it, and orphan cleanup only erases private
  // functions, so a public copy would be exported and never reclaimed.
  newFunc.setPrivate();
  return newFunc;
}

void eraseOrphanedSpecializations(SymbolTable& symbolTable,
                                  SmallVector<func::FuncOp>& candidates) {
  // Duplicates would leave dangling handles once the first copy is erased.
  SmallVector<func::FuncOp> unique;
  llvm::DenseSet<Operation*> seen;
  for (auto candidate : candidates) {
    if (seen.insert(candidate.getOperation()).second) {
      unique.emplace_back(candidate);
    }
  }
  candidates = std::move(unique);

  auto erasedAny = true;
  while (erasedAny) {
    erasedAny = false;
    SmallVector<func::FuncOp> remaining;
    for (auto candidate : candidates) {
      if (candidate.isPrivate() && SymbolTable::symbolKnownUseEmpty(
                                       candidate, candidate->getParentOp())) {
        symbolTable.erase(candidate);
        erasedAny = true;
        continue;
      }
      remaining.emplace_back(candidate);
    }
    candidates = std::move(remaining);
  }
}

} // namespace mlir::qco
