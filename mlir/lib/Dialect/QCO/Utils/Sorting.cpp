/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Utils/Sorting.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstddef>

namespace mlir::qco {

void reorderTopologically(Block& block, IRRewriter& rewriter) {
  Operation* const terminator = block.getTerminator();

  struct Dependencies {
    size_t pending = 0;
    SmallVector<Operation*, 2> successors;
  };
  const auto numOperations = llvm::range_size(block);
  DenseMap<Operation*, Dependencies> dependencies;
  dependencies.reserve(numOperations);
  DenseMap<Value, Operation*> lastEffect;

  /// Count repeated edges on both ends instead of maintaining deduplication
  /// sets.
  const auto addDependency = [&](Operation* predecessor, Operation* successor) {
    assert(predecessor != successor);
    ++dependencies[successor].pending;
    dependencies[predecessor].successors.push_back(successor);
  };

  for (Operation& op : block) {
    dependencies.try_emplace(&op);

    /// Preserve the order of effects on each SSA value, including nested
    /// effects.
    const auto effects = getEffectsRecursively(&op);
    if (effects) {
      for (const auto& effect : *effects) {
        auto value = effect.getValue();
        if (!value) {
          continue;
        }
        auto [it, inserted] = lastEffect.try_emplace(value, &op);
        if (!inserted && it->second != &op) {
          addDependency(it->second, &op);
          it->second = &op;
        }
      }
    }

    /// An effect edge need not lead back to the value's defining operation.
    /// Always retain SSA dependencies, including those of effect-bearing
    /// inputs.
    for (auto v : op.getOperands()) {
      Operation* def = v.getDefiningOp();
      if (def != nullptr && v.getParentBlock() == &block) {
        addDependency(def, &op);
      }
    }

    /// A nested capture makes its enclosing operation depend on the producer.
    for (Operation* user : op.getUsers()) {
      if (user->getBlock() == &block) {
        continue;
      }

      if (Operation* parent = block.findAncestorOpInBlock(*user);
          parent != nullptr) {
        addDependency(&op, parent);
      }
    }
  }

  assert(dependencies.size() == numOperations);

  SmallVector<Operation*> worklist;
  worklist.reserve(numOperations);
  for (Operation& op : block) {
    if (dependencies[&op].pending == 0) {
      worklist.emplace_back(&op);
    }
  }

  for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
    Operation* ready = worklist[cursor];

    rewriter.moveOpBefore(ready, &block, block.end());

    for (Operation* user : dependencies[ready].successors) {
      if (--dependencies[user].pending == 0) {
        worklist.push_back(user);
      }
    }
  }

  assert(worklist.size() == numOperations && "cyclic operation dependencies");

  rewriter.moveOpBefore(terminator, &block, block.end());
}
} // namespace mlir::qco
