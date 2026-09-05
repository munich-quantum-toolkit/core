/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/Sorting.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;

/// Find the nearest neighbour in a given block.
static Operation* findParentInBlock(Operation* op, Block& block) {
  Operation* parent = op->getParentOp();
  while (parent != nullptr && parent->getBlock() != &block) {
    parent = parent->getParentOp();
  }
  return parent;
}

/// Return the vector of locations for each block argument.
static SmallVector<Location> getArgumentLocs(Block& block) {
  return to_vector(map_range(block.getArguments(),
                             [](BlockArgument& arg) { return arg.getLoc(); }));
}

namespace mlir::qco {
void reorderTopologically(Block& block, IRRewriter& rewriter) {
  Operation* const terminator = block.getTerminator();

  // Construct unresolved map: The dependencies of each operation.

  DenseMap<Operation*, size_t> inDegree;
  DenseMap<Operation*, size_t> blockOrder;
  DenseMap<Operation*, llvm::SmallSetVector<Operation*, 16>> successors;
  DenseMap<Operation*, llvm::SmallDenseSet<Operation*, 16>> predecessors;

  const auto addDependency = [&](Operation* predecessor, Operation* successor) {
    if (predecessor == successor ||
        !predecessors[successor].insert(predecessor).second) {
      return;
    }
    ++inDegree[successor];
    successors[predecessor].insert(successor);
  };

  DenseMap<Value, Operation*> lastEffectForValue;

  for (Operation& op : block) {

    // Collect the in-block dependencies of the current operation.

    successors.try_emplace(&op);
    predecessors.try_emplace(&op);
    inDegree.try_emplace(&op, 0);
    blockOrder.try_emplace(&op, blockOrder.size());

    for (Value v : op.getOperands()) {
      Operation* def = v.getDefiningOp();
      if (def != nullptr && v.getParentBlock() == &block) {
        addDependency(def, &op);
      }
    }

    // For each user of the current operation that is *not* in the targeted
    // block, find the nearest parent operation in the targeted block, and
    // increase its pending count. Thus, this parent operation also depends on
    // the release of the current operation.

    for (Operation* user : op.getUsers()) {
      if (user->getBlock() == &block) {
        addDependency(&op, user);
        continue;
      }

      if (Operation* parent = findParentInBlock(user, block);
          parent != nullptr) {
        addDependency(&op, parent);
      }
    }

    // SSA use-def chains do not capture ordering constraints on mutable
    // storage. Preserve the original order of effects on the same concrete
    // SSA-backed resource, such as a non-aliasing CBit register. Value-less
    // effects cannot safely impose block-order dependencies here: routing may
    // temporarily require those operations to move while repairing SSA order.
    const auto effects = getEffectsRecursively(&op);
    if (!effects) {
      continue;
    }

    llvm::SmallDenseSet<Value, 4> affectedValues;
    for (const auto& effect : *effects) {
      Value value = effect.getValue();
      if (!value || !affectedValues.insert(value).second) {
        continue;
      }
      if (Operation* previous = lastEffectForValue.lookup(value)) {
        addDependency(previous, &op);
      }
      lastEffectForValue[value] = &op;
    }
  }

  assert((inDegree.size() == range_size(block)));

  const auto laterInBlock = [&blockOrder](Operation* lhs, Operation* rhs) {
    return blockOrder.lookup(lhs) > blockOrder.lookup(rhs);
  };
  llvm::PriorityQueue<Operation*, std::vector<Operation*>,
                      decltype(laterInBlock)>
      worklist(laterInBlock);
  for (Operation& op : block) {
    if (inDegree.lookup(&op) == 0) {
      worklist.push(&op);
    }
  }

  Block* newBlock = rewriter.createBlock(&block, block.getArgumentTypes(),
                                         getArgumentLocs(block));

  while (!worklist.empty()) {
    Operation* ready = worklist.top();
    worklist.pop();

    rewriter.moveOpBefore(ready, newBlock, newBlock->end());

    for (Operation* user : successors[ready]) {
      inDegree[user]--;
      if (inDegree[user] == 0) {
        worklist.push(user);
      }
    }
  }

  assert(all_of(inDegree, [](const auto& kv) { return kv.second == 0; }));

  // Finally replace the old block arguments with the new ones, move the
  // terminator back at its place, and erase the old block.

  for (size_t i = 0; i < block.getNumArguments(); ++i) {
    rewriter.replaceAllUsesWith(block.getArgument(i), newBlock->getArgument(i));
  }

  rewriter.moveOpBefore(terminator, newBlock, newBlock->end());
  rewriter.eraseBlock(&block);
}
} // namespace mlir::qco
