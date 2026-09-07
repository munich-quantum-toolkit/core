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
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace llvm;

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
  return map_to_vector(block.getArguments(),
                       [](BlockArgument& arg) { return arg.getLoc(); });
}

namespace mlir::qco {
void reorderTopologically(Block& block, IRRewriter& rewriter) {
  Operation* const terminator = block.getTerminator();

  // Construct unresolved map: The dependencies of each operation.

  SmallDenseSet<Value> effectedValues;
  DenseMap<Operation*, size_t> inDegree;
  DenseMap<Value, Operation*> lastEffect;
  DenseMap<Operation*, SmallSetVector<Operation*, 16>> successors;
  DenseMap<Operation*, SmallDenseSet<Operation*, 16>> predecessors;

  const auto addDependency = [&](Operation* predecessor, Operation* successor) {
    assert(predecessor != successor);
    if (!predecessors[successor].insert(predecessor).second) {
      return;
    }
    ++inDegree[successor];
    successors[predecessor].insert(successor);
  };

  for (Operation& op : block) {
    successors.try_emplace(&op);
    predecessors.try_emplace(&op);
    inDegree.try_emplace(&op, 0);

    // Collect the in-block dependencies of the current operation.

    // First, process the side-effect dependencies. This includes all operations
    // with memory effects. For example, classical register operations which
    // don't fulfill linear typing.

    const auto effects = getEffectsRecursively(&op);
    if (effects) {
      for (const auto& effect : *effects) {
        auto value = effect.getValue();
        if (!(value && effectedValues.insert(value).second)) {
          continue;
        }

        if (Operation* last = lastEffect.lookup(value)) {
          addDependency(last, &op);
        }

        lastEffect[value] = &op;
      }
    }

    // Then, process the def-use dependencies, where each operand depends on its
    // defining operation.

    for (auto v : op.getOperands()) {
      if (effectedValues.contains(v)) {
        continue;
      }

      Operation* def = v.getDefiningOp();
      if (def != nullptr && v.getParentBlock() == &block) {
        addDependency(def, &op);
      }
    }

    // Finally, for each user of the current operation that is *not* in the
    // targeted block, find the nearest parent operation in the targeted block,
    // and increase its pending count. Thus, this parent operation also depends
    // on the release of the current operation.

    for (Operation* user : op.getUsers()) {
      if (user->getBlock() == &block) {
        continue;
      }

      if (Operation* parent = findParentInBlock(user, block);
          parent != nullptr) {
        addDependency(&op, parent);
      }
    }

    effectedValues.clear();
  }

  assert((inDegree.size() == range_size(block)));

  SmallVector<Operation*> worklist;
  worklist.reserve(range_size(block));
  for (Operation& op : block) {
    if (inDegree.lookup(&op) == 0) {
      worklist.emplace_back(&op);
    }
  }

  Block* newBlock = rewriter.createBlock(&block, block.getArgumentTypes(),
                                         getArgumentLocs(block));

  for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
    Operation* ready = worklist[cursor];

    rewriter.moveOpBefore(ready, newBlock, newBlock->end());

    for (Operation* user : successors[ready]) {
      inDegree[user]--;
      if (inDegree[user] == 0) {
        worklist.push_back(user);
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
