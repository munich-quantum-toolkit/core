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
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;

namespace {
/// Find the nearest neighbour in a given block.
Operation* findParentInBlock(Operation* op, Block& block) {
  Operation* parent = op->getParentOp();
  while (parent != nullptr && parent->getBlock() != &block) {
    parent = parent->getParentOp();
  }
  return parent;
}

/// Return the vector of locations for each block argument.
SmallVector<Location> getArgumentLocs(Block& block) {
  return to_vector(map_range(block.getArguments(),
                             [](BlockArgument& arg) { return arg.getLoc(); }));
}
} // namespace

namespace mlir::qco {
void reorderTopologically(Block& block, IRRewriter& rewriter) {
  Operation* const terminator = block.getTerminator();

  // Construct unresolved map: The dependencies of each operation.

  DenseMap<Operation*, size_t> inDegree;
  DenseMap<Operation*, llvm::SmallDenseSet<Operation*, 16>> successors;
  DenseMap<Operation*, llvm::SmallDenseSet<Operation*, 16>> predecessors;

  for (Operation& op : block) {

    // Collect the in-block dependencies of the current operation.

    auto& succs = successors[&op];
    auto& pres = predecessors[&op];
    inDegree.try_emplace(&op, 0);

    for (Value v : op.getOperands()) {
      Operation* def = v.getDefiningOp();
      if (def != nullptr && v.getParentBlock() == &block &&
          !pres.contains(def)) {
        pres.insert(def);
        ++inDegree[&op];
      }
    }

    // For each user of the current operation that is *not* in the targeted
    // block, find the nearest parent operation in the targeted block, and
    // increase its pending count. Thus, this parent operation also depends on
    // the release of the current operation.

    for (Operation* user : op.getUsers()) {
      if (user->getBlock() == &block) {
        if (!succs.contains(user)) {
          succs.insert(user);
        }
        continue;
      }

      if (Operation* parent = findParentInBlock(user, block);
          parent != nullptr) {

        auto& parentPre = predecessors[parent];
        if (!parentPre.contains(&op)) {
          parentPre.insert(&op);
          ++inDegree[parent];
        }

        if (!succs.contains(parent)) {
          succs.insert(parent);
        }
      }
    }
  }

  assert((inDegree.size() == range_size(block)));

  SmallVector<Operation*> worklist;
  worklist.reserve(range_size(block));
  for (const auto& [op, ndeps] : inDegree) {
    if (ndeps == 0) {
      worklist.emplace_back(op);
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
