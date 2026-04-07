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
#include "mlir/Dialect/QTensor/Transforms/Passes.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir::qtensor {

#define GEN_PASS_DEF_SHRINKQTENSORTOFITPASS
#include "mlir/Dialect/QTensor/Transforms/Passes.h.inc"

/**
 * @brief Get the single user operation of a linear QTensor value.
 *
 * Asserts that `tensor` has exactly one use; the program will abort if this
 * condition is not met.
 *
 * @param tensor The QTensor value expected to be used exactly once.
 * @return Operation* The unique user operation of `tensor`.
 */
[[nodiscard]] static Operation* getLinearTensorUser(const Value tensor) {
  assert(tensor.hasOneUse() && "Expected a linear tensor with exactly one use");
  return *tensor.getUsers().begin();
}

/**
 * @brief Mark the specified index as live in the provided bit vector.
 *
 * @param index The index to set as live; must be in the range [0, liveIndices.size()).
 * @param liveIndices Bit vector tracking live indices; the bit at `index` will be set on success.
 * @return LogicalResult `success()` if `index` is within bounds and was marked, `failure()` otherwise.
 */
[[nodiscard]] static LogicalResult markLiveIndex(const int64_t index,
                                                 llvm::BitVector& liveIndices) {
  if (index < 0 || std::cmp_greater_equal(index, liveIndices.size())) {
    return failure();
  }
  liveIndices.set(static_cast<size_t>(index));
  return success();
}

/**
 * @brief Replace a tensor operand referring to one Value with another for supported ops.
 *
 * Attempts to replace uses of `from` with `to` on the tensor operand of the provided
 * operation when the operation is an `ExtractOp`, `InsertOp`, or `DeallocOp`.
 *
 * @param op Operation to inspect and potentially modify.
 * @param from The tensor Value to be replaced.
 * @param to The tensor Value to use as the replacement.
 * @return LogicalResult `success()` if `op` is one of the supported ops and the matching
 * operand referred to `from` and was updated to `to`; `failure()` otherwise.
 */
[[nodiscard]] static LogicalResult remapTensorOperand(Operation* op, Value from,
                                                      Value to) {
  if (auto extractOp = llvm::dyn_cast<ExtractOp>(op)) {
    if (extractOp.getTensor() != from) {
      return failure();
    }
    extractOp->setOperand(0, to);
    return success();
  }
  if (auto insertOp = llvm::dyn_cast<InsertOp>(op)) {
    if (insertOp.getDest() != from) {
      return failure();
    }
    insertOp->setOperand(1, to);
    return success();
  }
  if (auto deallocOp = llvm::dyn_cast<DeallocOp>(op)) {
    if (deallocOp.getTensor() != from) {
      return failure();
    }
    deallocOp->setOperand(0, to);
    return success();
  }
  return failure();
}

/**
 * @brief Traverse the single-use tensor chain from an allocation and mark indices that are accessed.
 *
 * Walks forward from `allocOp` following the linear (single-user) tensor chain composed of
 * `qtensor.extract`, `qtensor.insert`, and terminating `qtensor.dealloc`. For each visited
 * `extract` or `insert` with a constant index, marks that index in `live`. On success returns
 * the terminating `dealloc` operation via `deallocOp`.
 *
 * @param allocOp The allocation operation whose result tensor is the start of the chain.
 * @param[out] live BitVector sized to the allocation's length; bits corresponding to accessed
 *                   indices are set to 1.
 * @param[out] deallocOp Will be assigned the `qtensor.dealloc` operation that terminates the chain.
 * @return LogicalResult `success()` if a matching linear chain was found, all accessed indices
 *         were constant and within bounds, and the terminating dealloc was located; `failure()`
 *         otherwise.
 */
[[nodiscard]] static LogicalResult collectLiveIndices(AllocOp allocOp,
                                                      llvm::BitVector& live,
                                                      DeallocOp& deallocOp) {
  auto tensor = allocOp.getResult();
  while (true) {
    auto* user = getLinearTensorUser(tensor);
    if (user == nullptr) {
      return failure();
    }

    if (auto currentDealloc = llvm::dyn_cast<DeallocOp>(user)) {
      if (currentDealloc.getTensor() != tensor) {
        return failure();
      }
      deallocOp = currentDealloc;
      return success();
    }

    if (auto extractOp = llvm::dyn_cast<ExtractOp>(user)) {
      if (extractOp.getTensor() != tensor) {
        return failure();
      }
      auto index = getConstantIntValue(extractOp.getIndex());
      if (!index || failed(markLiveIndex(*index, live))) {
        return failure();
      }
      tensor = extractOp.getOutTensor();
      continue;
    }

    if (auto insertOp = llvm::dyn_cast<InsertOp>(user)) {
      if (insertOp.getDest() != tensor) {
        return failure();
      }
      auto index = getConstantIntValue(insertOp.getIndex());
      if (!index || failed(markLiveIndex(*index, live))) {
        return failure();
      }
      tensor = insertOp.getResult();
      continue;
    }

    return failure();
  }
}

namespace {

/**
 * @brief Shrink static qtensors by removing never-accessed indices.
 * @details QTensor is linear, so this rewrite follows a single use-def chain.
 */
struct ShrinkStaticQTensor final : OpRewritePattern<AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Attempts to shrink a statically-sized `qtensor.alloc` by removing unused indices and rewriting its linear user chain.
   *
   * Validates the allocation has a positive constant size, collects the set of accessed indices along the single-use alloc→...→dealloc chain, builds a dense remapping for live indices, creates a smaller `qtensor.alloc`, and rewrites subsequent `qtensor.extract`/`qtensor.insert` and the final `qtensor.dealloc` to operate on the new, compact tensor.
   *
   * @param allocOp The `qtensor.alloc` operation to match and potentially rewrite.
   * @param rewriter PatternRewriter used to create and replace operations.
   * @returns LogicalResult `success()` if the alloc was shrunk and the chain was successfully rewritten; `failure()` if any validation fails (missing/invalid constant size, live-index collection failure, missing or mismatched dealloc, new size is zero or unchanged, unexpected user pattern, non-constant or out-of-range indices, or any remapping failure).
   */
  LogicalResult matchAndRewrite(AllocOp allocOp,
                                PatternRewriter& rewriter) const override {
    auto oldSize = getConstantIntValue(allocOp.getSize());
    if (!oldSize || *oldSize <= 0) {
      return failure();
    }

    llvm::BitVector live(static_cast<size_t>(*oldSize), false);
    DeallocOp oldDeallocOp{};
    if (failed(collectLiveIndices(allocOp, live, oldDeallocOp))) {
      return failure();
    }

    if (!oldDeallocOp) {
      return failure();
    }

    llvm::SmallVector<int64_t> newIndexByOldIndex(static_cast<size_t>(*oldSize),
                                                  -1);
    int64_t newSize = 0;
    for (int64_t index = 0; index < *oldSize; ++index) {
      if (live.test(static_cast<size_t>(index))) {
        newIndexByOldIndex[static_cast<size_t>(index)] = newSize++;
      }
    }

    if (newSize <= 0 || newSize == *oldSize) {
      return failure();
    }

    rewriter.setInsertionPoint(allocOp);
    auto size =
        arith::ConstantIndexOp::create(rewriter, allocOp.getLoc(), newSize);
    auto newAlloc =
        AllocOp::create(rewriter, allocOp.getLoc(), size.getResult());

    auto oldTensor = allocOp.getResult();
    auto currentTensor = newAlloc.getResult();
    while (true) {
      Operation* currentOp = getLinearTensorUser(oldTensor);
      if (currentOp == nullptr) {
        return failure();
      }

      if (auto deallocOp = llvm::dyn_cast<DeallocOp>(currentOp)) {
        if (deallocOp != oldDeallocOp || deallocOp.getTensor() != oldTensor) {
          return failure();
        }
        rewriter.setInsertionPoint(deallocOp);
        DeallocOp::create(rewriter, deallocOp.getLoc(), currentTensor);
        rewriter.eraseOp(deallocOp);
        break;
      }

      if (auto extractOp = llvm::dyn_cast<ExtractOp>(currentOp)) {
        if (extractOp.getTensor() != oldTensor) {
          return failure();
        }
        const auto oldIndex = *getConstantIntValue(extractOp.getIndex());
        if (oldIndex < 0 ||
            std::cmp_greater_equal(oldIndex, newIndexByOldIndex.size())) {
          return failure();
        }
        const auto mappedIndex =
            newIndexByOldIndex[static_cast<size_t>(oldIndex)];
        if (mappedIndex < 0) {
          return failure();
        }
        auto oldOutTensor = extractOp.getOutTensor();
        auto* nextOp = getLinearTensorUser(oldOutTensor);
        if (nextOp == nullptr) {
          return failure();
        }

        rewriter.setInsertionPoint(extractOp);
        auto index = arith::ConstantIndexOp::create(
            rewriter, extractOp.getLoc(), mappedIndex);
        auto newExtract = ExtractOp::create(rewriter, extractOp.getLoc(),
                                            currentTensor, index.getResult());
        rewriter.replaceAllUsesWith(extractOp.getResult(),
                                    newExtract.getResult());

        currentTensor = newExtract.getOutTensor();
        if (failed(remapTensorOperand(nextOp, oldOutTensor, oldTensor))) {
          return failure();
        }
        rewriter.eraseOp(extractOp);
        continue;
      }

      if (auto insertOp = llvm::dyn_cast<InsertOp>(currentOp)) {
        if (insertOp.getDest() != oldTensor) {
          return failure();
        }
        const auto oldIndex = *getConstantIntValue(insertOp.getIndex());
        if (oldIndex < 0 ||
            std::cmp_greater_equal(oldIndex, newIndexByOldIndex.size())) {
          return failure();
        }
        const auto mappedIndex =
            newIndexByOldIndex[static_cast<size_t>(oldIndex)];
        if (mappedIndex < 0) {
          return failure();
        }
        auto oldResultTensor = insertOp.getResult();
        auto* nextOp = getLinearTensorUser(oldResultTensor);
        if (nextOp == nullptr) {
          return failure();
        }

        rewriter.setInsertionPoint(insertOp);
        auto index = arith::ConstantIndexOp::create(rewriter, insertOp.getLoc(),
                                                    mappedIndex);
        auto newInsert =
            InsertOp::create(rewriter, insertOp.getLoc(), insertOp.getScalar(),
                             currentTensor, index.getResult());

        currentTensor = newInsert.getResult();
        if (failed(remapTensorOperand(nextOp, oldResultTensor, oldTensor))) {
          return failure();
        }
        rewriter.eraseOp(insertOp);
        continue;
      }

      return failure();
    }

    rewriter.eraseOp(allocOp);
    return success();
  }
};

struct ShrinkQTensorToFitPass final
    : impl::ShrinkQTensorToFitPassBase<ShrinkQTensorToFitPass> {
protected:
  /**
   * @brief Run the pass: register and apply the shrink-qtensor rewrite patterns.
   *
   * Registers the ShrinkStaticQTensor rewrite pattern and applies all patterns
   * greedily to the current operation, signaling pass failure if pattern
   * application fails.
   */
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ShrinkStaticQTensor>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qtensor
