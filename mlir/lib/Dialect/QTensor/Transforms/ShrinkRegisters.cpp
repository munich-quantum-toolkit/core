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
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <utility>

namespace mlir::qtensor {

#define GEN_PASS_DEF_SHRINKQTENSORTOFITPASS
#include "mlir/Dialect/QTensor/Transforms/Passes.h.inc"

/**
 * @brief Return the unique user of a linear qtensor value.
 */
[[nodiscard]] static Operation* getLinearTensorUser(const Value tensor) {
  assert(tensor.hasOneUse() && "Expected a linear tensor with exactly one use");
  return *tensor.getUsers().begin();
}

/**
 * @brief Mark a single live index.
 */
[[nodiscard]] static LogicalResult markLiveIndex(const int64_t index,
                                                 llvm::BitVector& liveIndices) {
  if (index < 0 || index >= static_cast<int64_t>(liveIndices.size())) {
    return failure();
  }
  liveIndices.set(static_cast<size_t>(index));
  return success();
}

/**
 * @brief Mark a contiguous live range.
 */
[[nodiscard]] static LogicalResult markLiveRange(const int64_t offset,
                                                 const int64_t size,
                                                 llvm::BitVector& liveIndices) {
  if (offset < 0 || size <= 0 ||
      offset + size > static_cast<int64_t>(liveIndices.size())) {
    return failure();
  }
  for (int64_t index = offset; index < offset + size; ++index) {
    liveIndices.set(static_cast<size_t>(index));
  }
  return success();
}

/**
 * @brief Redirect the tensor operand from @p from to @p to.
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
  if (auto extractSliceOp = llvm::dyn_cast<ExtractSliceOp>(op)) {
    if (extractSliceOp.getTensor() != from) {
      return failure();
    }
    extractSliceOp->setOperand(0, to);
    return success();
  }
  if (auto insertSliceOp = llvm::dyn_cast<InsertSliceOp>(op)) {
    if (insertSliceOp.getDest() != from) {
      return failure();
    }
    insertSliceOp->setOperand(1, to);
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
 * @brief Walk alloc->dealloc and collect all touched indices.
 */
[[nodiscard]] static LogicalResult collectLiveIndices(AllocOp allocOp,
                                                      llvm::BitVector& live,
                                                      DeallocOp& deallocOp) {
  Value tensor = allocOp.getResult();
  while (true) {
    auto* user = getLinearTensorUser(tensor);
    if (!user) {
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

    if (auto extractSliceOp = llvm::dyn_cast<ExtractSliceOp>(user)) {
      if (extractSliceOp.getTensor() != tensor) {
        return failure();
      }
      auto offset = getConstantIntValue(extractSliceOp.getOffset());
      auto size = getConstantIntValue(extractSliceOp.getSize());
      if (!offset || !size || failed(markLiveRange(*offset, *size, live))) {
        return failure();
      }
      tensor = extractSliceOp.getOutTensor();
      continue;
    }

    if (auto insertSliceOp = llvm::dyn_cast<InsertSliceOp>(user)) {
      if (insertSliceOp.getDest() != tensor) {
        return failure();
      }
      auto offset = getConstantIntValue(insertSliceOp.getOffset());
      auto size = getConstantIntValue(insertSliceOp.getSize());
      if (!offset || !size || failed(markLiveRange(*offset, *size, live))) {
        return failure();
      }
      tensor = insertSliceOp.getResult();
      continue;
    }

    return failure();
  }
}

/**
 * @brief Shrink static qtensors by removing never-accessed indices.
 * @details QTensor is linear, so this rewrite follows a single use-def chain.
 */
struct ShrinkStaticQTensor final : OpRewritePattern<AllocOp> {
  using OpRewritePattern::OpRewritePattern;

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

    Value oldTensor = allocOp.getResult();
    Value currentTensor = newAlloc.getResult();
    while (true) {
      Operation* currentOp = getLinearTensorUser(oldTensor);
      if (!currentOp) {
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
            oldIndex >= static_cast<int64_t>(newIndexByOldIndex.size())) {
          return failure();
        }
        const auto mappedIndex =
            newIndexByOldIndex[static_cast<size_t>(oldIndex)];
        if (mappedIndex < 0) {
          return failure();
        }
        Value oldOutTensor = extractOp.getOutTensor();
        Operation* nextOp = getLinearTensorUser(oldOutTensor);
        if (!nextOp) {
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
            oldIndex >= static_cast<int64_t>(newIndexByOldIndex.size())) {
          return failure();
        }
        const auto mappedIndex =
            newIndexByOldIndex[static_cast<size_t>(oldIndex)];
        if (mappedIndex < 0) {
          return failure();
        }
        Value oldResultTensor = insertOp.getResult();
        Operation* nextOp = getLinearTensorUser(oldResultTensor);
        if (!nextOp) {
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

      if (auto extractSliceOp = llvm::dyn_cast<ExtractSliceOp>(currentOp)) {
        if (extractSliceOp.getTensor() != oldTensor) {
          return failure();
        }
        const auto oldOffset = *getConstantIntValue(extractSliceOp.getOffset());
        const auto oldSliceSize =
            *getConstantIntValue(extractSliceOp.getSize());
        if (oldOffset < 0 || oldSliceSize <= 0 ||
            oldOffset + oldSliceSize >
                static_cast<int64_t>(newIndexByOldIndex.size())) {
          return failure();
        }
        const auto mappedOffset =
            newIndexByOldIndex[static_cast<size_t>(oldOffset)];
        if (mappedOffset < 0) {
          return failure();
        }
        Value oldOutTensor = extractSliceOp.getOutTensor();
        Operation* nextOp = getLinearTensorUser(oldOutTensor);
        if (!nextOp) {
          return failure();
        }
        rewriter.setInsertionPoint(extractSliceOp);
        auto newOffset = arith::ConstantIndexOp::create(
            rewriter, extractSliceOp.getLoc(), mappedOffset);
        auto newSliceSize = arith::ConstantIndexOp::create(
            rewriter, extractSliceOp.getLoc(), oldSliceSize);
        auto newExtractSlice = ExtractSliceOp::create(
            rewriter, extractSliceOp.getLoc(), currentTensor,
            newOffset.getResult(), newSliceSize.getResult());
        rewriter.replaceAllUsesWith(extractSliceOp.getResult(),
                                    newExtractSlice.getResult());

        currentTensor = newExtractSlice.getOutTensor();
        if (failed(remapTensorOperand(nextOp, oldOutTensor, oldTensor))) {
          return failure();
        }
        rewriter.eraseOp(extractSliceOp);
        continue;
      }

      if (auto insertSliceOp = llvm::dyn_cast<InsertSliceOp>(currentOp)) {
        if (insertSliceOp.getDest() != oldTensor) {
          return failure();
        }
        const auto oldOffset = *getConstantIntValue(insertSliceOp.getOffset());
        const auto oldSliceSize = *getConstantIntValue(insertSliceOp.getSize());
        if (oldOffset < 0 || oldSliceSize <= 0 ||
            oldOffset + oldSliceSize >
                static_cast<int64_t>(newIndexByOldIndex.size())) {
          return failure();
        }
        const auto mappedOffset =
            newIndexByOldIndex[static_cast<size_t>(oldOffset)];
        if (mappedOffset < 0) {
          return failure();
        }
        Value oldResultTensor = insertSliceOp.getResult();
        Operation* nextOp = getLinearTensorUser(oldResultTensor);
        if (!nextOp) {
          return failure();
        }

        rewriter.setInsertionPoint(insertSliceOp);
        auto newOffset = arith::ConstantIndexOp::create(
            rewriter, insertSliceOp.getLoc(), mappedOffset);
        auto newSliceSize = arith::ConstantIndexOp::create(
            rewriter, insertSliceOp.getLoc(), oldSliceSize);
        auto newInsertSlice = InsertSliceOp::create(
            rewriter, insertSliceOp.getLoc(), insertSliceOp.getSource(),
            currentTensor, newOffset.getResult(), newSliceSize.getResult());

        currentTensor = newInsertSlice.getResult();
        if (failed(remapTensorOperand(nextOp, oldResultTensor, oldTensor))) {
          return failure();
        }
        rewriter.eraseOp(insertSliceOp);
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
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ShrinkStaticQTensor>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::qtensor
