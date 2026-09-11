/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Dialect/QTensor/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir::qtensor {

#define GEN_PASS_DEF_SHRINKQTENSORTOFITPASS
#include "mqt/Dialect/QTensor/Transforms/Passes.h.inc"

/// Mark a single live index.
[[nodiscard]] static LogicalResult
markLiveIndex(int64_t index, int64_t tensorSize,
              llvm::SmallDenseSet<int64_t>& liveIndices) {
  if (index < 0 || index >= tensorSize) {
    return failure();
  }
  liveIndices.insert(index);
  return success();
}

namespace {

struct TensorAccess {
  Operation* operation;
  int64_t index;
};

} // namespace

/// Walk alloc → dealloc and plan all accesses without changing the IR.
[[nodiscard]] static LogicalResult collectTensorChain(
    AllocOp allocOp, int64_t tensorSize, llvm::SmallDenseSet<int64_t>& live,
    SmallVectorImpl<TensorAccess>& accesses, DeallocOp& deallocOp) {
  auto tensor = allocOp.getResult();
  while (true) {
    auto* user = *tensor.getUsers().begin();

    if (auto currentDealloc = dyn_cast<DeallocOp>(user)) {
      deallocOp = currentDealloc;
      return success();
    }

    if (auto extractOp = dyn_cast<ExtractOp>(user)) {
      auto index = getConstantIntValue(extractOp.getIndex());
      if (!index || failed(markLiveIndex(*index, tensorSize, live))) {
        return failure();
      }
      accesses.push_back({.operation = extractOp, .index = *index});
      tensor = extractOp.getOutTensor();
      continue;
    }

    if (auto insertOp = dyn_cast<InsertOp>(user)) {
      auto index = getConstantIntValue(insertOp.getIndex());
      if (!index || failed(markLiveIndex(*index, tensorSize, live))) {
        return failure();
      }
      accesses.push_back({.operation = insertOp, .index = *index});
      tensor = insertOp.getResult();
      continue;
    }

    return failure();
  }
}

namespace {

/// Shrink static qtensors by removing never-accessed indices.
///
/// QTensor is linear, so this rewrite follows a single use-def chain.
struct ShrinkStaticQTensor final : OpRewritePattern<AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp allocOp,
                                PatternRewriter& rewriter) const override {
    auto oldSize = getConstantIntValue(allocOp.getSize());
    if (!oldSize || *oldSize <= 0) {
      return failure();
    }

    llvm::SmallDenseSet<int64_t> live;
    SmallVector<TensorAccess> accesses;
    DeallocOp oldDeallocOp{};
    if (failed(collectTensorChain(allocOp, *oldSize, live, accesses,
                                  oldDeallocOp)) ||
        live.empty() || std::cmp_equal(live.size(), *oldSize)) {
      return failure();
    }

    SmallVector<int64_t> liveIndices(live.begin(), live.end());
    llvm::sort(liveIndices);
    const auto newSize = static_cast<int64_t>(liveIndices.size());
    DenseMap<int64_t, int64_t> newIndexByOldIndex;
    for (auto [newIndex, oldIndex] : llvm::enumerate(liveIndices)) {
      newIndexByOldIndex.try_emplace(oldIndex, static_cast<int64_t>(newIndex));
    }

    rewriter.setInsertionPoint(allocOp);
    auto size =
        arith::ConstantIndexOp::create(rewriter, allocOp.getLoc(), newSize);
    auto newAlloc =
        AllocOp::create(rewriter, allocOp.getLoc(), size.getResult());
    rewriter.modifyOpInPlace(newAlloc, [&] {
      newAlloc->setDiscardableAttrs(allocOp->getDiscardableAttrDictionary());
    });

    auto currentTensor = newAlloc.getResult();
    for (const auto& access : accesses) {
      const auto mappedIndex = newIndexByOldIndex.at(access.index);
      if (auto extractOp = dyn_cast<ExtractOp>(access.operation)) {
        rewriter.setInsertionPoint(extractOp);
        auto index = arith::ConstantIndexOp::create(
            rewriter, extractOp.getLoc(), mappedIndex);
        auto newExtract = ExtractOp::create(rewriter, extractOp.getLoc(),
                                            currentTensor, index.getResult());
        rewriter.replaceAllUsesWith(extractOp.getResult(),
                                    newExtract.getResult());
        currentTensor = newExtract.getOutTensor();
        continue;
      }

      auto insertOp = cast<InsertOp>(access.operation);
      rewriter.setInsertionPoint(insertOp);
      auto index = arith::ConstantIndexOp::create(rewriter, insertOp.getLoc(),
                                                  mappedIndex);
      auto newInsert =
          InsertOp::create(rewriter, insertOp.getLoc(), insertOp.getScalar(),
                           currentTensor, index.getResult());

      currentTensor = newInsert.getResult();
    }

    rewriter.setInsertionPoint(oldDeallocOp);
    DeallocOp::create(rewriter, oldDeallocOp.getLoc(), currentTensor);

    rewriter.eraseOp(oldDeallocOp);
    for (const auto& access : llvm::reverse(accesses)) {
      rewriter.eraseOp(access.operation);
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

} // namespace

} // namespace mlir::qtensor
