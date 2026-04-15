/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Transforms/Passes.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::qc {

#define GEN_PASS_DEF_SHRINKQUBITREGISTERSPASS
#include "mlir/Dialect/QC/Transforms/Passes.h.inc"

/**
 * @brief Return the constant index of a one-dimensional `memref.load`
 * operation.
 */
[[nodiscard]] static std::optional<int64_t>
getLoadIndex(memref::LoadOp loadOp) {
  if (loadOp.getIndices().size() != 1) {
    return std::nullopt;
  }
  return getConstantIntValue(loadOp.getIndices().front());
}

namespace {
/**
 * @brief Shrink static qubit registers to actually read indices.
 */
struct ShrinkQubitRegister final : OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter& rewriter) const override {
    auto allocOp = op.getMemref().getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      return failure();
    }

    auto memRefType = llvm::dyn_cast<MemRefType>(op.getMemref().getType());
    if (!memRefType || memRefType.getRank() != 1 ||
        !memRefType.hasStaticShape()) {
      return failure();
    }
    if (!llvm::isa<QubitType>(memRefType.getElementType())) {
      return failure();
    }
    if (!memRefType.getLayout().isIdentity()) {
      return failure();
    }
    if (memRefType.getMemorySpace() != nullptr) {
      return failure();
    }

    llvm::SmallVector<memref::LoadOp> loadOps;
    llvm::SmallVector<int64_t> liveIndices;
    llvm::DenseMap<int64_t, size_t> newIndexByOldIndex;

    for (auto* user : op.getMemref().getUsers()) {
      if (user == op.getOperation()) {
        continue;
      }
      auto loadOp = llvm::dyn_cast<memref::LoadOp>(user);
      if (!loadOp) {
        return failure();
      }
      auto index = getLoadIndex(loadOp);
      if (!index || *index < 0 || *index >= memRefType.getDimSize(0)) {
        return failure();
      }
      loadOps.push_back(loadOp);
      if (!loadOp.getResult().use_empty() &&
          !newIndexByOldIndex.contains(*index)) {
        newIndexByOldIndex.try_emplace(*index, 0U);
        liveIndices.push_back(*index);
      }
    }

    if (liveIndices.empty()) {
      for (auto loadOp : loadOps) {
        rewriter.eraseOp(loadOp);
      }
      rewriter.eraseOp(op);
      rewriter.eraseOp(allocOp);
      return success();
    }

    llvm::sort(liveIndices);
    if (static_cast<int64_t>(liveIndices.size()) == memRefType.getDimSize(0) &&
        llvm::all_of(llvm::enumerate(liveIndices), [](const auto& indexed) {
          return static_cast<int64_t>(indexed.index()) == indexed.value();
        })) {
      return failure();
    }

    newIndexByOldIndex.clear();
    for (size_t i = 0; i < liveIndices.size(); ++i) {
      newIndexByOldIndex.try_emplace(liveIndices[i], i);
    }

    rewriter.setInsertionPoint(allocOp);
    auto newMemRefType =
        MemRefType::get({static_cast<int64_t>(liveIndices.size())},
                        memRefType.getElementType());
    auto newAlloc =
        memref::AllocOp::create(rewriter, allocOp.getLoc(), newMemRefType);

    for (auto loadOp : loadOps) {
      if (loadOp.getResult().use_empty()) {
        rewriter.eraseOp(loadOp);
        continue;
      }

      const auto oldIndex = *getLoadIndex(loadOp);
      const auto newIndex =
          static_cast<int64_t>(newIndexByOldIndex.lookup(oldIndex));
      rewriter.setInsertionPoint(loadOp);
      auto indexConst =
          arith::ConstantIndexOp::create(rewriter, loadOp.getLoc(), newIndex);
      auto newLoad = memref::LoadOp::create(rewriter, loadOp.getLoc(),
                                            newAlloc.getResult(),
                                            ValueRange{indexConst.getResult()});
      rewriter.replaceOp(loadOp, newLoad);
    }

    rewriter.setInsertionPoint(op);
    memref::DeallocOp::create(rewriter, op.getLoc(), newAlloc.getResult());
    rewriter.eraseOp(op);
    rewriter.eraseOp(allocOp);
    return success();
  }
};

struct ShrinkQubitRegistersPass final
    : impl::ShrinkQubitRegistersPassBase<ShrinkQubitRegistersPass> {
protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ShrinkQubitRegister>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::qc
