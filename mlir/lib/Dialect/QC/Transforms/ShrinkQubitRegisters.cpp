/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir::qc {

#define GEN_PASS_DEF_SHRINKQUBITREGISTERSPASS
#include "mqt/Dialect/QC/Transforms/Passes.h.inc"

namespace {
/// Shrink static qubit registers to actually read indices.
struct ShrinkQubitRegister final : OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter& rewriter) const override {
    auto allocOp = op.getMemref().getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      return failure();
    }

    auto memRefType = dyn_cast<MemRefType>(op.getMemref().getType());
    if (!memRefType || memRefType.getRank() != 1 ||
        !memRefType.hasStaticShape()) {
      return failure();
    }
    if (!isa<QubitType>(memRefType.getElementType())) {
      return failure();
    }
    if (!memRefType.getLayout().isIdentity()) {
      return failure();
    }
    if (memRefType.getMemorySpace() != nullptr) {
      return failure();
    }

    SmallVector<memref::LoadOp> loadOps;
    SmallVector<int64_t> liveIndices;
    DenseMap<int64_t, size_t> newIndexByOldIndex;

    for (auto* user : op.getMemref().getUsers()) {
      if (user == op.getOperation()) {
        continue;
      }
      auto loadOp = dyn_cast<memref::LoadOp>(user);
      if (!loadOp) {
        return failure();
      }
      auto index = getConstantIntValue(loadOp.getIndices().front());
      if (!index || *index < 0 || *index >= memRefType.getDimSize(0)) {
        return failure();
      }
      loadOps.push_back(loadOp);
      if (!loadOp.getResult().use_empty() &&
          newIndexByOldIndex.try_emplace(*index, 0U).second) {
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

    if (std::cmp_equal(liveIndices.size(), memRefType.getDimSize(0))) {
      return failure();
    }
    llvm::sort(liveIndices);

    for (size_t i = 0; i < liveIndices.size(); ++i) {
      newIndexByOldIndex[liveIndices[i]] = i;
    }

    rewriter.setInsertionPoint(allocOp);
    auto newMemRefType =
        MemRefType::get({static_cast<int64_t>(liveIndices.size())},
                        memRefType.getElementType());
    auto newAlloc =
        memref::AllocOp::create(rewriter, allocOp.getLoc(), newMemRefType);
    newAlloc->setDiscardableAttrs(allocOp->getDiscardableAttrDictionary());

    for (auto loadOp : loadOps) {
      if (loadOp.getResult().use_empty()) {
        rewriter.eraseOp(loadOp);
        continue;
      }

      const auto oldIndex = *getConstantIntValue(loadOp.getIndices().front());
      const auto newIndex =
          static_cast<int64_t>(newIndexByOldIndex.lookup(oldIndex));
      rewriter.setInsertionPoint(loadOp);
      auto index = loadOp.getIndices().front();
      if (newIndex != oldIndex) {
        index =
            arith::ConstantIndexOp::create(rewriter, loadOp.getLoc(), newIndex);
      }
      auto newLoad = memref::LoadOp::create(
          rewriter, loadOp.getLoc(), newAlloc.getResult(), ValueRange{index});
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
    SmallVector<Operation*> candidates;
    getOperation()->walk(
        [&](memref::DeallocOp op) { candidates.push_back(op); });
    if (candidates.empty()) {
      markAllAnalysesPreserved();
      return;
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<ShrinkQubitRegister>(&getContext());

    GreedyRewriteConfig config;
    config.setScope(&getOperation().getRegion());
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    if (failed(
            applyOpPatternsGreedily(candidates, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::qc
