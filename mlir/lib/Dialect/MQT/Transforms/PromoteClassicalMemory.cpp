/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <utility>

namespace mlir::mqt {

#define GEN_PASS_DEF_PROMOTECLASSICALMEMORY
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

namespace {
struct PromoteClassicalMemory final
    : impl::PromoteClassicalMemoryBase<PromoteClassicalMemory> {
  using PromoteClassicalMemoryBase::PromoteClassicalMemoryBase;
  OpPassManager pipeline_{ModuleOp::getOperationName()};

  PromoteClassicalMemory() {
    pipeline_.addPass(createSROA());
    pipeline_.addPass(createMem2Reg());
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect, cbit::CBitDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
    memref::registerMemorySlotExternalModels(registry);
    pipeline_.getDependentDialects(registry);
  }

protected:
  void runOnOperation() override {
    // Index arithmetic, static control flow and views decide which accesses
    // become static; quantum operations never do.
    SmallVector<Operation*> normalized;
    bool promotable = false;
    getOperation().walk([&](Operation* op) {
      promotable |= isa<PromotableAllocationOpInterface,
                        DestructurableAllocationOpInterface>(op);
      if (isa<arith::ArithDialect, cbit::CBitDialect, scf::SCFDialect,
              memref::MemRefDialect>(op->getDialect())) {
        normalized.push_back(op);
      }
    });
    if (!promotable) {
      markAllAnalysesPreserved();
      return;
    }
    RewritePatternSet normalization(&getContext());
    // Reads forward written registers, constant conditions select static
    // indices, loop-carried views are forwarded, and constant view offsets
    // need their result type updated.
    cbit::ReadOp::getCanonicalizationPatterns(normalization, &getContext());
    scf::IfOp::getCanonicalizationPatterns(normalization, &getContext());
    scf::ForOp::getCanonicalizationPatterns(normalization, &getContext());
    memref::SubViewOp::getCanonicalizationPatterns(normalization,
                                                   &getContext());
    // Like the canonicalizer, keep whatever converged.
    (void)applyOpPatternsGreedily(normalized, std::move(normalization));
    // Only wide snapshots need comparison lowering before promotion.
    const auto wideRead = getOperation().walk([](cbit::ReadOp read) {
      return read.getType().getWidth() > 64 ? WalkResult::interrupt()
                                            : WalkResult::advance();
    });
    if (wideRead.wasInterrupted()) {
      RewritePatternSet comparisons(&getContext());
      cbit::populateRegisterComparisonPatterns(comparisons,
                                               /*forPromotion=*/true);
      if (failed(
              applyPatternsGreedily(getOperation(), std::move(comparisons)))) {
        signalPassFailure();
        return;
      }
    }
    if (failed(runPipeline(pipeline_, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::mqt
