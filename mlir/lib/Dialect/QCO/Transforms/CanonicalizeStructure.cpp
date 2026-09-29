/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <memory>
#include <utility>

namespace mlir::qco {
namespace {

/// Gate-changing canonicalization can leave a target's native gate set.
class CanonicalizeStructurePass final
    : public PassWrapper<CanonicalizeStructurePass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeStructurePass)

  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns(context);
    for (auto* dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(patterns);
    }
    for (auto operation : context->getRegisteredOperations()) {
      if (!operation.hasInterface<UnitaryOpInterface>()) {
        operation.getCanonicalizationPatterns(patterns, context);
      }
    }
    patterns_ = FrozenRewritePatternSet(std::move(patterns));
    return success();
  }

protected:
  void runOnOperation() override {
    if (failed(applyPatternsGreedily(getOperation(), patterns_,
                                     GreedyRewriteConfig{}.setMaxIterations(
                                         GreedyRewriteConfig::kNoLimit)))) {
      signalPassFailure();
    }
  }

private:
  FrozenRewritePatternSet patterns_;
};

} // namespace

std::unique_ptr<Pass> createQCOCanonicalizer(bool preserveGates) {
  if (!preserveGates) {
    return createCanonicalizerPass(
        GreedyRewriteConfig{}.setMaxIterations(GreedyRewriteConfig::kNoLimit));
  }
  return std::make_unique<CanonicalizeStructurePass>();
}

} // namespace mlir::qco
