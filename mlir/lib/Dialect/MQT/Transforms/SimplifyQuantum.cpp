/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/MQT/Utils/DenseUnitary.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <utility>

namespace mlir::mqt {
#define GEN_PASS_DEF_SIMPLIFYQUANTUM
#define GEN_PASS_DEF_LOWERQUANTUMMODIFIERS
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"
namespace {
struct LowerQuantumModifiers final
    : impl::LowerQuantumModifiersBase<LowerQuantumModifiers> {
  using LowerQuantumModifiersBase::LowerQuantumModifiersBase;

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    qc::CtrlOp::getLoweringPatterns(patterns, &getContext());
    qc::InvOp::getLoweringPatterns(patterns, &getContext());
    qc::PowOp::getLoweringPatterns(patterns, &getContext());
    qco::CtrlOp::getLoweringPatterns(patterns, &getContext());
    qco::InvOp::getLoweringPatterns(patterns, &getContext());
    qco::PowOp::getLoweringPatterns(patterns, &getContext());
    // Structural cleanup owns CFG simplification.
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};
struct SimplifyQuantum final : impl::SimplifyQuantumBase<SimplifyQuantum> {
  using SimplifyQuantumBase::SimplifyQuantumBase;

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    qc::UnitaryOp::getOptimizationPatterns(patterns, &getContext());
    qc::PowOp::getLoweringPatterns(patterns, &getContext());
    qc::CtrlOp::getLoweringPatterns(patterns, &getContext());
    qc::InvOp::getLoweringPatterns(patterns, &getContext());
    qco::PowOp::getLoweringPatterns(patterns, &getContext());
    qco::CtrlOp::getOptimizationPatterns(patterns, &getContext());
    qco::InvOp::getLoweringPatterns(patterns, &getContext());
    qco::DCXOp::getOptimizationPatterns(patterns, &getContext());
    qco::ECROp::getOptimizationPatterns(patterns, &getContext());
    qco::GPhaseOp::getOptimizationPatterns(patterns, &getContext());
    qco::HOp::getOptimizationPatterns(patterns, &getContext());
    qco::POp::getOptimizationPatterns(patterns, &getContext());
    qco::RCCXOp::getOptimizationPatterns(patterns, &getContext());
    qco::ROp::getOptimizationPatterns(patterns, &getContext());
    qco::RXOp::getOptimizationPatterns(patterns, &getContext());
    qco::RXXOp::getOptimizationPatterns(patterns, &getContext());
    qco::RYOp::getOptimizationPatterns(patterns, &getContext());
    qco::RYYOp::getOptimizationPatterns(patterns, &getContext());
    qco::RZOp::getOptimizationPatterns(patterns, &getContext());
    qco::RZXOp::getOptimizationPatterns(patterns, &getContext());
    qco::RZZOp::getOptimizationPatterns(patterns, &getContext());
    qco::SOp::getOptimizationPatterns(patterns, &getContext());
    qco::SWAPOp::getOptimizationPatterns(patterns, &getContext());
    qco::SXOp::getOptimizationPatterns(patterns, &getContext());
    qco::SXdgOp::getOptimizationPatterns(patterns, &getContext());
    qco::SdgOp::getOptimizationPatterns(patterns, &getContext());
    qco::TOp::getOptimizationPatterns(patterns, &getContext());
    qco::TdgOp::getOptimizationPatterns(patterns, &getContext());
    qco::U2Op::getOptimizationPatterns(patterns, &getContext());
    qco::UOp::getOptimizationPatterns(patterns, &getContext());
    qco::XOp::getOptimizationPatterns(patterns, &getContext());
    qco::XXMinusYYOp::getOptimizationPatterns(patterns, &getContext());
    qco::XXPlusYYOp::getOptimizationPatterns(patterns, &getContext());
    qco::YOp::getOptimizationPatterns(patterns, &getContext());
    qco::ZOp::getOptimizationPatterns(patterns, &getContext());

    patterns.add(+[](qco::IdOp op, PatternRewriter& rewriter) {
      rewriter.replaceOp(op, op.getQubitIn());
      return success();
    });
    patterns.add(+[](qco::UnitaryOp op, PatternRewriter& rewriter) {
      if (!isExactIdentityMatrix(op.getMatrix())) {
        return failure();
      }
      rewriter.replaceOp(op, op.getQubitsIn());
      return success();
    });
    // Structural cleanup owns CFG simplification.
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::mqt
