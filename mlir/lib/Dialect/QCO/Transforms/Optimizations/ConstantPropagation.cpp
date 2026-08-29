/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ConstantPropagation/ConstantPropagationAnalysis.hpp"
#include "ConstantPropagation/Decisions.hpp"
#include "ConstantPropagation/Rewriter.hpp"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::qco {

#define GEN_PASS_DEF_CONSTANTPROPAGATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/**
 * @brief Quantum constant propagation.
 *
 * Assumes every input qubit of the entry-point function starts in |0>,
 * propagates the quantum/classical state through the circuit up to a complexity
 * threshold (an MLIR `DenseForwardDataFlowAnalysis` over a `UnionTable`
 * lattice), then removes operations that are superfluous given that state.
 *
 * Rewrites: delete a `qco.ctrl` whose controls can never all hold, and remove
 * the always-satisfied controls from a `qco.ctrl` - rebuilding it with the rest,
 * or inlining its body when every control was redundant. Analyze and rewrite
 * alternate until a fixpoint because a removed gate can change a later gate's
 * control facts. Classical controls are not reasoned about yet.
 */
struct ConstantPropagation final
    : impl::ConstantPropagationBase<ConstantPropagation> {
  using ConstantPropagationBase::ConstantPropagationBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    func::FuncOp entry;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (!mqt::isEntryPoint(func)) {
        continue;
      }
      if (entry) {
        module.emitError(
            "constant propagation supports a single entry-point function");
        return signalPassFailure();
      }
      entry = func;
    }
    if (!entry) {
      return;
    }

    IRRewriter rewriter(&getContext());
    constexpr unsigned maxRounds = 64;
    for (unsigned round = 0; round < maxRounds; ++round) {
      DataFlowSolver solver;
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<dataflow::SparseConstantPropagation>();
      solver.load<ConstantPropagationAnalysis>(maximumNonzeroAmplitudes,
                                               maximumHybridStates);
      if (failed(solver.initializeAndRun(module))) {
        return signalPassFailure(); // the analysis emitted the diagnostic
      }

      const SmallVector<Decision> decisions = collectDecisions(entry, solver);
      if (decisions.empty()) {
        return; // fixpoint reached
      }
      applyDecisions(decisions, rewriter);
    }

    entry.emitError("constant propagation did not converge within ")
        << maxRounds << " rounds";
    signalPassFailure();
  }
};

} // namespace

} // namespace mlir::qco
