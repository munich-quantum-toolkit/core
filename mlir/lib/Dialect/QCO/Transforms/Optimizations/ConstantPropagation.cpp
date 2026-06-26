/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>

#include <mlir/IR/BuiltinOps.h.inc>
namespace mlir::qco {

#define GEN_PASS_DEF_CONSTANTPROPAGATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/**
 * This method moves all measurements as far to the front as possible, in order
 * to execute QCP more efficiently.
 */
bool moveMeasurementsToFront(ModuleOp module, MLIRContext* ctx) {
  bool changed = false;
  // PatternRewriter rewriter(ctx);
  // module.walk([&](MeasureOp op) {
  //   Operation* previousInstruction = op.getInQubit().getDefiningOp();
  //   Operation* previousNode = op->getPrevNode();
  //   while (llvm::dyn_cast<MeasureOp>(previousNode) &&
  //          previousInstruction != previousNode) {
  //     previousNode = previousNode->getPrevNode();
  //   }
  //   if (previousNode != previousInstruction) {
  //     rewriter.moveOpAfter(op, previousInstruction);
  //     changed = true;
  //   }
  // });

  return changed;
}

/**
 * @brief This pass applies constant propagation to a circuit. It assumes that
 * all states start in |0> and removes quantum instructions that are superfluous
 * when the current state is considered. It also replaces quantum resources by
 * classical resources.
 */
struct ConstantPropagation final
    : impl::ConstantPropagationBase<ConstantPropagation> {
  using ConstantPropagationBase::ConstantPropagationBase;

  void runOnOperation() override {
    // if (failed(route(getOperation(), &getContext()))) {
    //   signalPassFailure();
    // }
  }
};

} // namespace

} // namespace mlir::qco
