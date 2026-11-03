/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

namespace mlir::mqt {

#define GEN_PASS_DEF_DROPGLOBALPHASES
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

void dropGlobalPhases(ModuleOp moduleOp, RewriterBase& rewriter) {
  auto entryPoint = getEntryPoint(moduleOp);
  if (!entryPoint) {
    return;
  }
  entryPoint.walk([&](Operation* op) {
    if (isa<qc::GPhaseOp, qco::GPhaseOp>(op) &&
        !op->getParentOfType<qc::CtrlOp, qc::InvOp, qc::PowOp, qco::CtrlOp,
                             qco::InvOp, qco::PowOp>()) {
      rewriter.eraseOp(op);
    }
  });
}

namespace {
struct DropGlobalPhases final : impl::DropGlobalPhasesBase<DropGlobalPhases> {
protected:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    dropGlobalPhases(getOperation(), rewriter);
  }
};
} // namespace
} // namespace mlir::mqt
