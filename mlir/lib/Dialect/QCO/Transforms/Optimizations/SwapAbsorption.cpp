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
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

namespace mlir::qco {
#define GEN_PASS_DEF_SWAPABSORPTION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {
struct SwapAbsorption : impl::SwapAbsorptionBase<SwapAbsorption> {
  using SwapAbsorptionBase::SwapAbsorptionBase;

protected:
  void runOnOperation() override {
    ModuleOp anchor = getOperation();
    IRRewriter rewriter(&getContext());

    anchor.walk([&rewriter](mlir::Operation *op) {
      if(auto swap = mlir::dyn_cast<SWAPOp>(op))
      {
          rewriter.replaceOp(swap,
                             {swap.getQubit1In(), swap.getQubit0In()});
      }
    });
  }
};
} // namespace
} // namespace mlir::qco
