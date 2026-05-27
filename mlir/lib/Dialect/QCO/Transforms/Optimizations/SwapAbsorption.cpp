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

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

namespace mlir::qco {
#define GEN_PASS_DEF_SWAPABSORPTIONPASS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {
struct SwapAbsorption : impl::SwapAbsorptionPassBase<SwapAbsorption> {
public:
  using SwapAbsorptionPassBase::SwapAbsorptionPassBase;

protected:
  void runOnOperation() override {
    ModuleOp anchor = getOperation();
    IRRewriter rewriter(&getContext());

    for (auto func : anchor.getOps<func::FuncOp>()) {
      SmallVector<WireIterator> wires;
      for (auto op : func.getOps<StaticOp>()) {
        wires.emplace_back(op.getQubit());
      }

      SmallVector<SWAPOp> readyToAbsorb;
      readyToAbsorb.reserve((wires.size() + 1) / 2);

      std::ignore = walkProgramGraph<WireDirection::Forward>(
          wires, [&](const ReadyRange& ready, ReleasedOps& released) {
            for (const auto& [op, indices] : ready) {
              if (isa<SWAPOp>(op)) {
                readyToAbsorb.emplace_back(op);
              }
              released.emplace_back(op);
            }
            return WalkResult::interrupt();
          });

      for (auto swapOp : readyToAbsorb) {
        auto in0 = swapOp.getQubit0In();
        auto in1 = swapOp.getQubit1In();

        auto out0 = swapOp.getQubit0Out();
        auto out1 = swapOp.getQubit1Out();

        // TODO: What if single qubit gates are in front of the SWAP input?
        // Tipp: Use the WireIterator.
        StaticOp op0 = cast<StaticOp>(in0.getDefiningOp());
        StaticOp op1 = cast<StaticOp>(in1.getDefiningOp());

        rewriter.replaceAllUsesWith(out0, op1.getQubit());
        rewriter.replaceAllUsesWith(out1, op0.getQubit());
        rewriter.eraseOp(swapOp);
      }
    }
  }
};
} // namespace
} // namespace mlir::qco
