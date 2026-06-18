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

    for (auto func : anchor.getOps<func::FuncOp>()) {
      SmallVector<SWAPOp> readyToAbsorb;
      SmallVector<WireIterator> wires;
      do {
        wires.clear();
        for (auto op : func.getOps<StaticOp>()) {
          wires.emplace_back(op.getQubit());
        }
        if (wires.empty()) {
          return;
        }

        readyToAbsorb.clear();
        findSwapsReadyForAbsorption(wires, readyToAbsorb);

        for (auto swapOp : readyToAbsorb) {
          rewriter.replaceOp(swapOp,
                             {swapOp.getQubit1In(), swapOp.getQubit0In()});
        }
      } while (!readyToAbsorb.empty());
    }
  }

private:
  static void findSwapsReadyForAbsorption(MutableArrayRef<WireIterator> wires,
                                          SmallVector<SWAPOp>& readyToAbsorb) {
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
  }
};
} // namespace
} // namespace mlir::qco
