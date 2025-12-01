/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/LayeredUnit.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Router.h"

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <queue>

#define DEBUG_TYPE "route-astar-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ASTARROUTINGPASSSC
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/**
 * @brief Insert SWAP ops at the rewriter's insertion point.
 *
 * @param loc The location of the inserted SWAP ops.
 * @param swaps The hardware indices of the SWAPs.
 * @param layout The current layout.
 * @param rewriter The pattern rewriter.
 */
void insertSWAPs(Location loc, ArrayRef<QubitIndexPair> swaps, Layout& layout,
                 PatternRewriter& rewriter) {
  for (const auto [hw0, hw1] : swaps) {
    const Value in0 = layout.lookupHardwareValue(hw0);
    const Value in1 = layout.lookupHardwareValue(hw1);
    [[maybe_unused]] const auto [prog0, prog1] =
        layout.getProgramIndices(hw0, hw1);

    LLVM_DEBUG({
      llvm::dbgs() << llvm::format(
          "route: swap= p%d:h%d, p%d:h%d <- p%d:h%d, p%d:h%d\n", prog1, hw0,
          prog0, hw1, prog0, hw0, prog1, hw1);
    });

    auto swap = createSwap(loc, in0, in1, rewriter);
    const auto [out0, out1] = getOuts(swap);

    rewriter.setInsertionPointAfter(swap);
    replaceAllUsesInRegionAndChildrenExcept(in0, out1, swap->getParentRegion(),
                                            swap, rewriter);
    replaceAllUsesInRegionAndChildrenExcept(in1, out0, swap->getParentRegion(),
                                            swap, rewriter);

    layout.swap(in0, in1);
    layout.remapQubitValue(in0, out0);
    layout.remapQubitValue(in1, out1);
  }
}

/**
 * @brief Routes the program by dividing the circuit into layers of parallel
 * two-qubit gates and iteratively searches and inserts SWAPs for each layer
 * using A*-search.
 */
struct AStarRoutingPassSC final
    : impl::AStarRoutingPassSCBase<AStarRoutingPassSC> {
  using AStarRoutingPassSCBase<AStarRoutingPassSC>::AStarRoutingPassSCBase;

  void runOnOperation() override {
    if (failed(preflight())) {
      signalPassFailure();
      return;
    }

    if (failed(route())) {
      signalPassFailure();
      return;
    }
  }

private:
  /**
   * @brief Route the given module for the targeted architecture using
   * A*-search. Processes each entry_point function separately.
   */
  LogicalResult route() {
    ModuleOp module(getOperation());
    PatternRewriter rewriter(module->getContext());
    const AStarHeuristicRouter router(
        HeuristicWeights(alpha, lambda, nlookahead));
    std::unique_ptr<Architecture> arch(getArchitecture(archName));

    if (!arch) {
      const Location loc = UnknownLoc::get(&getContext());
      emitError(loc) << "unsupported architecture '" << archName << "'";
      return failure();
    }

    for (auto func : module.getOps<func::FuncOp>()) {
      LLVM_DEBUG(llvm::dbgs() << "handleFunc: " << func.getSymName() << '\n');

      if (!isEntryPoint(func)) {
        LLVM_DEBUG(llvm::dbgs() << "\tskip non entry\n");
        continue;
      }

      /// Iteratively process each unit in the function.
      std::queue<LayeredUnit> units;
      units.emplace(LayeredUnit::fromEntryPointFunction(func, arch->nqubits()));
      for (; !units.empty(); units.pop()) {
        LayeredUnit& unit = units.front();

        LLVM_DEBUG(unit.dump());

        SmallVector<QubitIndexPair> history;
        for (const auto window : unit.slidingWindow(nlookahead)) {
          Operation* anchor = window.opLayer->anchor;
          const ArrayRef<GateLayer> layers = window.gateLayers;
          const ArrayRef<Operation*> ops = window.opLayer->ops;

          /// Find and insert SWAPs.
          rewriter.setInsertionPoint(anchor);
          const auto swaps = router.route(layers, unit.layout(), *arch);
          if (!swaps) {
            const Location loc = UnknownLoc::get(&getContext());
            return emitError(loc, "A* failed to find a valid SWAP sequence");
          }

          if (!swaps->empty()) {
            history.append(*swaps);
            insertSWAPs(anchor->getLoc(), *swaps, unit.layout(), rewriter);
            numSwaps += swaps->size();
          }

          /// Process all operations contained in the layer.
          for (Operation* curr : ops) {
            rewriter.setInsertionPoint(curr);

            /// Re-order to fix any SSA Dominance issues.
            if (window.nextAnchor != nullptr) {
              rewriter.moveOpBefore(curr, window.nextAnchor);
            }

            /// Forward layout.
            TypeSwitch<Operation*>(curr)
                .Case<UnitaryInterface>([&](UnitaryInterface op) {
                  if (isa<SWAPOp>(op)) {
                    const auto ins = getIns(op);
                    unit.layout().swap(ins.first, ins.second);
                    history.push_back(
                        {unit.layout().lookupHardwareIndex(ins.first),
                         unit.layout().lookupHardwareIndex(ins.second)});
                  }
                  remap(op, unit.layout());
                })
                .Case<ResetOp>([&](ResetOp op) { remap(op, unit.layout()); })
                .Case<MeasureOp>(
                    [&](MeasureOp op) { remap(op, unit.layout()); })
                .Case<scf::YieldOp>([&](scf::YieldOp op) {
                  if (unit.restore()) {
                    rewriter.setInsertionPointAfter(op->getPrevNode());
                    insertSWAPs(op.getLoc(),
                                llvm::to_vector(llvm::reverse(history)),
                                unit.layout(), rewriter);
                  }
                })
                .Default([](auto) {
                  llvm_unreachable("unhandled 'curr' operation");
                });
          }
        }

        for (const auto& next : unit.next()) {
          units.emplace(next);
        }
      }
    }

    return success();
  }

  LogicalResult preflight() {
    if (archName.empty()) {
      const Location loc = UnknownLoc::get(&getContext());
      return emitError(loc, "required option 'arch' not provided");
    }

    return success();
  }
};

} // namespace
} // namespace mqt::ir::opt
