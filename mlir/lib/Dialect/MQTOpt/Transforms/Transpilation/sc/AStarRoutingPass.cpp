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

#include <algorithm>
#include <cassert>
#include <iterator>
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
        for (const auto& [i, layer] : llvm::enumerate(unit)) {

          /// Compute sliding window.
          const auto len = std::min(1 + nlookahead, unit.size() - i);
          SmallVector<ArrayRef<QubitIndexPair>> window;
          window.reserve(len);
          llvm::transform(ArrayRef(unit.begin(), unit.end()).slice(i, len),
                          std::back_inserter(window), [&](const Layer& l) {
                            return ArrayRef(l.twoQubitProgs);
                          });

          /// Find and insert SWAPs.
          rewriter.setInsertionPoint(layer.anchor);
          const auto swaps = router.route(window, unit.layout(), *arch);
          if (!swaps) {
            const Location loc = UnknownLoc::get(&getContext());
            return emitError(loc, "A* failed to find a valid SWAP sequence");
          }

          if (!swaps->empty()) {
            history.append(*swaps);
            insertSWAPs(layer.anchor->getLoc(), *swaps, unit.layout(),
                        rewriter);
            numSwaps += swaps->size();

            LLVM_DEBUG({
              for (const auto [hw0, hw1] : *swaps) {
                llvm::dbgs()
                    << llvm::format("route: swap= hw(%d, %d)\n", hw0, hw1);
              }
            });
          }

          /// Process all operations contained in the layer.
          for (Operation* curr : layer.ops) {
            rewriter.setInsertionPoint(curr);

            /// Re-order to fix any SSA Dominance issues.
            if (i + 1 < unit.size()) {
              rewriter.moveOpBefore(curr, unit[i + 1].anchor);
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
                  unit.layout().remap(op);
                })
                .Case<ResetOp>([&](ResetOp op) { unit.layout().remap(op); })
                .Case<MeasureOp>([&](MeasureOp op) { unit.layout().remap(op); })
                .Case<scf::YieldOp>([&](scf::YieldOp op) {
                  if (unit.restore()) {
                    rewriter.setInsertionPoint(op);
                    insertSWAPs(op.getLoc(), llvm::reverse(history),
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
