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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Router.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/SequentialUnit.h"

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
#include <utility>

#define DEBUG_TYPE "route-naive-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_NAIVEROUTINGPASSSC
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/**
 * @brief Simple pre-order traversal of the IR that routes any non-executable
 * gates by inserting SWAPs along the shortest path.
 */
struct NaiveRoutingPassSC final
    : impl::NaiveRoutingPassSCBase<NaiveRoutingPassSC> {
  using NaiveRoutingPassSCBase<NaiveRoutingPassSC>::NaiveRoutingPassSCBase;

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
  LogicalResult route() {
    ModuleOp module(getOperation());
    PatternRewriter rewriter(module->getContext());
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
      std::queue<SequentialUnit> units;
      units.emplace(
          SequentialUnit::fromEntryPointFunction(func, arch->nqubits()));
      for (; !units.empty(); units.pop()) {
        SequentialUnit& unit = units.front();

        SmallVector<QubitIndexPair> history;
        for (Operation& curr : unit) {
          rewriter.setInsertionPoint(&curr);

          /// Forward layout.
          TypeSwitch<Operation*>(&curr)
              .Case<UnitaryInterface>([&](UnitaryInterface op) {
                if (isTwoQubitGate(op)) {
                  if (!isExecutable(op, unit.layout(), *arch)) {
                    const auto ins = getIns(op);
                    const auto gate = std::make_pair(
                        unit.layout().lookupProgramIndex(ins.first),
                        unit.layout().lookupProgramIndex(ins.second));
                    const auto swaps =
                        NaiveRouter::route(gate, unit.layout(), *arch);
                    if (!swaps.empty()) {
                      history.append(swaps);
                      insertSWAPs(op->getLoc(), swaps, unit.layout(), rewriter);
                      numSwaps += swaps.size();

                      LLVM_DEBUG({
                        for (const auto [hw0, hw1] : swaps) {
                          llvm::dbgs() << llvm::format(
                              "route: swap= hw(%d, %d)\n", hw0, hw1);
                        }
                      });
                    }
                  }
                }

                if (auto swap = dyn_cast<SWAPOp>(op.getOperation())) {
                  const auto in0 = swap.getInQubits()[0];
                  history.emplace_back(unit.layout().lookupHardwareIndex(in0),
                                       unit.layout().lookupHardwareIndex(in1));
                }
                unit.layout().remap(op);
              })
              .Case<ResetOp>([&](ResetOp op) { unit.layout().remap(op); })
              .Case<MeasureOp>([&](MeasureOp op) { unit.layout().remap(op); })
              .Case<scf::YieldOp>([&](scf::YieldOp op) {
                if (unit.restore()) {
                  rewriter.setInsertionPointAfter(op->getPrevNode());
                  insertSWAPs(op.getLoc(), llvm::reverse(history),
                              unit.layout(), rewriter);
                }
              });
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
      return emitError(UnknownLoc::get(&getContext()),
                       "required option 'arch' not provided");
    }

    return success();
  }
};

} // namespace
} // namespace mqt::ir::opt
