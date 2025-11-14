/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <cassert>
#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
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

#define DEBUG_TYPE "route-naive-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_NAIVEROUTINGPASSSC
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/// @brief A composite datastructure for LLVM Statistics.
struct Statistics {
  llvm::Statistic* numSwaps;
};

/// @brief Commonly passed parameters for the routing functions.
struct RoutingContext {
  /// @brief The targeted architecture.
  std::unique_ptr<Architecture> arch;
  /// @brief LLVM/MLIR statistics.
  Statistics stats;
  /// @brief A pattern rewriter.
  PatternRewriter rewriter;
};

LogicalResult processRegion(Region& region, Layout& layout,
                            SmallVector<QubitIndexPair>& history,
                            RoutingContext& ctx);

/**
 * @brief Insert SWAP ops at the rewriter's insertion point.
 *
 * @param location The location of the inserted SWAP ops.
 * @param swaps The hardware indices of the SWAPs.
 * @param layout The current layout.
 * @param rewriter The pattern rewriter.
 */
void insertSWAPs(Location location, ArrayRef<QubitIndexPair> swaps,
                 Layout& layout, PatternRewriter& rewriter) {
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

    auto swap = createSwap(location, in0, in1, rewriter);
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
 * @brief Given a layout, validate if the two-qubit unitary op is executable on
 * the targeted architecture.
 *
 * @param op The two-qubit unitary.
 * @param layout The current layout.
 * @param arch The targeted architecture.
 */
[[nodiscard]] bool isExecutable(UnitaryInterface op, const Layout& layout,
                                const Architecture& arch) {
  const auto ins = getIns(op);
  return arch.areAdjacent(layout.lookupHardwareIndex(ins.first),
                          layout.lookupHardwareIndex(ins.second));
}

/**
 * @brief Copy the layout and recursively process the loop body.
 */
WalkResult handle(scf::ForOp op, Layout& layout, RoutingContext& ctx) {
  /// Copy layout.
  Layout forLayout(layout);

  /// Forward out-of-loop and in-loop values.
  const auto initArgs = op.getInitArgs().take_front(ctx.arch->nqubits());
  const auto results = op.getResults().take_front(ctx.arch->nqubits());
  const auto iterArgs = op.getRegionIterArgs().take_front(ctx.arch->nqubits());
  for (const auto [arg, res, iter] : llvm::zip(initArgs, results, iterArgs)) {
    layout.remapQubitValue(arg, res);
    forLayout.remapQubitValue(arg, iter);
  }

  /// Recursively handle loop region.
  SmallVector<QubitIndexPair> history;
  return processRegion(op.getRegion(), forLayout, history, ctx);
}

/**
 * @brief Copy the layout for each branch and recursively map the branches.
 */
WalkResult handle(scf::IfOp op, Layout& layout, RoutingContext& ctx) {
  /// Recursively handle each branch region.
  Layout ifLayout(layout);
  SmallVector<QubitIndexPair> ifHistory;
  const auto ifRes =
      processRegion(op.getThenRegion(), ifLayout, ifHistory, ctx);
  if (ifRes.failed()) {
    return ifRes;
  }

  Layout elseLayout(layout);
  SmallVector<QubitIndexPair> elseHistory;
  const auto elseRes =
      processRegion(op.getElseRegion(), elseLayout, elseHistory, ctx);
  if (elseRes.failed()) {
    return elseRes;
  }

  /// Forward out-of-if values.
  const auto results = op->getResults().take_front(ctx.arch->nqubits());
  for (const auto [in, out] : llvm::zip(layout.getHardwareQubits(), results)) {
    layout.remapQubitValue(in, out);
  }

  return WalkResult::advance();
}

/**
 * @brief Indicates the end of a region defined by a scf op.
 *
 * Restores layout by uncomputation.
 */
WalkResult handle(scf::YieldOp op, Layout& layout,
                  ArrayRef<QubitIndexPair> history, RoutingContext& ctx) {
  /// Uncompute SWAPs.
  insertSWAPs(op.getLoc(), llvm::to_vector(llvm::reverse(history)), layout,
              ctx.rewriter);
  /// Count SWAPs.
  *(ctx.stats.numSwaps) += history.size();
  return WalkResult::advance();
}

/**
 * @brief Add hardware qubit with respective program & hardware index to
 * layout.
 *
 * Thanks to the placement pass, we can apply the identity layout here.
 */
WalkResult handle(QubitOp op, Layout& layout) {
  const std::size_t index = op.getIndex();
  layout.add(index, index, op.getQubit());
  return WalkResult::advance();
}

/**
 * @brief Use shortest path swapping to make the given unitary executable.
 * @details Optimized for an avg. SWAP count of 16.
 */
void findAndInsertSWAPs(UnitaryInterface op, Layout& layout,
                        SmallVector<QubitIndexPair>& history,
                        RoutingContext& ctx) {
  /// Find SWAPs.
  SmallVector<QubitIndexPair, 16> swaps;
  const auto ins = getIns(op);
  const auto hw0 = layout.lookupHardwareIndex(ins.first);
  const auto hw1 = layout.lookupHardwareIndex(ins.second);
  const auto path = ctx.arch->shortestPathBetween(hw0, hw1);
  for (std::size_t i = 0; i < path.size() - 2; ++i) {
    swaps.emplace_back(path[i], path[i + 1]);
  }

  /// Append SWAPs to history.
  history.append(swaps);

  /// Insert SWAPs.
  insertSWAPs(op.getLoc(), swaps, layout, ctx.rewriter);

  /// Count SWAPs.
  *(ctx.stats.numSwaps) += swaps.size();
}

/**
 * @brief Ensures the executability of two-qubit gates on the given target
 * architecture by inserting SWAPs.
 */
WalkResult handle(UnitaryInterface op, Layout& layout,
                  SmallVector<QubitIndexPair>& history, RoutingContext& ctx) {
  const std::vector<Value> inQubits = op.getAllInQubits();
  const std::vector<Value> outQubits = op.getAllOutQubits();
  const std::size_t nacts = inQubits.size();

  // Global-phase or zero-qubit unitary: Nothing to do.
  if (nacts == 0) {
    return WalkResult::advance();
  }

  if (isa<BarrierOp>(op)) {
    for (const auto [in, out] : llvm::zip(inQubits, outQubits)) {
      layout.remapQubitValue(in, out);
    }
    return WalkResult::advance();
  }

  /// Expect two-qubit gate decomposition.
  if (nacts > 2) {
    return op->emitOpError() << "acts on more than two qubits";
  }

  /// Single-qubit: Forward mapping.
  if (nacts == 1) {
    layout.remapQubitValue(inQubits[0], outQubits[0]);
    return WalkResult::advance();
  }

  if (!isExecutable(op, layout, *ctx.arch)) {
    findAndInsertSWAPs(op, layout, history, ctx);
  }

  const auto [execIn0, execIn1] = getIns(op);
  const auto [execOut0, execOut1] = getOuts(op);

  LLVM_DEBUG({
    llvm::dbgs() << llvm::format("handleUnitary: gate= p%d:h%d, p%d:h%d\n",
                                 layout.lookupProgramIndex(execIn0),
                                 layout.lookupHardwareIndex(execIn0),
                                 layout.lookupProgramIndex(execIn1),
                                 layout.lookupHardwareIndex(execIn1));
  });

  if (isa<SWAPOp>(op)) {
    layout.swap(execIn0, execIn1);
    history.push_back({layout.lookupHardwareIndex(execIn0),
                       layout.lookupHardwareIndex(execIn1)});
  }

  layout.remapQubitValue(execIn0, execOut0);
  layout.remapQubitValue(execIn1, execOut1);

  return WalkResult::advance();
}

LogicalResult processRegion(Region& region, Layout& layout,
                            SmallVector<QubitIndexPair>& history,
                            RoutingContext& ctx) {
  for (Operation& curr : region.getOps()) {
    const OpBuilder::InsertionGuard guard(ctx.rewriter);
    ctx.rewriter.setInsertionPoint(&curr);

    const auto res =
        TypeSwitch<Operation*, WalkResult>(&curr)
            /// mqtopt Dialect
            .Case<UnitaryInterface>([&](UnitaryInterface op) {
              return handle(op, layout, history, ctx);
            })
            .Case<QubitOp>([&](QubitOp op) { return handle(op, layout); })
            .Case<ResetOp>([&](ResetOp op) {
              remap(op, layout);
              return WalkResult::advance();
            })
            .Case<MeasureOp>([&](MeasureOp op) {
              remap(op, layout);
              return WalkResult::advance();
            })
            /// built-in Dialect
            .Case<ModuleOp>([&]([[maybe_unused]] ModuleOp op) {
              return WalkResult::advance();
            })
            /// func Dialect
            .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
              return WalkResult::advance();
            })
            /// scf Dialect
            .Case<scf::ForOp>(
                [&](scf::ForOp op) { return handle(op, layout, ctx); })
            .Case<scf::IfOp>(
                [&](scf::IfOp op) { return handle(op, layout, ctx); })
            .Case<scf::YieldOp>([&](scf::YieldOp op) {
              return handle(op, layout, history, ctx);
            })
            /// Skip the rest.
            .Default([](auto) { return WalkResult::skip(); });

    if (res.wasInterrupted()) {
      return failure();
    }
  }

  return success();
}

LogicalResult processFunction(func::FuncOp func, RoutingContext& ctx) {
  LLVM_DEBUG(llvm::dbgs() << "handleFunc: " << func.getSymName() << '\n');

  if (!isEntryPoint(func)) {
    LLVM_DEBUG(llvm::dbgs() << "\tskip non entry\n");
    return success(); // Ignore non entry_point functions for now.
  }

  Layout layout(ctx.arch->nqubits());
  SmallVector<QubitIndexPair> history;
  return processRegion(func.getBody(), layout, history, ctx);
}

/**
 * @brief Naively route the given module for the targeted architecture.
 *
 * @param module The module to route.
 * @param arch The targeted architecture.
 * @param stats The composite statistics datastructure.
 */
LogicalResult route(ModuleOp module, std::unique_ptr<Architecture> arch,
                    Statistics& stats) {
  RoutingContext ctx{.arch = std::move(arch),
                     .stats = stats,
                     .rewriter = PatternRewriter(module->getContext())};
  for (auto func : module.getOps<func::FuncOp>()) {
    if (processFunction(func, ctx).failed()) {
      return failure();
    }
  }
  return success();
}

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct NaiveRoutingPassSC final
    : impl::NaiveRoutingPassSCBase<NaiveRoutingPassSC> {
  using NaiveRoutingPassSCBase<NaiveRoutingPassSC>::NaiveRoutingPassSCBase;

  void runOnOperation() override {
    if (preflight().failed()) {
      signalPassFailure();
      return;
    }

    auto arch = getArchitecture(archName);
    if (!arch) {
      emitError(UnknownLoc::get(&getContext()))
          << "unsupported architecture '" << archName << "'";
      signalPassFailure();
      return;
    }

    Statistics stats{.numSwaps = &numSwaps};
    if (route(getOperation(), std::move(arch), stats).failed()) {
      signalPassFailure();
    };
  }

private:
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
