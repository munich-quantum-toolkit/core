/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/RoutingDriverBase.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <utility>

#define DEBUG_TYPE "route-sc"

namespace mqt::ir::opt {

using namespace mlir;

class NaiveDriver final : public RoutingDriverBase {
  using SWAPHistory = SmallVector<QubitIndexPair>;

public:
  using RoutingDriverBase::RoutingDriverBase;

private:
  LogicalResult rewrite(func::FuncOp func, PatternRewriter& rewriter) override {
    LLVM_DEBUG(llvm::dbgs() << "handleFunc: " << func.getSymName() << '\n');

    if (!isEntryPoint(func)) {
      LLVM_DEBUG(llvm::dbgs() << "\tskip non entry\n");
      return success(); // Ignore non entry_point functions for now.
    }

    Layout layout(arch->nqubits());
    SWAPHistory history;
    return rewrite(func.getBody(), layout, history, rewriter);
  }

  LogicalResult rewrite(Region& region, Layout& layout, SWAPHistory& history,
                        PatternRewriter& rewriter) const {
    for (Operation& curr : region.getOps()) {
      const OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(&curr);

      const auto res =
          TypeSwitch<Operation*, WalkResult>(&curr)
              /// mqtopt Dialect
              .Case<UnitaryInterface>([&](UnitaryInterface op) {
                return handleUnitary(op, layout, history, rewriter);
              })
              .Case<QubitOp>(
                  [&](QubitOp op) { return handleQubit(op, layout); })
              .Case<ResetOp>(
                  [&](ResetOp op) { return handleReset(op, layout); })
              .Case<MeasureOp>(
                  [&](MeasureOp op) { return handleMeasure(op, layout); })
              /// built-in Dialect
              .Case<ModuleOp>([&]([[maybe_unused]] ModuleOp op) {
                return WalkResult::advance();
              })
              /// func Dialect
              .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
                return WalkResult::advance();
              })
              /// scf Dialect
              .Case<scf::ForOp>([&](scf::ForOp op) {
                return handleFor(op, layout, rewriter);
              })
              .Case<scf::IfOp>(
                  [&](scf::IfOp op) { return handleIf(op, layout, rewriter); })
              .Case<scf::YieldOp>([&](scf::YieldOp op) {
                return handleYield(op, layout, history, rewriter);
              })
              /// Skip the rest.
              .Default([](auto) { return WalkResult::skip(); });

      if (res.wasInterrupted()) {
        return failure();
      }
    }

    return success();
  }

  /**
   * @brief Copy the layout and recursively map the loop body.
   */
  WalkResult handleFor(scf::ForOp op, Layout& layout,
                       PatternRewriter& rewriter) const {
    /// Copy layout.
    Layout forLayout(layout);

    /// Forward out-of-loop and in-loop values.
    const auto initArgs = op.getInitArgs().take_front(arch->nqubits());
    const auto results = op.getResults().take_front(arch->nqubits());
    const auto iterArgs = op.getRegionIterArgs().take_front(arch->nqubits());
    for (const auto [arg, res, iter] : llvm::zip(initArgs, results, iterArgs)) {
      layout.remapQubitValue(arg, res);
      forLayout.remapQubitValue(arg, iter);
    }

    /// Recursively handle loop region.
    SWAPHistory history;
    return rewrite(op.getRegion(), forLayout, history, rewriter);
  }

  /**
   * @brief Copy the layout for each branch and recursively map the branches.
   */
  WalkResult handleIf(scf::IfOp op, Layout& layout,
                      PatternRewriter& rewriter) const {
    /// Recursively handle each branch region.
    Layout ifLayout(layout);
    SWAPHistory ifHistory;
    const auto ifRes =
        rewrite(op.getThenRegion(), ifLayout, ifHistory, rewriter);
    if (ifRes.failed()) {
      return ifRes;
    }

    Layout elseLayout(layout);
    SWAPHistory elseHistory;
    const auto elseRes =
        rewrite(op.getElseRegion(), elseLayout, elseHistory, rewriter);
    if (elseRes.failed()) {
      return elseRes;
    }

    /// Forward out-of-if values.
    const auto results = op->getResults().take_front(arch->nqubits());
    for (const auto [hw, res] : llvm::enumerate(results)) {
      const Value q = layout.lookupHardwareValue(hw);
      layout.remapQubitValue(q, res);
    }

    return WalkResult::advance();
  }

  /**
   * @brief Indicates the end of a region defined by a scf op.
   *
   * Restores layout by uncomputation and replaces (invalid) yield.
   *
   * Using uncompute has the advantages of (1) being intuitive and
   * (2) preserving the optimality of the original SWAP sequence.
   * Essentially the better the routing algorithm the better the
   * uncompute. Moreover, this has the nice property that routing
   * a 'for' of 'if' region always requires 2 * #(SWAPs required for region)
   * additional SWAPS.
   */
  static WalkResult handleYield(scf::YieldOp op, Layout& layout,
                                SWAPHistory& history,
                                PatternRewriter& rewriter) {
    if (!isa<scf::ForOp>(op->getParentOp()) &&
        !isa<scf::IfOp>(op->getParentOp())) {
      return WalkResult::skip();
    }

    /// Uncompute SWAPs.
    RoutingDriverBase::insertSWAPs(llvm::to_vector(llvm::reverse(history)),
                                   layout, op.getLoc(), rewriter);

    return WalkResult::advance();
  }

  /**
   * @brief Add hardware qubit with respective program & hardware index to
   * layout.
   *
   * Thanks to the placement pass, we can apply the identity layout here.
   */
  static WalkResult handleQubit(QubitOp op, Layout& layout) {
    const std::size_t index = op.getIndex();
    layout.add(index, index, op.getQubit());
    return WalkResult::advance();
  }

  /**
   * @brief Ensures the executability of two-qubit gates on the given target
   * architecture by inserting SWAPs.
   */
  WalkResult handleUnitary(UnitaryInterface op, Layout& layout,
                           SWAPHistory& history,
                           PatternRewriter& rewriter) const {
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

    if (!isExecutable(op, layout)) {
      route(op, layout, history, rewriter);
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

  /**
   * @brief Update layout.
   */
  static WalkResult handleReset(ResetOp op, Layout& layout) {
    layout.remapQubitValue(op.getInQubit(), op.getOutQubit());
    return WalkResult::advance();
  }

  /**
   * @brief Update layout.
   */
  static WalkResult handleMeasure(MeasureOp op, Layout& layout) {
    layout.remapQubitValue(op.getInQubit(), op.getOutQubit());
    return WalkResult::advance();
  }

  /**
   * @brief Use shortest path swapping to make the given unitary executable.
   * @details Optimized for an avg. SWAP count of 16.
   */
  void route(UnitaryInterface op, Layout& layout, SWAPHistory& history,
             PatternRewriter& rewriter) const {
    /// Find SWAPs.
    SmallVector<QubitIndexPair, 16> swaps;
    const auto ins = getIns(op);
    const auto hw0 = layout.lookupHardwareIndex(ins.first);
    const auto hw1 = layout.lookupHardwareIndex(ins.second);
    const auto path = arch->shortestPathBetween(hw0, hw1);
    for (std::size_t i = 0; i < path.size() - 2; ++i) {
      swaps.emplace_back(path[i], path[i + 1]);
    }

    /// Append SWAPs to history.
    history.append(swaps);

    /// Insert SWAPs.
    RoutingDriverBase::insertSWAPs(swaps, layout, op.getLoc(), rewriter);
  }
};
} // namespace mqt::ir::opt
