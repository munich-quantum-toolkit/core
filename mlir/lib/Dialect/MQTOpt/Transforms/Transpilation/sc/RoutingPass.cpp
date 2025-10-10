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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Planner.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Scheduler.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Stack.h"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
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
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <utility>
#include <vector>

#define DEBUG_TYPE "route-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGPASSSC
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/**
 * @brief Create and return SWAPOp for two qubits.
 *
 * Expects the rewriter to be set to the correct position.
 *
 * @param location The Location to attach to the created op.
 * @param in0 First input qubit SSA value.
 * @param in1 Second input qubit SSA value.
 * @param rewriter A PatternRewriter.
 * @return The created SWAPOp.
 */
[[nodiscard]] SWAPOp createSwap(Location location, Value in0, Value in1,
                                PatternRewriter& rewriter) {
  const SmallVector<Type> resultTypes{in0.getType(), in1.getType()};
  const SmallVector<Value> inQubits{in0, in1};

  return rewriter.create<SWAPOp>(
      /* location = */ location,
      /* out_qubits = */ resultTypes,
      /* pos_ctrl_out_qubits = */ TypeRange{},
      /* neg_ctrl_out_qubits = */ TypeRange{},
      /* static_params = */ nullptr,
      /* params_mask = */ nullptr,
      /* params = */ ValueRange{},
      /* in_qubits = */ inQubits,
      /* pos_ctrl_in_qubits = */ ValueRange{},
      /* neg_ctrl_in_qubits = */ ValueRange{});
}

/**
 * @brief Replace all uses of a value within a region and its nested regions,
 * except for a specific operation.
 *
 * @param oldValue The value to replace
 * @param newValue The new value to use
 * @param region The region in which to perform replacements
 * @param exceptOp Operation to exclude from replacements
 * @param rewriter The pattern rewriter
 */
void replaceAllUsesInRegionAndChildrenExcept(Value oldValue, Value newValue,
                                             Region* region,
                                             Operation* exceptOp,
                                             PatternRewriter& rewriter) {
  if (oldValue == newValue) {
    return;
  }

  rewriter.replaceUsesWithIf(oldValue, newValue, [&](OpOperand& use) {
    Operation* user = use.getOwner();
    if (user == exceptOp) {
      return false;
    }

    // For other blocks, check if in region tree
    Region* userRegion = user->getParentRegion();
    while (userRegion) {
      if (userRegion == region) {
        return true;
      }
      userRegion = userRegion->getParentRegion();
    }
    return false;
  });
}

class Router {
public:
  explicit Router(std::unique_ptr<Architecture> arch,
                  std::unique_ptr<SchedulerBase> layerizer,
                  std::unique_ptr<PlannerBase> planner, Pass::Statistic& nadd)
      : arch_(std::move(arch)), layerizer_(std::move(layerizer)),
        planner_(std::move(planner)), nadd_(&nadd) {}

  /**
   * @returns true iff @p op is executable on the targeted architecture.
   */
  [[nodiscard]] bool isExecutable(UnitaryInterface op) {
    const auto [in0, in1] = getIns(op);
    return arch().areAdjacent(stack().top().lookupHardwareIndex(in0),
                              stack().top().lookupHardwareIndex(in1));
  }

  /**
   * @brief Insert SWAPs such that the gates provided by the layerizer are
   * executable.
   */
  void route(UnitaryInterface op, PatternRewriter& rewriter) {
    const auto layers = layerizer_->schedule(op, stack().top());
    const auto swaps = planner_->plan(layers, stack().top(), arch());
    insert(swaps, op->getLoc(), rewriter);
    historyStack_.top().append(swaps.begin(), swaps.end());
  }

  /**
   * @brief Restore layout by uncomputing.
   *
   * History is cleared by the caller (e.g., via stack/history pop in handlers).
   *
   * @todo Remove SWAP history and use advanced strategies.
   */
  void restore(Location location, PatternRewriter& rewriter) {
    const auto swaps = llvm::to_vector(llvm::reverse(historyStack_.top()));
    insert(swaps, location, rewriter);
  }

  /**
   * @returns reference to the stack object.
   */
  [[nodiscard]] LayoutStack<Layout>& stack() { return stack_; }

  /**
   * @returns reference to the history stack object.
   */
  [[nodiscard]] LayoutStack<SmallVector<QubitIndexPair>>& historyStack() {
    return historyStack_;
  }

  /**
   * @returns reference to architecture object.
   */
  [[nodiscard]] Architecture& arch() const { return *arch_; }

private:
  void insert(ArrayRef<QubitIndexPair> swaps, Location location,
              PatternRewriter& rewriter) {
    for (const auto [hw0, hw1] : swaps) {
      const Value in0 = stack().top().lookupHardwareValue(hw0);
      const Value in1 = stack().top().lookupHardwareValue(hw1);

      LLVM_DEBUG({
        const auto [prog0, prog1] = stack().top().getProgramIndices(hw0, hw1);
        llvm::dbgs() << llvm::format(
            "route: swap= p%d:h%d, p%d:h%d <- p%d:h%d, p%d:h%d\n", prog1, hw0,
            prog0, hw1, prog0, hw0, prog1, hw1);
      });

      auto swap = createSwap(location, in0, in1, rewriter);
      const auto [out0, out1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      replaceAllUsesInRegionAndChildrenExcept(
          in0, out1, swap->getParentRegion(), swap, rewriter);
      replaceAllUsesInRegionAndChildrenExcept(
          in1, out0, swap->getParentRegion(), swap, rewriter);

      stack().top().swap(in0, in1);
      stack().top().remapQubitValue(in0, out0);
      stack().top().remapQubitValue(in1, out1);

      (*nadd_)++;
    }
  }

  std::unique_ptr<Architecture> arch_;
  std::unique_ptr<SchedulerBase> layerizer_;
  std::unique_ptr<PlannerBase> planner_;

  LayoutStack<Layout> stack_{};
  LayoutStack<SmallVector<QubitIndexPair>> historyStack_{};

  Pass::Statistic* nadd_;
};

/**
 * @brief Push new state onto the stack.
 */
WalkResult handleFunc([[maybe_unused]] func::FuncOp op, Router& router) {
  assert(router.stack().empty() && "handleFunc: stack must be empty");

  LLVM_DEBUG({
    llvm::dbgs() << "handleFunc: entry_point= " << op.getSymName() << '\n';
  });

  /// Function body state.
  router.stack().emplace(router.arch().nqubits());
  router.historyStack().emplace();

  return WalkResult::advance();
}

/**
 * @brief Indicates the end of a region defined by a function. Consequently,
 * we pop the region's state from the stack.
 */
WalkResult handleReturn(Router& router) {
  router.stack().pop();
  router.historyStack().pop();
  return WalkResult::advance();
}

/**
 * @brief Push new state for the loop body onto the stack.
 */
WalkResult handleFor(scf::ForOp op, Router& router) {
  /// Loop body state.
  router.stack().duplicateTop();
  router.historyStack().emplace();

  /// Forward out-of-loop and in-loop values.
  const auto initArgs = op.getInitArgs().take_front(router.arch().nqubits());
  const auto results = op.getResults().take_front(router.arch().nqubits());
  const auto iterArgs =
      op.getRegionIterArgs().take_front(router.arch().nqubits());
  for (const auto [arg, res, iter] : llvm::zip(initArgs, results, iterArgs)) {
    router.stack().getItemAtDepth(FOR_PARENT_DEPTH).remapQubitValue(arg, res);
    router.stack().top().remapQubitValue(arg, iter);
  }

  return WalkResult::advance();
}

/**
 * @brief Push two new states for the then and else branches onto the stack.
 */
WalkResult handleIf(scf::IfOp op, Router& router) {
  /// Prepare stack.
  router.stack().duplicateTop(); /// Else.
  router.stack().duplicateTop(); /// Then.
  router.historyStack().emplace();
  router.historyStack().emplace();

  /// Forward out-of-if values.
  const auto results = op->getResults().take_front(router.arch().nqubits());
  Layout& layoutBeforeIf = router.stack().getItemAtDepth(IF_PARENT_DEPTH);
  for (const auto [hw, res] : llvm::enumerate(results)) {
    const Value q = layoutBeforeIf.lookupHardwareValue(hw);
    layoutBeforeIf.remapQubitValue(q, res);
  }

  return WalkResult::advance();
}

/**
 * @brief Indicates the end of a region defined by a branching op.
 * Consequently, we pop the region's state from the stack.
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
WalkResult handleYield(scf::YieldOp op, Router& router,
                       [[maybe_unused]] PatternRewriter& rewriter) {
  if (!isa<scf::ForOp>(op->getParentOp()) &&
      !isa<scf::IfOp>(op->getParentOp())) {
    return WalkResult::skip();
  }

  router.restore(op->getLoc(), rewriter);
  router.stack().pop();
  router.historyStack().pop();

  return WalkResult::advance();
}

/**
 * @brief Add hardware qubit with respective program & hardware index to
 * layout.
 *
 * Thanks to the placement pass, we can apply the identity layout here.
 */
WalkResult handleQubit(QubitOp op, Router& router) {
  const std::size_t index = op.getIndex();
  router.stack().top().add(index, index, op.getQubit());
  return WalkResult::advance();
}

/**
 * @brief Ensures the executability of two-qubit gates on the given target
 * architecture by inserting SWAPs.
 */
WalkResult handleUnitary(UnitaryInterface op, Router& router,
                         PatternRewriter& rewriter) {
  const std::vector<Value> inQubits = op.getAllInQubits();
  const std::vector<Value> outQubits = op.getAllOutQubits();
  const std::size_t nacts = inQubits.size();

  // Global-phase or zero-qubit unitary: Nothing to do.
  if (nacts == 0) {
    return WalkResult::advance();
  }

  /// Expect two-qubit gate decomposition.
  if (nacts > 2) {
    if (isa<BarrierOp>(op)) {
      for (const auto [in, out] : llvm::zip(inQubits, outQubits)) {
        router.stack().top().remapQubitValue(in, out);
      }
      return WalkResult::advance();
    }
    return op->emitOpError() << "acts on more than two qubits";
  }

  /// Single-qubit: Forward mapping.
  if (nacts == 1) {
    router.stack().top().remapQubitValue(inQubits[0], outQubits[0]);
    return WalkResult::advance();
  }

  if (!router.isExecutable(op)) {
    router.route(op, rewriter);
  }

  const auto [execIn0, execIn1] = getIns(op);
  const auto [execOut0, execOut1] = getOuts(op);

  LLVM_DEBUG({
    llvm::dbgs() << llvm::format(
        "handleUnitary: gate= p%d:h%d, p%d:h%d\n",
        router.stack().top().lookupProgramIndex(execIn0),
        router.stack().top().lookupHardwareIndex(execIn0),
        router.stack().top().lookupProgramIndex(execIn1),
        router.stack().top().lookupHardwareIndex(execIn1));
  });

  if (isa<SWAPOp>(op)) {
    router.stack().top().swap(execIn0, execIn1);
  }

  router.stack().top().remapQubitValue(execIn0, execOut0);
  router.stack().top().remapQubitValue(execIn1, execOut1);

  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleReset(ResetOp op, Router& router) {
  router.stack().top().remapQubitValue(op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleMeasure(MeasureOp op, Router& router) {
  router.stack().top().remapQubitValue(op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Route the given module by inserting SWAPs.
 *
 * @details
 * Collects all functions marked with the 'entry_point' attribute, builds a
 * preorder worklist of their operations, and processes that list. Each
 * operation is handled via a TypeSwitch and may rewrite the IR in place via
 * the provided PatternRewriter. If any handler signals an error (interrupt),
 * this function returns failure.
 *
 * @note
 * We consciously avoid MLIR pattern drivers: Idiomatic MLIR transformation
 * patterns are independent and order-agnostic. Since we require state-sharing
 * between patterns for the transformation we violate this assumption.
 * Essentially this is also the reason why we can't utilize MLIR's
 * `applyPatternsGreedily` function. Moreover, we require pre-order traversal
 * which current drivers of MLIR don't support. However, even if such a driver
 * would exist, it would probably not return logical results which we require
 * for error-handling (similarly to `walkAndApplyPatterns`). Consequently, a
 * custom driver would be required in any case, which adds unnecessary code to
 * maintain.
 */
LogicalResult route(ModuleOp module, MLIRContext* ctx, Router& router) {
  PatternRewriter rewriter(ctx);

  /// Prepare work-list.
  SmallVector<Operation*> worklist;
  for (const auto func : module.getOps<func::FuncOp>()) {
    if (!isEntryPoint(func)) {
      continue; // Ignore non entry_point functions for now.
    }
    func->walk<WalkOrder::PreOrder>(
        [&](Operation* op) { worklist.push_back(op); });
  }

  /// Iterate work-list.
  for (Operation* curr : worklist) {
    if (curr == nullptr) {
      continue; // Skip erased ops.
    }

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(curr);

    const auto res =
        TypeSwitch<Operation*, WalkResult>(curr)
            /// built-in Dialect
            .Case<ModuleOp>([&]([[maybe_unused]] ModuleOp op) {
              return WalkResult::advance();
            })
            /// func Dialect
            .Case<func::FuncOp>(
                [&](func::FuncOp op) { return handleFunc(op, router); })
            .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
              return handleReturn(router);
            })
            /// scf Dialect
            .Case<scf::ForOp>(
                [&](scf::ForOp op) { return handleFor(op, router); })
            .Case<scf::IfOp>([&](scf::IfOp op) { return handleIf(op, router); })
            .Case<scf::YieldOp>([&](scf::YieldOp op) {
              return handleYield(op, router, rewriter);
            })
            /// mqtopt Dialect
            .Case<QubitOp>([&](QubitOp op) { return handleQubit(op, router); })
            .Case<ResetOp>([&](ResetOp op) { return handleReset(op, router); })
            .Case<MeasureOp>(
                [&](MeasureOp op) { return handleMeasure(op, router); })
            .Case<UnitaryInterface>([&](UnitaryInterface op) {
              return handleUnitary(op, router, rewriter);
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
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPassSC final : impl::RoutingPassSCBase<RoutingPassSC> {
  using RoutingPassSCBase<RoutingPassSC>::RoutingPassSCBase;

  void runOnOperation() override {
    Router router = getRouter();

    const auto start = std::chrono::steady_clock::now();
    if (failed(route(getOperation(), &getContext(), router))) {
      signalPassFailure();
    }
    const auto end = std::chrono::steady_clock::now();
    tms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
  }

private:
  [[nodiscard]] Router getRouter() {
    /// TODO: Configurable Architecture.
    auto arch = getArchitecture(ArchitectureName::MQTTest);

    switch (static_cast<RoutingMethod>(method)) {
    case RoutingMethod::Naive:
      LLVM_DEBUG({ llvm::dbgs() << "getRouter: method=naive\n"; });
      return Router(std::move(arch), std::make_unique<OneOpScheduler>(),
                    std::make_unique<NaivePlanner>(), nadd);
    case RoutingMethod::QMAP:
      LLVM_DEBUG({ llvm::dbgs() << "getRouter: method=qmap\n"; });
      const HeuristicWeights weights(alpha, beta, lambda, nlookahead);
      return Router(std::move(arch),
                    std::make_unique<CrawlScheduler>(nlookahead),
                    std::make_unique<QMAPPlanner>(weights), nadd);
    }

    llvm_unreachable("Unknown method");
  }
};

} // namespace
} // namespace mqt::ir::opt
