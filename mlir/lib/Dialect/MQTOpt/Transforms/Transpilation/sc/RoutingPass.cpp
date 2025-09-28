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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Stack.h"

#include <cassert>
#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
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
 * @brief A function attribute that specifies an (QIR) entry point function.
 */
constexpr llvm::StringLiteral ENTRY_POINT_ATTR{"entry_point"};

/**
 * @brief Attribute to forward function-level attributes to LLVM IR.
 */
constexpr llvm::StringLiteral PASSTHROUGH_ATTR{"passthrough"};

/**
 * @brief 'For' pushes once onto the stack, hence the parent is at depth one.
 */
constexpr std::size_t FOR_PARENT_DEPTH = 1UL;

/**
 * @brief 'If' pushes twice onto the stack, hence the parent is at depth two.
 */
constexpr std::size_t IF_PARENT_DEPTH = 2UL;

/**
 * @brief The datatype for qubit indices. For now, 64bit.
 */
using QubitIndex = std::size_t;

/**
 * @brief A pair of program indices.
 */
using ProgramIndexPair = std::pair<QubitIndex, QubitIndex>;

/**
 * @brief Check if a unitary acts on two qubits.
 * @param u A unitary.
 * @returns True iff the qubit gate acts on two qubits.
 */
[[nodiscard]] bool isTwoQubitGate(UnitaryInterface u) {
  return u.getAllInQubits().size() == 2;
}

/**
 * @brief Return input qubit pair for a two-qubit unitary.
 * @param u A two-qubit unitary.
 * @return Pair of SSA values consisting of the first and second in-qubits.
 */
[[nodiscard]] std::pair<Value, Value> getIns(UnitaryInterface op) {
  assert(isTwoQubitGate(op));
  const std::vector<Value> inQubits = op.getAllInQubits();
  return {inQubits[0], inQubits[1]};
}

/**
 * @brief Return output qubit pair for a two-qubit unitary.
 * @param u A two-qubit unitary.
 * @return Pair of SSA values consisting of the first and second out-qubits.
 */
[[nodiscard]] std::pair<Value, Value> getOuts(UnitaryInterface op) {
  assert(isTwoQubitGate(op));
  const std::vector<Value> outQubits = op.getAllOutQubits();
  return {outQubits[0], outQubits[1]};
}

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

  // Create a predicate function that checks if the use is:
  // 1. In the specified region or one of its nested regions
  // 2. Not in the excepted operation
  const auto isInRegionAndNotExcepted = [&](OpOperand& ops) -> bool {
    Operation* user = ops.getOwner();

    // Skip the excepted operation
    if (user == exceptOp) {
      return false;
    }

    // Check if the user is in the specified region or a child region
    Region* userRegion = user->getParentRegion();
    while (userRegion != nullptr) {
      if (userRegion == region) {
        return true;
      }
      userRegion = userRegion->getParentRegion();
    }

    return false;
  };

  rewriter.replaceUsesWithIf(oldValue, newValue, isInRegionAndNotExcepted);
}

/**
 * @brief Return true if the function contains "entry_point" in the passthrough
 * attribute.
 */
bool isEntryPoint(func::FuncOp op) {
  const auto passthroughAttr = op->getAttrOfType<ArrayAttr>(PASSTHROUGH_ATTR);
  if (!passthroughAttr) {
    return false;
  }

  return llvm::any_of(passthroughAttr, [](const Attribute attr) {
    return isa<StringAttr>(attr) && cast<StringAttr>(attr) == ENTRY_POINT_ATTR;
  });
}

struct StackItem {
  explicit StackItem(const std::size_t nqubits) : state(nqubits) {}
  Layout<QubitIndex> state;
  SmallVector<ProgramIndexPair, 32> history;
};

class StateStack : public TranspilationStack<StackItem> {
public:
  /**
   * @brief Returns the most recent state of the stack.
   */
  [[nodiscard]] Layout<QubitIndex>& topState() { return top().state; }

  /**
   * @brief Returns the item at the specified depth from the top of the stack.
   */
  [[nodiscard]] Layout<QubitIndex>& getStateAtDepth(std::size_t depth) {
    return getItemAtDepth(depth).state;
  }

  /**
   * @brief Duplicates the top state.
   */
  void duplicateTopState() {
    duplicateTop();
    top().history.clear();
  }

  /**
   * @brief Return the current (most recent) swap history.
   */
  [[nodiscard]] ArrayRef<ProgramIndexPair> getHistory() {
    return top().history;
  }

  /**
   * @brief Record a swap.
   */
  void recordSwap(QubitIndex programIdx0, QubitIndex programIdx1) {
    top().history.emplace_back(programIdx0, programIdx1);
  }
};

/**
 * @brief Returns true iff @p op is executable on the targeted architecture.
 */
[[nodiscard]] bool isExecutable(UnitaryInterface op, Layout<QubitIndex>& state,
                                const Architecture& arch) {
  const auto [in0, in1] = getIns(op);
  return arch.areAdjacent(state.lookupHardware(in0), state.lookupHardware(in1));
}

/**
 * @brief Get shortest path between @p qStart and @p qEnd.
 */
[[nodiscard]] llvm::SmallVector<std::size_t> getPath(const Value qStart,
                                                     const Value qEnd,
                                                     Layout<QubitIndex>& state,
                                                     const Architecture& arch) {
  return arch.shortestPathBetween(state.lookupHardware(qStart),
                                  state.lookupHardware(qEnd));
}

/**
 * @brief The necessary datastructures to route a quantum-classical module.
 */
struct RoutingContext {
  explicit RoutingContext(Architecture& arch) : arch(&arch) {}

  Architecture* arch;
  StateStack stack{};
};

/**
 * @brief Base class for all routing algorithms.
 */
class RouterBase {
public:
  virtual ~RouterBase() = default;

  /**
   * @brief Ensures the executability of two-qubit gates on the given target
   * architecture by inserting SWAPs greedily.
   */
  WalkResult handleUnitary(UnitaryInterface op, RoutingContext& ctx,
                           PatternRewriter& rewriter) {
    const std::vector<Value> inQubits = op.getAllInQubits();
    const std::vector<Value> outQubits = op.getAllOutQubits();
    const std::size_t nacts = inQubits.size();

    // Global-phase or zero-qubit unitary: Nothing to do.
    if (nacts == 0) {
      return WalkResult::advance();
    }

    if (nacts > 2) {
      return op->emitOpError() << "acts on more than two qubits";
    }

    // Single-qubit: Forward mapping.
    if (nacts == 1) {
      ctx.stack.topState().remapQubitValue(inQubits[0], outQubits[0]);
      return WalkResult::advance();
    }

    if (!isExecutable(op, ctx.stack.topState(), *ctx.arch)) {
      makeExecutable(op, ctx, rewriter);
    }

    const auto [execIn0, execIn1] = getIns(op);
    const auto [execOut0, execOut1] = getOuts(op);

    LLVM_DEBUG({
      llvm::dbgs() << llvm::format(
          "handleUnitary: gate= s%d/h%d, s%d/h%d\n",
          ctx.stack.topState().lookupProgram(execIn0),
          ctx.stack.topState().lookupHardware(execIn0),
          ctx.stack.topState().lookupProgram(execIn1),
          ctx.stack.topState().lookupHardware(execIn1));
    });

    ctx.stack.topState().remapQubitValue(execIn0, execOut0);
    ctx.stack.topState().remapQubitValue(execIn1, execOut1);

    return WalkResult::advance();
  }

protected:
  /**
   * @brief Insert SWAPs such that @p u is executable.
   */
  virtual void makeExecutable(UnitaryInterface op, RoutingContext& ctx,
                              PatternRewriter& rewriter) = 0;
};

/**
 * @brief Inserts SWAPs along the shortest path between two hardware
 * qubits.
 */
class NaiveRouter final : public RouterBase {
protected:
  void makeExecutable(UnitaryInterface op, RoutingContext& ctx,
                      PatternRewriter& rewriter) final {
    assert(isTwoQubitGate(op) && "makeExecutable: must be two-qubit gate");

    const auto [qStart, qEnd] = getIns(op);
    const auto path = getPath(qStart, qEnd, ctx.stack.topState(), *ctx.arch);

    for (std::size_t i = 0; i < path.size() - 2; ++i) {
      const QubitIndex hardwareIdx0 = path[i];
      const QubitIndex hardwareIdx1 = path[i + 1];

      const Value qIn0 = ctx.stack.topState().lookupHardware(hardwareIdx0);
      const Value qIn1 = ctx.stack.topState().lookupHardware(hardwareIdx1);

      const QubitIndex programIdx0 = ctx.stack.topState().lookupProgram(qIn0);
      const QubitIndex programIdx1 = ctx.stack.topState().lookupProgram(qIn1);

      LLVM_DEBUG({
        llvm::dbgs() << llvm::format(
            "makeExecutable: swap= s%d/h%d, s%d/h%d <- s%d/h%d, s%d/h%d\n",
            programIdx1, hardwareIdx0, programIdx0, hardwareIdx1, programIdx0,
            hardwareIdx0, programIdx1, hardwareIdx1);
      });

      auto swap = createSwap(op->getLoc(), qIn0, qIn1, rewriter);
      const auto [qOut0, qOut1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      replaceAllUsesInRegionAndChildrenExcept(
          qIn0, qOut1, swap->getParentRegion(), swap, rewriter);
      replaceAllUsesInRegionAndChildrenExcept(
          qIn1, qOut0, swap->getParentRegion(), swap, rewriter);

      ctx.stack.recordSwap(programIdx0, programIdx1);

      auto& state = ctx.stack.topState();
      state.swap(qIn0, qIn1);
      state.remapQubitValue(qIn0, qOut0);
      state.remapQubitValue(qIn1, qOut1);
    }
  }
};

/**
 * @brief Push new state onto the stack.
 */
WalkResult handleFunc([[maybe_unused]] func::FuncOp op, RoutingContext& ctx) {
  assert(ctx.stack.empty() && "handleFunc: stack must be empty");

  LLVM_DEBUG({
    llvm::dbgs() << "handleFunc: entry_point= " << op.getSymName() << '\n';
  });

  /// Function body state.
  ctx.stack.emplace(ctx.arch->nqubits());

  return WalkResult::advance();
}

/**
 * @brief Defines the end of a region (and hence routing state) defined by a
 * function. Consequently, we pop the region's state from the stack. Since
 * we currently only route entry_point functions we do not need to return
 * all static qubits here.
 */
WalkResult handleReturn(RoutingContext& ctx) {
  ctx.stack.pop();
  return WalkResult::advance();
}

/**
 * @brief Push new state for the loop body onto the stack.
 */
WalkResult handleFor(scf::ForOp op, RoutingContext& ctx) {
  /// Loop body state.
  ctx.stack.duplicateTopState();

  /// Forward out-of-loop and in-loop values.
  const auto initArgs = op.getInitArgs().take_front(ctx.arch->nqubits());
  const auto results = op.getResults().take_front(ctx.arch->nqubits());
  const auto iterArgs = op.getRegionIterArgs().take_front(ctx.arch->nqubits());
  for (const auto [arg, res, iter] : llvm::zip(initArgs, results, iterArgs)) {
    ctx.stack.getStateAtDepth(FOR_PARENT_DEPTH).remapQubitValue(arg, res);
    ctx.stack.topState().remapQubitValue(arg, iter);
  }

  return WalkResult::advance();
}

/**
 * @brief Push two new states for the then and else branches onto the stack.
 */
WalkResult handleIf(scf::IfOp op, RoutingContext& ctx) {
  /// Prepare stack.
  ctx.stack.duplicateTopState(); /// Else.
  ctx.stack.duplicateTopState(); /// Then.

  /// Forward out-of-if values.
  const auto results = op->getResults().take_front(ctx.arch->nqubits());
  Layout<QubitIndex>& stateBeforeIf =
      ctx.stack.getStateAtDepth(IF_PARENT_DEPTH);
  for (const auto [hardwareIdx, res] : llvm::enumerate(results)) {
    const Value q = stateBeforeIf.lookupHardware(hardwareIdx);
    stateBeforeIf.remapQubitValue(q, res);
  }

  return WalkResult::advance();
}

/**
 * @brief Defines the end of a region (and hence routing state) defined by a
 * branching op. Consequently, we pop the region's state from the stack.
 *
 * Restores layout by uncomputation and replaces (invalid) yield.
 *
 * The results of the yield op are extended by the missing hardware
 * qubits, similarly to the 'for' and 'if' op. This is only possible
 * because we restore the layout - the mapping from hardware to program
 * qubits (and vice versa).
 *
 * Using uncompute has the advantages of (1) being intuitive and
 * (2) preserving the optimality of the original SWAP sequence.
 * Essentially the better the routing algorithm the better the
 * uncompute. Moreover, this has the nice property that routing
 * a 'for' of 'if' region always requires 2 * #(SWAPs required for region)
 * additional SWAPS.
 */
WalkResult handleYield(scf::YieldOp op, RoutingContext& ctx,
                       PatternRewriter& rewriter) {
  if (!isa<scf::ForOp>(op->getParentOp()) &&
      !isa<scf::IfOp>(op->getParentOp())) {
    return WalkResult::skip();
  }

  for (const auto [programIdx0, programIdx1] :
       llvm::reverse(ctx.stack.getHistory())) {
    const Value qIn0 = ctx.stack.topState().lookupProgram(programIdx0);
    const Value qIn1 = ctx.stack.topState().lookupProgram(programIdx1);

    auto swap = createSwap(op->getLoc(), qIn0, qIn1, rewriter);
    const auto [qOut0, qOut1] = getOuts(swap);

    rewriter.setInsertionPointAfter(swap);
    replaceAllUsesInRegionAndChildrenExcept(
        qIn0, qOut1, swap->getParentRegion(), swap, rewriter);
    replaceAllUsesInRegionAndChildrenExcept(
        qIn1, qOut0, swap->getParentRegion(), swap, rewriter);

    ctx.stack.topState().swap(qIn0, qIn1);
    ctx.stack.topState().remapQubitValue(qIn0, qOut0);
    ctx.stack.topState().remapQubitValue(qIn1, qOut1);
  }

  assert(llvm::equal(ctx.stack.topState().getCurrentLayout(),
                     ctx.stack.getStateAtDepth(1).getCurrentLayout()) &&
         "layouts must match after restoration");

  ctx.stack.pop();

  return WalkResult::advance();
}

/**
 * @brief Add hardware qubit with respective program & hardware index to layout.
 *
 * Thanks to the placement pass, we can apply the identity layout here.
 */
WalkResult handleQubit(QubitOp op, RoutingContext& ctx) {
  const std::size_t index = op.getIndex();
  ctx.stack.topState().add(index, index, op.getQubit());
  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleReset(ResetOp op, RoutingContext& ctx) {
  ctx.stack.topState().remapQubitValue(op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleMeasure(MeasureOp op, RoutingContext& ctx) {
  ctx.stack.topState().remapQubitValue(op.getInQubit(), op.getOutQubit());
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
LogicalResult route(ModuleOp module, MLIRContext* mlirCtx, RoutingContext& ctx,
                    RouterBase& router) {
  PatternRewriter rewriter(mlirCtx);

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
            .Case<ModuleOp>(
                [&](ModuleOp /* op */) { return WalkResult::advance(); })
            /// func Dialect
            .Case<func::FuncOp>(
                [&](func::FuncOp op) { return handleFunc(op, ctx); })
            .Case<func::ReturnOp>(
                [&](func::ReturnOp /* op */) { return handleReturn(ctx); })
            /// scf Dialect
            .Case<scf::ForOp>([&](scf::ForOp op) { return handleFor(op, ctx); })
            .Case<scf::IfOp>([&](scf::IfOp op) { return handleIf(op, ctx); })
            .Case<scf::YieldOp>(
                [&](scf::YieldOp op) { return handleYield(op, ctx, rewriter); })
            /// mqtopt Dialect
            .Case<QubitOp>([&](QubitOp op) { return handleQubit(op, ctx); })
            .Case<ResetOp>([&](ResetOp op) { return handleReset(op, ctx); })
            .Case<MeasureOp>(
                [&](MeasureOp op) { return handleMeasure(op, ctx); })
            .Case<UnitaryInterface>([&](UnitaryInterface op) {
              return router.handleUnitary(op, ctx, rewriter);
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
  void runOnOperation() override {
    auto arch = getArchitecture(ArchitectureName::MQTTest);
    auto router = std::make_unique<NaiveRouter>();

    RoutingContext ctx(*arch);

    if (failed(route(getOperation(), &getContext(), ctx, *router))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mqt::ir::opt
