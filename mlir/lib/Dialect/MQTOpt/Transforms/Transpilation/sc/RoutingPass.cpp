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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Stack.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseSet.h"

#include <cassert>
#include <cstddef>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
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

struct StackItem {
  explicit StackItem(const std::size_t nqubits) : state(nqubits) {}
  Layout<QubitIndex> state;
  SmallVector<ProgramIndexPair, 32> history;
};

class StateStack : public LayoutStack<StackItem> {
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
 * @brief Base class for all routing algorithms.
 */
class RouterBase {
public:
  explicit RouterBase(Pass::Statistic& nadd) : nadd(&nadd) {}

  virtual ~RouterBase() = default;

  virtual SmallVector<QubitIndexPair>
  getSwapPlan(UnitaryInterface op, const Layout<QubitIndex>& layout,
              const Architecture& arch) = 0;

  /**
   * @brief Insert SWAPs such that @p u is executable.
   */
  void route(UnitaryInterface op, StateStack& stack, const Architecture& arch,
             PatternRewriter& rewriter) {
    assert(isTwoQubitGate(op) && "route: must be two-qubit gate");

    for (const auto [hardwareIdx0, hardwareIdx1] :
         getSwapPlan(op, stack.topState(), arch)) {
      const Value qIn0 = stack.topState().lookupHardware(hardwareIdx0);
      const Value qIn1 = stack.topState().lookupHardware(hardwareIdx1);

      const QubitIndex programIdx0 = stack.topState().lookupProgram(qIn0);
      const QubitIndex programIdx1 = stack.topState().lookupProgram(qIn1);

      LLVM_DEBUG({
        llvm::dbgs() << llvm::format(
            "route: swap= s%d/h%d, s%d/h%d <- s%d/h%d, s%d/h%d\n", programIdx1,
            hardwareIdx0, programIdx0, hardwareIdx1, programIdx0, hardwareIdx0,
            programIdx1, hardwareIdx1);
      });

      auto swap = createSwap(op->getLoc(), qIn0, qIn1, rewriter);
      const auto [qOut0, qOut1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      replaceAllUsesInRegionAndChildrenExcept(
          qIn0, qOut1, swap->getParentRegion(), swap, rewriter);
      replaceAllUsesInRegionAndChildrenExcept(
          qIn1, qOut0, swap->getParentRegion(), swap, rewriter);

      stack.recordSwap(programIdx0, programIdx1);

      stack.topState().swap(qIn0, qIn1);
      stack.topState().remapQubitValue(qIn0, qOut0);
      stack.topState().remapQubitValue(qIn1, qOut1);

      (*nadd)++;
    }
  }

  /**
   * @brief Restore layout by uncomputing.
   *
   * @todo Remove SWAP history and use advanced strategies.
   */
  virtual void restore(Layout<QubitIndex>& layout,
                       ArrayRef<ProgramIndexPair> history, Location location,
                       PatternRewriter& rewriter) {
    for (const auto [programIdx0, programIdx1] : llvm::reverse(history)) {
      const Value qIn0 = layout.lookupProgram(programIdx0);
      const Value qIn1 = layout.lookupProgram(programIdx1);

      auto swap = createSwap(location, qIn0, qIn1, rewriter);
      const auto [qOut0, qOut1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      replaceAllUsesInRegionAndChildrenExcept(
          qIn0, qOut1, swap->getParentRegion(), swap, rewriter);
      replaceAllUsesInRegionAndChildrenExcept(
          qIn1, qOut0, swap->getParentRegion(), swap, rewriter);

      layout.swap(qIn0, qIn1);
      layout.remapQubitValue(qIn0, qOut0);
      layout.remapQubitValue(qIn1, qOut1);

      (*nadd)++;
    }
  }

protected:
  Pass::Statistic* nadd;
};

/**
 * @brief Inserts SWAPs along the shortest path between two hardware
 * qubits.
 */
class NaiveRouter final : public RouterBase {
public:
  using RouterBase::RouterBase;

  SmallVector<QubitIndexPair> getSwapPlan(UnitaryInterface op,
                                          const Layout<QubitIndex>& layout,
                                          const Architecture& arch) final {
    const auto [qStart, qEnd] = getIns(op);
    const auto path = arch.shortestPathBetween(layout.lookupHardware(qStart),
                                               layout.lookupHardware(qEnd));

    SmallVector<QubitIndexPair> swaps(path.size() - 2);
    for (std::size_t i = 0; i < path.size() - 2; ++i) {
      swaps[i] = {path[i], path[i + 1]};
    }

    return swaps;
  }
};

class QMAPRouter final : public RouterBase {
public:
  using RouterBase::RouterBase;

  SmallVector<QubitIndexPair> getSwapPlan(UnitaryInterface op,
                                          const Layout<QubitIndex>& layout,
                                          const Architecture& arch) final {
    /// Copy layout.
    Layout<QubitIndex> copy = layout;
    /// Collect all front-gates starting from 'op'.
    const llvm::DenseSet<QubitIndexPair> gates = collectGates(copy);
    /// Convert to thin layout. TODO: Layout redesign.
    ArrayRef<QubitIndex> curr = copy.getCurrentLayout();
    ThinLayout<QubitIndex> thinLayout(curr.size());
    for (const auto [programIdx, hardwareIdx] : llvm::enumerate(curr)) {
      thinLayout.add(programIdx, hardwareIdx);
    }

    return search(thinLayout, gates, arch);
  }

private:
  struct SearchNode {
    llvm::SmallVector<QubitIndexPair> seq;
    ThinLayout<QubitIndex> layout;

    double cost{};
    std::size_t depth{};

    SearchNode(llvm::SmallVector<QubitIndexPair> seq, QubitIndexPair swap,
               ThinLayout<QubitIndex> layout)
        : seq(std::move(seq)), layout(std::move(layout)) {
      /// Apply node-specific swap to given layout.
      this->layout.swap(this->layout.lookupHardware(swap.first),
                        this->layout.lookupHardware(swap.second));
      /// TODO: Retrigger unnecessary (2 * size) resize? Linked List?
      this->seq.push_back(swap);
    }

    bool operator>(const SearchNode& rhs) const { return cost > rhs.cost; }
  };

  using MinQueue =
      std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<>>;

  static SmallVector<QubitIndexPair>
  search(const ThinLayout<QubitIndex>& layout,
         const llvm::DenseSet<QubitIndexPair>& gates,
         const Architecture& arch) {
    /// The heuristic cost function counts the number of SWAPs that were
    /// required if we were to route the gate set naively.
    const auto heuristic = [&](const SearchNode& node) {
      double h{};
      for (const auto [p0, p1] : gates) {
        const std::size_t nswaps =
            arch.lengthOfShortestPathBetween(node.layout.lookupHardware(p0),
                                             node.layout.lookupHardware(p1)) -
            2;
        h += static_cast<double>(nswaps);
      }
      return h;
    };

    const auto isGoal = [&](const SearchNode& node) {
      return std::ranges::all_of(gates, [&](const QubitIndexPair gate) {
        const auto [p0, p1] = gate;
        return arch.areAdjacent(node.layout.lookupHardware(p0),
                                node.layout.lookupHardware(p1));
      });
    };

    /// Initialize queue.
    MinQueue queue{};
    for (const Exchange swap : collectSWAPs(layout, gates, arch)) {
      SearchNode node({}, swap, layout);
      node.cost = heuristic(node);

      queue.emplace(node);
    }

    /// Iterative searching and expanding.
    while (!queue.empty()) {
      SearchNode curr = queue.top();
      queue.pop();

      if (isGoal(curr)) {
        return curr.seq;
      }

      for (const Exchange swap : collectSWAPs(curr.layout, gates, arch)) {
        SearchNode node(curr.seq, swap, curr.layout);
        node.depth = curr.depth + 1;
        node.cost = static_cast<double>(node.depth) + heuristic(node);

        queue.emplace(node);
      }
    }

    return {};
  }

  /// TODO: I don't like the Exchange class here, especially with
  /// makeExchange.
  static SmallVector<Exchange>
  collectSWAPs(const ThinLayout<QubitIndex>& layout,
               const llvm::DenseSet<QubitIndexPair>& gates,
               const Architecture& arch) {
    // TODO: Adjust (or remove?) assumption of 16.
    llvm::SmallDenseSet<Exchange, 16> candidates{};

    const auto collect = [&](const QubitIndex p) {
      const std::size_t h = layout.lookupHardware(p);
      for (const std::size_t n : arch.neighboursOf(h)) {
        candidates.insert(makeExchange(h, n));
      }
    };

    for (const auto [p0, p1] : gates) {
      collect(p0);
      collect(p1);
    }

    return {candidates.begin(), candidates.end()};
  }

  [[nodiscard]] static llvm::DenseSet<QubitIndexPair>
  collectGates(Layout<QubitIndex>& layout) {
    llvm::DenseSet<UnitaryInterface> candidates;

    for (const Value in : layout.getHardwareQubits()) {
      Value out = in;

      while (!out.getUsers().empty()) {
        Operation* user = *out.getUsers().begin();

        if (auto op = dyn_cast<ResetOp>(user)) {
          out = op.getOutQubit();
          continue;
        }

        if (auto op = dyn_cast<UnitaryInterface>(user)) {
          if (isTwoQubitGate(op)) {
            candidates.insert(op);
            break;
          }

          if (!dyn_cast<GPhaseOp>(user)) {
            out = op.getOutQubits().front();
          }

          continue;
        }

        if (auto measure = dyn_cast<MeasureOp>(user)) {
          out = measure.getOutQubit();
          continue;
        }

        break;
      }

      if (in != out) {
        layout.remapQubitValue(in, out);
      }
    }

    llvm::DenseSet<QubitIndexPair> gates;
    for (UnitaryInterface op : candidates) {
      const auto [in0, in1] = getIns(op);
      if (layout.contains(in0) && layout.contains(in1)) {
        gates.insert({layout.lookupProgram(in0), layout.lookupProgram(in1)});
      }
    }

    return gates;
  }
};

/**
 * @brief The necessary datastructures to route a quantum-classical module.
 */
struct RoutingContext {
  explicit RoutingContext(Architecture& arch, RouterBase& router)
      : arch(&arch), router(&router) {}

  Architecture* arch;
  RouterBase* router;
  StateStack stack{};
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
 * @brief Indicates the end of a region defined by a function. Consequently,
 * we pop the region's state from the stack.
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
WalkResult handleYield(scf::YieldOp op, RoutingContext& ctx,
                       PatternRewriter& rewriter) {
  if (!isa<scf::ForOp>(op->getParentOp()) &&
      !isa<scf::IfOp>(op->getParentOp())) {
    return WalkResult::skip();
  }

  ctx.router->restore(ctx.stack.topState(), ctx.stack.getHistory(),
                      op->getLoc(), rewriter);

  assert(llvm::equal(ctx.stack.topState().getCurrentLayout(),
                     ctx.stack.getStateAtDepth(1).getCurrentLayout()) &&
         "layouts must match after restoration");

  ctx.stack.pop();

  return WalkResult::advance();
}

/**
 * @brief Add hardware qubit with respective program & hardware index to
 * layout.
 *
 * Thanks to the placement pass, we can apply the identity layout here.
 */
WalkResult handleQubit(QubitOp op, RoutingContext& ctx) {
  const std::size_t index = op.getIndex();
  ctx.stack.topState().add(index, index, op.getQubit());
  return WalkResult::advance();
}

/**
 * @brief Ensures the executability of two-qubit gates on the given target
 * architecture by inserting SWAPs.
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
    ctx.router->route(op, ctx.stack, *ctx.arch, rewriter);
  }

  const auto [execIn0, execIn1] = getIns(op);
  const auto [execOut0, execOut1] = getOuts(op);

  LLVM_DEBUG({
    llvm::dbgs() << llvm::format("handleUnitary: gate= s%d/h%d, s%d/h%d\n",
                                 ctx.stack.topState().lookupProgram(execIn0),
                                 ctx.stack.topState().lookupHardware(execIn0),
                                 ctx.stack.topState().lookupProgram(execIn1),
                                 ctx.stack.topState().lookupHardware(execIn1));
  });

  ctx.stack.topState().remapQubitValue(execIn0, execOut0);
  ctx.stack.topState().remapQubitValue(execIn1, execOut1);

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
LogicalResult route(ModuleOp module, MLIRContext* mlirCtx,
                    RoutingContext& ctx) {
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
            .Case<ModuleOp>([&]([[maybe_unused]] ModuleOp op) {
              return WalkResult::advance();
            })
            /// func Dialect
            .Case<func::FuncOp>(
                [&](func::FuncOp op) { return handleFunc(op, ctx); })
            .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
              return handleReturn(ctx);
            })
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
              return handleUnitary(op, ctx, rewriter);
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
    const auto arch = getArchitecture(ArchitectureName::MQTTest);
    const auto router = getRouter();

    if (RoutingContext ctx(*arch, *router);
        failed(route(getOperation(), &getContext(), ctx))) {
      signalPassFailure();
    }
  }

private:
  [[nodiscard]] std::unique_ptr<RouterBase> getRouter() {
    switch (static_cast<RoutingMethod>(method)) {
    case RoutingMethod::Naive:
      LLVM_DEBUG({ llvm::dbgs() << "getRouter: method=naive\n"; });
      return std::make_unique<NaiveRouter>(nadd);
    case RoutingMethod::QMAP:
      LLVM_DEBUG({ llvm::dbgs() << "getRouter: method=qmap\n"; });
      return std::make_unique<QMAPRouter>(nadd);
    }

    llvm_unreachable("Unknown method");
  }
};

} // namespace
} // namespace mqt::ir::opt
