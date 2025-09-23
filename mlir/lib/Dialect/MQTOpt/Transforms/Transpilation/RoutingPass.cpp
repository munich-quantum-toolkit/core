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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/RoutingStack.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
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
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <numeric>
#include <optional>
#include <random>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "routing"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/**
 * @brief A function attribute that specifies an (QIR) entry point function.
 */
constexpr llvm::StringLiteral ENTRY_POINT_ATTR{"entry_point"};

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
 * @brief A pair of software indices.
 */
using SoftwareIndexPair = std::pair<QubitIndex, QubitIndex>;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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
[[nodiscard]] std::pair<Value, Value> getIns(UnitaryInterface u) {
  assert(isTwoQubitGate(u));
  return {u.getAllInQubits()[0], u.getAllInQubits()[1]};
}

/**
 * @brief Return output qubit pair for a two-qubit unitary.
 * @param u A two-qubit unitary.
 * @return Pair of SSA values consisting of the first and second out-qubits.
 */
[[nodiscard]] std::pair<Value, Value> getOuts(UnitaryInterface u) {
  assert(isTwoQubitGate(u));
  return {u.getAllOutQubits()[0], u.getAllOutQubits()[1]};
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

//===----------------------------------------------------------------------===//
// Initial Layouts
//===----------------------------------------------------------------------===//

/**
 * @brief Return identity layout.
 * @param nqubits The number of qubits.
 */
[[nodiscard, maybe_unused]] SmallVector<std::size_t>
getIdentityLayout(const std::size_t nqubits) {
  SmallVector<std::size_t> layout(nqubits);
  std::iota(layout.begin(), layout.end(), 0);
  return layout;
}

/**
 * @brief Generate random layout.
 * @param nqubits The number of qubits.
 * @param seed The seed used for randomization.
 */
[[nodiscard, maybe_unused]] SmallVector<std::size_t>
getRandomLayout(const std::size_t nqubits, const std::size_t seed) {
  std::mt19937_64 rng(seed);
  auto layout = getIdentityLayout(nqubits);
  std::shuffle(layout.begin(), layout.end(), rng);
  return layout;
}

//===----------------------------------------------------------------------===//
// State (Permutation) Management
//===----------------------------------------------------------------------===//

/**
 * @brief Manage mapping between SSA values and physical hardware indices.
 * This class maintains the bi-directional mapping between software and hardware
 * qubits.
 *
 * Note that we use the terminology "hardware" and "software" qubits here,
 * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
 * are C++ keywords.
 */
class [[nodiscard]] LayoutState {
public:
  /**
   * @brief Initialize the layout state.
   *
   * Applies initial layout to the given array-ref of hardware qubits.
   *
   * @param qubits The hardware qubits.
   * @param initialLayout A map from software qubit index to hardware qubit
   * index.
   */
  explicit LayoutState(ArrayRef<Value> qubits,
                       ArrayRef<std::size_t> initialLayout)
      : qubits_(qubits.size()), softwareToHardware_(qubits.size()) {
    valueToMapping_.reserve(qubits.size());

    for (const std::size_t softwareIdx : initialLayout) {
      const QubitInfo info{.hardwareIdx = initialLayout[softwareIdx],
                           .softwareIdx = softwareIdx};
      const Value q = qubits[info.hardwareIdx];

      qubits_[info.hardwareIdx] = q;
      softwareToHardware_[softwareIdx] = info.hardwareIdx;
      valueToMapping_.try_emplace(q, info);
    }
  }

  /**
   * @brief Look up hardware index for a qubit value.
   * @param q The SSA Value representing the qubit.
   * @return The hardware index where this qubit currently resides.
   */
  [[nodiscard]] std::size_t lookupHardware(const Value q) const {
    return valueToMapping_.at(q).hardwareIdx;
  }

  /**
   * @brief Look up qubit value for a hardware index.
   * @param hardwareIdx The hardware index.
   * @return The SSA value currently representing the qubit at the hardware
   * location.
   */
  [[nodiscard]] Value lookupHardware(const std::size_t hardwareIdx) const {
    assert(hardwareIdx < qubits_.size() && "Hardware index out of bounds");
    return qubits_[hardwareIdx];
  }

  /**
   * @brief Look up software index for a qubit value.
   * @param q The SSA Value representing the qubit.
   * @return The software index where this qubit currently resides.
   */
  [[nodiscard]] std::size_t lookupSoftware(const Value q) const {
    return valueToMapping_.at(q).softwareIdx;
  }

  /**
   * @brief Look up qubit value for a software index.
   * @param softwareIdx The software index.
   * @return The SSA value currently representing the qubit at the software
   * location.
   */
  [[nodiscard]] Value lookupSoftware(const std::size_t softwareIdx) const {
    const std::size_t hardwareIdx = softwareToHardware_[softwareIdx];
    return lookupHardware(hardwareIdx);
  }

  /**
   * @brief Replace an old SSA value with a new one.
   */
  void remapQubitValue(const Value in, const Value out) {
    const auto it = valueToMapping_.find(in);
    assert(it != valueToMapping_.end() && "forward: unknown input value");

    const QubitInfo map = it->second;
    qubits_[map.hardwareIdx] = out;

    assert(!valueToMapping_.contains(out) &&
           "forward: output value already mapped");

    valueToMapping_.try_emplace(out, map);
    valueToMapping_.erase(in);
  }

  /**
   * @brief Swap the locations of two software qubits. This is the effect of a
   * SWAP gate.
   */
  void swapSoftwareIndices(const Value q0, const Value q1) {
    auto ita = valueToMapping_.find(q0);
    auto itb = valueToMapping_.find(q1);
    assert(ita != valueToMapping_.end() && itb != valueToMapping_.end() &&
           "swapSoftwareIndices: unknown values");
    std::swap(ita->second.softwareIdx, itb->second.softwareIdx);
    std::swap(softwareToHardware_[ita->second.softwareIdx],
              softwareToHardware_[itb->second.softwareIdx]);
  }

  /**
   * @brief Return the current layout.
   */
  ArrayRef<std::size_t> getCurrentLayout() { return softwareToHardware_; }

  /**
   * @brief Return the SSA values for hardware indices from 0...nqubits.
   */
  [[nodiscard]] ArrayRef<Value> getHardwareQubits() const { return qubits_; }

private:
  struct QubitInfo {
    std::size_t hardwareIdx;
    std::size_t softwareIdx;
  };

  /**
   * @brief Maps an SSA value to its `QubitInfo`.
   */
  DenseMap<Value, QubitInfo> valueToMapping_;

  /**
   * @brief Maps hardware qubit indices to SSA values.
   */
  SmallVector<Value> qubits_;

  /**
   * @brief Maps a software qubit index to its hardware index.
   */
  SmallVector<std::size_t> softwareToHardware_;
};

struct StackItem {
  explicit StackItem(LayoutState state) : state(std::move(state)) {}
  LayoutState state;
  SmallVector<SoftwareIndexPair, 32> history;
};

class StateStack : public RoutingStack<StackItem> {
public:
  /**
   * @brief Returns the most recent state of the stack.
   */
  [[nodiscard]] LayoutState& topState() { return top().state; }

  /**
   * @brief Returns the item at the specified depth from the top of the stack.
   */
  [[nodiscard]] LayoutState& getStateAtDepth(std::size_t depth) {
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
  [[nodiscard]] ArrayRef<SoftwareIndexPair> getHistory() {
    return top().history;
  }

  /**
   * @brief Record a swap.
   */
  void recordSwap(std::size_t softwareIdx0, std::size_t softwareIdx1) {
    top().history.emplace_back(softwareIdx0, softwareIdx1);
  }
};

/**
 * @brief Manages free / used hardware indices.
 */
struct HardwareIndexPool {

  /**
   * @brief Fill the pool with indices determined by the given layout.
   */
  void fill(ArrayRef<std::size_t> layout) {
    freeHardwareIndices_.clear();
    for (const std::size_t i : llvm::reverse(layout)) {
      freeHardwareIndices_.insert(i);
    }
  }

  /**
   * @brief Re-insert hardware index to set of free indices.
   */
  void release(const std::size_t index) { freeHardwareIndices_.insert(index); }

  /**
   * @brief Retrieve free hardware index if available.
   * @returns The index, or std::nullopt if none is available.
   */
  [[nodiscard]] std::optional<std::size_t> retrieve() {
    if (freeHardwareIndices_.empty()) {
      return std::nullopt;
    }
    const std::size_t index = freeHardwareIndices_.back();
    freeHardwareIndices_.pop_back();
    return index;
  }

  /**
   * @brief Return true if the index is in-use, i.e., it has been allocated
   * before.
   */
  [[nodiscard]] bool isUsed(const std::size_t index) const {
    return !freeHardwareIndices_.contains(index);
  }

private:
  /**
   * @brief Set of free hardware indices.
   *
   * The SetVector ensures a deterministic iteration order.
   */
  llvm::SetVector<std::size_t> freeHardwareIndices_;
};

/**
 * @brief Base class for all routing algorithms.
 */
class Router {
public:
  virtual ~Router() = default;

  /**
   * @brief Ensures the executability of two-qubit gates on the given target
   * architecture by inserting SWAPs greedily.
   */
  WalkResult handleUnitary(UnitaryInterface op, StateStack& stack,
                           const Architecture& arch, HardwareIndexPool& pool,
                           PatternRewriter& rewriter) {
    // Global-phase or zero-qubit unitary: Nothing to do.
    if (op.getAllInQubits().empty()) {
      return WalkResult::advance();
    }

    // Single-qubit: Forward mapping.
    if (!isTwoQubitGate(op)) {
      stack.topState().remapQubitValue(op.getAllInQubits()[0],
                                       op.getAllOutQubits()[0]);
      return WalkResult::advance();
    }

    if (!isExecutable(op, stack, arch)) {
      makeExecutable(op, stack, arch, pool, rewriter);
    }

    const auto [execIn0, execIn1] = getIns(op);
    const auto [execOut0, execOut1] = getOuts(op);

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG({
      llvm::dbgs() << llvm::format("two-qubit gate: s%d/h%d, s%d/h%d\n",
                                   stack.topState().lookupSoftware(execIn0),
                                   stack.topState().lookupHardware(execIn0),
                                   stack.topState().lookupSoftware(execIn1),
                                   stack.topState().lookupHardware(execIn1));
    });

    stack.topState().remapQubitValue(execIn0, execOut0);
    stack.topState().remapQubitValue(execIn1, execOut1);

    return WalkResult::advance();
  }

protected:
  /**
   * @brief Insert SWAPs such that @p u is executable.
   */
  virtual void makeExecutable(UnitaryInterface op, StateStack& stack,
                              const Architecture& arch, HardwareIndexPool& pool,
                              PatternRewriter& rewriter) = 0;

  /**
   * @brief Returns true iff @p u is executable on the targeted architecture.
   */
  [[nodiscard]] static bool isExecutable(UnitaryInterface op, StateStack& stack,
                                         const Architecture& arch) {
    const auto [in0, in1] = getIns(op);
    return arch.areAdjacent(stack.topState().lookupHardware(in0),
                            stack.topState().lookupHardware(in1));
  }

  /**
   * @brief Get shortest path between @p qStart and @p qEnd.
   */
  [[nodiscard]] static llvm::SmallVector<std::size_t>
  getPath(const Value qStart, const Value qEnd, StateStack& stack,
          const Architecture& arch) {
    return arch.shortestPathBetween(stack.topState().lookupHardware(qStart),
                                    stack.topState().lookupHardware(qEnd));
  }

  /**
   * @brief Check if a qubit's hardware index has been allocated before and now
   * has zero uses.
   *
   * Due to the SWAPs it is possible to deallocate a qubit with a
   * different hardware index than the one we originally allocated. This
   * function ensures that we release hardware indices when a qubit that has
   * been allocated and now has zero uses.
   */
  static void checkZeroUse(const Value q, StateStack& stack,
                           HardwareIndexPool& pool) {
    const std::size_t hardwareIdx = stack.topState().lookupHardware(q);
    if (q.use_empty() && pool.isUsed(hardwareIdx)) {

      // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
      LLVM_DEBUG({
        llvm::dbgs() << llvm::format("free index with zero uses: %d\n",
                                     hardwareIdx);
      });

      pool.release(hardwareIdx);
    }
  }
};

/**
 * @brief Inserts SWAPs along the shortest path between two hardware
 * qubits.
 */
class NaiveRouter final : public Router {
protected:
  void makeExecutable(UnitaryInterface op, StateStack& stack,
                      const Architecture& arch, HardwareIndexPool& pool,
                      PatternRewriter& rewriter) final {
    assert(isTwoQubitGate(op) && "makeExecutable: must be two-qubit gate");

    const auto [qStart, qEnd] = getIns(op);
    const auto path = getPath(qStart, qEnd, stack, arch);

    for (std::size_t i = 0; i < path.size() - 2; ++i) {
      const std::size_t hardwareIdx0 = path[i];
      const std::size_t hardwareIdx1 = path[i + 1];

      const Value qIn0 = stack.topState().lookupHardware(hardwareIdx0);
      const Value qIn1 = stack.topState().lookupHardware(hardwareIdx1);

      const std::size_t softwareIdx0 = stack.topState().lookupSoftware(qIn0);
      const std::size_t softwareIdx1 = stack.topState().lookupSoftware(qIn1);

      // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
      LLVM_DEBUG({
        llvm::dbgs() << llvm::format(
            "swap: s%d/h%d, s%d/h%d <- s%d/h%d, s%d/h%d\n", softwareIdx1,
            hardwareIdx0, softwareIdx0, hardwareIdx1, softwareIdx0,
            hardwareIdx0, softwareIdx1, hardwareIdx1);
      });

      auto swap = createSwap(op->getLoc(), qIn0, qIn1, rewriter);
      const auto [qOut0, qOut1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      replaceAllUsesInRegionAndChildrenExcept(
          qIn0, qOut1, swap->getParentRegion(), swap, rewriter);
      replaceAllUsesInRegionAndChildrenExcept(
          qIn1, qOut0, swap->getParentRegion(), swap, rewriter);

      stack.recordSwap(softwareIdx0, softwareIdx1);

      auto& state = stack.topState();
      state.swapSoftwareIndices(qIn0, qIn1);
      state.remapQubitValue(qIn0, qOut0);
      state.remapQubitValue(qIn1, qOut1);

      checkZeroUse(qOut0, stack, pool);
      checkZeroUse(qOut1, stack, pool);
    }
  }
};

/**
 * @brief Contains all necessary datastructures to route a quantum-classical
 * program.
 */
struct RoutingContext {
  explicit RoutingContext(std::unique_ptr<Architecture> arch,
                          std::unique_ptr<Router> router)
      : arch(std::move(arch)), router(std::move(router)) {
    initialLayout = getIdentityLayout(this->arch->nqubits());
  }

  explicit RoutingContext(std::unique_ptr<Architecture> arch,
                          std::unique_ptr<Router> router,
                          const std::size_t seed)
      : arch(std::move(arch)), router(std::move(router)) {
    initialLayout = getRandomLayout(this->arch->nqubits(), seed);
  }

  std::unique_ptr<Architecture> arch;
  std::unique_ptr<Router> router;
  SmallVector<std::size_t> initialLayout;

  StateStack stack{};
  HardwareIndexPool pool{};
};

/**
 * @brief Adds nqubit 'mqtopt.qubit' ops for entry_point functions. Initializes
 * the pool and hence applies the initial layout. Consequently, the initial
 * layout is always applied at the beginning of a function. Pushes
 * newly-initialized state on stack.
 */
WalkResult handleFunc(func::FuncOp op, RoutingContext& routingCtx,
                      PatternRewriter& rewriter) {
  assert(routingCtx.stack.empty() && "handleFunc: stack must be empty");

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
  LLVM_DEBUG({
    llvm::dbgs() << "handleFunc: entry_point: " << op.getSymName() << '\n';
  });

  rewriter.setInsertionPointToStart(&op.getBody().front());

  llvm::SmallVector<Value> qubits(routingCtx.arch->nqubits());
  for (std::size_t i = 0; i < qubits.size(); ++i) {
    auto qubitOp =
        rewriter.create<QubitOp>(rewriter.getInsertionPoint()->getLoc(), i);
    rewriter.setInsertionPointAfter(qubitOp);
    qubits[i] = qubitOp.getQubit();
  }

  routingCtx.pool.fill(routingCtx.initialLayout);
  routingCtx.stack.emplace(LayoutState(qubits, routingCtx.initialLayout));

  return WalkResult::advance();
}

/**
 * @brief Defines the end of a region (and hence routing state) defined by a
 * function. Consequently, we pop the region's state from the stack. Since
 * we currently only route entry_point functions we do not need to return
 * all static qubits here.
 */
WalkResult handleReturn(StateStack& stack) {
  assert(!stack.empty() && "handleReturn: stack must not be empty");
  stack.pop();
  return WalkResult::advance();
}

/**
 * @brief Replaces the 'for' loop with one that has all hardware qubits as init
 * arguments.
 *
 * Prepares the stack for the routing of the loop body by adding a copy of the
 * current state to the stack and resetting its SWAP history. Forwards the
 * results in the parent state.
 */
WalkResult handleFor(scf::ForOp op, StateStack& stack,
                     PatternRewriter& rewriter) {
  const std::size_t nargs = op.getBody()->getNumArguments();
  const std::size_t nresults = op->getNumResults();

  /// Construct new init arguments.
  const ArrayRef<Value> qubits = stack.topState().getHardwareQubits();
  const SmallVector<Value> newInitArgs(qubits);

  /// Replace old for with a new one with updated init arguments.
  auto forOp = rewriter.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                           op.getUpperBound(), op.getStep(),
                                           newInitArgs);

  rewriter.mergeBlocks(op.getBody(), forOp.getBody(),
                       forOp.getBody()->getArguments().take_front(nargs));

  if (nresults > 0) {
    rewriter.replaceOp(op, forOp.getResults().take_front(nresults));
  } else {
    rewriter.eraseOp(op);
  }

  /// Prepare stack.
  stack.duplicateTopState();

  // Forward out-of-loop and in-loop state.
  for (const auto [arg, res, iter] :
       llvm::zip(forOp.getInitArgs(), forOp.getResults(),
                 forOp.getRegionIterArgs())) {
    if (isa<QubitType>(arg.getType())) {
      stack.getStateAtDepth(FOR_PARENT_DEPTH).remapQubitValue(arg, res);
      stack.topState().remapQubitValue(arg, iter);
    }
  }

  return WalkResult::advance();
}

/**
 * @brief Replaces the 'if' statement with one that has all hardware qubits as
 * result.
 *
 * Prepares the stack for the routing of the 'then' and 'else' body by adding a
 * copy of the current state to the stack and resetting its SWAP history for
 * each branch. Forwards the results in the parent state.
 */
WalkResult handleIf(scf::IfOp op, StateStack& stack,
                    PatternRewriter& rewriter) {
  const std::size_t nresults = op->getNumResults();

  /// Construct new result types.
  const ArrayRef<Value> qubits = stack.topState().getHardwareQubits();
  const auto rng = llvm::map_range(qubits, [](Value q) { return q.getType(); });
  const SmallVector<Type> resultTypes(rng.begin(), rng.end());

  /// Replace old if with a new one with updated result types.
  const bool hasElse = !op.getElseRegion().empty();
  auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), resultTypes,
                                         op.getCondition(), hasElse);

  rewriter.mergeBlocks(&op.getThenRegion().front(),
                       &ifOp.getThenRegion().front());
  if (hasElse) {
    rewriter.mergeBlocks(&op.getElseRegion().front(),
                         &ifOp.getElseRegion().front());
  }

  if (nresults > 0) {
    rewriter.replaceOp(op, ifOp.getResults().take_front(nresults));
  } else {
    rewriter.eraseOp(op);
  }

  /// Prepare stack.
  stack.duplicateTopState(); // Else
  stack.duplicateTopState(); // Then

  /// Forward results for all hardware qubits.
  LayoutState& stateBeforeIf = stack.getStateAtDepth(IF_PARENT_DEPTH);
  for (std::size_t i = 0; i < qubits.size(); ++i) {
    const Value in = stateBeforeIf.getHardwareQubits()[i];
    const Value out = ifOp->getResult(i);
    stateBeforeIf.remapQubitValue(in, out);
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
 * because we restore the layout - the mapping from hardware to software
 * qubits (and vice versa).
 *
 * Using uncompute has the advantages of (1) being intuitive and
 * (2) preserving the optimally of the original SWAP sequence.
 * Essentially the better the routing algorithm the better the
 * uncompute. Moreover, this has the nice property that routing
 * a 'for' of 'if' region always requires 2 * #(SWAPs required for region)
 * additional SWAPS.
 */
WalkResult handleYield(scf::YieldOp op, StateStack& stack,
                       PatternRewriter& rewriter) {
  if (!isa<scf::ForOp>(op->getParentOp()) &&
      !isa<scf::IfOp>(op->getParentOp())) {
    return WalkResult::skip();
  }

  for (const auto [softwareIdx0, softwareIdx1] :
       llvm::reverse(stack.getHistory())) {
    const Value qIn0 = stack.topState().lookupSoftware(softwareIdx0);
    const Value qIn1 = stack.topState().lookupSoftware(softwareIdx1);

    auto swap = createSwap(op->getLoc(), qIn0, qIn1, rewriter);
    const auto [qOut0, qOut1] = getOuts(swap);

    rewriter.setInsertionPointAfter(swap);
    replaceAllUsesInRegionAndChildrenExcept(
        qIn0, qOut1, swap->getParentRegion(), swap, rewriter);
    replaceAllUsesInRegionAndChildrenExcept(
        qIn1, qOut0, swap->getParentRegion(), swap, rewriter);

    stack.topState().swapSoftwareIndices(qIn0, qIn1);
    stack.topState().remapQubitValue(qIn0, qOut0);
    stack.topState().remapQubitValue(qIn1, qOut1);
  }

  assert(llvm::equal(stack.topState().getCurrentLayout(),
                     stack.getStateAtDepth(1).getCurrentLayout()) &&
         "layouts must match after restoration");

  const SmallVector<Value> newResults(stack.topState().getHardwareQubits());
  rewriter.replaceOpWithNewOp<scf::YieldOp>(op, newResults);

  stack.pop();

  return WalkResult::advance();
}

/**
 * @brief Retrieves free hardware index from pool, gets the respective SSA value
 * of the qubit, and replaces the alloc statement with the reset SSA value of
 * the retrieved hardware qubit.
 */
WalkResult handleAlloc(AllocQubitOp op, StateStack& stack,
                       HardwareIndexPool& pool, PatternRewriter& rewriter) {
  const std::optional<std::size_t> index = pool.retrieve();
  if (!index) {
    return op.emitOpError(
        "requires one too many qubits for the targeted architecture");
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
  LLVM_DEBUG({ llvm::dbgs() << "alloc index: " << *index << '\n'; });
  const Value q = stack.topState().lookupHardware(*index);

  auto reset = rewriter.create<ResetOp>(op.getLoc(), q);
  rewriter.replaceOp(op, reset);

  stack.topState().remapQubitValue(reset.getInQubit(), reset.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Release hardware index of deallocated qubit. Note that due to the
 * inserted SWAPs the deallocated qubit might not be the same hardware qubit as
 * the one allocated. Consequently, we add a `isUsed` check here and deal with
 * allocated, but not properly deallocated, qubits in the `NaiveUnitaryPattern`.
 */
WalkResult handleDealloc(DeallocQubitOp op, StateStack& stack,
                         HardwareIndexPool& pool, PatternRewriter& rewriter) {
  const Value q = op.getQubit();
  const std::size_t index = stack.topState().lookupHardware(q);
  if (pool.isUsed(index)) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG({ llvm::dbgs() << "dealloc index: " << index << '\n'; });
    pool.release(index);
  }

  rewriter.eraseOp(op);
  return WalkResult::advance();
}

/**
 * @brief Forwards SSA Values, i.e., forward 'in' to 'out'.
 */
WalkResult handleReset(ResetOp op, StateStack& stack) {
  stack.topState().remapQubitValue(op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Forwards SSA Values, i.e., forward 'in' to 'out'.
 */
WalkResult handleMeasure(MeasureOp op, StateStack& stack) {
  stack.topState().remapQubitValue(op.getInQubit(), op.getOutQubit());
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
    if (!func->hasAttr(ENTRY_POINT_ATTR)) {
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
                [&](func::FuncOp op) { return handleFunc(op, ctx, rewriter); })
            .Case<func::ReturnOp>([&](func::ReturnOp /* op */) {
              return handleReturn(ctx.stack);
            })
            /// scf Dialect
            .Case<scf::ForOp>([&](scf::ForOp op) {
              return handleFor(op, ctx.stack, rewriter);
            })
            .Case<scf::IfOp>(
                [&](scf::IfOp op) { return handleIf(op, ctx.stack, rewriter); })
            .Case<scf::YieldOp>([&](scf::YieldOp op) {
              return handleYield(op, ctx.stack, rewriter);
            })
            /// mqtopt Dialect
            .Case<AllocQubitOp>([&](AllocQubitOp op) {
              return handleAlloc(op, ctx.stack, ctx.pool, rewriter);
            })
            .Case<DeallocQubitOp>([&](DeallocQubitOp op) {
              return handleDealloc(op, ctx.stack, ctx.pool, rewriter);
            })
            .Case<ResetOp>(
                [&](ResetOp op) { return handleReset(op, ctx.stack); })
            .Case<MeasureOp>(
                [&](MeasureOp op) { return handleMeasure(op, ctx.stack); })
            .Case<UnitaryInterface>([&](UnitaryInterface op) {
              return ctx.router->handleUnitary(op, ctx.stack, *ctx.arch,
                                               ctx.pool, rewriter);
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
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    std::random_device rd;
    const std::size_t seed = rd();
    RoutingContext routingCtx(getArchitecture(ArchitectureName::MQTTest),
                              std::make_unique<NaiveRouter>(), seed);

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG({
      llvm::dbgs() << "initial layout with seed " << seed << ": ";
      for (const std::size_t i : routingCtx.initialLayout) {
        llvm::dbgs() << i << ' ';
      }
      llvm::dbgs() << '\n';
    });

    if (failed(route(getOperation(), &getContext(), routingCtx))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mqt::ir::opt
