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
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Action.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Support/WalkResult.h>
#include <numeric>
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

//===----------------------------------------------------------------------===//
// Initial Layouts
//===----------------------------------------------------------------------===//

/**
 * @brief A mapping from software to hardware qubits.
 */
using Layout = ArrayRef<std::size_t>;

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
 * @brief A pair of indices.
 */
using QubitIndexPair = std::pair<std::size_t, std::size_t>;

/**
 * @brief Manage mapping between SSA values and physical hardware indices.
 *
 * Note that we use the terminology "hardware" and "software" qubits here,
 * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
 * are C++ keywords.
 */
class [[nodiscard]] QubitState {
public:
  explicit QubitState(const std::size_t nqubits)
      : hardwareQubits_(nqubits), layout_(nqubits) {
    mapping_.reserve(nqubits);
  }

  /**
   * @brief Initialize the state.
   *
   * Applies initial layout to the given array-ref of hardware qubits. That is,
   * it sets the permutation.
   *
   * @param hwQubits An array-ref of SSA values.
   * @param l0 The initial layout to apply.
   */
  void initialize(ArrayRef<Value> hwQubits, Layout l0) {
    mapping_.clear();
    for (std::size_t s = 0; s < getNumQubits(); ++s) {
      const std::size_t h = l0[s];
      const Value q = hwQubits[h];
      hardwareQubits_[h] = q;
      layout_[s] = h;
      mapping_.try_emplace(q, std::make_pair(h, s));
    }
  }

  /**
   * @brief Return hardware index from SSA value.
   */
  [[nodiscard]] std::size_t getHardwareIndex(const Value q) const {
    return mapping_.at(q).first;
  }

  /**
   * @brief Return software index from SSA value.
   */
  [[nodiscard]] std::size_t getSoftwareIndex(const Value q) const {
    return mapping_.at(q).second;
  }

  /**
   * @brief Return SSA Value from hardware index.
   * @param index The hardware index.
   */
  [[nodiscard]] Value getHardwareValue(const std::size_t index) const {
    assert(index < hardwareQubits_.size() &&
           "getHardwareValue: index out of bounds");
    return hardwareQubits_[index];
  }

  /**
   * @brief Return the SSA values for hardware indices from 0...nqubits.
   */
  [[nodiscard]] ArrayRef<Value> getHardwareQubits() const {
    return hardwareQubits_;
  }

  /**
   * @brief Return the number of hardware qubits.
   */
  [[nodiscard]] std::size_t getNumQubits() const {
    return hardwareQubits_.size();
  }

  /**
   * @brief Return the swap history.
   */
  [[nodiscard]] ArrayRef<QubitIndexPair> getSwapHistory() const {
    return swapHistory_;
  }

  /**
   * @brief Record a swap.
   */
  void recordSwap(std::size_t idx1, std::size_t idx2) {
    swapHistory_.emplace_back(idx1, idx2);
  }

  /**
   * @brief Clear swap history.
   */
  void clearHistory() { swapHistory_.clear(); }

  /**
   * @brief Forward SSA values.
   * @details Replace @p in with @p out in maps.
   */
  void forward(const Value in, const Value out) {
    const auto it = mapping_.find(in);
    assert(it != mapping_.end() && "forward: unknown input value");

    const QubitIndexPair map = it->second;
    const std::size_t h = map.first;
    hardwareQubits_[h] = out;

    assert(!mapping_.contains(out) && "forward: output value already mapped");

    mapping_.try_emplace(out, map);
    mapping_.erase(in);
  }

  /**
   * @brief Return the current layout.
   */
  Layout getLayout() { return layout_; }

  /**
   * @brief Forward range of SSA values.
   */
  void forwardRange(ValueRange in, ValueRange out) {
    assert(in.size() == out.size() && "'in' must have same size as 'out'");
    for (std::size_t i = 0; i < in.size(); ++i) {
      forward(in[i], out[i]);
    }
  }

  /**
   * @brief Swap software indices.
   */
  void swapSoftwareIndices(const Value a, const Value b) {
    auto ita = mapping_.find(a);
    auto itb = mapping_.find(b);
    assert(ita != mapping_.end() && itb != mapping_.end() &&
           "swapSoftwareIndices: unknown values");
    std::swap(ita->second.second, itb->second.second);
    std::swap(layout_[ita->second.second], layout_[itb->second.second]);
  }

private:
  /**
   * @brief Maps SSA values to (hardware, software) index pair.
   */
  DenseMap<Value, QubitIndexPair> mapping_;

  /**
   * @brief Maps hardware indices to SSA values.
   *
   * The size of this vector is the number of hardware qubits.
   */
  SmallVector<Value> hardwareQubits_;

  /**
   * @brief The current layout.
   */
  SmallVector<std::size_t> layout_;

  /**
   * @brief A history of SWAPs inserted.
   */
  SmallVector<QubitIndexPair, 32> swapHistory_;
};

using RoutingStack = SmallVector<QubitState, 2>;

/**
 * @brief Return the element on the top of the stack.
 * @param stack The routing stack.
 */
QubitState& top(RoutingStack& stack) {
  assert(!stack.empty() && "state: expected at least one stack item");
  return stack.back();
}

/**
 * @brief Return the element below the top of the stack.
 * @param stack The routing stack.
 */
QubitState& belowTop(RoutingStack& stack) {
  assert(stack.size() >= 2 && "parentState: expected at least two stack items");
  return stack[stack.size() - 2];
}

/**
 * @brief Copy the top of the stack.
 * @param stack The routing stack.
 */
void duplicateTop(RoutingStack& stack) {
  assert(!stack.empty() && "state: expected at least one stack item");
  stack.push_back(stack.back());
  stack.back().clearHistory();
}

struct HardwareIndexPool {

  /**
   * @brief Fill the pool with indices determined by the given layout.
   */
  void fill(Layout layout) {
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
   * @brief Retrieve free hardware index.
   */
  [[nodiscard]] std::size_t retrieve() {
    const std::size_t index = freeHardwareIndices_.back();
    freeHardwareIndices_.pop_back();
    return index;
  }

  /**
   * @brief Return true if the index is in-use, i.e., it has been allocated
   * before.
   */
  [[nodiscard]] bool isUsed(const std::size_t index) {
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

//===----------------------------------------------------------------------===//
// Pre-Order Walk Pattern Rewrite Driver
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(misc-const-correctness)
struct PreOrderWalkDriverAction final
    : tracing::ActionImpl<PreOrderWalkDriverAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreOrderWalkDriverAction)
  using ActionImpl::ActionImpl;
  // NOLINTNEXTLINE(readability-identifier-naming) // MLIR requires lowercase.
  static constexpr StringLiteral tag = "walk-and-apply-patterns-pre-order";
  void print(raw_ostream& os) const override { os << tag; }
};
// NOLINTEND(misc-const-correctness)

/**
 * @brief A pre-order version of the walkAndApplyPatterns driver.
 *
 * Walks the IR pre-order and fills the worklist to ensure that each
 * operation is visited exactly once (even if we change parent ops).
 *
 * @param op The operation to walk. Does not visit the op itself.
 * @param patterns A set of patterns to apply.
 * @link Adapted from
 * https://mlir.llvm.org/doxygen/WalkPatternRewriteDriver_8cpp_source.html
 */
void walkPreOrderAndApplyPatterns(Operation* op,
                                  const FrozenRewritePatternSet& patterns) {
  MLIRContext* ctx = op->getContext();
  PatternRewriter rewriter(ctx);

  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  SmallVector<Operation*> worklist;
  ctx->executeAction<PreOrderWalkDriverAction>(
      [&] {
        op->walk<WalkOrder::PreOrder>(
            [&](Operation* current) { worklist.push_back(current); });

        for (const auto& curr : worklist) {
          if (!curr) {
            continue; // Skip erased ops.
          }
          std::ignore = applicator.matchAndRewrite(curr, rewriter);
        }
      },
      {op});
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Check if an operation contains a unitary.
 *
 * @param op The operation to walk.
 * @return True iff the walk finds a unitary.
 */
bool containsUnitary(Operation* op) {
  return op->walk([&](UnitaryInterface) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

/**
 * @brief Initialize entry point function.
 *
 * Inserts static qubits (via the `mqtopt.qubit` op) at the current insertion
 * point of the rewriter. Stores the SSA values in the provided qubit vector.
 *
 * @param qubits The reference to the vector of qubits. The function creates
 * `qubits.size()` many static qubits.
 * @param rewriter A PatternRewriter.
 */
void initEntryPoint(SmallVectorImpl<Value>& qubits, PatternRewriter& rewriter) {
  for (std::size_t i = 0; i < qubits.size(); ++i) {
    auto op =
        rewriter.create<QubitOp>(rewriter.getInsertionPoint()->getLoc(), i);
    rewriter.setInsertionPointAfter(op);
    qubits[i] = op.getQubit();
  }
}

/**
 * @brief Collect all missing hardware qubits from the results.
 *
 * @param old A range of values to check for included qubits.
 * @param hardwareQubits The hardware qubits.
 * @return Vector of missing hardware qubits with increasing index.
 */
[[nodiscard]] SmallVector<Value, 0>
getMissingQubits(ValueRange old, ArrayRef<Value> hardwareQubits) {
  const SmallVector<Value, 0> oldResults(old.begin(), old.end());
  llvm::DenseSet<Value> included;
  included.insert(oldResults.begin(), oldResults.end());

  SmallVector<Value, 0> missing;
  missing.reserve(hardwareQubits.size());
  for (const Value q : hardwareQubits) {
    if (!included.contains(q)) {
      missing.push_back(q);
    }
  }

  return missing;
}

/**
 * @brief Base class for all routing rewrite patterns.
 */
template <class BasePattern> struct ContextualPattern : BasePattern {
public:
  ContextualPattern(MLIRContext* ctx, Architecture& arch, Layout layout,
                    RoutingStack& stack, HardwareIndexPool& pool)
      : BasePattern(ctx), arch_(&arch), layout_(layout), stack_(&stack),
        pool_(&pool) {}

  [[nodiscard]] Architecture& arch() const { return *arch_; }
  [[nodiscard]] Layout layout() const { return layout_; }
  [[nodiscard]] RoutingStack& stack() const { return *stack_; }
  [[nodiscard]] HardwareIndexPool& pool() const { return *pool_; }

private:
  Architecture* arch_;
  Layout layout_;
  RoutingStack* stack_;
  HardwareIndexPool* pool_;
};

/**
 * @brief Contextual rewrite pattern for ops.
 */
template <class OpType>
struct ContextualOpPattern : ContextualPattern<OpRewritePattern<OpType>> {
  using ContextualPattern<OpRewritePattern<OpType>>::ContextualPattern;
};

/**
 * @brief Contextual rewrite pattern for interfaces.
 */
template <class SourceOp>
struct ContextualInterfacePattern
    : ContextualPattern<OpInterfaceRewritePattern<SourceOp>> {
  using ContextualPattern<
      OpInterfaceRewritePattern<SourceOp>>::ContextualPattern;
};

struct FuncOpPattern final : ContextualOpPattern<func::FuncOp> {
  using ContextualOpPattern<func::FuncOp>::ContextualOpPattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter& rewriter) const final {
    if (!containsUnitary(op)) {
      return failure();
    }

    // In this if-branch we would collect (or add) the qubits
    // from (to) the argument-list.
    if (!op->hasAttr(ENTRY_POINT_ATTR)) {
      return failure();
    }

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG({
      llvm::dbgs() << "routing entry_point function: " << op.getSymName()
                   << '\n';
    });

    rewriter.setInsertionPointToStart(&op.getBody().front());
    llvm::SmallVector<Value> qubits(arch().nqubits());
    initEntryPoint(qubits, rewriter);

    pool().fill(layout());
    stack().emplace_back(arch().nqubits());
    top(stack()).initialize(qubits, layout());

    return success();
  }
};

/**
 * @brief Replaces the 'for' loop with one that includes all hardware qubits in
 * the init arguments. The missing hardware qubits are added to the end of the
 * arguments.
 *
 * Prepares the stack for the routing of the loop body by adding a copy of the
 * current state to the stack, resetting its SWAP history, and forwarding the
 * respective SSA values.
 */
struct ForOpPattern final : ContextualOpPattern<scf::ForOp> {
  using ContextualOpPattern<scf::ForOp>::ContextualOpPattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter& rewriter) const final {
    const auto missingQubits =
        getMissingQubits(op.getInitArgs(), top(stack()).getHardwareQubits());

    SmallVector<Value, 0> newInitArgs;
    newInitArgs.reserve(op.getInitArgs().size() + missingQubits.size());
    newInitArgs.append(op.getInitArgs().begin(), op.getInitArgs().end());
    newInitArgs.append(missingQubits.begin(), missingQubits.end());

    auto forOp = rewriter.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                             op.getUpperBound(), op.getStep(),
                                             newInitArgs);

    // Clone body from the old 'for' to the new one and map block arguments.
    const std::size_t nargs = op.getBody()->getNumArguments();
    rewriter.mergeBlocks(op.getBody(), forOp.getBody(),
                         forOp.getBody()->getArguments().take_front(nargs));

    // Replace or erase old 'for' op.
    const std::size_t nresults = op->getNumResults();
    if (nresults > 0) {
      rewriter.replaceOp(op, forOp.getResults().take_front(nresults));
    } else {
      rewriter.eraseOp(op);
    }

    // Prepare stack.
    duplicateTop(stack());

    // Forward out-of-loop and in-loop state.
    for (const auto [arg, res, iter] :
         llvm::zip(forOp.getInitArgs(), forOp.getResults(),
                   forOp.getRegionIterArgs())) {
      if (isa<QubitType>(arg.getType())) {
        belowTop(stack()).forward(arg, res);
        top(stack()).forward(arg, iter);
      }
    }

    return success();
  }
};

/**
 * @brief Restores layout by uncomputation and replaces (invalid) yield.
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
 * a for-loop always requires 2 * #(SWAPs required for loop-body)
 * additional SWAPS.
 */
struct YieldOpPattern final : ContextualOpPattern<scf::YieldOp> {
  using ContextualOpPattern<scf::YieldOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(scf::YieldOp op,
                                PatternRewriter& rewriter) const final {
    if (!isa<scf::ForOp>(op->getParentOp())) {
      return failure();
    }

    auto& state = top(stack());

    const auto swaps = llvm::reverse(state.getSwapHistory());

    for (const auto& [s0, s1] : swaps) {
      const auto layout = state.getLayout();
      const std::size_t h0 = layout[s0];
      const std::size_t h1 = layout[s1];
      const Value in0 = state.getHardwareValue(h0);
      const Value in1 = state.getHardwareValue(h1);

      auto swap = createSwap(op->getLoc(), in0, in1, rewriter);
      const auto [out0, out1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      rewriter.replaceAllUsesExcept(in0, out1, swap);
      rewriter.replaceAllUsesExcept(in1, out0, swap);

      state.swapSoftwareIndices(in0, in1);
      state.forward(in0, out0);
      state.forward(in1, out1);
    }

    assert(llvm::equal(state.getLayout(), belowTop(stack()).getLayout()) &&
           "layouts must match after restoration");

    const auto missingQubits =
        getMissingQubits(op.getResults(), state.getHardwareQubits());

    SmallVector<Value, 0> newResults;
    newResults.reserve(op.getResults().size() + missingQubits.size());
    newResults.append(op.getResults().begin(), op.getResults().end());
    newResults.append(missingQubits.begin(), missingQubits.end());

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, newResults);

    stack().pop_back();

    return success();
  }
};

struct AllocQubitPattern final : ContextualOpPattern<AllocQubitOp> {
  using ContextualOpPattern<AllocQubitOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(AllocQubitOp alloc,
                                PatternRewriter& rewriter) const final {
    auto& state = top(stack());

    const std::size_t index = pool().retrieve();
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG({ llvm::dbgs() << "alloc index: " << index << '\n'; });
    const Value q = top(stack()).getHardwareValue(index);

    auto reset = rewriter.create<ResetOp>(alloc.getLoc(), q);
    rewriter.replaceOp(alloc, reset);

    state.forward(reset.getInQubit(), reset.getOutQubit());
    return success();
  }
};

struct DeallocQubitPattern final : ContextualOpPattern<DeallocQubitOp> {
  using ContextualOpPattern<DeallocQubitOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(DeallocQubitOp dealloc,
                                PatternRewriter& rewriter) const final {
    const Value q = dealloc.getQubit();
    const std::size_t index = top(stack()).getHardwareIndex(q);

    if (pool().isUsed(index)) {
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
      LLVM_DEBUG({ llvm::dbgs() << "dealloc index: " << index << '\n'; });
      pool().release(index);
    }

    rewriter.eraseOp(dealloc);
    return success();
  }
};

struct ResetPattern final : ContextualOpPattern<ResetOp> {
  using ContextualOpPattern<ResetOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(ResetOp reset,
                                PatternRewriter& /*rewriter*/) const final {
    top(stack()).forward(reset.getInQubit(), reset.getOutQubit());
    return success();
  }
};

struct NaiveUnitaryPattern final
    : ContextualInterfacePattern<UnitaryInterface> {
  using ContextualInterfacePattern<
      UnitaryInterface>::ContextualInterfacePattern;

  LogicalResult matchAndRewrite(UnitaryInterface u,
                                PatternRewriter& rewriter) const final {
    // Global-phase or zero-qubit unitary: Nothing to do.
    if (u.getAllInQubits().empty()) {
      return success();
    }

    auto& state = top(stack());

    // Single-qubit: Forward mapping.
    if (!isTwoQubitGate(u)) {
      state.forward(u.getAllInQubits()[0], u.getAllOutQubits()[0]);
      return success();
    }

    if (!isExecutable(u)) {
      makeExecutable(u, rewriter); // Ensure executability on hardware.
    }

    const auto& [execIn0, execIn1] = getIns(u);
    const auto& [execOut0, execOut1] = getOuts(u);

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG({
      const std::size_t s0 = state.getSoftwareIndex(execIn0);
      const std::size_t s1 = state.getSoftwareIndex(execIn1);
      const std::size_t h0 = state.getHardwareIndex(execIn0);
      const std::size_t h1 = state.getHardwareIndex(execIn1);

      llvm::dbgs() << llvm::format("gate: s%d/h%d, s%d/h%d\n", s0, h0, s1, h1);
    });

    state.forward(execIn0, execOut0);
    state.forward(execIn1, execOut1);

    return success();
  }

private:
  /**
   * @brief Returns true iff @p u is executable on the targeted architecture.
   */
  [[nodiscard]] bool isExecutable(UnitaryInterface u) const {
    const auto& state = top(stack());
    const auto& [in0, in1] = getIns(u);
    return arch().areAdjacent(state.getHardwareIndex(in0),
                              state.getHardwareIndex(in1));
  }

  /**
   * @brief Get shortest path between @p in0 and @p in1.
   */
  [[nodiscard]] llvm::SmallVector<std::size_t> getPath(const Value in0,
                                                       const Value in1) const {
    const auto& state = top(stack());
    return arch().shortestPathBetween(state.getHardwareIndex(in0),
                                      state.getHardwareIndex(in1));
  }

  /**
   * @brief Insert SWAPs such that @p u is executable.
   */
  void makeExecutable(UnitaryInterface u, PatternRewriter& rewriter) const {
    auto& state = top(stack());

    const auto& [q0, q1] = getIns(u);
    const auto path = getPath(q0, q1);
    for (std::size_t i = 0; i < path.size() - 2; ++i) {
      const std::size_t h0 = path[i];
      const std::size_t h1 = path[i + 1];

      const Value in0 = state.getHardwareValue(h0);
      const Value in1 = state.getHardwareValue(h1);

      const std::size_t s0 = state.getSoftwareIndex(in0);
      const std::size_t s1 = state.getSoftwareIndex(in1);

      // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
      LLVM_DEBUG({
        llvm::dbgs() << llvm::format(
            "swap: s%d/h%d, s%d/h%d <- s%d/h%d, s%d/h%d\n", s1, h0, s0, h1, s0,
            h0, s1, h1);
      });

      auto swap = createSwap(u->getLoc(), in0, in1, rewriter);
      const auto& [out0, out1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      rewriter.replaceAllUsesExcept(in0, out1, swap);
      rewriter.replaceAllUsesExcept(in1, out0, swap);

      state.recordSwap(s0, s1);
      state.swapSoftwareIndices(in0, in1);
      state.forward(in0, out0);
      state.forward(in1, out1);

      checkZeroUse(out0);
      checkZeroUse(out1);
    }
  }

  /**
   * @brief Check if a qubit's hardware index has been allocated before and now
   * has zero uses.
   *
   * Due to the SWAPs the 'dealloc' statements might dealloc a qubit with a
   * different hardware index than the one we originally allocated. This
   * function ensures that we release hardware indices when a qubit that has
   * been allocated previously and now has zero uses.
   */
  void checkZeroUse(const Value q) const {
    const std::size_t index = top(stack()).getHardwareIndex(q);
    if (!q.hasOneUse() && pool().isUsed(index)) {

      // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
      LLVM_DEBUG({
        llvm::dbgs() << llvm::format("free index with nuse=0: %d\n", index);
      });

      pool().release(index);
    }
  }
};

struct MeasurePattern final : ContextualOpPattern<MeasureOp> {
  using ContextualOpPattern<MeasureOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(MeasureOp m,
                                PatternRewriter& /*rewriter*/) const final {
    top(stack()).forward(m.getInQubit(), m.getOutQubit());
    return success();
  }
};

/**
 * @brief Collect patterns for the naive routing algorithm.
 */
void populateNaiveRoutingPatterns(RewritePatternSet& patterns,
                                  Architecture& arch, Layout layout,
                                  RoutingStack& stack,
                                  HardwareIndexPool& pool) {
  patterns.add<FuncOpPattern, ForOpPattern, YieldOpPattern, AllocQubitPattern,
               ResetPattern, NaiveUnitaryPattern, MeasurePattern,
               DeallocQubitPattern>(patterns.getContext(), arch, layout, stack,
                                    pool);
}

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    Architecture arch = getArchitecture(ArchitectureName::MQTTest);

    RoutingStack stack{};
    HardwareIndexPool pool{};

    std::random_device rd;
    const std::size_t seed = rd();
    const auto layout = getRandomLayout(arch.nqubits(), seed);

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG({
      llvm::dbgs() << "initial layout with seed " << seed << ": ";
      for (const std::size_t i : layout) {
        llvm::dbgs() << i << ' ';
      }
      llvm::dbgs() << '\n';
    });

    RewritePatternSet patterns(module.getContext());
    populateNaiveRoutingPatterns(patterns, arch, layout, stack, pool);
    walkPreOrderAndApplyPatterns(module, std::move(patterns));
  }
};

} // namespace
} // namespace mqt::ir::opt
