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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Action.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
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
constexpr llvm::StringLiteral ATTRIBUTE_ENTRY_POINT{"entry_point"};

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
 * @brief Manage mapping between SSA values and physical hardware indices.
 *
 * Note that we use the terminology "hardware" and "software" qubits here,
 * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
 * are C++ keywords.
 */
class QubitState {
public:
  /**
   * @brief A pair of indices.
   */
  using QubitIndexPair = std::pair<std::size_t, std::size_t>;

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
   * @brief Reset the state.
   * @param l0 The initial layout to apply.
   */
  void reset(Layout l0) { initialize(hardwareQubits_, l0); }

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
  [[nodiscard]] SmallVector<QubitIndexPair, 32> getSwapHistory() const {
    return swapHistory_;
  }

  /**
   * @brief Record a swap.
   */
  void recordSwap(std::size_t idx1, std::size_t idx2) {
    swapHistory_.emplace_back(idx1, idx2);
  }

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
    mapping_.erase(it);
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

using QubitStateStack = SmallVector<QubitState, 2>;

/**
 * @brief Manages qubit allocations.
 *
 * Provides get(..) and free(...) methods that are to be called when
 * replacing alloc's and dealloc's. Internally, these methods use a vector
 * to manage free / unused hardware qubit SSA values. This ensures that
 * alloc's and dealloc's can be in any arbitrary order.
 */
class QubitAllocator {
public:
  explicit QubitAllocator(const std::size_t nqubits) : indices_(nqubits) {}

  /**
   * @brief Initialize the allocator.
   *
   * Adds the indices of the given layout to the vector of free indices.
   * Note that we reverse the layout here s.t. nqubit consecutive allocations
   * are exactly the initial layout indices.
   *
   * @param layout A layout to apply.
   */
  void initialize(Layout layout) {
    llvm::copy(llvm::reverse(layout), indices_.begin());
  }

  /**
   * @brief Get unused, free, hardware index.
   * @return Hardware index or std::nullopt if no more free are left.
   */
  [[nodiscard]] std::optional<std::size_t> get() {
    if (indices_.empty()) {
      return std::nullopt;
    }

    const std::size_t i = indices_.back();
    indices_.pop_back();
    return i;
  }

  /**
   * @brief Release hardware index.
   */
  void free(const std::size_t i) { indices_.push_back(i); }

  /**
   * @brief Return the number of free indices.
   */
  [[nodiscard]] std::size_t getNumFree() const { return indices_.size(); }

private:
  /**
   * @brief LIFO of available hardware indices.
   */
  SmallVector<std::size_t, 0> indices_;
};

/**
 * @brief Retrieve free, unused, hardware qubit.
 *
 * @return Hardware qubit SSA value OR `nullptr` if there are no more free
 * qubits left.
 */
Value retrieveFreeQubit(QubitAllocator& allocator, QubitState& state) {
  const std::optional<std::size_t> index = allocator.get();
  if (!index.has_value()) {
    return {};
  }

  return state.getHardwareValue(index.value());
}

/**
 * @brief Release used hardware qubit.
 */
void releaseUsedQubit(Value q, QubitAllocator& allocator, QubitState& state) {
  allocator.free(state.getHardwareIndex(q));
}

//===----------------------------------------------------------------------===//
// Pre-Order Walk Pattern Rewrite Driver
//===----------------------------------------------------------------------===//

struct PreOrderWalkDriverAction final
    : tracing::ActionImpl<PreOrderWalkDriverAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreOrderWalkDriverAction)
  using ActionImpl::ActionImpl;
  // NOLINTNEXTLINE(readability-identifier-naming) // MLIR requires lowercase.
  static constexpr StringLiteral tag = "walk-and-apply-patterns-pre-order";
  void print(raw_ostream& os) const override { os << tag; }
};

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
 * @brief Base class for rewrite patterns that require the current routing
 * context.
 *
 * The routing context consists of
 *  - The targeted architecture
 *  - An initial layout generator
 *  - A qubit state stack
 *  - A qubit allocator
 */
template <class BasePattern> struct ContextualPattern : BasePattern {
public:
  ContextualPattern(MLIRContext* ctx, Architecture& arch, Layout layout,
                    QubitStateStack& stack, QubitAllocator& allocator)
      : BasePattern(ctx), arch_(&arch), layout_(layout), stack_(&stack),
        allocator_(&allocator) {}

protected:
  [[nodiscard]] QubitState& parentState() const {
    assert(stack().size() >= 2 &&
           "parentState: expected at least two stack items");
    return *std::next(stack().end(), -2);
  }
  [[nodiscard]] Architecture& arch() const { return *arch_; }
  [[nodiscard]] Layout layout() const { return layout_; }
  [[nodiscard]] QubitStateStack& stack() const { return *stack_; }
  [[nodiscard]] QubitState& state() const {
    assert(!stack().empty() && "state: expected at least one stack item");
    return stack().back();
  }
  [[nodiscard]] QubitAllocator& allocator() const { return *allocator_; }

private:
  Architecture* arch_;
  Layout layout_;
  QubitStateStack* stack_;
  QubitAllocator* allocator_;
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

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter& rewriter) const final {
    if (!containsUnitary(func)) {
      return failure();
    }

    llvm::SmallVector<Value> qubits(arch().nqubits());

    // In this if-branch we would collect (or add) the qubits
    // from (to) the argument-list.
    if (!func->hasAttr(ATTRIBUTE_ENTRY_POINT)) {
      return failure();
    }

    rewriter.setInsertionPointToStart(&func.getBody().front());
    initEntryPoint(qubits, rewriter);

    stack().emplace_back(arch().nqubits());
    state().initialize(qubits, layout());
    allocator().initialize(layout());

    return success();
  }
};

struct IfOpPattern final : OpRewritePattern<scf::IfOp> {
  explicit IfOpPattern(MLIRContext* ctx) : OpRewritePattern<scf::IfOp>(ctx) {}
  LogicalResult matchAndRewrite(scf::IfOp /*op*/,
                                PatternRewriter& /*rewriter*/) const final {
    return success();
  }
};

struct ForOpPattern final : ContextualOpPattern<scf::ForOp> {
  using ContextualOpPattern<scf::ForOp>::ContextualOpPattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter& rewriter) const final {
    if (!containsUnitary(forOp)) {
      return failure();
    }

    auto newFor = extendForArgs(forOp, rewriter);
    if (newFor != forOp) {
      cloneBody(forOp, newFor, rewriter);
      extendYield(newFor, rewriter);

      // Replace old results with new ones or remove if none.
      const std::size_t nresults = forOp->getNumResults();
      if (nresults > 0) {
        const auto res = newFor.getResults().take_front(nresults);
        rewriter.replaceOp(forOp, res);
      } else {
        rewriter.eraseOp(forOp);
      }
    }

    stack().push_back(state()); // Add copy to stack.
    state().clearHistory();     // The for-loop region gets its own history.

    // Forward out-of-loop and in-loop state.
    parentState().forwardRange(newFor.getInitArgs(), newFor->getResults());
    state().forwardRange(newFor.getInitArgs(), newFor.getRegionIterArgs());

    return success();
  }

private:
  /**
   * @brief Clone body from one 'for' op to the other and map
   * block arguments.
   */
  static void cloneBody(scf::ForOp forOp, scf::ForOp newForOp,
                        PatternRewriter& rewriter) {
    SmallVector<Value> mapping(newForOp.getBody()->getArguments());
    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(), mapping);
  }

  /**
   * @brief Extend the results of the existing 'yield' operation with the
   * missing qubits from the init args.
   *
   * Instead of replacing the op, we (temporarily) create an invalid IR
   * with two yield operations and remove the second yield in the respective
   * YieldOpPattern. This allows us to restore the permutation without
   * having to manage a worklist and listen to replacement events.
   */
  static void extendYield(scf::ForOp newFor, PatternRewriter& rewriter) {
    const Operation* term = newFor.getBody()->getTerminator();
    scf::YieldOp yield = dyn_cast<scf::YieldOp>(term);

    const BlockArgument* begin = newFor.getRegionIterArgs().begin();
    const BlockArgument* end = newFor.getRegionIterArgs().end();

    SmallVector<Value> results;
    results.reserve(newFor.getNumRegionIterArgs());

    const auto existing = llvm::to_vector(yield.getResults());
    results.append(existing.begin(), existing.end());
    std::advance(begin, static_cast<std::ptrdiff_t>(existing.size()));
    results.append(begin, end);

    rewriter.setInsertionPointAfter(yield);
    rewriter.create<scf::YieldOp>(yield->getLoc(), results);
  }

  /**
   * @brief Create new for op with all static qubits as init args.
   *
   * Returns the old loop if the init args already contain all qubits.
   *
   * Keeps the order of the existing init args the same. Simply adds
   * the missing qubits.
   */
  scf::ForOp extendForArgs(scf::ForOp forOp, PatternRewriter& rewriter) const {
    DenseSet<Value> included;
    for (const auto& arg : forOp.getInitArgs()) {
      if (arg.getType() == rewriter.getType<QubitType>()) {
        included.insert(arg);
      }
    }

    if (included.size() == state().getNumQubits()) {
      return forOp;
    }

    SmallVector<Value> missing;
    missing.reserve(state().getNumQubits() - included.size());
    for (const auto& q : state().getHardwareQubits()) {
      if (!included.contains(q)) {
        missing.push_back(q);
      }
    }

    SmallVector<Value> newInitArgs(forOp.getInitArgs());
    newInitArgs.append(missing.begin(), missing.end());

    return rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(),
                                       forOp.getUpperBound(), forOp.getStep(),
                                       newInitArgs);
  }
};

struct YieldOpPattern final : ContextualOpPattern<scf::YieldOp> {
  using ContextualOpPattern<scf::YieldOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(scf::YieldOp yield,
                                PatternRewriter& rewriter) const final {
    restoreLayout(yield, rewriter);
    stack().pop_back();
    rewriter.eraseOp(yield);
    return success();
  }

private:
  /**
   * @brief Restore layout by uncompute.
   *
   * Using uncompute has the advantages of (1) being intuitive and
   * (2) preserving the optimally of the original SWAP sequence.
   * Essentially the better the routing algorithm the better the
   * uncompute. Moreover, this has the nice property that routing
   * a for-loop always requires 2 * #(SWAPs required for loop-body)
   * additional SWAPS.
   *
   * @param yield The yield operation.
   * @param rewriter The PatternRewriter.
   */
  void restoreLayout(scf::YieldOp yield, PatternRewriter& rewriter) const {
    const auto swaps = llvm::reverse(state().getSwapHistory());
    for (const auto& [s0, s1] : swaps) {
      const auto layout = state().getLayout();
      const std::size_t h0 = layout[s0];
      const std::size_t h1 = layout[s1];
      const Value in0 = state().getHardwareValue(h0);
      const Value in1 = state().getHardwareValue(h1);

      auto swap = createSwap(yield->getLoc(), in0, in1, rewriter);
      const auto [out0, out1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      rewriter.replaceAllUsesExcept(in0, out1, swap);
      rewriter.replaceAllUsesExcept(in1, out0, swap);

      state().swapSoftwareIndices(in0, in1);
      state().forward(in0, out0);
      state().forward(in1, out1);
    }

    assert(llvm::equal(state().getLayout(), parentState().getLayout()) &&
           "layouts must match after restoration");
  }
};

struct AllocQubitPattern final : ContextualOpPattern<AllocQubitOp> {
  using ContextualOpPattern<AllocQubitOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(AllocQubitOp alloc,
                                PatternRewriter& rewriter) const final {
    if (const Value q = retrieveFreeQubit(allocator(), state())) {
      auto reset = rewriter.create<ResetOp>(alloc.getLoc(), q);
      rewriter.replaceOp(alloc, reset);
      state().forward(reset.getInQubit(), reset.getOutQubit());
      return success();
    }

    return alloc.emitOpError()
           << "requires " << (arch().nqubits() + 1)
           << " qubits but target architecture '" << arch().name()
           << "' only supports " << arch().nqubits() << " qubits";
  }
};

struct ResetPattern final : ContextualOpPattern<ResetOp> {
  using ContextualOpPattern<ResetOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(ResetOp reset,
                                PatternRewriter& /*rewriter*/) const final {
    state().forward(reset.getInQubit(), reset.getOutQubit());
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

    // Single-qubit: Forward mapping.
    if (!isTwoQubitGate(u)) {
      state().forward(u.getAllInQubits()[0], u.getAllOutQubits()[0]);
      return success();
    }

    if (!isExecutable(u)) {
      makeExecutable(u, rewriter); // Ensure executability on hardware.
    }

    const auto& [execIn0, execIn1] = getIns(u);
    const auto& [execOut0, execOut1] = getOuts(u);
    state().forward(execIn0, execOut0);
    state().forward(execIn1, execOut1);
    return success();
  }

private:
  /**
   * @brief Returns true iff @p u is executable on the targeted architecture.
   */
  [[nodiscard]] bool isExecutable(UnitaryInterface u) const {
    const auto& [in0, in1] = getIns(u);
    return arch().areAdjacent(state().getHardwareIndex(in0),
                              state().getHardwareIndex(in1));
  }

  /**
   * @brief Get shortest path between @p in0 and @p in1.
   */
  [[nodiscard]] llvm::SmallVector<std::size_t> getPath(const Value in0,
                                                       const Value in1) const {
    return arch().shortestPathBetween(state().getHardwareIndex(in0),
                                      state().getHardwareIndex(in1));
  }

  /**
   * @brief Insert SWAPs such that @p u is executable.
   */
  void makeExecutable(UnitaryInterface u, PatternRewriter& rewriter) const {
    const auto& [q0, q1] = getIns(u);
    const auto path = getPath(q0, q1);
    for (std::size_t i = 0; i < path.size() - 2; ++i) {
      const Value in0 = state().getHardwareValue(path[i]);
      const Value in1 = state().getHardwareValue(path[i + 1]);

      const std::size_t s0 = state().getSoftwareIndex(in0);
      const std::size_t s1 = state().getSoftwareIndex(in1);
      state().recordSwap(s0, s1);

      auto swap = createSwap(u->getLoc(), in0, in1, rewriter);
      const auto& [out0, out1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      rewriter.replaceAllUsesExcept(in0, out1, swap);
      rewriter.replaceAllUsesExcept(in1, out0, swap);

      state().swapSoftwareIndices(in0, in1);
      state().forward(in0, out0);
      state().forward(in1, out1);
    }
  }
};

struct MeasurePattern final : ContextualOpPattern<MeasureOp> {
  using ContextualOpPattern<MeasureOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(MeasureOp m,
                                PatternRewriter& /*rewriter*/) const final {
    state().forward(m.getInQubit(), m.getOutQubit());
    return success();
  }
};

struct DeallocQubitPattern final : ContextualOpPattern<DeallocQubitOp> {
  using ContextualOpPattern<DeallocQubitOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(DeallocQubitOp dealloc,
                                PatternRewriter& rewriter) const final {
    releaseUsedQubit(dealloc.getQubit(), allocator(), state());
    if (allocator().getNumFree() == arch().nqubits()) {
      allocator().initialize(layout());
      state().reset(layout());
    }
    rewriter.eraseOp(dealloc);
    return success();
  }
};

/**
 * @brief Collect patterns for the naive routing algorithm.
 */
void populateNaiveRoutingPatterns(RewritePatternSet& patterns,
                                  Architecture& arch, Layout layout,
                                  QubitStateStack& stack,
                                  QubitAllocator& allocator) {
  patterns.add<IfOpPattern>(patterns.getContext());
  patterns.add<FuncOpPattern, ForOpPattern, YieldOpPattern, AllocQubitPattern,
               ResetPattern, NaiveUnitaryPattern, MeasurePattern,
               DeallocQubitPattern>(patterns.getContext(), arch, layout, stack,
                                    allocator);
}

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    Architecture arch = getArchitecture(ArchitectureName::MQTTest);

    QubitStateStack stack{};
    QubitAllocator allocator(arch.nqubits());

    std::random_device rd;
    const auto initialLayout = getRandomLayout(arch.nqubits(), rd());

    RewritePatternSet patterns(module.getContext());
    populateNaiveRoutingPatterns(patterns, arch, initialLayout, stack,
                                 allocator);
    walkPreOrderAndApplyPatterns(module, std::move(patterns));
  }
};

} // namespace
} // namespace mqt::ir::opt
