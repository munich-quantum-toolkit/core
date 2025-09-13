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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <llvm/ADT/MapVector.h>
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
#include <queue>
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

/**
 * @brief The expected number of qubits a architecture supports for optimizing
 * inline capacities of SmallVectors.
 */
constexpr std::size_t AVG_NQUBITS_ARCH = 32;

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
 * @brief Base class for all initial layout mapping functions.
 */
class Layout {
public:
  [[nodiscard]] virtual SmallVector<std::size_t> generate() const = 0;
  explicit Layout(const std::size_t nqubits) : nqubits_(nqubits) {}
  virtual ~Layout() = default;

protected:
  [[nodiscard]] std::size_t nqubits() const { return nqubits_; }

private:
  std::size_t nqubits_;
};

/**
 * @brief Identity mapping.
 */
class IdentityLayout final : public Layout {
public:
  using Layout::Layout;

  [[nodiscard]] SmallVector<std::size_t> generate() const final {
    SmallVector<std::size_t, AVG_NQUBITS_ARCH> mapping(nqubits());
    std::iota(mapping.begin(), mapping.end(), 0);
    return mapping;
  }
};

/**
 * @brief Random mapping.
 */
class RandomLayout final : public Layout {
public:
  using Layout::Layout;

  [[nodiscard]] SmallVector<std::size_t> generate() const final {
    std::random_device rd;
    std::mt19937_64 rng(rd());

    SmallVector<std::size_t, AVG_NQUBITS_ARCH> mapping(nqubits());
    std::iota(mapping.begin(), mapping.end(), 0);
    std::shuffle(mapping.begin(), mapping.end(), rng);
    return mapping;
  };
};

/**
 * @brief Hard-coded mapping.
 */
class ConstantLayout final : public Layout {
public:
  explicit ConstantLayout(ArrayRef<std::size_t> mapping)
      : Layout(mapping.size()), mapping_(nqubits()) {
    std::ranges::copy(mapping, mapping_.begin());
  }

  [[nodiscard]] SmallVector<std::size_t> generate() const final {
    return mapping_;
  }

private:
  SmallVector<std::size_t> mapping_;
};

//===----------------------------------------------------------------------===//
// State (Permutation) Management
//===----------------------------------------------------------------------===//

/**
 * @brief Manage mapping between SSA values and physical hardware indices.
 *
 * Provides retrieve(..) and release(...) methods that are to be called when
 * replacing alloc's and dealloc's. Internally, these methods use a queue
 * to manage free / unused hardware qubit SSA values. Using a queue ensures
 * that alloc's and dealloc's can be in any arbitrary order.
 *
 * This class generates and applies the initial layout at construction.
 * Consequently, `nqubits` back-to-back retrievals will receive the hardware
 * qubits defined by the initial layout.
 *
 * Note that we use the terminology "hardware" and "software" qubits here,
 * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
 * are C++ keywords.
 */
class QubitState {
public:
  QubitState(const SmallVectorImpl<Value>& hwQubits,
             ArrayRef<std::size_t> layout)
      : hardwareQubits_(hwQubits.size()) {
    mapping_.reserve(hwQubits.size());
    reset(hwQubits, layout);
  }

  /**
   * @brief Return hardware index from SSA value.
   */
  [[nodiscard]] std::size_t getHardwareIndex(const Value q) const {
    return mapping_.at(q).first;
  }

  /**
   * @brief Return hardware index from SSA value.
   * @param q The SSA value.
   */
  [[nodiscard]] std::size_t getSoftwareIndex(const Value q) const {
    return mapping_.at(q).second;
  }

  /**
   * @brief Return SSA Value from hardware index.
   * @param hardwareIndex The hardware index.
   */
  [[nodiscard]] Value getHardwareValue(const std::size_t index) const {
    assert(index < hardwareQubits_.size() &&
           "getHardwareValue: index out of bounds");
    return hardwareQubits_[index];
  }

  /**
   * @brief Return the SSA values for hardware indices from 0...nqubits.
   */
  [[nodiscard]] SmallVector<Value> getHardwareQubits() const {
    return hardwareQubits_;
  }

  /**
   * @brief Return the number of hardware qubits.
   */
  [[nodiscard]] std::size_t getNumQubits() const {
    return hardwareQubits_.size();
  }

  /**
   * @brief Return the current software to hardware qubit mapping.
   */
  SmallVector<std::size_t> getPermutation() {
    llvm::SmallVector<std::size_t> perm(getNumQubits());
    for (const auto& [q, map] : mapping_) {
      perm[map.second] = map.first;
    }
    return perm;
  }

  /**
   * @brief Retrieve free, unused, hardware qubit.
   * @return SSA value of static qubit or `nullptr` if there are no more free
   * qubits.
   */
  Value retrieve() {
    if (free_.empty()) {
      return nullptr;
    }

    const std::size_t i = free_.front();
    const Value q = getHardwareValue(i);
    free_.pop();
    return q;
  }

  /**
   * @brief Release hardware qubit.
   */
  void release(Value q) { free_.push(getHardwareIndex(q)); }

  /**
   * @brief Forward SSA values.
   * @details Replace @p in with @p out in maps.
   */
  void forward(const Value in, const Value out) {
    const auto it = mapping_.find(in);
    assert(it != mapping_.end() && "forward: unknown input value");

    const QubitMapping map = it->second;
    const std::size_t h = map.first;

    hardwareQubits_[h] = out;

    assert(!mapping_.contains(out) && "forward: output value already mapped");

    mapping_.try_emplace(out, map);
    mapping_.erase(it);
  }

  /**
   * @brief Swap software indices.
   */
  void swapSoftwareIndices(const Value a, const Value b) {
    std::swap(mapping_[a].second, mapping_[b].second);
  }

  /**
   * @brief Reset the state.
   *
   * Fills the queue with static qubit SSA values, permutated by the layout.
   *
   * @param qubits A vector of SSA values.
   * @param layout A layout to apply.
   */
  void reset(const SmallVectorImpl<Value>& hwQubits,
             ArrayRef<std::size_t> layout) {
    clear();
    for (std::size_t s = 0; s < getNumQubits(); ++s) {
      const std::size_t h = layout[s];
      const Value q = hwQubits[h];
      hardwareQubits_[h] = q;
      mapping_.try_emplace(q, std::make_pair(h, s));
      free_.push(h);
    }
  }

  /**
   * @brief Return the number of free qubits.
   */
  [[nodiscard]] std::size_t getNumFree() const { return free_.size(); }

private:
  /**
   * @brief A pair of indices, where first is the hardware index
   * and second is the software index.
   */
  using QubitMapping = std::pair<std::size_t, std::size_t>;

  /**
   * @brief Clear the state.
   */
  void clear() {
    free_ = std::queue<std::size_t>(); // Clear queue.
    mapping_.clear();
  }

  /**
   * @brief Maps SSA values to hardware indices;
   */
  DenseMap<Value, QubitMapping> mapping_;

  /**
   * @brief Maps hardware indices to SSA values.
   *
   * The size of this vector is the number of hardware qubits.
   */
  SmallVector<Value> hardwareQubits_;

  /**
   * @brief Queue of available hardware indices.
   * TODO: Change to smallvector because max-size of queue will always be
   * nqubits.
   */
  std::queue<std::size_t> free_;
};

using QubitStateStack = SmallVector<QubitState, 2>;

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
          std::ignore = applicator.matchAndRewrite(curr, rewriter);
        }
      },
      {op});
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Verify if a operation requires routing.
 *
 * Walk the IR and interrupt the traversal if any unitary is found.
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
 *  - A qubit state manager
 *  - The targeted architecture
 *  - An initial layout generator
 */
template <class BasePattern> struct ContextualPattern : BasePattern {
public:
  ContextualPattern(MLIRContext* ctx, QubitStateStack& stack,
                    Architecture& arch, Layout& layout)
      : BasePattern(ctx), stack_(&stack), arch_(&arch), layout_(&layout) {}

protected:
  [[nodiscard]] QubitState& parentState() const {
    return *std::next(stack().end(), -2);
  }
  [[nodiscard]] QubitState& state() const { return stack().back(); }
  [[nodiscard]] QubitStateStack& stack() const { return *stack_; }
  [[nodiscard]] Architecture& arch() const { return *arch_; }
  [[nodiscard]] Layout& layout() const { return *layout_; }

private:
  QubitStateStack* stack_;
  Architecture* arch_;
  Layout* layout_;
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

    auto mapping = layout().generate();
    stack().emplace_back(qubits,
                         mapping); // TODO: Naming 'mapping' vs 'layout'.

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
    cloneBody(forOp, newFor, rewriter);
    extendYield(newFor, rewriter);

    // Replace old results with new ones or remove if none.
    if (forOp.getNumResults() > 0) {
      const auto res = newFor.getResults().take_front(forOp.getNumResults());
      rewriter.replaceOp(forOp, res);
    } else {
      rewriter.eraseOp(forOp);
    }

    stack().push_back(state()); // Add copy to stack.

    // Forward out-of-loop and in-loop state.
    std::size_t i = 0;
    for (const auto& arg : newFor.getInitArgs()) {
      parentState().forward(arg, newFor->getResult(i));
      state().forward(arg, newFor.getRegionIterArg(i));
      i++;
    }

    return success();
  }

private:
  /**
   * @brief Clone body from one for op to the other and map
   * block arguments.
   */
  static void cloneBody(scf::ForOp forOp, scf::ForOp newForOp,
                        PatternRewriter& rewriter) {
    SmallVector<Value> mapping(newForOp.getBody()->getArguments());
    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(), mapping);
  }

  /**
   * @brief Extend the results of the existing yield operation with the
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
    results.append(yield->getResults().begin(), yield->getResults().end());
    results.append(std::next(begin, yield->getNumResults()), end);

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
    assert(stack().size() >= 2);
    const auto from = state().getPermutation();
    const auto to = parentState().getPermutation();
    restorePermutation(from, to, yield->getLoc(), rewriter);
    stack().pop_back();
    rewriter.eraseOp(yield);
    return success();
  }

private:
  /**
   * @brief Restore permutation by adding SWAPs in linear runtime.
   *
   * Finds all cycles of a permutation and swaps along those cycles.
   *
   * @example The permutation (1, 2, 3, 4) -> (1, 3, 2, 4) has the
   * cycles (1)(2, 3)(4). Single element cycles are already at their
   * desired position. Consequently, we swap 2 and 3.
   *
   * @param from The current permutation
   * @param to   The desired permutation
   * @param location The location of the SWAPs.
   * @param rewriter The PatternRewriter.
   */
  void restorePermutation(const SmallVectorImpl<std::size_t>& from,
                          const SmallVectorImpl<std::size_t>& to,
                          Location location, PatternRewriter& rewriter) const {
    assert(from.size() == to.size());

    const std::size_t n = from.size();

    DenseMap<std::size_t, std::size_t> pos(n);
    for (std::size_t i = 0; i < n; ++i) {
      pos[to[i]] = i;
    }

    SmallVector<std::size_t> f(n);
    for (std::size_t i = 0; i < n; ++i) {
      f[i] = pos[from[i]];
    }

    llvm::BitVector visited(n);
    for (std::size_t i = 0; i < n; ++i) {
      if (visited[i] || f[i] == i) {
        visited[i] = true;
        continue;
      }

      SmallVector<std::size_t, 8> cycles;
      std::size_t j = i;
      while (!visited[j]) {
        visited[j] = true;
        cycles.push_back(j);
        j = f[j];
      }

      if (cycles.size() == 1) {
        continue;
      }

      std::size_t a = cycles[0];
      for (std::size_t k = 1; k < cycles.size(); ++k) {
        std::size_t b = cycles[k];

        const Value in0 = state().getHardwareValue(a);
        const Value in1 = state().getHardwareValue(b);

        auto swap = createSwap(location, in0, in1, rewriter);
        const auto& [out0, out1] = getOuts(swap);

        rewriter.setInsertionPointAfter(swap);
        rewriter.replaceAllUsesExcept(in0, out1, swap);
        rewriter.replaceAllUsesExcept(in1, out0, swap);

        state().swapSoftwareIndices(in0, in1);
        state().forward(in0, out0);
        state().forward(in1, out1);
      }
    }
  }
};

struct AllocQubitPattern final : ContextualOpPattern<AllocQubitOp> {
  using ContextualOpPattern<AllocQubitOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(AllocQubitOp alloc,
                                PatternRewriter& rewriter) const final {
    if (const Value q = state().retrieve()) {
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
    return success(); // Gate is now executable: Don't revisit.
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
    auto path = getPath(q0, q1);

    for (std::size_t i = 0; i < path.size() - 2; ++i) {
      const Value in0 = state().getHardwareValue(path[i]);
      const Value in1 = state().getHardwareValue(path[i + 1]);

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

/**
 * @brief Apply pattern for mqtopt.deallocQubit.
 *
 * If all allocated qubits are deallocated again, e.g. we reach the end of a
 * "circuit", the initial layout will be re-generated and re-applied.
 */
struct DeallocQubitPattern final : ContextualOpPattern<DeallocQubitOp> {
  using ContextualOpPattern<DeallocQubitOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(DeallocQubitOp dealloc,
                                PatternRewriter& rewriter) const final {
    state().release(dealloc.getQubit());
    if (state().getNumFree() == arch().nqubits()) {
      const auto mapping = layout().generate();
      state().reset(state().getHardwareQubits(), mapping);
    }
    rewriter.eraseOp(dealloc);
    return success();
  }
};

/**
 * @brief Collect patterns for the naive routing algorithm.
 */
void populateNaiveRoutingPatterns(RewritePatternSet& patterns,
                                  QubitStateStack& stack, Architecture& arch,
                                  Layout& layout) {
  patterns.add<IfOpPattern>(patterns.getContext());
  patterns.add<FuncOpPattern, ForOpPattern, YieldOpPattern, AllocQubitPattern,
               ResetPattern, NaiveUnitaryPattern, MeasurePattern,
               DeallocQubitPattern>(patterns.getContext(), stack, arch, layout);
}

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    QubitStateStack stack{};
    Architecture arch = getArchitecture(ArchitectureName::MQTTest);
    RandomLayout layout(arch.nqubits());

    RewritePatternSet patterns(module.getContext());
    populateNaiveRoutingPatterns(patterns, stack, arch, layout);
    walkPreOrderAndApplyPatterns(module, std::move(patterns));
  }
};

} // namespace
} // namespace mqt::ir::opt
