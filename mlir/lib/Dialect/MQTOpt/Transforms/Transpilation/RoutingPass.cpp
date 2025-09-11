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
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

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
#include <ranges>
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
    SmallVector<std::size_t, 0> mapping(nqubits());
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
    std::mt19937 g(rd());

    SmallVector<std::size_t, 0> mapping(nqubits());
    std::iota(mapping.begin(), mapping.end(), 0);
    std::shuffle(mapping.begin(), mapping.end(), g);
    return mapping;
  };
};

/**
 * @brief Hard-coded mapping.
 */
class ConstantLayout final : public Layout {
public:
  explicit ConstantLayout(const SmallVectorImpl<std::size_t>& mapping)
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
 * If all allocated qubits are deallocated again, e.g. we reach the end of a
 * "circuit", the initial layout will be re-generated and re-applied.
 *
 * Note that we use the terminology "hardware" and "software" qubits here,
 * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
 * are C++ keywords.
 */
class QubitStateManager {
public:
  explicit QubitStateManager(const SmallVectorImpl<Value>& hardwareQubits,
                             Layout& layout)
      : hardwareQubits_(hardwareQubits.size()), layout_(&layout) {
    reset(hardwareQubits);
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
  [[nodiscard]] Value getHardwareValue(const std::size_t hardwareIndex) const {
    return hardwareQubits_[hardwareIndex];
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
  void release(Value q) {
    free_.push(getHardwareIndex(q));
    if (free_.size() == getNumQubits()) {
      reset(hardwareQubits_);
    }
  }

  /**
   * @brief Forward SSA values.
   * @details Replace @p in with @p out in maps.
   */
  void forward(const Value in, const Value out) {
    const std::size_t h = getHardwareIndex(in);
    mapping_[out] = mapping_.at(in);
    hardwareQubits_[h] = out;
    mapping_.erase(in);
  }

  /**
   * @brief Swap software indices.
   */
  void swapSoftwareIndices(const Value a, const Value b) {
    std::swap(mapping_[a].second, mapping_[b].second);
  }

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
   * @brief Reset the state.
   *
   * Clears the queue and generates the initial layout. Then fills the queue
   * with static qubit SSA values, permutated by the initial layout.
   *
   * @param qubits A vector of SSA values.
   */
  void reset(const SmallVectorImpl<Value>& hardwareQubits) {
    const auto mapping = layout().generate();

    clear();
    for (std::size_t s = 0; s < getNumQubits(); ++s) {
      const std::size_t h = mapping[s];
      const Value q = hardwareQubits[h];
      hardwareQubits_[h] = q;
      mapping_.try_emplace(q, std::make_pair(h, s));
      free_.push(h);
    }
  }

  /**
   * @brief Return a reference to the initial layout.
   */
  [[nodiscard]] Layout& layout() const { return *layout_; }

  /**
   * @brief Maps SSA values to hardware and software indices.
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
   */
  std::queue<std::size_t> free_;

  /**
   * @brief The initial layout generator.
   */
  Layout* layout_;
};

using QubitStateStack = SmallVector<QubitStateManager, 2>;

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
  [[nodiscard]] QubitStateManager& parentState() const {
    return *std::next(stack().end(), -2);
  }
  [[nodiscard]] QubitStateManager& state() const { return stack().back(); }
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

    stack().emplace_back(qubits, layout());

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
  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter& rewriter) const final {
    if (!containsUnitary(loop)) {
      return failure();
    }

    SmallVector<Value> physical = state().getHardwareQubits();

    // SetVector of already included static qubits.
    llvm::SetVector<Value> included;
    // Vector of classical arguments such as integers, floats, etc.
    SmallVector<Value, 4> classicalArgs;
    // Vector of missing (ALL \ INCLUDED) static qubits.
    SmallVector<Value> extra;

    for (const auto& arg : loop.getInitArgs()) {
      if (arg.getType() != rewriter.getType<QubitType>()) {
        classicalArgs.push_back(arg);
        continue;
      }
      included.insert(arg);
    }

    for (const auto& q : physical) {
      if (!included.contains(q)) {
        extra.push_back(q);
      }
    }
    // TODO: Just add extra to existing once.
    SmallVector<Value> newInitArgs;
    newInitArgs.reserve(classicalArgs.size() + physical.size());
    newInitArgs.append(classicalArgs.begin(), classicalArgs.end());
    newInitArgs.append(included.begin(), included.end());
    newInitArgs.append(extra.begin(), extra.end());

    auto newLoop = rewriter.create<scf::ForOp>(
        loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
        loop.getStep(), newInitArgs);

    {
      // Map new arguments to old in the same order (which we haven't changed.)
      SmallVector<Value> mapping;
      for (const auto& arg : newLoop.getBody()->getArguments()) {
        mapping.push_back(arg);
      }
      rewriter.mergeBlocks(loop.getBody(), newLoop.getBody(), mapping);
    }

    {
      scf::YieldOp yield =
          dyn_cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
      assert(yield);
      const std::size_t nresults = yield.getResults().size();

      SmallVector<Value> newYieldOperands;
      newYieldOperands.reserve(newLoop.getNumRegionIterArgs());

      for (const auto& operand : yield.getResults()) {
        newYieldOperands.push_back(operand);
      }

      for (const auto& arg :
           newLoop.getRegionIterArgs() | std::views::drop(nresults)) {
        newYieldOperands.push_back(arg);
      }

      rewriter.setInsertionPoint(yield);
      rewriter.create<scf::YieldOp>(yield->getLoc(), newYieldOperands);
    }

    if (loop.getNumResults() > 0) {
      // Replace old results with new.
      auto newResults = newLoop.getResults().take_front(loop.getNumResults());
      rewriter.replaceOp(loop, newResults);
    }

    stack().push_back(state()); // Add copy to stack.

    // Forward out-of-loop and in-loop state.
    std::size_t i = 0;
    for (const auto& arg : newLoop.getInitArgs()) {
      parentState().forward(arg, newLoop->getResult(i));
      state().forward(arg, newLoop.getRegionIterArg(i));
      i++;
    }

    return success();
  }
};

struct YieldOpPattern final : ContextualOpPattern<scf::YieldOp> {
  using ContextualOpPattern<scf::YieldOp>::ContextualOpPattern;
  LogicalResult matchAndRewrite(scf::YieldOp yield,
                                PatternRewriter& rewriter) const final {
    llvm::outs() << "restore permutation.\n";
    for (const auto& pI : parentState().getPermutation()) {
      llvm::outs() << pI << " ";
    }
    llvm::outs() << "\n";
    for (const auto& i : state().getPermutation()) {
      llvm::outs() << i << " ";
    }
    llvm::outs() << "\n";

    stack().pop_back();
    rewriter.eraseOp(yield);
    return success();
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

      // ○: in0   ■: in1
      //
      // ■, ○ = SWAP(○, ■)
      //
      //  ○──%q0_0──X──%q0_1──■
      //            │
      //  ■──%q1_0──X──%q1_1──○

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
    state().release(dealloc.getQubit());
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
    IdentityLayout layout(arch.nqubits());

    RewritePatternSet patterns(module.getContext());
    populateNaiveRoutingPatterns(patterns, stack, arch, layout);
    walkPreOrderAndApplyPatterns(module, std::move(patterns));
  }
};

} // namespace
} // namespace mqt::ir::opt
