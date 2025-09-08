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

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
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
 * @brief Enumerates the initial layout methods.
 */
enum class LayoutMethod : std::uint8_t {
  /// @brief Map virtual qubit 0 to physical qubit 0, 1 to 1, etc.
  Identity,
  /// @brief Create random bijective mapping from virtual to physical qubits.
  Random,
  /// @brief Use hard-coded bijective mapping from virtual to physical qubits.
  Constant
};

/**
 * @brief Base class for all initial layout mapping functions.
 */
class Layout {
public:
  [[nodiscard]] virtual llvm::SmallVector<std::size_t> generate() const = 0;
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

  [[nodiscard]] llvm::SmallVector<std::size_t> generate() const final {
    llvm::SmallVector<std::size_t, 0> mapping(nqubits());
    std::iota(mapping.begin(), mapping.end(), 0);
    return mapping;
  }
};

/**
 * @brief Random mapping.
 */
class RandomLayout final : public Layout {
public:
  explicit RandomLayout(const std::size_t nqubits, const std::size_t seed = 42U)
      : Layout(nqubits), seed_(seed) {}
  [[nodiscard]] llvm::SmallVector<std::size_t> generate() const final {
    std::mt19937 g(seed_);

    llvm::SmallVector<std::size_t, 0> mapping(nqubits());
    std::iota(mapping.begin(), mapping.end(), 0);
    std::shuffle(mapping.begin(), mapping.end(), g);
    return mapping;
  }

private:
  std::size_t seed_;
};

/**
 * @brief Hard-coded mapping.
 */
class ConstantLayout final : public Layout {
  explicit ConstantLayout(const llvm::SmallVectorImpl<std::size_t>& mapping)
      : Layout(mapping.size()), mapping_(nqubits()) {
    std::ranges::copy(mapping, mapping_.begin());
  }

  [[nodiscard]] llvm::SmallVector<std::size_t> generate() const final {
    return mapping_;
  }

private:
  llvm::SmallVector<std::size_t> mapping_;
};

//===----------------------------------------------------------------------===//
// State (Permutation) Management
//===----------------------------------------------------------------------===//

/**
 * @brief Manage mapping between SSA values and static hardware indices.
 *
 * Provides retrieve(..) and release(...) methods that are to be called when
 * replacing alloc's and dealloc's. Internally, these methods use a queue
 * to manage free / unused static qubit SSA values. Using a queue ensures
 * that alloc's and dealloc's can be in any arbitrary order.
 *
 * The initial layout is applied once at construction when initializing the
 * queue. Consequently, `nqubits` back-to-back retrievals will receive
 * the static qubits defined by the initial layout.
 */
class QubitStateManager {
public:
  explicit QubitStateManager(const llvm::SmallVectorImpl<Value>& qubits,
                             const std::shared_ptr<Layout>& layout)
      : indexToValue_(qubits.size()), layout_(layout) {
    const auto mapping = layout_->generate();
    for (std::size_t j = 0; j < qubits.size(); ++j) {
      const std::size_t i = mapping[j];
      const Value q = qubits[i];
      valueToIndex_[q] = i;
      indexToValue_[i] = q;
      free_.push(i);
    }
  }

  /**
   * @brief Return static index from SSA value @p v.
   */
  [[nodiscard]] std::size_t get(const Value q) const {
    return valueToIndex_.lookup(q);
  }

  /**
   * @brief Return SSA Value from static index @p i.
   */
  [[nodiscard]] Value get(const std::size_t i) const {
    return indexToValue_[i];
  }

  /**
   * @brief Retrieve free, unused, static qubit.
   * @return SSA value of static qubit or `nullptr` if there are no more free
   * qubits.
   */
  Value retrieve() {
    if (free_.empty()) {
      return nullptr;
    }

    const std::size_t i = free_.front();
    const Value q = get(i);
    free_.pop();
    return q;
  }

  /**
   * @brief Release static qubit.
   */
  void release(Value q) { free_.push(get(q)); }

  /**
   * @brief Forward SSA values.
   * @details Replace @p in with @p out in maps.
   */
  void forward(const Value in, const Value out) {
    const std::size_t i = get(in);
    valueToIndex_[out] = i;
    indexToValue_[i] = out;
    valueToIndex_.erase(in);
  }

  /**
   * @brief Return the number of static qubits.
   */
  [[nodiscard]] std::size_t nqubits() const { return indexToValue_.size(); }

private:
  /**
   * @brief Maps SSA values to static indices.
   *
   * Using MapVector enables insertion-order iteration.
   */
  llvm::MapVector<Value, std::size_t> valueToIndex_;

  /**
   * @brief Maps static indices to SSA values.
   *
   * The size of this vector is the number of static qubits.
   */
  llvm::SmallVector<Value, 0> indexToValue_;

  /**
   * @brief Queue of available static qubit indices.
   */
  std::queue<std::size_t> free_;

  /**
   * @brief The initial layout method.
   */
  std::shared_ptr<Layout> layout_;
};

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {

/**
 * @brief Base class for rewrite patterns that require the current routing
 * context.
 *
 * The routing context consists of
 *  - A qubit state manager
 *  - The targeted architecture
 *  - An initial layout method
 *
 * @details The use of shared pointers indicates clearly that this
 * is a non-owning class.
 */
template <class OpType>
struct ContextualRewritePattern : OpRewritePattern<OpType> {
  ContextualRewritePattern(MLIRContext* ctx,
                           const std::shared_ptr<QubitStateManager>& state,
                           const std::shared_ptr<Architecture>& arch,
                           const std::shared_ptr<Layout>& layout)
      : OpRewritePattern<OpType>(ctx), state_(state), arch_(arch),
        layout_(layout) {}

  [[nodiscard]] QubitStateManager& state() const { return *state_; }
  [[nodiscard]] Architecture& arch() const { return *arch_; }
  [[nodiscard]] Layout& layout() const { return *layout_; }

private:
  std::shared_ptr<QubitStateManager> state_;
  std::shared_ptr<Architecture> arch_;
  std::shared_ptr<Layout> layout_;
};

struct IfOpPattern final : OpRewritePattern<scf::IfOp> {
  explicit IfOpPattern(MLIRContext* ctx) : OpRewritePattern<scf::IfOp>(ctx) {}
  LogicalResult matchAndRewrite(scf::IfOp /*op*/,
                                PatternRewriter& /*rewriter*/) const final {
    return failure();
  }
};

struct ForOpPattern final : OpRewritePattern<scf::ForOp> {
  explicit ForOpPattern(MLIRContext* ctx) : OpRewritePattern<scf::ForOp>(ctx) {}
  LogicalResult matchAndRewrite(scf::ForOp /*op*/,
                                PatternRewriter& /*rewriter*/) const final {
    return failure();
  }
};

struct AllocQubitPattern final : ContextualRewritePattern<AllocQubitOp> {
  using ContextualRewritePattern<AllocQubitOp>::ContextualRewritePattern;
  LogicalResult matchAndRewrite(AllocQubitOp alloc,
                                PatternRewriter& rewriter) const final {
    if (const Value q = state().retrieve()) {
      auto reset = rewriter.create<ResetOp>(q.getLoc(), q);
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

struct ResetPattern final : ContextualRewritePattern<ResetOp> {
  using ContextualRewritePattern<ResetOp>::ContextualRewritePattern;
  LogicalResult matchAndRewrite(ResetOp reset,
                                PatternRewriter& /*rewriter*/) const final {
    state().forward(reset.getInQubit(), reset.getOutQubit());
    return failure();
  }
};

struct NaiveUnitaryPattern final : OpInterfaceRewritePattern<UnitaryInterface> {
  NaiveUnitaryPattern(MLIRContext* ctx,
                      const std::shared_ptr<QubitStateManager>& state,
                      const std::shared_ptr<Architecture>& arch,
                      const std::shared_ptr<Layout>& layout)
      : OpInterfaceRewritePattern<UnitaryInterface>(ctx), state_(state),
        arch_(arch), layout_(layout) {}
  LogicalResult matchAndRewrite(UnitaryInterface u,
                                PatternRewriter& rewriter) const final {
    // Global-phase or zero-qubit unitary: Nothing to do.
    if (u.getAllInQubits().empty()) {
      return failure();
    }

    // Single-qubit: Forward mapping
    if (!isTwoQubitGate(u)) {
      state().forward(u.getAllInQubits()[0], u.getAllOutQubits()[0]);
      return failure();
    }

    // Two-qubit: Ensure executable on hardware.
    if (!isExecutable(u)) {
      makeExecutable(u, rewriter);
    }

    // After ensuring executability, forward outputs.
    const auto& [execIn0, execIn1] = getIns(u);
    const auto& [execOut0, execOut1] = getOuts(u);

    state().forward(execIn0, execOut0);
    state().forward(execIn1, execOut1);

    return failure();
  }

private:
  /**
   * @brief Returns true iff @p u is executable on the targeted architecture.
   */
  [[nodiscard]] bool isExecutable(UnitaryInterface u) const {
    const auto& [in0, in1] = getIns(u);
    return arch().areAdjacent(state().get(in0), state().get(in1));
  }

  /**
   * @brief Get shortest path between @p in0 and @p in1.
   */
  [[nodiscard]] llvm::SmallVector<std::size_t> getPath(const Value in0,
                                                       const Value in1) const {
    return arch().shortestPathBetween(state().get(in0), state().get(in1));
  }

  /**
   * @brief Insert SWAPs such that @p u is executable.
   */
  void makeExecutable(UnitaryInterface u, PatternRewriter& rewriter) const {
    const auto& [q0, q1] = getIns(u);
    auto path = getPath(q0, q1);

    for (std::size_t i = 0; i < path.size() - 1; i += 2) {
      const Value in0 = state().get(path[i]);
      const Value in1 = state().get(path[i + 1]);

      // Y(out), X(out) = SWAP X(in), Y(in)
      auto swap = createSwap(u->getLoc(), in0, in1, rewriter);
      const auto& [out1, out0] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      rewriter.replaceAllUsesExcept(in0, out0, swap);
      rewriter.replaceAllUsesExcept(in1, out1, swap);

      state().forward(in0, out1);
      state().forward(in1, out0);
    }
  }

  [[nodiscard]] QubitStateManager& state() const { return *state_; }
  [[nodiscard]] Architecture& arch() const { return *arch_; }
  [[nodiscard]] Layout& layout() const { return *layout_; }

  std::shared_ptr<QubitStateManager> state_;
  std::shared_ptr<Architecture> arch_;
  std::shared_ptr<Layout> layout_;
};

struct MeasurePattern final : ContextualRewritePattern<MeasureOp> {
  using ContextualRewritePattern<MeasureOp>::ContextualRewritePattern;
  LogicalResult matchAndRewrite(MeasureOp m,
                                PatternRewriter& /*rewriter*/) const final {
    state().forward(m.getInQubit(), m.getOutQubit());
    return failure();
  }
};

struct DeallocQubitPattern final : ContextualRewritePattern<DeallocQubitOp> {
  using ContextualRewritePattern<DeallocQubitOp>::ContextualRewritePattern;
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
void populateNaiveRoutingPatterns(
    RewritePatternSet& patterns,
    const std::shared_ptr<QubitStateManager>& state,
    const std::shared_ptr<Architecture>& arch,
    const std::shared_ptr<Layout>& layout) {
  patterns.add<IfOpPattern, ForOpPattern>(patterns.getContext());
  patterns.add<AllocQubitPattern, ResetPattern, NaiveUnitaryPattern,
               MeasurePattern, DeallocQubitPattern>(patterns.getContext(),
                                                    state, arch, layout);
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
void initEntryPoint(llvm::SmallVectorImpl<Value>& qubits,
                    PatternRewriter& rewriter) {
  for (std::size_t i = 0; i < qubits.size(); ++i) {
    auto op =
        rewriter.create<QubitOp>(rewriter.getInsertionPoint()->getLoc(), i);
    rewriter.setInsertionPointAfter(op);
    qubits[i] = op.getQubit();
  }
}

struct PreOrderWalkDriverAction final
    : tracing::ActionImpl<PreOrderWalkDriverAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreOrderWalkDriverAction)
  using ActionImpl::ActionImpl;
  static constexpr StringLiteral tag = "walk-and-apply-patterns-pre-order";
  void print(raw_ostream& os) const override { os << tag; }
};

/**
 * @brief A pre-order version of the walkAndApplyPatterns driver.
 *
 * Instead of a post-order worklist this driver simply walks the IR in pre-order
 * (parents first).
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

  ctx->executeAction<PreOrderWalkDriverAction>(
      [&] {
        op->walk<WalkOrder::PreOrder>([&](Operation* x) {
          std::ignore = applicator.matchAndRewrite(x, rewriter);
        });
      },
      {op});
}

[[nodiscard]] LogicalResult route(ModuleOp module,
                                  const ArchitectureName& archName,
                                  const LayoutMethod& layoutMethod) {
  std::random_device rd;
  const auto arch = std::make_shared<Architecture>(getArchitecture(archName));
  const auto layout = std::make_shared<RandomLayout>(arch->nqubits(), rd());

  auto res = module->walk([&](func::FuncOp func) {
    PatternRewriter rewriter(func->getContext());
    rewriter.setInsertionPointToStart(&func.getBody().front());

    llvm::SmallVector<Value, 0> qubits(arch->nqubits());

    if (!func->hasAttr(ATTRIBUTE_ENTRY_POINT)) {
      // For now we don't route non-entry point functions.
      // This would be the point where we would collect the qubits from the
      // argument-list.
      return WalkResult::skip();
    }

    initEntryPoint(qubits, rewriter);

    const auto state = std::make_shared<QubitStateManager>(qubits, layout);

    RewritePatternSet patterns(func.getContext());
    populateNaiveRoutingPatterns(patterns, state, arch, layout);

    walkPreOrderAndApplyPatterns(func, std::move(patterns));

    return WalkResult::advance();
  });

  return res.wasInterrupted() ? failure() : success();
}

}; // namespace

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    if (failed(route(getOperation(), ArchitectureName::MQTTest,
                     LayoutMethod::Random))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mqt::ir::opt
