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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <numeric>
#include <random>
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
 * @param rewriter PatternRewriter used to create the op.
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
 * @brief InitialLayout class for all initial layout mapping functions.
 */
struct InitialLayout {
  explicit InitialLayout(const std::size_t nqubits) : mapping_(nqubits) {}
  [[nodiscard]] std::size_t operator()(std::size_t i) const {
    return mapping_[i];
  }

protected:
  llvm::SmallVector<std::size_t, 0> mapping_;
};

/**
 * @brief Identity mapping.
 */
struct Identity : InitialLayout {
  explicit Identity(const std::size_t nqubits) : InitialLayout(nqubits) {
    std::iota(mapping_.begin(), mapping_.end(), 0);
  }
};

/**
 * @brief Random mapping.
 */
struct Random : Identity {
  explicit Random(const std::size_t nqubits) : Identity(nqubits) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(mapping_.begin(), mapping_.end(), g);
  }
};

/**
 * @brief Custom mapping.
 */
struct Custom : InitialLayout {
  Custom(const std::size_t nqubits,
         const llvm::SmallVectorImpl<std::size_t>& mapping)
      : InitialLayout(nqubits) {
    std::ranges::copy(mapping, mapping_.begin());
  }
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
                             const InitialLayout& layout) {
    for (std::size_t j = 0; j < qubits.size(); ++j) {
      const std::size_t i = layout(j);
      const Value q = qubits[i];
      valueToIndex_[q] = i;
      indexToValue_[i] = q;
      free_.push_back(q);
    }
  }

  /**
   * @brief Return static index from SSA value @p v.
   */
  [[nodiscard]] std::size_t get(const Value q) const {
    return valueToIndex_.at(q);
  }

  /**
   * @brief Return SSA Value from static index @p i.
   */
  [[nodiscard]] Value get(const std::size_t i) const {
    return indexToValue_.at(i);
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

    Value q = free_.back();
    free_.pop_back();
    return q;
  }

  /**
   * @brief Release static qubit.
   */
  void release(Value q) { free_.push_back(q); }

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

private:
  /**
   * @brief Maps SSA values to static indices.
   */
  llvm::DenseMap<Value, std::size_t> valueToIndex_;

  /**
   * @brief Maps static indices to SSA values.
   */
  llvm::DenseMap<std::size_t, Value> indexToValue_;

  /**
   * @brief Queue of available qubits indices.
   */
  llvm::SmallVector<Value> free_;
};

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {

struct RoutingEnvironment {
  RoutingEnvironment(const llvm::SmallVectorImpl<Value>& qubits,
                     const InitialLayout& layout, const Architecture& arch)
      : state(qubits, layout), arch(&arch) {}
  QubitStateManager state;
  const Architecture* arch;
};

template <class OpType> struct PatternWithEnv : OpRewritePattern<OpType> {
  PatternWithEnv(MLIRContext* ctx, std::shared_ptr<RoutingEnvironment> env)
      : OpRewritePattern<OpType>(ctx), env_(std::move(env)) {}

  [[nodiscard]] RoutingEnvironment& env() const { return *env_; }

private:
  std::shared_ptr<RoutingEnvironment> env_;
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

struct AllocQubitPattern final : PatternWithEnv<AllocQubitOp> {
  using PatternWithEnv<AllocQubitOp>::PatternWithEnv;
  LogicalResult matchAndRewrite(AllocQubitOp alloc,
                                PatternRewriter& rewriter) const final {
    if (Value q = env().state.retrieve()) {
      auto reset = rewriter.create<ResetOp>(q.getLoc(), q);
      rewriter.replaceOp(alloc, reset);
      env().state.forward(reset.getInQubit(), reset.getOutQubit());
      return success();
    }

    return alloc.emitOpError()
           << "requires " << (env().arch->nqubits() + 1)
           << " qubits but target architecture '" << env().arch->name()
           << "' only supports " << env().arch->nqubits() << " qubits";
  }
};

struct ResetPattern final : PatternWithEnv<ResetOp> {
  using PatternWithEnv<ResetOp>::PatternWithEnv;
  LogicalResult matchAndRewrite(ResetOp reset,
                                PatternRewriter& /*rewriter*/) const final {
    env().state.forward(reset.getInQubit(), reset.getOutQubit());
    return failure();
  }
};

struct NaiveUnitaryPattern final : OpInterfaceRewritePattern<UnitaryInterface> {
  NaiveUnitaryPattern(MLIRContext* ctx, std::shared_ptr<RoutingEnvironment> env)
      : OpInterfaceRewritePattern<UnitaryInterface>(ctx), env_(std::move(env)) {
  }
  LogicalResult matchAndRewrite(UnitaryInterface u,
                                PatternRewriter& rewriter) const final {
    // Global-phase or zero-qubit unitary: Nothing to do.
    if (u.getAllInQubits().empty()) {
      return failure();
    }

    // Single-qubit: Forward mapping
    if (!isTwoQubitGate(u)) {
      env().state.forward(u.getAllInQubits()[0], u.getAllOutQubits()[0]);
      return failure();
    }

    // Two-qubit: Ensure executable on hardware.
    if (!isExecutable(u)) {
      makeExecutable(u, rewriter);
    }

    // After ensuring executability, forward outputs.
    const auto& [execIn0, execIn1] = getIns(u);
    const auto& [execOut0, execOut1] = getOuts(u);

    env().state.forward(execIn0, execOut0);
    env().state.forward(execIn1, execOut1);

    return failure();
  }

private:
  /**
   * @brief Returns true iff @p u is executable on the targeted architecture.
   */
  [[nodiscard]] bool isExecutable(UnitaryInterface u) const {
    const auto& [in0, in1] = getIns(u);
    return env().arch->areAdjacent(env().state.get(in0), env().state.get(in1));
  }

  /**
   * @brief Get shortest path between @p in0 and @p in1.
   */
  [[nodiscard]] llvm::SmallVector<std::size_t> getPath(const Value in0,
                                                       const Value in1) const {
    return env().arch->shortestPathBetween(env().state.get(in0),
                                           env().state.get(in1));
  }

  /**
   * @brief Insert SWAPs such that @p u is executable.
   */
  void makeExecutable(UnitaryInterface u, PatternRewriter& rewriter) const {
    const auto& [q0, q1] = getIns(u);
    auto path = getPath(q0, q1);

    for (std::size_t i = 0; i < path.size() - 1; i += 2) {
      const Value in0 = env().state.get(path[i]);
      const Value in1 = env().state.get(path[i + 1]);

      // Y(out), X(out) = SWAP X(in), Y(in)
      auto swap = createSwap(u->getLoc(), in0, in1, rewriter);
      const auto& [out1, out0] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      rewriter.replaceAllUsesExcept(in0, out0, swap);
      rewriter.replaceAllUsesExcept(in1, out1, swap);

      env().state.forward(in0, out1);
      env().state.forward(in1, out0);
    }
  }

  [[nodiscard]] RoutingEnvironment& env() const { return *env_; }

  std::shared_ptr<RoutingEnvironment> env_;
};

struct MeasurePattern final : PatternWithEnv<MeasureOp> {
  using PatternWithEnv<MeasureOp>::PatternWithEnv;
  LogicalResult matchAndRewrite(MeasureOp m,
                                PatternRewriter& /*rewriter*/) const final {
    env().state.forward(m.getInQubit(), m.getOutQubit());
    return failure();
  }
};

struct DeallocQubitPattern final : PatternWithEnv<DeallocQubitOp> {
  using PatternWithEnv<DeallocQubitOp>::PatternWithEnv;
  LogicalResult matchAndRewrite(DeallocQubitOp dealloc,
                                PatternRewriter& rewriter) const final {
    env().state.release(dealloc.getQubit());
    rewriter.eraseOp(dealloc);
    return success();
  }
};

/**
 * @brief Collect patterns for the naive routing algorithm.
 */
void populateNaiveRoutingPatterns(RewritePatternSet& patterns,
                                  std::shared_ptr<RoutingEnvironment> env) {
  patterns.add<IfOpPattern, ForOpPattern>(patterns.getContext());
  patterns.add<AllocQubitPattern, ResetPattern, NaiveUnitaryPattern,
               MeasurePattern, DeallocQubitPattern>(patterns.getContext(), env);
}

void initEntryPoint(llvm::SmallVectorImpl<Value>& qubits,
                    PatternRewriter& rewriter) {
  const auto location = rewriter.getInsertionPoint()->getLoc();
  for (std::size_t i = 0; i < qubits.size(); ++i) {
    auto op = rewriter.create<QubitOp>(location, i);
    qubits[i] = op.getQubit();
  }
}

[[nodiscard]] LogicalResult route(ModuleOp module,
                                  std::unique_ptr<InitialLayout> layout,
                                  std::unique_ptr<Architecture> arch) {
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
    RewritePatternSet patterns(func.getContext());
    populateNaiveRoutingPatterns(
        patterns, std::make_shared<RoutingEnvironment>(qubits, *layout, *arch));

    walkAndApplyPatterns(func, std::move(patterns));

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
    auto arch = getArchitecture("MQT-Test");
    auto layout = std::make_unique<Random>(arch->nqubits());

    if (failed(route(getOperation(), std::move(layout), std::move(arch)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mqt::ir::opt
