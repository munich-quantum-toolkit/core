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
#include <functional>
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
#include <numeric>
#include <queue>
#include <random>
#include <utility>

#define DEBUG_TYPE "routing"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace transpilation {
using namespace mlir;

/// @brief A function attribute that specifies an (QIR) entry point function.
constexpr llvm::StringLiteral ATTRIBUTE_ENTRY_POINT{"entry_point"};

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {
/// @brief Return true iff the qubit gate acts on two qubits.
[[nodiscard]] bool isTwoQubitGate(UnitaryInterface u) {
  return u.getAllInQubits().size() == 2;
}

/// @brief Return input qubit pair for two-qubit unitary @p u.
[[nodiscard]] std::pair<Value, Value> getIns(UnitaryInterface u) {
  assert(isTwoQubitGate(u));
  return {u.getAllInQubits()[0], u.getAllInQubits()[1]};
}

/// @brief Return output qubit pair for two-qubit unitary @p u.
[[nodiscard]] std::pair<Value, Value> getOuts(UnitaryInterface u) {
  assert(isTwoQubitGate(u));
  return {u.getAllOutQubits()[0], u.getAllOutQubits()[1]};
}
} // namespace

//===----------------------------------------------------------------------===//
// Initial Layouts
//===----------------------------------------------------------------------===//

/// @brief InitialLayout class for all initial layout mapping functions.
struct InitialLayout {
  explicit InitialLayout(const std::size_t nqubits) : mapping_(nqubits) {}
  [[nodiscard]] std::size_t operator()(std::size_t i) const {
    return mapping_[i];
  }

protected:
  llvm::SmallVector<std::size_t, 0> mapping_;
};

/// @brief Identity mapping.
struct Identity : InitialLayout {
  explicit Identity(const std::size_t nqubits) : InitialLayout(nqubits) {
    std::iota(mapping_.begin(), mapping_.end(), 0);
  }
};

/// @brief Random mapping.
struct Random : Identity {
  explicit Random(const std::size_t nqubits) : Identity(nqubits) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(mapping_.begin(), mapping_.end(), g);
  }
};

/// @brief Custom mapping.
struct Custom : InitialLayout {
  Custom(const std::size_t nqubits,
         const llvm::SmallVectorImpl<std::size_t>& mapping)
      : InitialLayout(nqubits) {
    std::ranges::copy(mapping, mapping_.begin());
  }
};

//===----------------------------------------------------------------------===//
// Routing
//===----------------------------------------------------------------------===//

class QubitStateManager {
public:
  /**
   * @brief Return empty circuit state.
   */
  static QubitStateManager empty() { return QubitStateManager(); }

  /**
   * @brief Initialize circuit state.
   * @details TODO
   * @param qubits A vector of static qubits.
   * @param layout The mapping from dynamic numbers to static indices.
   */
  explicit QubitStateManager(const llvm::SmallVector<Value>& qubits,
                             const InitialLayout& layout)
      : staticReg_(qubits.size()), layout_(qubits.size()),
        invLayout_(qubits.size()) {
    for (std::size_t dn = 0; dn < qubits.size(); ++dn) {
      const std::size_t i = layout(dn);
      const Value q = qubits[i];

      layout_[dn] = i;
      invLayout_[i] = dn;
      staticReg_[dn] = q;
    }
  }

  /**
   * @brief Return static index from SSA value @p v.
   */
  [[nodiscard]] std::size_t get(const Value v) const {
    return valueToIndex_.at(v);
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
    if (nalloc() == nqubits()) {
      return nullptr;
    }

    Value q;
    for (std::size_t dn = 0; dn < nqubits(); ++dn) {
      if (staticReg_[dn] != nullptr) {
        const std::size_t i = layout_[dn];

        q = staticReg_[dn];       // Retrieve free qubit.
        staticReg_[dn] = nullptr; // Set as used.

        valueToIndex_[q] = i;
        indexToValue_[i] = q;

        break;
      }
    }
    nalloc_++;
    return q;
  }

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
   * @brief Release static qubit.
   */
  void release(Value q) {
    const std::size_t i = get(q);
    const std::size_t dn = invLayout_[i];
    staticReg_[dn] = q;
    nalloc_--;
  }

  /**
   * @brief Return the number of static qubits.
   */
  [[nodiscard]] std::size_t nqubits() const { return staticReg_.size(); }

  /**
   * @brief Return the number of allocated qubits.
   */
  [[nodiscard]] std::size_t nalloc() const { return nalloc_; }

private:
  explicit QubitStateManager() = default;

  /// @brief Maps a SSA value to its static index.
  llvm::DenseMap<Value, std::size_t> valueToIndex_;

  /// @brief Maps a static index to its SSA value.
  llvm::DenseMap<std::size_t, Value> indexToValue_;

  /// @brief Vector / Register of static qubits.
  llvm::SmallVector<Value> staticReg_;

  /// @brief The number of allocated qubits.
  std::size_t nalloc_{};

  /// @brief Maps from dynamic number to static index.
  llvm::SmallVector<std::size_t, 0> layout_;

  /// @brief Maps from static index to dynamic number.
  llvm::SmallVector<std::size_t, 0> invLayout_;
};

class Router {
public:
  explicit Router(std::unique_ptr<Architecture> arch)
      : arch_(std::move(arch)) {}

  virtual ~Router() = default;

  /// @brief Use SWAP-based routing to fit to the target's coupling map.
  virtual LogicalResult route(ModuleOp) const = 0;

protected:
  /// @brief Swap @p in0 and @p in1 at @p location.
  [[nodiscard]] static SWAPOp createSwap(Location location, Value in0,
                                         Value in1, PatternRewriter& rewriter) {
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

  /// @brief Insert and return static qubits at current insertion point.
  [[nodiscard]] llvm::SmallVector<Value>
  initStaticQubits(PatternRewriter& rewriter) const {
    llvm::SmallVector<Value> staticQubits;
    staticQubits.reserve(arch().nqubits());

    for (std::size_t i = 0; i < arch().nqubits(); ++i) {
      const auto location = rewriter.getInsertionPoint()->getLoc();
      auto qubit = rewriter.create<QubitOp>(location, i);
      staticQubits.push_back(qubit);
    }

    return staticQubits;
  }

  /// @brief Return targeted architecture.
  [[nodiscard]] const Architecture& arch() const { return *arch_; }

private:
  std::unique_ptr<Architecture> arch_;
};

class NaiveRouter final : public Router {
public:
  using Router::Router;

  [[nodiscard]] LogicalResult route(ModuleOp module) const override {
    PatternRewriter rewriter(module->getContext());

    auto res = module->walk([&](func::FuncOp func) {
      if (!func->hasAttr(ATTRIBUTE_ENTRY_POINT)) {
        return WalkResult::skip(); // For now we don't route non-entry point
                                   // functions.
      }

      rewriter.setInsertionPointToStart(&func.getBody().front());
      auto staticQubits = initStaticQubits(rewriter);

      Custom layout(arch().nqubits(),
                    llvm::SmallVector<std::size_t>{2, 0, 4, 1, 3, 5});

      QubitStateManager state(staticQubits, layout);

      if (failed(forward(func, state, rewriter))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return res.wasInterrupted() ? failure() : success();
  }

private:
  /// @brief Returns true if @p u is executable on the targeted architecture.
  [[nodiscard]] bool isExecutable(UnitaryInterface u,
                                  const QubitStateManager& state) const {
    const auto& [in0, in1] = getIns(u);
    return arch().areAdjacent(state.get(in0), state.get(in1));
  }

  /// @brief Get shortest path between @p in0 and @p in1.
  [[nodiscard]] llvm::SmallVector<std::size_t>
  getPath(const Value in0, const Value in1,
          const QubitStateManager& state) const {
    return arch().shortestPathBetween(state.get(in0), state.get(in1));
  }

  /// @brief Insert SWAPs such that @p u is executable.
  void makeExecutable(UnitaryInterface u, QubitStateManager& state,
                      PatternRewriter& rewriter) const {
    const auto& [q0, q1] = getIns(u);
    auto path = getPath(q0, q1, state);

    for (std::size_t i = 0; i < path.size() - 1; i += 2) {
      const Value in0 = state.get(path[i]);
      const Value in1 = state.get(path[i + 1]);

      // Y(out), X(out) = SWAP X(in), Y(in)
      auto swap = createSwap(u->getLoc(), in0, in1, rewriter);
      const auto& [out1, out0] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      rewriter.replaceAllUsesExcept(in0, out0, swap);
      rewriter.replaceAllUsesExcept(in1, out1, swap);

      state.forward(in0, out1);
      state.forward(in1, out0);
    }
  }

  [[nodiscard]] LogicalResult forward(func::FuncOp func,
                                      QubitStateManager& state,
                                      PatternRewriter& rewriter) const {

    auto res = func->walk<WalkOrder::PreOrder>([&](Operation* op) {
      rewriter.setInsertionPoint(op);

      // Skip any initialized static qubits.
      if (auto qubit = dyn_cast<QubitOp>(op)) {
        return WalkResult::skip();
      }

      // As of now, we don't support conditionals. Hence, skip.
      if (auto cond = dyn_cast<scf::IfOp>(op)) {
        return WalkResult::skip();
      }

      // As of now, we don't support loops with qubit dependencies. Hence, skip.
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (loop.getRegionIterArgs().empty()) {
          return WalkResult::advance();
        }
        return WalkResult::skip();
      }

      if (auto alloc = dyn_cast<AllocQubitOp>(op)) {
        if (auto q = state.retrieve()) {
          auto reset = rewriter.create<ResetOp>(q.getLoc(), q);
          state.forward(reset.getInQubit(), reset.getOutQubit());

          rewriter.replaceAllUsesWith(alloc.getQubit(), reset);
          rewriter.eraseOp(alloc);
          return WalkResult::advance();
        }

        return WalkResult(func->emitOpError()
                          << "requires " << (state.nalloc() + 1)
                          << " qubits but target architecture '"
                          << arch().name() << "' only supports "
                          << arch().nqubits() << " qubits");

        return WalkResult::advance();
      }

      if (auto reset = dyn_cast<ResetOp>(op)) {
        state.forward(reset.getInQubit(), reset.getOutQubit());
        return WalkResult::advance();
      }

      if (auto u = dyn_cast<UnitaryInterface>(op)) {
        if (u.getAllInQubits().empty()) { // Handle Global Phase Gates.
          return WalkResult::advance();
        }

        if (!isTwoQubitGate(u)) {
          state.forward(u.getAllInQubits()[0], u.getAllOutQubits()[0]);
          return WalkResult::advance();
        }

        if (!isExecutable(u, state)) {
          makeExecutable(u, state, rewriter);
        }

        // Gate is (now) executable on hardware:

        const auto& [execIn0, execIn1] = getIns(u);
        const auto& [execOut0, execOut1] = getOuts(u);

        state.forward(execIn0, execOut0);
        state.forward(execIn1, execOut1);

        return WalkResult::advance();
      }

      if (auto measure = dyn_cast<MeasureOp>(op)) {
        state.forward(measure.getInQubit(), measure.getOutQubit());
        return WalkResult::advance();
      }

      if (auto dealloc = dyn_cast<DeallocQubitOp>(op)) {
        state.release(dealloc.getQubit());
        rewriter.eraseOp(dealloc);
        return WalkResult::advance();
      }

      return WalkResult::advance();
    });

    if (res.wasInterrupted()) {
      return failure();
    }

    assert(state.nalloc() == 0);

    return success();
  }
};

} // namespace transpilation

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    using namespace transpilation;

    auto arch = getArchitecture("MQT-Test");

    const NaiveRouter router(std::move(arch));

    if (failed(router.route(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
