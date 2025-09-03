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
#include <llvm/ADT/STLExtras.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <numeric>
#include <queue>
#include <random>
#include <utility>

#define DEBUG_TYPE "routing"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace transpilation;
using namespace mlir;

/// @brief A function attribute that specifies an (QIR) entry point function.
constexpr llvm::StringLiteral ATTRIBUTE_ENTRY_POINT{"entry_point"};

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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

class CircuitState {
public:
  explicit CircuitState(llvm::SmallVector<Value> staticQubits)
      : staticQubits_(std::move(staticQubits)) {};

  /// @brief Return static index from SSA value @p in.
  [[nodiscard]] std::size_t valueToStaticIndex(const Value in) const {
    const std::size_t dynNum = valueToDynamic_.at(in);
    return dynamicToStatic_.at(dynNum);
  }

  /// @brief Return SSA value from static index @p statIdx.
  [[nodiscard]] Value staticIndexToValue(const std::size_t statIdx) const {
    const std::size_t dynNum = staticToDynamic_.at(statIdx);
    return dynamicToValue_.at(dynNum);
  }

  /// @brief Expand circuit to have nqubits dynamic qubits.
  void expand(const InitialLayout& layout) {
    const std::size_t nqubits = staticQubits_.size();
    for (std::size_t i = 0; i < nqubits; ++i) {
      assign(i, layout);
    }
  }

  /// @brief Retrieve free static qubit.
  Value alloc() {
    Value qubit = free_.front();
    free_.pop();
    return qubit;
  }

  void forward(const Value in, const Value out) {
    const std::size_t dynNum = valueToDynamic_[in];
    const std::size_t statIdx = dynamicToStatic_[dynNum];

    valueToDynamic_[out] = dynNum;
    dynamicToValue_[dynNum] = out;

    staticToDynamic_[statIdx] = dynNum;
    dynamicToStatic_[dynNum] = statIdx;

    valueToDynamic_.erase(in);
  }

  /// @brief Release used qubit.
  void dealloc(Value in) { free_.emplace(in); }

  /// @brief Return the number of allocated qubits.
  [[nodiscard]] std::size_t nallocated() const {
    return staticQubits_.size() - free_.size();
  }

private:
  /// @brief Assign a permutated static qubit (and index) to @p dynNum.
  void assign(const std::size_t dynNum, const InitialLayout& layout) {
    const std::size_t statIdx = layout(dynNum);
    const Value staticQubit = staticQubits_[statIdx];

    valueToDynamic_[staticQubit] = dynNum;
    dynamicToValue_[dynNum] = staticQubit;

    staticToDynamic_[statIdx] = dynNum;
    dynamicToStatic_[dynNum] = statIdx;

    free_.emplace(staticQubit);
  }

  /// @brief Maps a SSA value to its dynamic number.
  llvm::DenseMap<Value, std::size_t> valueToDynamic_;
  /// @brief Maps a dynamic number to its SSA value.
  llvm::DenseMap<std::size_t, Value> dynamicToValue_;

  /// @brief Maps a static index to its dynamic number.
  llvm::DenseMap<std::size_t, std::size_t> staticToDynamic_;
  /// @brief Maps a dynamic number to its static index.
  llvm::DenseMap<std::size_t, std::size_t> dynamicToStatic_;

  /// @brief Vector of initialized static qubits.
  llvm::SmallVector<Value> staticQubits_;

  /// @brief Currently free to allocate qubits.
  std::queue<Value> free_;
};

class Router {
public:
  Router(std::unique_ptr<Architecture> arch,
         std::unique_ptr<InitialLayout> layout)
      : arch_(std::move(arch)), layout_(std::move(layout)) {}

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

  /// @brief Return initial layout mapping.
  [[nodiscard]] const InitialLayout& layout() const { return *layout_; }

private:
  std::unique_ptr<Architecture> arch_;
  std::unique_ptr<InitialLayout> layout_;
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

      if (failed(routeFunc(func, staticQubits, rewriter))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return res.wasInterrupted() ? failure() : success();
  }

private:
  /// @brief Returns true if @p u is executable on the targeted architecture.
  [[nodiscard]] bool isExecutable(UnitaryInterface u,
                                  const CircuitState& state) const {
    const auto& [in0, in1] = getIns(u);
    return arch().areAdjacent(state.valueToStaticIndex(in0),
                              state.valueToStaticIndex(in1));
  }

  /// @brief Get shortest path between @p in0 and @p in1.
  [[nodiscard]] llvm::SmallVector<std::size_t>
  getPath(const Value in0, const Value in1, const CircuitState& state) const {
    return arch().shortestPathBetween(state.valueToStaticIndex(in0),
                                      state.valueToStaticIndex(in1));
  }

  /// @brief Insert SWAPs such that @p u is executable.
  void makeExecutable(UnitaryInterface u, CircuitState& state,
                      PatternRewriter& rewriter) const {
    const auto& [in0, in1] = getIns(u);
    auto path = getPath(in0, in1, state);
    for (std::size_t i = 0; i < path.size() - 1; i += 2) {
      const Value in0 = state.staticIndexToValue(path[i]);
      const Value in1 = state.staticIndexToValue(path[i + 1]);

      auto swap = createSwap(u->getLoc(), in0, in1, rewriter);

      const auto& [swapOut0, swapOut1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      rewriter.replaceAllUsesExcept(in0, swapOut0, swap);
      rewriter.replaceAllUsesExcept(in1, swapOut1, swap);

      state.forward(in0, swapOut1);
      state.forward(in1, swapOut0);
    }
  }

  [[nodiscard]] LogicalResult
  routeFunc(func::FuncOp func, const llvm::SmallVector<Value>& staticQubits,
            PatternRewriter& rewriter) const {

    CircuitState state(staticQubits);
    state.expand(layout());

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
        if (loop.getRegionIterArgs().size() == 0) {
          return WalkResult::advance();
        }
        return WalkResult::skip();
      }

      if (auto alloc = dyn_cast<AllocQubitOp>(op)) {
        if (state.nallocated() == arch().nqubits()) {
          return WalkResult(func->emitOpError()
                            << "requires " << (state.nallocated() + 1)
                            << " qubits but target architecture '"
                            << arch().name() << "' only supports "
                            << arch().nqubits() << " qubits");
        }

        auto staticQubit = state.alloc();
        rewriter.replaceAllUsesWith(alloc.getQubit(), staticQubit);
        rewriter.eraseOp(alloc);
        return WalkResult::advance();
      }

      if (auto reset = dyn_cast<ResetOp>(op)) {
        state.forward(reset.getInQubit(), reset.getOutQubit());
        return WalkResult::advance();
      }

      if (auto u = dyn_cast<UnitaryInterface>(op)) {
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
        state.dealloc(dealloc.getQubit());
        rewriter.eraseOp(dealloc);
        return WalkResult::advance();
      }

      return WalkResult::advance();
    });

    if (res.wasInterrupted()) {
      return failure();
    }

    assert(state.nallocated() == 0);

    return success();
  }
};

} // namespace

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    auto arch = transpilation::getArchitecture("MQT-Test");
    auto layout = std::make_unique<Identity>(arch->nqubits());

    const NaiveRouter router(std::move(arch), std::move(layout));

    if (failed(router.route(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
