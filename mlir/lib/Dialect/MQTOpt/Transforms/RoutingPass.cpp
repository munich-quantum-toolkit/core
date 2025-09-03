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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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
#include <string>
#include <string_view>
#include <utility>

#define DEBUG_TYPE "routing"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/// @brief A function attribute that specifies an (QIR) entry point function.
constexpr llvm::StringLiteral ATTRIBUTE_ENTRY_POINT{"entry_point"};

[[nodiscard]] bool isTwoQubitGate(UnitaryInterface unitary) {
  return unitary.getAllInQubits().size() == 2;
}

//===----------------------------------------------------------------------===//
// Architecture
//===----------------------------------------------------------------------===//

/// @brief A quantum accelerator's architecture.
class Architecture {
public:
  using CouplingMap = llvm::DenseSet<std::pair<std::size_t, std::size_t>>;

  explicit Architecture(std::string name, std::size_t nqubits,
                        CouplingMap couplingMap)
      : name_(std::move(name)), nqubits_(nqubits),
        couplingMap_(std::move(couplingMap)),
        dist_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)),
        prev_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)) {
    floydWarshallWithPathReconstruction();
  }

  /// @brief Return the architecture's name.
  [[nodiscard]] constexpr std::string_view name() const { return name_; }

  /// @brief Return the architecture's number of qubits.
  [[nodiscard]] constexpr std::size_t nqubits() const { return nqubits_; }

  /// @brief Return true if @p u and @p are adjacent.
  [[nodiscard]] bool areAdjacent(std::size_t u, std::size_t v) const {
    return couplingMap_.contains({u, v});
  }

  /// @brief Collect the shortest path between @p u and @p v.
  [[nodiscard]] llvm::SmallVector<std::size_t>
  shortestPathBetween(std::size_t u, std::size_t v) const {
    llvm::SmallVector<std::size_t> path;

    if (prev_[u][v] == UINT64_MAX) {
      return {};
    }

    path.push_back(v);
    while (u != v) {
      v = prev_[u][v];
      path.push_back(v);
    }

    return path;
  }

private:
  using Matrix = llvm::SmallVector<llvm::SmallVector<std::size_t>>;

  /**
   * @brief Find all shortest paths in the coupling map between two qubits.
   * @details Vertices are the qubits. Edges connected two qubits.
   * @link Adapted from https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
   */
  void floydWarshallWithPathReconstruction() {
    for (const auto& [u, v] : couplingMap_) {
      dist_[u][v] = 1;
      prev_[u][v] = u;
    }
    for (std::size_t v = 0; v < nqubits(); ++v) {
      dist_[v][v] = 0;
      prev_[v][v] = v;
    }

    for (std::size_t k = 0; k < nqubits(); ++k) {
      for (std::size_t i = 0; i < nqubits(); ++i) {
        for (std::size_t j = 0; j < nqubits(); ++j) {
          if (dist_[i][k] == UINT64_MAX || dist_[k][j] == UINT64_MAX) {
            continue; // avoid overflow with "infinite" distances
          }
          const std::size_t sum = dist_[i][k] + dist_[k][j];
          if (dist_[i][j] > sum) {
            dist_[i][j] = sum;
            prev_[i][j] = prev_[k][j];
          }
        }
      }
    }
  }

  std::string name_;
  std::size_t nqubits_;
  CouplingMap couplingMap_;

  Matrix dist_;
  Matrix prev_;
};

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

struct CircuitState {
  explicit CircuitState(llvm::SmallVector<Value> staticQubits)
      : staticQubits(std::move(staticQubits)) {};

  /// @brief Maps a SSA value to its dynamic number.
  llvm::DenseMap<Value, std::size_t> valueToDynamic;
  /// @brief Maps a dynamic number to its SSA value.
  llvm::DenseMap<std::size_t, Value> dynamicToValue;

  /// @brief Maps a static index to its dynamic number.
  llvm::DenseMap<std::size_t, std::size_t> staticToDynamic;
  /// @brief Maps a dynamic number to its static index.
  llvm::DenseMap<std::size_t, std::size_t> dynamicToStatic;

  /// @brief Vector of initialized static qubits.
  llvm::SmallVector<Value> staticQubits;

  /// @brief Currently free to allocate qubits.
  std::queue<Value> free;

  /// @brief Expand circuit to have nqubits dynamic qubits.
  void expand(const InitialLayout& layout) {
    const std::size_t nqubits = staticQubits.size();
    for (std::size_t i = 0; i < nqubits; ++i) {
      assign(i, layout);
    }
  }

  Value alloc() {
    Value qubit = free.front();
    free.pop();
    return qubit;
  }

  void forward(const Value in, const Value out) {
    const std::size_t dynamicNumber = valueToDynamic[in];
    const std::size_t staticIndex = dynamicToStatic[dynamicNumber];

    valueToDynamic[out] = dynamicNumber;
    dynamicToValue[dynamicNumber] = out;

    staticToDynamic[staticIndex] = dynamicNumber;
    dynamicToStatic[dynamicNumber] = staticIndex;

    valueToDynamic.erase(in);
  }

  void dealloc(Value in) { free.emplace(in); }

  /// @brief Return the number of allocated qubits.
  [[nodiscard]] std::size_t nallocated() const {
    return staticQubits.size() - free.size();
  }

  [[nodiscard]] bool empty() const {
    return valueToDynamic.empty() && dynamicToValue.empty() &&
           staticToDynamic.empty() && dynamicToStatic.empty();
  }

private:
  /// @brief Assign a permutated static qubit (and index) to @p dynNum.
  void assign(const std::size_t dynNum, const InitialLayout& layout) {
    const std::size_t statIdx = layout(dynNum);
    const Value staticQubit = staticQubits[statIdx];

    valueToDynamic[staticQubit] = dynNum;
    dynamicToValue[dynNum] = staticQubit;

    staticToDynamic[statIdx] = dynNum;
    dynamicToStatic[dynNum] = statIdx;

    free.emplace(staticQubit);
  }
};

class Router {
public:
  Router(std::unique_ptr<Architecture> arch,
         std::unique_ptr<InitialLayout> layout)
      : arch_(std::move(arch)), layout_(std::move(layout)) {}

  virtual ~Router() = default;

  virtual LogicalResult route(ModuleOp) const = 0;

protected:
  [[nodiscard]] static SWAPOp createSwap(Location location, Value qubitA,
                                         Value qubitB,
                                         PatternRewriter& rewriter) {
    const SmallVector<Type> resultTypes{qubitA.getType(), qubitB.getType()};
    const SmallVector<Value> inQubits{qubitA, qubitB};

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
   * @brief Insert and return static qubits at current insertion point.
   */
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

  /**
   * @brief Return targeted architecture.
   */
  [[nodiscard]] const Architecture& arch() const { return *arch_; }

  /**
   * @brief Return initial layout mapping.
   */
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
  [[nodiscard]] bool isExecutable(UnitaryInterface u,
                                  const CircuitState& state) const {
    const Value in0 = u.getAllInQubits()[0];
    const Value in1 = u.getAllInQubits()[1];

    const std::size_t dynNum0 = state.valueToDynamic.at(in0);
    const std::size_t dynNum1 = state.valueToDynamic.at(in1);
    const std::size_t statIdx0 = state.dynamicToStatic.at(dynNum0);
    const std::size_t statIdx1 = state.dynamicToStatic.at(dynNum1);

    return arch().areAdjacent(statIdx0, statIdx1);
  }

  void makeExecutable(UnitaryInterface u, CircuitState& state,
                      PatternRewriter& rewriter) const {
    const Value in0 = u.getAllInQubits()[0];
    const Value in1 = u.getAllInQubits()[1];

    const std::size_t dynNum0 = state.valueToDynamic.at(in0);
    const std::size_t dynNum1 = state.valueToDynamic.at(in1);
    const std::size_t statIdx0 = state.dynamicToStatic.at(dynNum0);
    const std::size_t statIdx1 = state.dynamicToStatic.at(dynNum1);

    auto path = arch().shortestPathBetween(statIdx0, statIdx1);
    for (std::size_t i = 0; i < path.size() - 1; i += 2) {
      const std::size_t pathStatIdx0 = path[i];
      const std::size_t pathStatIdx1 = path[i + 1];

      const std::size_t pathDynNum0 = state.staticToDynamic.at(pathStatIdx0);
      const std::size_t pathDynNum1 = state.staticToDynamic.at(pathStatIdx1);

      const Value pathIn0 = state.dynamicToValue.at(pathDynNum0);
      const Value pathIn1 = state.dynamicToValue.at(pathDynNum1);

      auto swap = createSwap(u->getLoc(), pathIn0, pathIn1, rewriter);

      const Value swapOut0 = swap.getOutQubits()[0];
      const Value swapOut1 = swap.getOutQubits()[1];

      rewriter.setInsertionPointAfter(swap);
      rewriter.replaceAllUsesExcept(pathIn0, swapOut0, swap);
      rewriter.replaceAllUsesExcept(pathIn1, swapOut1, swap);

      // Update permutation maps.
      state.forward(pathIn0, swapOut0);
      state.forward(pathIn1, swapOut1);
    }
  }

  [[nodiscard]] LogicalResult
  routeFunc(func::FuncOp func, const llvm::SmallVector<Value>& staticQubits,
            PatternRewriter& rewriter) const {

    CircuitState state(staticQubits);
    state.expand(layout());

    auto res = func->walk([&](Operation* op) {
      rewriter.setInsertionPoint(op);

      if (auto qubit = dyn_cast<QubitOp>(op)) {
        return WalkResult::skip(); // Skip any initialized static qubits.
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

        const Value execIn0 = u.getAllInQubits()[0];
        const Value execIn1 = u.getAllInQubits()[1];
        const Value execOut0 = u.getAllOutQubits()[0];
        const Value execOut1 = u.getAllOutQubits()[1];

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

    // 0 -- 1
    // |    |
    // 2 -- 3
    // |    |
    // 4 -- 5

    const Architecture::CouplingMap couplingMap{
        {0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 3}, {3, 1}, {2, 3},
        {3, 2}, {2, 4}, {4, 2}, {3, 5}, {5, 3}, {4, 5}, {5, 4}};

    auto arch = std::make_unique<Architecture>("MQT-Test", 6, couplingMap);
    auto layout = std::make_unique<Custom>(
        arch->nqubits(), llvm::SmallVector<std::size_t>{4, 0, 2, 5, 3, 1});

    const NaiveRouter router(std::move(arch), std::move(layout));

    if (failed(router.route(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
