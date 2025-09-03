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

/// @brief Maps qubits to indices.
using QubitIndexMap = llvm::DenseMap<Value, std::size_t>;

/// @brief Maps indices to qubits.
using IndexQubitMap = llvm::DenseMap<std::size_t, Value>;

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

namespace layout {

/// @brief Base class for all initial layout mapping functions.
struct Base {
  explicit Base(const std::size_t nqubits) : mapping_(nqubits) {}
  [[nodiscard]] std::size_t operator()(std::size_t i) const {
    return mapping_[i];
  }

protected:
  llvm::SmallVector<std::size_t, 0> mapping_;
};

/// @brief Identity mapping.
struct Identity : Base {
  explicit Identity(const std::size_t nqubits) : Base(nqubits) {
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
struct Custom : Base {
  Custom(const std::size_t nqubits,
         const llvm::SmallVectorImpl<std::size_t>& mapping)
      : Base(nqubits) {
    std::ranges::copy(mapping, mapping_.begin());
  }
};

}; // namespace layout

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

  /// @brief Currently allocated number of qubits.
  std::size_t allocated{};

  Value alloc(const layout::Base& layout) {
    const std::size_t staticIndex = layout(allocated);
    const Value staticQubit = staticQubits[staticIndex];

    valueToDynamic[staticQubit] = allocated;
    dynamicToValue[allocated] = staticQubit;

    staticToDynamic[staticIndex] = allocated;
    dynamicToStatic[allocated] = staticIndex;

    allocated++;

    return staticQubit;
  }
};

class Router {
public:
  virtual ~Router() = default;

  virtual LogicalResult route(ModuleOp, const Architecture&,
                              const layout::Base&, PatternRewriter&) const = 0;

protected:
  /**
   * @brief Insert and return static qubits at current insertion point.
   */
  [[nodiscard]] static llvm::SmallVector<Value>
  initStaticQubits(const Architecture& arch, PatternRewriter& rewriter) {
    llvm::SmallVector<Value> staticQubits;
    staticQubits.reserve(arch.nqubits());

    for (std::size_t i = 0; i < arch.nqubits(); ++i) {
      const auto location = rewriter.getInsertionPoint()->getLoc();
      auto qubit = rewriter.create<QubitOp>(location, i);
      staticQubits.push_back(qubit);
    }

    return staticQubits;
  }

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
};

class NaiveRouter final : public Router {
public:
  [[nodiscard]] LogicalResult route(ModuleOp module, const Architecture& arch,
                                    const layout::Base& layout,
                                    PatternRewriter& rewriter) const override {
    auto res = module->walk([&](func::FuncOp func) {
      if (!func->hasAttr(ATTRIBUTE_ENTRY_POINT)) {
        return WalkResult::skip(); // For now we don't route non-entry point
                                   // functions.
      }

      rewriter.setInsertionPointToStart(&func.getBody().front());
      auto staticQubits = initStaticQubits(arch, rewriter);

      if (failed(routeFunc(func, staticQubits, arch, layout, rewriter))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return res.wasInterrupted() ? failure() : success();
  }

private:
  [[nodiscard]] static LogicalResult
  routeFunc(func::FuncOp func, const llvm::SmallVector<Value>& staticQubits,
            const Architecture& arch, const layout::Base& layout,
            PatternRewriter& rewriter) {

    CircuitState state(staticQubits);

    auto res = func->walk([&](Operation* op) {
      rewriter.setInsertionPoint(op);

      if (auto qubit = dyn_cast<QubitOp>(op)) {
        return WalkResult::skip(); // Skip any initialized static qubits.
      }

      if (auto alloc = dyn_cast<AllocQubitOp>(op)) {
        if (state.allocated == arch.nqubits()) {
          return WalkResult(func->emitOpError()
                            << "requires " << (state.allocated + 1)
                            << " qubits but target architecture '"
                            << arch.name() << "' only supports "
                            << arch.nqubits() << " qubits");
        }

        auto staticQubit = state.alloc(layout);
        rewriter.replaceAllUsesWith(alloc.getQubit(), staticQubit);
        rewriter.eraseOp(alloc);

        return WalkResult::advance();
      }

      if (auto reset = dyn_cast<ResetOp>(op)) {
        return WalkResult::advance();
      }

      if (auto u = dyn_cast<UnitaryInterface>(op)) {
        if (!isTwoQubitGate(u)) {
        }
        return WalkResult::advance();
      }

      if (auto measure = dyn_cast<MeasureOp>(op)) {
        return WalkResult::advance();
      }

      if (auto dealloc = dyn_cast<DeallocQubitOp>(op)) {
        return WalkResult::advance();
      }

      return WalkResult::interrupt(); // Future proofing: fail on missing 'if'.
    });

    return res.wasInterrupted() ? failure() : success();
  }
};

} // namespace

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    const ModuleOp module = getOperation();
    PatternRewriter rewriter(module->getContext());

    // 0 -- 1
    // |    |
    // 2 -- 3
    // |    |
    // 4 -- 5
    const Architecture::CouplingMap couplingMap{
        {0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 3}, {3, 1}, {2, 3},
        {3, 2}, {2, 4}, {4, 2}, {3, 5}, {5, 3}, {4, 5}, {5, 4}};

    auto arch = std::make_unique<Architecture>("MQT-Test", 6, couplingMap);

    const llvm::SmallVector<std::size_t> mapping{4, 0, 2, 5, 3, 1};
    const NaiveRouter router;
    const layout::Custom layout(arch->nqubits(), mapping);

    auto res = router.route(module, *arch, layout, rewriter);
    if (res.failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
