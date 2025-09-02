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

void forwardMaps(QubitIndexMap& map, IndexQubitMap& invMap, const Value in,
                 const Value out) {
  map[out] = map[in];
  invMap[map[in]] = out;
  map.erase(in);
}

struct Permutation {
  explicit Permutation(const std::size_t nqubits) : staticQubits(nqubits) {}

  /// @brief A vector of static qubits
  llvm::SmallVector<Value, 0> staticQubits;
  /// @brief A mapping from dynamic numbers to their static index.
  llvm::DenseMap<std::size_t, std::size_t> layout;
  /// @brief A mapping from current SSA values to their assigned dynamic number.
  llvm::DenseMap<Value, std::size_t> dynamic;
};

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

/// @brief Base class for all initial layout mapping functions.
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
// Layering with (lookahead)
//===----------------------------------------------------------------------===//

struct Layer {
  /// @brief Vector of two-qubit gates (e.g. CNOTs).
  llvm::SmallVector<Operation*> gates;
};

/// @brief Absorb all one qubit gates on the def-use chain of @p q.
Value absorbOneQubitGates(const Value q) {
  Value newQ = q;
  while (!newQ.getUsers().empty()) {
    Operation* op = *newQ.getUsers().begin();
    if (auto reset = dyn_cast<ResetOp>(op)) {
      newQ = reset.getOutQubit();
    } else if (auto u = dyn_cast<UnitaryInterface>(op)) {
      if (isTwoQubitGate(u)) {
        break;
      }

      if (!dyn_cast<GPhaseOp>(op)) {
        newQ = u.getOutQubits().front();
      }
    } else if (auto measure = dyn_cast<MeasureOp>(op)) {
      newQ = measure.getOutQubit();
    } else { // DeallocQubitOp.
      assert(dyn_cast<DeallocQubitOp>(op));
      break;
    }
  }
  return newQ;
}

/// @brief Collect all "ready" two-qubit gates from @p map.
llvm::SmallVector<Operation*> collectTwoQubitGates(const QubitIndexMap& map) {
  llvm::DenseSet<Value> visited;
  llvm::SmallVector<Operation*> gates;

  for (const auto& [q, _] : map) {
    if (visited.contains(q)) {
      continue;
    }

    if (q.getUsers().empty()) {
      visited.insert(q);
      continue;
    }

    Operation* user = *(q.getUsers().begin());

    auto u = dyn_cast<UnitaryInterface>(user);
    if (!u) {
      visited.insert(q);
      continue;
    }

    // All single qubit operations should have been absorbed. What's
    // left are two qubit interaction gates.

    assert(isTwoQubitGate(u));

    const Value first = u.getAllInQubits()[0];
    const Value second = u.getAllInQubits()[1];

    if (map.contains(first) && map.contains(second)) {
      gates.push_back(u);
    }

    visited.insert(first);
    visited.insert(second);
  }

  return gates;
}

Layer findLayer(const QubitIndexMap& map) {
  QubitIndexMap absorbedMap;
  for (const auto& [q, i] : map) {
    absorbedMap[absorbOneQubitGates(q)] = i;
  }
  return Layer{.gates = collectTwoQubitGates(absorbedMap)};
}

//===----------------------------------------------------------------------===//
// Router
//===----------------------------------------------------------------------===//

class Router {
public:
  virtual ~Router() = default;

  [[nodiscard]] virtual llvm::DenseSet<Operation*>
  route(Location location, const Architecture& arch, QubitIndexMap& map,
        IndexQubitMap& invMap, PatternRewriter& rewriter) const = 0;

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
};

/// @brief Naively routes one two-qubit gate after the other.
class NaiveRouter final : public Router {
public:
  [[nodiscard]] llvm::DenseSet<Operation*>
  route(Location location, const Architecture& arch, QubitIndexMap& map,
        IndexQubitMap& invMap, PatternRewriter& rewriter) const override {

    QubitIndexMap copyMap(map);
    IndexQubitMap copyInvMap(invMap);
    Layer layer = findLayer(copyMap);

    llvm::DenseSet<Operation*> routed;
    for (Operation* op : layer.gates) {
      UnitaryInterface u = dyn_cast<UnitaryInterface>(op);

      const Value first = u.getAllInQubits()[0];
      const Value second = u.getAllInQubits()[1];
      const std::size_t idxFirst = copyMap.at(first);
      const std::size_t idxSecond = copyMap.at(second);

      // If the qubits are not adjacent swap along the shortest path.

      if (!arch.areAdjacent(idxFirst, idxSecond)) {
        const auto path = arch.shortestPathBetween(idxFirst, idxSecond);

        for (std::size_t i = 0; i < path.size() - 1; i += 2) {
          const Value inFirst = copyInvMap.at(path[i]);
          const Value inSecond = copyInvMap.at(path[i + 1]);

          auto swap = createSwap(location, inFirst, inSecond, rewriter);

          const Value outFirst = swap.getOutQubits()[0];
          const Value outSecond = swap.getOutQubits()[1];

          rewriter.setInsertionPointAfter(swap); // Move rewriter ahead.
          rewriter.replaceAllUsesExcept(inFirst, outFirst, swap);
          rewriter.replaceAllUsesExcept(inSecond, outSecond, swap);

          forwardMaps(copyMap, copyInvMap, inFirst, outFirst);
          forwardMaps(copyMap, copyInvMap, inSecond, outSecond);
        }
      }

      forwardMaps(copyMap, copyInvMap, u.getAllInQubits()[0],
                  u.getAllOutQubits()[0]);
      forwardMaps(copyMap, copyInvMap, u.getAllInQubits()[1],
                  u.getAllOutQubits()[1]);

      routed.insert(u);
    }

    return routed;
  }
};

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Route entry point function @p func.
 * @details If the function is marked as entry point, create static qubits
 * that can later be passed to other quantum functions.
 */
[[nodiscard]] LogicalResult routeFunc(func::FuncOp func,
                                      const Architecture& arch,
                                      const InitialLayout& layout,
                                      PatternRewriter& rewriter) {
  Region& body = func.getBody();
  rewriter.setInsertionPointToStart(&body.front());

  if (!func->hasAttr(ATTRIBUTE_ENTRY_POINT)) {
    return success(); // For now, we don't route non-entry point functions.
  }

  Permutation p(arch.nqubits());

  llvm::SmallVector<Value> staticQubits(arch.nqubits());
  for (std::size_t i = 0; i < arch.nqubits(); ++i) {
    auto qubit = rewriter.create<QubitOp>(body.getLoc(), i);
    staticQubits[i] = qubit.getQubit();
    p.staticQubits[i] = qubit.getQubit();
  }

  QubitIndexMap map;
  IndexQubitMap invMap;

  for (std::size_t i = 0; i < staticQubits.size(); ++i) {
    map.try_emplace(staticQubits[i], i);
    invMap.try_emplace(i, staticQubits[i]);
  }

  std::size_t allocated = 0;
  llvm::DenseSet<Operation*> routed;

  auto result = body.walk([&](Operation* op) {
    rewriter.setInsertionPoint(op);

    if (auto alloc = dyn_cast<AllocQubitOp>(op)) {

      
      // Replace dynamic with reseted static qubit.
      
      const Value dynamicQubit = alloc.getQubit();
      const Value staticQubit = invMap[layout(allocated)];
      
      p.dynamic[dynamicQubit] = allocated;
      p.layout[allocated] = layout(allocated);

      auto reset = rewriter.create<ResetOp>(alloc->getLoc(), staticQubit);
      rewriter.replaceAllUsesWith(dynamicQubit, reset.getOutQubit());

      // Update front map.
      forwardMaps(map, invMap, staticQubit, reset.getOutQubit());

      // Increase allocated count and validate for architecture.

      if ((++allocated) > arch.nqubits()) {
        return WalkResult(func->emitOpError()
                          << "requires " << allocated
                          << " qubits but target architecture '" << arch.name()
                          << "' only supports " << arch.nqubits() << " qubits");
      }

      // Finally, delete alloc operation.

      rewriter.eraseOp(alloc);

      return WalkResult::advance();
    }

    if (auto reset = dyn_cast<ResetOp>(op)) {
      forwardMaps(map, invMap, reset.getInQubit(), reset.getOutQubit());
      return WalkResult::advance();
    }

    if (auto u = dyn_cast<UnitaryInterface>(op)) {
      if (!isTwoQubitGate(u)) {
        forwardMaps(map, invMap, u.getInQubits().front(),
                    u.getOutQubits().front());

        return WalkResult::advance();
      }

      // The first unrouted two-qubit gate we encounter is
      // the first two-qubit gate of a new layer. Hence,
      // compute the layer's gates using the def-use chain.
      // To keep valid SSA value chains, the initial qubit
      // map must be the one before the first two-qubit gate.

      if (!routed.contains(u)) {
      }

      forwardMaps(map, invMap, u.getAllInQubits()[0], u.getAllOutQubits()[0]);
      forwardMaps(map, invMap, u.getAllInQubits()[1], u.getAllOutQubits()[1]);

      return WalkResult::advance();
    }

    if (auto measure = dyn_cast<MeasureOp>(op)) {
      forwardMaps(map, invMap, measure.getInQubit(), measure.getOutQubit());
      return WalkResult::advance();
    }

    if (auto dealloc = dyn_cast<DeallocQubitOp>(op)) {
      // Decrease allocation count.
      --allocated;

      // Then, erase dealloc operation.
      rewriter.eraseOp(op);

      return WalkResult::advance();
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }

  return success();
}

/**
 * @brief Collect functions of a module and route each.
 * @details Assume that there are no nested functions.
 */
[[nodiscard]] LogicalResult routeModule(ModuleOp module,
                                        const Architecture& arch,
                                        const InitialLayout& layout,
                                        PatternRewriter& rewriter) {
  rewriter.setInsertionPoint(module);
  for (const func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (failed(routeFunc(func, arch, layout, rewriter))) {
      return failure();
    }
  }
  return success();
}

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
    const auto arch =
        std::make_unique<Architecture>("MQT-Test", 6, couplingMap);

    const llvm::SmallVector<std::size_t> mapping{4, 0, 2, 5, 3, 1};
    const Custom layout(arch->nqubits(), mapping);

    if (failed(routeModule(module, *arch, layout, rewriter))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
