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
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
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
  explicit InitialLayout(const std::size_t nqubits) : mapping(nqubits) {}

  /// @brief Apply initial layout and replace dynamic qubits with permutated
  /// static qubits.
  [[nodiscard]] QubitIndexMap apply(const ArrayRef<Value> staticQubits,
                                    const ArrayRef<Value>& dynamicQubits,
                                    PatternRewriter& rewriter) const {
    QubitIndexMap map;
    for (std::size_t i = 0; i < staticQubits.size(); ++i) {
      map.try_emplace(staticQubits[i], i);

      if (mapping[i] == i) {
        rewriter.replaceAllUsesWith(dynamicQubits[i], staticQubits[i]);
        rewriter.eraseOp(dynamicQubits[i].getDefiningOp());
      }
    }
    return map;
  }

protected:
  llvm::SmallVector<std::size_t, 0> mapping;
};

/// @brief Identity mapping.
struct Identity : InitialLayout {
  explicit Identity(const std::size_t nqubits) : InitialLayout(nqubits) {
    std::iota(mapping.begin(), mapping.end(), 0);
  }
};

/// @brief Random mapping.
struct Random : Identity {
  explicit Random(const std::size_t nqubits) : Identity(nqubits) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(mapping.begin(), mapping.end(), g);
  }
};

/// @brief A qubit-pair defines any two-qubit interaction.
struct QubitPair {
  explicit(false) QubitPair(UnitaryInterface unitary) {
    if (unitary.getAllCtrlInQubits().empty()) {
      assert(unitary.getAllInQubits().size() == 2);
      in = std::make_pair(unitary.getAllInQubits().front(),
                          unitary.getAllInQubits().back());
      out = std::make_pair(unitary.getAllOutQubits().front(),
                           unitary.getAllOutQubits().back());
      return;
    }

    assert(unitary.getInQubits().size() == 1);
    assert(unitary.getAllCtrlInQubits().size() == 1);

    in = std::make_pair(unitary.getAllInQubits().front(),
                        unitary.getAllCtrlInQubits().front());
    out = std::make_pair(unitary.getAllOutQubits().front(),
                         unitary.getAllCtrlOutQubits().front());
  }

  /// @brief Apply unitary to @p map.
  void forward(QubitIndexMap& map) const {
    map[out.first] = map[in.first];
    map[out.second] = map[in.second];
    map.erase(in.first);
    map.erase(in.second);
  }

  std::pair<Value, Value> in;
  std::pair<Value, Value> out;
};

//===----------------------------------------------------------------------===//
// Router
//===----------------------------------------------------------------------===//

/// @brief A layer consists of a qubit mapping and a set of 2Q gates.
class Layer {
public:
  explicit Layer(QubitIndexMap map) : map(std::move(map)) {
    absorbOneQubitInteractions();
    collectTwoQubitInteractions();
  }

  /// @brief The mapping from qubits to indices.
  QubitIndexMap map;

  /// @brief The mapping from indices to qubits.
  IndexQubitMap invMap;

  /// @brief Vector of two-qubit gates (e.g. CNOTs).
  llvm::SmallVector<Operation*> gates;

  /// @brief Return iterator of UnitaryInterfaces.
  [[nodiscard]] auto unitaries() const {
    return llvm::map_range(
        gates, [](Operation* op) { return dyn_cast<UnitaryInterface>(op); });
  }

  /// @brief Return true if the layer contains two-qubit gates.
  [[nodiscard]] bool hasRoutableGates() const { return !gates.empty(); }

private:
  /// @brief Absorb single qubit interactions for @p q.
  static Value forward(const Value q) {
    Value newQ = q;
    while (!newQ.getUsers().empty()) {
      Operation* op = *newQ.getUsers().begin();
      if (auto reset = dyn_cast<ResetOp>(op)) {
        newQ = reset.getOutQubit();
      } else if (auto unitary = dyn_cast<UnitaryInterface>(op)) {
        if (!unitary.getAllCtrlInQubits().empty()) {
          break;
        }

        if (!dyn_cast<GPhaseOp>(op)) {
          newQ = unitary.getOutQubits().front();
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

  /// @brief Absorb single qubit interactions for all qubits in map and
  /// initialize invMap.
  void absorbOneQubitInteractions() {
    QubitIndexMap forwardedMap{};

    for (const auto& [q, i] : map) {
      Value newQ = Layer::forward(q);
      forwardedMap[newQ] = i;
      invMap[i] = newQ;
    }

    map = std::move(forwardedMap);
  }

  /// @brief Collect two-qubit gates. Visits each qubit once.
  void collectTwoQubitInteractions() {
    llvm::DenseSet<Value> visited;
    for (const auto& entry : map) {
      Value qubit = entry.getFirst();
      if (visited.contains(qubit)) {
        continue;
      }

      if (qubit.getUsers().empty()) {
        visited.insert(qubit);
        continue;
      }

      Operation* user = *(qubit.getUsers().begin());

      auto unitary = dyn_cast<UnitaryInterface>(user);
      if (!unitary) {
        visited.insert(qubit);
        continue;
      }

      // All single qubit operations should have been absorbed. What's
      // left are two qubit interaction gates.

      auto target = unitary.getInQubits().front();
      auto control = unitary.getAllCtrlInQubits().front();
      if (map.contains(target) && map.contains(control)) {
        gates.push_back(unitary);
      }

      visited.insert(target);
      visited.insert(control);
    }
  }
};

[[maybe_unused]] llvm::SmallVector<Layer>
lookahead(const QubitIndexMap& front, const std::size_t depth = 0) {
  const std::size_t nlayers = 1 + depth;

  llvm::SmallVector<Layer> layers;
  layers.reserve(nlayers);
  layers.emplace_back(front);

  if (depth == 0) {
    return layers;
  }

  for (std::size_t i = 0; i < depth; ++i) {
    Layer& prev = layers.back();
    QubitIndexMap map(prev.map); // Copy previous.

    for (const QubitPair unitary : prev.unitaries()) {
      unitary.forward(map);
    }

    layers.emplace_back(std::move(map));
  }

  return layers;
}

class Router {
public:
  virtual ~Router() = default;

  [[nodiscard]] virtual QubitIndexMap
  route(const QubitIndexMap& front, const Architecture& arch,
        PatternRewriter& rewriter) const = 0;

protected:
  [[nodiscard]] static SWAPOp createSwap(Value qubitA, Value qubitB,
                                         Location location,
                                         PatternRewriter& rewriter) {
    ArrayRef<Type> resultTypes{rewriter.getType<QubitType>(),
                               rewriter.getType<QubitType>()};
    ArrayRef<Value> inQubits{qubitA, qubitB};

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
  [[nodiscard]] QubitIndexMap route(const QubitIndexMap& map,
                                    const Architecture& arch,
                                    PatternRewriter& rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);

    Layer curr(map);
    while (!curr.hasRoutableGates()) {

      for (UnitaryInterface unitary : curr.unitaries()) {
        rewriter.setInsertionPoint(unitary);

        std::size_t indexA = curr.map.at(unitary.getInQubits().front());
        std::size_t indexB = curr.map.at(unitary.getAllCtrlInQubits().front());

        // If the qubits are not adjacent swap along the shortest path:
        if (!arch.areAdjacent(indexA, indexB)) {
          llvm::SmallVector<std::size_t> path =
              arch.shortestPathBetween(indexA, indexB);

          for (std::size_t i = 0; i < path.size() - 1; i += 2) {
            Value qubitA = curr.invMap[path[i]];
            Value qubitB = curr.invMap[path[i + 1]];

            auto swap = createSwap(qubitA, qubitB, unitary->getLoc(), rewriter);

            Value swappedA = swap.getOutQubits()[0];
            Value swappedB = swap.getOutQubits()[1];

            rewriter.setInsertionPointAfter(swap);
            rewriter.replaceAllUsesExcept(qubitA, swappedA, swap);
            rewriter.replaceAllUsesExcept(qubitB, swappedB, swap);

            // Update permutation maps.

            curr.invMap[path[i]] = swappedA;
            curr.invMap[path[i + 1]] = swappedB;

            curr.map[swappedA] = path[i];
            curr.map[swappedB] = path[i + 1];

            curr.map.erase(qubitA);
            curr.map.erase(qubitB);
          }
        }

        Value qubitInA = unitary.getInQubits().front();
        Value qubitInB = unitary.getAllCtrlInQubits().front();
        Value qubitOutA = unitary.getOutQubits().front();
        Value qubitOutB = unitary.getAllCtrlOutQubits().front();

        curr.invMap[curr.map[qubitInA]] = qubitOutA;
        curr.invMap[curr.map[qubitInB]] = qubitOutB;

        curr.map[qubitOutA] = curr.map[qubitInA];
        curr.map[qubitOutB] = curr.map[qubitInB];

        curr.map.erase(qubitInA);
        curr.map.erase(qubitInB);
      }

      curr = Layer(curr.map);
    }

    return curr.map;
  }
};

/**
 * @brief Walk the IR and collect the dynamic qubits for each circuit.
 * @details Assumes that there are no allocations after a deallocation.
 * @return Vector of vectors, where each inner vector contains the dynamic
 * qubits allocated for a circuit.
 */
llvm::SmallVector<llvm::SmallVector<Value>, 2>
collectCircuits(Region& body, const Architecture& arch) {
  llvm::SmallVector<llvm::SmallVector<Value>, 2> circuits;
  llvm::SmallVector<Value> dynamicQubits;
  std::size_t allocated = 0;

  // A valid program has at most arch.nqubits qubits.
  dynamicQubits.reserve(arch.nqubits());

  body.walk([&](Operation* op) {
    if (auto alloc = dyn_cast<AllocQubitOp>(op)) {
      ++allocated;
      dynamicQubits.push_back(alloc.getQubit());
      return;
    }

    if (auto dealloc = dyn_cast<DeallocQubitOp>(op)) {
      if ((--allocated) == 0) {
        circuits.push_back(std::move(dynamicQubits));
        dynamicQubits.clear();
      }
      return;
    }
  });

  return circuits;
}

[[nodiscard]] LogicalResult
routeQuantumCircuit(llvm::ArrayRef<Value> staticQubits,
                    llvm::ArrayRef<Value> dynamicQubits,
                    const Architecture& arch, const InitialLayout& layout,
                    const Router& router, PatternRewriter& rewriter) {
  if (dynamicQubits.size() > staticQubits.size()) {
    return dynamicQubits.back().getDefiningOp()->emitError()
           << "program requires " << dynamicQubits.size()
           << " qubits but target architecture '" << arch.name()
           << "' only supports " << arch.nqubits() << " qubits";
  }

  QubitIndexMap front;
  front = layout.apply(staticQubits, dynamicQubits, rewriter);
  front = router.route(front, arch, rewriter);

  // Update static qubits with their current SSA value for the next
  // loop iteration.

  // for (auto& [q, i] : front) {
  //   staticQubits[i] = q;

  //   if (q.hasOneUse()) { // There may be unused qubits.
  //     DeallocQubitOp dealloc =
  //         dyn_cast<DeallocQubitOp>(*q.getUsers().begin());
  //     assert(dealloc);
  //     rewriter.eraseOp(dealloc);
  //   }
  // }

  return success();
}

/**
 * @brief Route entry point function @p func.
 * @details If the function is marked as entry point, create static qubits
 * that can later be passed to other quantum functions.
 */
[[nodiscard]] LogicalResult routeFunc(func::FuncOp func,
                                      const Architecture& arch,
                                      const InitialLayout& layout,
                                      const Router& router,
                                      PatternRewriter& rewriter) {
  Region& body = func.getBody();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&body.front());

  llvm::SmallVector<Value> staticQubits(arch.nqubits());

  if (!func->hasAttr(ATTRIBUTE_ENTRY_POINT)) {
    return success(); // For now, we don't route non-entry point functions.
  }

  for (std::size_t i = 0; i < arch.nqubits(); ++i) {
    auto qubit = rewriter.create<QubitOp>(body.getLoc(), i);
    staticQubits[i] = qubit.getQubit();
  }

  for (const auto& dynamicQubits : collectCircuits(body, arch)) {
    if (failed(routeQuantumCircuit(staticQubits, dynamicQubits, arch, layout,
                                   router, rewriter))) {
      return failure();
    }
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
                                        const Router& router,
                                        PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module);

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (failed(routeFunc(func, arch, layout, router, rewriter))) {
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
    ModuleOp module = getOperation();
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

    const Identity layout(arch->nqubits());
    const NaiveRouter router{};

    if (failed(routeModule(module, *arch, layout, router, rewriter))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
