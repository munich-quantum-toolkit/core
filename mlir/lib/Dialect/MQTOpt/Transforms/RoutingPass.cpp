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
using QubitMap = llvm::DenseMap<Value, uint64_t>;
/// @brief A qubit-qubit pair.
using TwoQubitGate = std::pair<Value, Value>;

constexpr std::string ATTRIBUTE_TARGET = "mqtopt.target";
constexpr std::string ATTRIBUTE_ENTRY_POINT = "entry_point";

/// @brief A quantum accelerator's architecture.
class Architecture {
public:
  using Edge = std::pair<uint64_t, uint64_t>;
  using CouplingMap = llvm::DenseSet<Edge>;
  using Matrix = llvm::SmallVector<llvm::SmallVector<uint64_t>>;

  explicit Architecture(std::string name, uint64_t nqubits,
                        CouplingMap couplingMap)
      : name_(std::move(name)), nqubits_(nqubits),
        couplingMap_(std::move(couplingMap)),
        dist(nqubits, llvm::SmallVector<uint64_t>(nqubits, UINT64_MAX)),
        prev(nqubits, llvm::SmallVector<uint64_t>(nqubits, UINT64_MAX)) {
    floydWarshallWithPathReconstruction();
  }

  /// @brief Return the architecture's name.
  [[nodiscard]] constexpr std::string_view name() const { return name_; }
  /// @brief Return the architecture's number of qubits.
  [[nodiscard]] constexpr uint64_t nqubits() const { return nqubits_; }
  /// @brief Return a const reference to the coupling map.
  [[nodiscard]] const CouplingMap& couplingMap() const { return couplingMap_; }

  [[nodiscard]] llvm::SmallVector<uint64_t> shortestPath(uint64_t u,
                                                         uint64_t v) const {
    llvm::SmallVector<uint64_t> path;

    if (prev[u][v] == UINT64_MAX) {
      return {};
    }

    path.push_back(v);
    while (u != v) {
      v = prev[u][v];
      path.push_back(v);
    }

    return path;
  }

private:
  /**
   * @brief Find all shortest paths in the coupling map between two qubits.
   * @details Vertices are the qubits. Edges connected two qubits.
   * @link Adapted from https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
   */
  void floydWarshallWithPathReconstruction() {
    for (const auto& [u, v] : couplingMap()) {
      dist[u][v] = 1;
      prev[u][v] = u;
    }
    for (uint64_t v = 0; v < nqubits(); ++v) {
      dist[v][v] = 0;
      prev[v][v] = v;
    }

    for (uint64_t k = 0; k < nqubits(); ++k) {
      for (uint64_t i = 0; i < nqubits(); ++i) {
        for (uint64_t j = 0; j < nqubits(); ++j) {
          const uint64_t sum = dist[i][k] + dist[k][j];
          if (dist[i][j] > sum) {
            dist[i][j] = sum;
            prev[i][j] = prev[k][j];
          }
        }
      }
    }
  }

  std::string name_;
  uint64_t nqubits_;
  CouplingMap couplingMap_;
  Matrix dist;
  Matrix prev;
};

/// @brief Base class for all initial layout mapping functions.
struct InitialLayout {
  explicit InitialLayout(const std::size_t nqubits) : mapping(nqubits) {}
  [[nodiscard]] std::size_t operator()(std::size_t i) const {
    return mapping[i];
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

/// @brief A layer consists of a qubit mapping and a set of 2Q gates.
struct Layer {
  /// @brief The mapping from qubits to indices.
  QubitMap qubitMap;
  /// @brief The mapping from indices to qubits.
  llvm::DenseMap<std::size_t, Value> indexMap;
  /// @brief Set of two-qubit gates (e.g. CNOTs).
  llvm::DenseSet<UnitaryInterface> gates;
};

/// @brief Compute layer, i.e., retrieve permutation and set of two-qubit gates.
Layer getLayer(const QubitMap& p) {
  Layer l;

  // Absorb operations that take and return a single qubit.
  for (const auto& [oldQ, i] : p) {
    Value newQ = oldQ;
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

    l.qubitMap[newQ] = i;
    l.indexMap[i] = newQ;
  }

  // Collect two-qubit gates of the layer.
  // Visit each qubit of a layer only once.

  llvm::DenseSet<Value> visited;
  for (const auto& [qubit, _] : l.qubitMap) {
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
    // left are one-control one-target gates.

    assert(!unitary.getInQubits().empty());
    assert(!unitary.getAllCtrlInQubits().empty());
    assert(unitary.getInQubits().size() == 1);
    assert(unitary.getAllCtrlInQubits().size() == 1);

    auto target = unitary.getInQubits().front();
    auto control = unitary.getAllCtrlInQubits().front();
    if (l.qubitMap.contains(target) && l.qubitMap.contains(control)) {
      l.gates.insert(unitary);
    }

    visited.insert(target);
    visited.insert(control);
  }

  return l;
}

llvm::SmallVector<Layer> lookahead(const QubitMap& front,
                                   const std::size_t depth = 0) {
  llvm::SmallVector<Layer> layers(1 + depth);
  layers[0] = getLayer(front);

  Layer& prev = layers.front();
  for (std::size_t i = 1; i < 1 + depth; ++i) {
    QubitMap p(prev.qubitMap); // Copy previous.

    // "Apply" two qubit gates.
    for (UnitaryInterface unitary : prev.gates) {
      const auto targetIn = unitary.getInQubits().front();
      const auto targetOut = unitary.getOutQubits().front();
      const auto controlIn = unitary.getAllCtrlInQubits().front();
      const auto controlOut = unitary.getAllCtrlOutQubits().front();

      p[targetOut] = p[targetIn];
      p[controlOut] = p[controlIn];
      p.erase(targetIn);
      p.erase(controlIn);
    }

    layers[i] = getLayer(p);
  }

  return layers;
}

/**
 * @brief Add attributes to module specifying the targeted architecture.
 */
void setModuleTarget(ModuleOp module, const Architecture& target,
                     PatternRewriter& rewriter) {
  const auto nameAttr = rewriter.getStringAttr(target.name());
  module->setAttr(ATTRIBUTE_TARGET, nameAttr);
}

/**
 * @brief Walk the IR and collect the dynamic qubits for each computation.
 * @details Assumes that all allocs come before all deallocs.
 * @return Vector of vectors, where each inner vector contains the dynamic
 * qubits allocated for a computation.
 */
llvm::SmallVector<llvm::SmallVector<Value>, 2>
collectComputations(Region& body, const Architecture& target) {
  llvm::SmallVector<llvm::SmallVector<Value>, 2> computations;
  llvm::SmallVector<Value> dynQubits;
  uint64_t allocated = 0;

  // A valid program has at most target.nqubits qubits.
  dynQubits.reserve(target.nqubits());

  body.walk([&](Operation* op) {
    if (auto alloc = dyn_cast<AllocQubitOp>(op)) {
      ++allocated;
      dynQubits.push_back(alloc.getQubit());
      return;
    }

    if (auto dealloc = dyn_cast<DeallocQubitOp>(op)) {
      if ((--allocated) == 0) {
        computations.push_back(std::move(dynQubits));
        dynQubits.clear();
      }
      return;
    }
  });

  return computations;
}

/**
 * @brief Route @p func.
 * @details If the function is marked as entry point, create static qubits that
 * can later be passed to other quantum functions.
 */
LogicalResult routeFunc(func::FuncOp func, const Architecture& arch,
                        const InitialLayout& layout,
                        PatternRewriter& rewriter) {
  Region& body = func.getBody();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&body.front());

  llvm::SmallVector<Value> statQubits(arch.nqubits());

  if (func->hasAttr(ATTRIBUTE_ENTRY_POINT)) {
    for (std::size_t i = 0; i < arch.nqubits(); ++i) {
      auto qubitOp = rewriter.create<QubitOp>(body.getLoc(), i);
      statQubits[i] = qubitOp.getQubit();
    }
  }

  for (const auto& dynQubits : collectComputations(body, arch)) {

    if (dynQubits.size() > statQubits.size()) {
      return dynQubits.back().getDefiningOp()->emitOpError()
             << "requires more qubits than the architecture supports";
    }

    QubitMap front;
    for (std::size_t i = 0; i < statQubits.size(); ++i) {
      front.try_emplace(statQubits[i], i);

      // Replace dynamic qubits with permutated static qubits.
      if (layout(i) == i) {
        rewriter.replaceAllUsesWith(dynQubits[i], statQubits[i]);
        rewriter.eraseOp(dynQubits[i].getDefiningOp());
      }
    }

    while (true) {
      const llvm::SmallVector<Layer>& layers = lookahead(front);

      const Layer& head = layers.front();
      if (head.gates.empty()) {
        break;
      }

      for (UnitaryInterface unitary : head.gates) {
        Value target = unitary.getInQubits().front();
        Value control = unitary.getAllCtrlInQubits().front();

        if (front.contains(target) && front.contains(control)) {
          continue;
        }

        uint64_t iA = head.qubitMap.at(target);
        uint64_t iB = head.qubitMap.at(control);

        llvm::outs() << "iA= " << iA << " ; iB= " << iB << '\n';
        for (const uint64_t i : arch.shortestPath(iA, iB)) {
          llvm::outs() << i << '\n';
        }
        llvm::outs() << "----\n";
      }

      return success();
    }

    // Update static qubits with their current SSA value for the next
    // loop iteration.

    for (const auto& [q, i] : front) {
      statQubits[i] = q;

      if (q.hasOneUse()) { // There may be unused qubits.
        DeallocQubitOp dealloc =
            dyn_cast<DeallocQubitOp>(*q.getUsers().begin());
        assert(dealloc);
        rewriter.eraseOp(dealloc);
      }
    }
  }

  return success();
}

/**
 * @brief Collect functions of a module and route each.
 * @details Assume that there are no nested functions.
 */
LogicalResult routeModule(ModuleOp module, const Architecture& arch,
                          const InitialLayout& layout,
                          PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module);

  setModuleTarget(module, arch, rewriter);

  // Copying because replacing functions invalidates their pointers.
  llvm::SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());

  for (func::FuncOp func : funcs) {
    LogicalResult res = routeFunc(func, arch, layout, rewriter);
    if (res.failed()) {
      return res;
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
    const auto arch = std::make_unique<Architecture>("MQT-6", 6, couplingMap);
    const Identity layout(arch->nqubits());

    if (routeModule(module, *arch, layout, rewriter).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
