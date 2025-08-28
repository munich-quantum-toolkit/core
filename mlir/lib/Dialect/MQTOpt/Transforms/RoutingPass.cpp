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

constexpr std::string ATTRIBUTE_TARGET = "mqtopt.target";

/// @brief A quantum accelerator's architecture.
struct Architecture {
  explicit Architecture(std::string name, uint64_t nqubits)
      : name_(std::move(name)), nqubits_(nqubits) {}
  /// @brief Return the architecture's name.
  [[nodiscard]] constexpr std::string_view name() const { return name_; }
  /// @brief Return the architecture's number of qubits.
  [[nodiscard]] constexpr uint64_t nqubits() const { return nqubits_; }

private:
  std::string name_;
  uint64_t nqubits_;
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
  llvm::DenseMap<Value, std::size_t> qubitMap;
  /// @brief The mapping from indices to qubits.
  llvm::DenseMap<std::size_t, Value> indexMap;
  /// @brief Set of two-qubit gates (e.g. CNOTs).
  llvm::DenseSet<UnitaryInterface> gates;
};

/// @brief Compute layer, i.e., retrieve permutation and set of two-qubit gates.
Layer getLayer(const llvm::DenseMap<Value, std::size_t>& p) {
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

template <std::size_t N = 0>
std::pair<Layer, std::array<Layer, N>>
lookahead(const llvm::DenseMap<Value, std::size_t>& p0) {
  Layer head(getLayer(p0));

  std::array<Layer, N> beyond;

  Layer& prev = head;
  for (std::size_t i = 0; i < N; ++i) {
    llvm::DenseMap<Value, std::size_t> p(prev.qubitMap); // Copy previous.

    // "Apply" two qubit gates.
    for (auto unitary : prev.gates) {
      const auto targetIn = unitary.getInQubits().front();
      const auto targetOut = unitary.getOutQubits().front();
      const auto controlIn = unitary.getAllCtrlInQubits().front();
      const auto controlOut = unitary.getAllCtrlOutQubits().front();

      p[targetOut] = p[targetIn];
      p[controlOut] = p[controlIn];
      p.erase(targetIn);
      p.erase(controlIn);
    }

    beyond[i] = getLayer(p);
  }

  return {head, beyond};
}

/// @brief Add attributes to module-like parent specifying the architecture.
void setModuleTarget(Operation* module, const Architecture& target,
                     PatternRewriter& rewriter) {
  const auto nameAttr = rewriter.getStringAttr(target.name());
  module->setAttr(ATTRIBUTE_TARGET, nameAttr);
}

/// @brief Initialize entry point, i.e., create and return static qubits.
llvm::SmallVector<Value> initEntry(func::FuncOp entry,
                                   PatternRewriter& rewriter,
                                   const Architecture& target) {
  llvm::SmallVector<Value, 0> statQubits(target.nqubits());
  for (std::size_t i = 0; i < target.nqubits(); ++i) {
    auto qubitOp = rewriter.create<QubitOp>(entry->getLoc(), i);
    statQubits[i] = qubitOp.getQubit();
  }
  return statQubits;
}

/**
 * @brief Walk the IR and collect the dynamic qubits for each computation.
 * @return Vector of vectors, where each inner vector contains the dynamic
 * qubits allocated for a computation.
 */
llvm::SmallVector<llvm::SmallVector<Value>, 2>
collectComputations(func::FuncOp f, const Architecture& target) {
  llvm::SmallVector<llvm::SmallVector<Value>, 2> computations;
  llvm::SmallVector<Value> dynQubits;
  uint64_t allocated = 0;

  // A valid program has at most target.nqubits qubits.
  dynQubits.reserve(target.nqubits());

  f.walk([&](Operation* op) {
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
 * @brief Apply topology constraints to @p f.
 * @note Expects rewriter set to body of @p f.
 * TODO / Remember: statQubits[i] is the static qubit i
 */
LogicalResult routeFunc(func::FuncOp f, const Architecture& target,
                        const InitialLayout& layout,
                        llvm::SmallVectorImpl<Value>& statQubits,
                        PatternRewriter& rewriter) {
  llvm::dbgs() << "routing func: " << f->getName() << '\n';

  for (const auto& dynQubits : collectComputations(f, target)) {
    OpBuilder::InsertionGuard guard(rewriter);

    if (dynQubits.size() > statQubits.size()) {
      return f.emitOpError()
             << "requires more qubits than the architecture supports";
    }

    llvm::DenseMap<Value, std::size_t> qubitMap; // Qubit <-> Index
    llvm::DenseMap<std::size_t, Value> indexMap; // Index <-> Qubit
    for (std::size_t i = 0; i < statQubits.size(); ++i) {
      qubitMap.try_emplace(statQubits[i], i);
      indexMap.try_emplace(i, statQubits[i]);
    }

    // Replace dynamic qubits with permutated static qubits.
    for (std::size_t i = 0; i < dynQubits.size(); ++i) {
      rewriter.replaceAllUsesWith(dynQubits[i], indexMap[layout(i)]);
      rewriter.eraseOp(dynQubits[i].getDefiningOp());
    }

    llvm::DenseMap<Value, std::size_t>& pIt = qubitMap;
    while (true) {
      const auto& [head, beyond] = lookahead<1UL>(pIt);

      if (head.gates.empty()) {
        llvm::dbgs() << "\tno more two-qubit gates. stop.\n";
        pIt = head.qubitMap;
        break;
      }

      llvm::dbgs() << "\thead:\n";
      for (const auto& g : head.gates) {
        llvm::dbgs() << "\t\t" << g->getLoc() << '\n';
      }
      llvm::dbgs() << "\tlookahead:\n";
      for (const auto& l : beyond) {
        for (const auto& g : l.gates) {
          llvm::dbgs() << "\t\t" << g->getLoc() << '\n';
        }
      }

      if (!beyond.empty()) {
        pIt = beyond.at(0).qubitMap;
      }
    }

    // Update static qubits with their current SSA value.
    for (const auto& [q, i] : pIt) {
      statQubits[i] = q;

      if (q.hasOneUse()) { // There may be unused qubits.
        llvm::outs() << q.getUsers().begin()->getName() << '\n';
        DeallocQubitOp dealloc =
            dyn_cast<DeallocQubitOp>(*q.getUsers().begin());
        assert(dealloc);
        rewriter.eraseOp(dealloc);
      }
    }
  }

  return success();
}

/// @brief Find the entry point of the module and route it.
/// @details Assume that there are no nested functions.
LogicalResult routeModule(ModuleOp module, const Architecture& target,
                          const InitialLayout& layout,
                          PatternRewriter& rewriter) {
  for (func::FuncOp f : module.getOps<func::FuncOp>()) {
    rewriter.setInsertionPointToStart(&f.getBody().front());

    if (f->hasAttr("entry_point")) {
      llvm::SmallVector<Value> statQubits = initEntry(f, rewriter, target);
      return routeFunc(f, target, layout, statQubits, rewriter);
    }
  }

  return module->emitOpError()
         << "expects func.func with attribute `entry_point`";
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

    const auto target = std::make_unique<Architecture>("iqm-spark+1", 6);
    const Identity layout(target->nqubits());

    setModuleTarget(module, *target, rewriter);
    if (routeModule(module, *target, layout, rewriter).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
