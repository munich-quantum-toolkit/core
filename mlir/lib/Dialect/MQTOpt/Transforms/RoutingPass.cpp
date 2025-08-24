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
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
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
struct InitalLayout {
  explicit InitalLayout(const std::size_t nqubits) : mapping(nqubits) {}
  [[nodiscard]] std::size_t operator()(std::size_t i) const {
    return mapping[i];
  }

protected:
  llvm::SmallVector<std::size_t, 0> mapping;
};

/// @brief Identity mapping.
struct Identity : InitalLayout {
  explicit Identity(const std::size_t nqubits) : InitalLayout(nqubits) {
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
void setModuleTarget(Operation* moduleLike, PatternRewriter& rewriter,
                     const Architecture& target) {
  const auto nameAttr = rewriter.getStringAttr(target.name());
  moduleLike->setAttr(ATTRIBUTE_TARGET, nameAttr);
}

/// @brief Create @p nqubits static qubits in module-like parent.
void addModuleQubits(Operation* module, PatternRewriter& rewriter,
                     const Architecture& target,
                     llvm::SmallVectorImpl<Value>& qubits) {
  const auto& qubit = QubitType::get(module->getContext());
  for (std::size_t i = 0; i < target.nqubits(); ++i) {
    auto qubitOp = rewriter.create<QubitOp>(module->getLoc(), qubit, i);
    qubits.push_back(qubitOp.getQubit());
    llvm::dbgs() << "\tcreated static qubit: " << qubitOp.getQubit() << '\n';
  }
}

LogicalResult spanAndFold(Operation* base, std::unique_ptr<Architecture> target,
                          const InitalLayout& layout) {
  MLIRContext* ctx = base->getContext();
  PatternRewriter rewriter(ctx);

  // CURRENT ASSUMPTIONS
  // -) A quantum computation has the following format:
  //       Q := (alloc -> compute -> dealloc)
  // -) Initial layout will be applied when creating the static qubits.

  Operation* module = nullptr;

  llvm::SmallVector<Value, 0> statQubits;
  llvm::SmallVector<Value, 0> dynQubits;

  statQubits.reserve(target->nqubits());
  dynQubits.reserve(target->nqubits());

  auto result = base->walk([&](AllocQubitOp alloc) {
    rewriter.setInsertionPoint(alloc);

    // ----------- Alloc Phase -----------
    // - Create static qubits in module-like parent.
    // - Collect dynamic qubits for computation.

    if (module != alloc->getParentOp()) {
      module = alloc->getParentOp();
      assert(module);

      llvm::dbgs() << "module found: " << module->getName() << '\n';

      statQubits.clear();
      dynQubits.clear();
    }

    if (!module->hasAttr(ATTRIBUTE_TARGET)) {
      setModuleTarget(module, rewriter, *target);
      addModuleQubits(module, rewriter, *target, statQubits);
    }

    dynQubits.emplace_back(alloc.getQubit());

    llvm::dbgs() << "\tcollected dynamic qubit: " << alloc.getQubit() << '\n';

    const Operation* successor = alloc->getNextNode();
    if (successor && dyn_cast<AllocQubitOp>(successor)) {
      return WalkResult::advance();
    }

    // ----------- Compute Phase -----------
    // - Assign dynamic qubits to permutated static qubits.
    // - Fit computation to the given topology.

    if (dynQubits.size() > statQubits.size()) {
      return WalkResult(
          alloc.emitOpError()
          << "requires more qubits than the architecture supports");
    }

    // Setup initial layout.
    llvm::DenseMap<Value, std::size_t> p0;
    for (std::size_t i = 0; i < statQubits.size(); ++i) {
      p0[statQubits[i]] = layout(i);
    }

    // TODO: Wrap in LLVM_DEBUG
    for (std::size_t i = 0; i < dynQubits.size(); ++i) {
      llvm::dbgs() << "\tassign: " << dynQubits[i] << " (dyn) to "
                   << statQubits[layout(i)] << " (stat)" << '\n';
    }

    // Replace dynamic qubits with permutated static qubits.
    for (std::size_t i = 0; i < dynQubits.size(); ++i) {
      rewriter.replaceAllUsesWith(dynQubits[i], statQubits[layout(i)]);
      rewriter.eraseOp(dynQubits[i].getDefiningOp());
    }

    llvm::DenseMap<Value, std::size_t>& pIt = p0;
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

    for (const auto& [q, i] : pIt) {
      llvm::dbgs() << "\treplace static[" << i << "] with " << q.getLoc()
                   << '\n';
      statQubits[i] = q;

      if (q.hasOneUse()) { // There may be unused qubits.
        DeallocQubitOp dealloc =
            dyn_cast<DeallocQubitOp>(*q.getUsers().begin());
        assert(dealloc);
        llvm::dbgs() << "\tremove: " << dealloc << '\n';
        rewriter.eraseOp(dealloc);
      }
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}
} // namespace

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    auto target = std::make_unique<Architecture>("iqm-spark+1", 6);

    const Identity layout(target->nqubits());
    if (spanAndFold(getOperation(), std::move(target), layout).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt