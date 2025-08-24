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
#include <functional>
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
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

namespace attribute_names {
constexpr std::string TARGET = "mqtopt.target";
} // namespace attribute_names

namespace initial_layout_funcs {
/// @brief Return identity mapping for @p nqubits.
llvm::SmallVector<std::size_t> identity(const std::size_t nqubits) {
  llvm::SmallVector<std::size_t, 0> mapping(nqubits);
  std::iota(mapping.begin(), mapping.end(), 0);
  return mapping;
}

/// @brief Return identity mapping for @p nqubits.
llvm::SmallVector<std::size_t> random(const std::size_t nqubits) {
  std::random_device rd;
  std::mt19937 g(rd());

  llvm::SmallVector<std::size_t, 0> mapping(nqubits);
  std::iota(mapping.begin(), mapping.end(), 0);
  std::shuffle(mapping.begin(), mapping.end(), g);

  return mapping;
}
}; // namespace initial_layout_funcs

/// @brief Maps from T1 to T2 and from T2 to T1.
template <class T1, class T2> struct BidirectionalMap {
  using Map = llvm::DenseMap<T1, T2>;
  using InvMap = llvm::DenseMap<T2, T1>;

  // Set bidirectional (T1 <-> T2) mapping.
  void set(const T1 t1, T2 t2) {
    map_[t1] = t2;
    invMap_[t2] = t1;
  }

  /// @brief Get T2 from T1.
  [[nodiscard]] T2 get(const T1 key) { return getForward().at(key); }

  /// @brief Get T1 from T2.
  [[nodiscard]] T1 get(const T2 key) { return getBackward().at(key); }

  /// @brief Return true if bi-map contains T1 @p key.
  [[nodiscard]] bool contains(const T1 key) {
    return getForward().contains(key);
  }

  /// @brief Return true if bi-map contains T2 @p key.
  [[nodiscard]] bool contains(const T2 key) {
    return getBackward().contains(key);
  }

  /// @brief Get reference of forward (T1 -> T2) map.
  [[nodiscard]] Map& getForward() { return map_; }

  /// @brief Get reference of backward (T1 <- T2) map.
  [[nodiscard]] InvMap& getBackward() { return invMap_; }

  /// @brief Erase from bi-map map.
  void erase(const T1 t1, const T2 t2) {
    map_.erase(t1);
    invMap_.erase(t2);
  }

  /// @brief Return the size of the bi-map.
  [[nodiscard]] std::size_t size() const { return map_.size(); }

  /// @brief Return true if the bidirectional map has no entries.
  [[nodiscard]] bool empty() const { return map_.empty() && invMap_.empty(); }

  /// @brief Clear the bi-map.
  void clear() {
    map_.clear();
    invMap_.clear();
  }

private:
  Map map_;
  InvMap invMap_;
};

/// @brief A QubitPermutation maps indices to qubits and qubits to indices.
using QubitPermutation = BidirectionalMap<std::size_t, Value>;

template <std::size_t N = 0> class Lookahead {
private:
  /// @brief A layer consists of a qubit mapping and a set of 2Q gates.
  struct Layer {
    QubitPermutation p;
    llvm::DenseSet<UnitaryInterface> gates;
  };

  /// @brief Replace @p oldQ with @p newQ but keep the same index.
  static void updateMapping(const Value oldQ, const Value newQ,
                            QubitPermutation& qubits) {
    qubits.set(qubits.get(oldQ), newQ);
    qubits.getBackward().erase(oldQ);
  }

  /// @brief Absorb operations that take and return a single qubit.
  static bool absorbWire(const Value oldQ, QubitPermutation& qubits) {
    bool hasDeallocated = false;

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
      } else { // Insert.
        assert(dyn_cast<DeallocQubitOp>(op));
        hasDeallocated = true;
        break;
      }
    }

    if (!hasDeallocated && oldQ != newQ) {
      Lookahead::updateMapping(oldQ, newQ, qubits);
    }

    return hasDeallocated;
  }

  /// @brief Absorb 1Q operations. Remove any deallocated qubits.
  static void absorb(Layer& l) {
    std::vector<std::pair<std::size_t, Value>> deallocated{};
    for (const auto& [i, q] : l.p.getForward()) {
      if (absorbWire(q, l.p)) {
        deallocated.emplace_back(i, q);
      }
    }

    for (const auto& [i, q] : deallocated) {
      l.p.erase(i, q);
    }
  }

  /// @brief Collect 2Q gates of the layer.
  static void collectGates(Layer& l) {
    for (auto [i, q] : l.p.getForward()) {
      if (q.getUsers().empty()) {
        continue;
      }

      Operation* user = *(q.getUsers().begin());
      UnitaryInterface unitary = dyn_cast<UnitaryInterface>(user);
      assert(unitary);

      // All single qubit operations should have been absorbed. What's
      // left are one-control one-target gates.

      assert(!unitary.getInQubits().empty());
      assert(!unitary.getAllCtrlInQubits().empty());
      assert(unitary.getInQubits().size() == 1);
      assert(unitary.getAllCtrlInQubits().size() == 1);

      auto target = unitary.getInQubits().front();
      auto control = unitary.getAllCtrlInQubits().front();
      if (l.p.contains(target) && l.p.contains(control)) {
        l.gates.insert(unitary);
      }
    }
  }

  /// @brief Return the permutation of the layer after @p l.
  static QubitPermutation nextPermutation(Layer& l) {
    QubitPermutation pNext = l.p;

    for (UnitaryInterface unitary : l.gates) {
      const auto targetIn = unitary.getInQubits().front();
      const auto targetOut = unitary.getOutQubits().front();
      const auto controlIn = unitary.getAllCtrlInQubits().front();
      const auto controlOut = unitary.getAllCtrlOutQubits().front();

      Lookahead::updateMapping(targetIn, targetOut, pNext);
      Lookahead::updateMapping(controlIn, controlOut, pNext);
    }

    return pNext;
  }

public:
  explicit Lookahead(const QubitPermutation& p0) {
    layers_[0].p = p0;

    for (std::size_t i = 0; i < layers_.size(); ++i) {
      Lookahead::absorb(layers_[i]);
      Lookahead::collectGates(layers_[i]);

      if (i < layers_.size() - 1) {
        layers_[i + 1].p = nextPermutation(layers_[i]);
      }
    }
  }

  /// @brief Return view of layers.
  [[nodiscard]] std::span<const Layer> layers() const { return layers_; }

  explicit operator bool() const { return !layers_[0].gates.empty(); }

private:
  /// @brief An array of maximally (1 + lookahead) layers.
  std::array<Layer, 1 + N> layers_;
};

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

/// @brief Add attributes to module-like parent specifying the architecture.
void setModuleTarget(Operation* moduleLike, PatternRewriter& rewriter,
                     const Architecture& target) {
  const auto nameAttr = rewriter.getStringAttr(target.name());
  moduleLike->setAttr(attribute_names::TARGET, nameAttr);
}

/// @brief Create @p nqubits static qubits in module-like parent.
void addModuleQubits(Operation* moduleLike, PatternRewriter& rewriter,
                     const Architecture& target,
                     llvm::SmallVectorImpl<Value>& qubits) {
  const auto& qubit = QubitType::get(moduleLike->getContext());
  for (std::size_t i = 0; i < target.nqubits(); ++i) {
    auto qubitOp = rewriter.create<QubitOp>(moduleLike->getLoc(), qubit, i);
    qubits[i] = qubitOp.getQubit();
  }
}

/// @brief Map static qubits to permutated indices of the initial layout.
void applyInitialLayout(const llvm::SmallVectorImpl<Value>& qubits,
                        const llvm::SmallVectorImpl<std::size_t>& mapping,
                        QubitPermutation& p) {
  for (std::size_t i = 0; i < qubits.size(); ++i) {
    p.set(mapping[i], qubits[i]);
  }
}

LogicalResult spanAndFold(Operation* base,
                          std::unique_ptr<Architecture> target) {
  MLIRContext* ctx = base->getContext();
  PatternRewriter rewriter(ctx);

  // CURRENT ASSUMPTIONS
  // -) A quantum computation has the following format:
  //       alloc - compute - dealloc
  // -) Initial layout will be applied, when creating the static qubits.

  QubitPermutation p{};

  llvm::SmallVector<Value, 0> statQubits(target->nqubits());
  llvm::SmallVector<Value, 0> dynQubits;
  dynQubits.reserve(target->nqubits());

  auto result = base->walk([&](AllocQubitOp alloc) {
    rewriter.setInsertionPoint(alloc);

    // ----------- Alloc Phase -----------
    // - Create static qubits in module-like parent.
    // - Collect dynamic qubits for computation.

    // TODO: What trait to use here best?
    Operation* moduleLike = alloc->getParentOp();
    assert(moduleLike);
    if (!moduleLike->hasAttr(attribute_names::TARGET)) {
      llvm::SmallVector<std::size_t> mapping =
          initial_layout_funcs::identity(target->nqubits());

      setModuleTarget(moduleLike, rewriter, *target);
      addModuleQubits(moduleLike, rewriter, *target, statQubits);
      applyInitialLayout(statQubits, mapping, p);
    }

    dynQubits.emplace_back(alloc.getQubit());

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

    for(std::size_t i = 0; i < dynQubits.size(); ++i) {
      rewriter.replaceAllUsesWith(dynQubits[i], statQubits[i]);
      rewriter.eraseOp(dynQubits[i].getDefiningOp());
    }

    while (const auto lookahead = Lookahead<1>(p)) {
      for (const auto& l : lookahead.layers()) {
        for (const auto& gate : l.gates) {
          llvm::outs() << gate->getLoc() << '\n';
        }
      }
      llvm::outs() << "\n";

      p = lookahead.layers()[1].p;
    }

    // ----------- Dealloc Phase -----------
    // - Erase dealloc ops.

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
    if (spanAndFold(getOperation(), std::move(target)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt