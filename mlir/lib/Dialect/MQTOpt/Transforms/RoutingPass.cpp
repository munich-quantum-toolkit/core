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

namespace initial_mapping_funcs {
/// @brief Insert identity mapping into @p perm.
void identity(std::vector<uint64_t>& perm) {
  std::iota(perm.begin(), perm.end(), 0);
}

/// @brief Insert random mapping into @p perm.
void random(std::vector<uint64_t>& perm) {
  std::random_device rd;
  std::mt19937 g(rd());

  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), g);
}
}; // namespace initial_mapping_funcs

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

  /// @brief Erase from bidirectional map.
  void erase(const T1 t1, const T2 t2) {
    map_.erase(t1);
    invMap_.erase(t2);
  }

  [[nodiscard]] std::size_t size() const { return map_.size(); }

  /// @brief Return true if the bidirectional map has no entries.
  [[nodiscard]] bool empty() const { return map_.empty() && invMap_.empty(); }

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
        assert(dyn_cast<InsertOp>(op));
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
    layers[0].p = p0;

    for (std::size_t i = 0; i < layers.size(); ++i) {
      Lookahead::absorb(layers[i]);
      Lookahead::collectGates(layers[i]);

      if (i < layers.size() - 1) {
        layers[i + 1].p = nextPermutation(layers[i]);
      }
    }
  }

  /// @brief Return view of layers.
  [[nodiscard]] std::span<const Layer> get() const { return layers; }

  explicit operator bool() const { return !layers[0].p.empty(); }

private:
  /// @brief An array of maximally (1 + lookahead) layers.
  std::array<Layer, 1 + N> layers;
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

LogicalResult spanAndFold(Operation* base,
                          std::unique_ptr<Architecture> target) {
  MLIRContext* ctx = base->getContext();
  PatternRewriter rewriter(ctx);

  // CURRENT ASSUMPTIONS
  // 1) All extract's are BUNDLED.
  // 2) Qubits are extracted ONCE.
  // 3) All used qubits are extracted before any computation.
  // 4) Extracts use static indices.

  std::vector<uint64_t> perm(target->nqubits()); // Holds initial layout.
  initial_mapping_funcs::identity(perm);

  llvm::SmallVector<Value, 0> qubits(target->nqubits());

  QubitPermutation p{}; // Holds index <-> qubit.
  auto result = base->walk([&](ExtractOp extract) {
    rewriter.setInsertionPoint(extract);

    const Operation* successor = extract->getNextNode();

    // TODO: What trait to use here best?
    Operation* moduleLike = extract->getParentOp();
    assert(moduleLike);
    if (!moduleLike->hasAttr(attribute_names::TARGET)) {
      setModuleTarget(moduleLike, rewriter, *target);
      addModuleQubits(moduleLike, rewriter, *target, qubits);
    }

    assert(extract.getIndexAttr().has_value());
    
    const std::size_t index = perm[extract.getIndexAttr().value()];
    const Value dynQ = extract.getOutQubit();
    const Value statQ = qubits[index];

    if (p.contains(index)) {
      return WalkResult(extract.emitOpError()
                        << "extracted the same qubit twice.");
    }

    if (p.size() == qubits.size()) {
      return WalkResult(
          extract.emitOpError()
          << "requires more qubits than the architecture supports");
    }

    p.set(index, statQ);

    rewriter.replaceAllUsesWith(dynQ, statQ);
    rewriter.replaceAllUsesWith(extract.getOutQreg(), extract.getInQreg());
    rewriter.eraseOp(extract);

    if (successor && dyn_cast<ExtractOp>(successor)) {
      return WalkResult::advance();
    }

    // ---------------------------------------------
    //  Extract phase over.
    //  Mapping circuit until inserts arrive.
    // ---------------------------------------------

    while (auto lookahead = Lookahead<1>(p)) {

      const auto layers = lookahead.get();

      llvm::outs() << "Current:" << '\n';
      for (const auto& gate : layers[0].gates) {
        llvm::outs() << "\t" << gate->getLoc() << '\n';
      }

      llvm::outs() << "Lookahead:" << '\n';
      for (const auto& l : layers.subspan(1)) {
        for (const auto& gate : l.gates) {
          llvm::outs() << "\t" << gate->getLoc() << '\n';
        }
        llvm::outs() << "\t------" << '\n';
      }

      p = layers[1].p;
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
    if (spanAndFold(getOperation(), std::move(target)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt