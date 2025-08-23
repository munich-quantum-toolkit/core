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
#include <iterator>
#include <list>
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

struct BidirectionalMap {
  using Map = llvm::DenseMap<std::size_t, Value>;
  using InvMap = llvm::DenseMap<Value, std::size_t>;

  using MapValueType = Map::value_type;

  // Set bidirectional (index <-> qubit) mapping.
  void set(const std::size_t i, const Value q) {
    map_[i] = q;
    invMap_[q] = i;
  }

  /// @brief Get qubit from index.
  [[nodiscard]] Value get(const std::size_t i) { return getForward().at(i); }

  /// @brief Get index from qubit.
  [[nodiscard]] std::size_t get(const Value q) { return getBackward().at(q); }

  /// @brief Returns True if @p i is in the forward map.
  [[nodiscard]] bool contains(const std::size_t i) {
    return getForward().contains(i);
  }

  /// @brief Returns True if @p q is in the backward map.
  [[nodiscard]] bool contains(const Value q) {
    return getBackward().contains(q);
  }

  /// @brief Get reference of forward (T1 -> T2) map.
  [[nodiscard]] Map& getForward() { return map_; }

  /// @brief Get reference of backward (T1 <- T2) map.
  [[nodiscard]] InvMap& getBackward() { return invMap_; }

  /// @brief Erase from bidirectional map.
  void erase(const std::size_t i, const Value q) {
    map_.erase(i);
    invMap_.erase(q);
  }

  /// @brief Returns True if the bidirectional map has no entries.
  [[nodiscard]] bool empty() const { return map_.empty() && invMap_.empty(); }

private:
  Map map_;
  InvMap invMap_;
};

template <std::size_t N = 0> class Lookahead {
private:
  /// @brief A layer consists of a qubit mapping and a set of 2Q gates.
  struct Layer {
    BidirectionalMap qubits;
    llvm::DenseSet<UnitaryInterface> gates;
  };

  /// @brief Replace @p oldQ with @p newQ but keep the same index.
  static void updateMapping(const Value oldQ, const Value newQ,
                            BidirectionalMap& qubits) {
    qubits.set(qubits.get(oldQ), newQ);
    qubits.getBackward().erase(oldQ);
  }

  /// @brief Absorb operations that take and return a single qubit.
  static bool absorbWire(const Value oldQ, BidirectionalMap& qubits) {
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

public:
  explicit Lookahead(const BidirectionalMap& p0) { init(p0); }

  [[nodiscard]] std::span<const Layer> get() const { return layers; }

private:
  /// @brief Absorb 1Q operations. Remove any deallocated qubits.
  void absorb(Layer& l) {
    std::vector<BidirectionalMap::MapValueType> deallocated{};
    for (const auto& [i, q] : l.qubits.getForward()) {
      if (absorbWire(q, l.qubits)) {
        deallocated.emplace_back(i, q);
      }
    }

    for (const auto& [i, q] : deallocated) {
      l.qubits.erase(i, q);
    }
  }

  /// @brief Collect 2Q gates of the layer.
  void collectGates(Layer& l) {
    for (auto [i, q] : l.qubits.getForward()) {
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
      if (l.qubits.contains(target) && l.qubits.contains(control)) {
        l.gates.insert(unitary);
      }
    }
  }

  /// @brief Return the permutation of the layer after @p l.
  BidirectionalMap nextPermutation(Layer& l) {
    BidirectionalMap p = l.qubits;

    for (UnitaryInterface unitary : l.gates) {
      const auto targetIn = unitary.getInQubits().front();
      const auto targetOut = unitary.getOutQubits().front();
      const auto controlIn = unitary.getAllCtrlInQubits().front();
      const auto controlOut = unitary.getAllCtrlOutQubits().front();

      Lookahead::updateMapping(targetIn, targetOut, p);
      Lookahead::updateMapping(controlIn, controlOut, p);
    }

    return p;
  }

  void init(const BidirectionalMap& p0) {
    layers[0] = Layer{.qubits = p0};

    for (std::size_t i = 0; i < layers.size(); ++i) {
      absorb(layers[i]);
      collectGates(layers[i]);

      if (i < layers.size() - 1) {
        layers[i + 1].qubits = nextPermutation(layers[i]);
      }
    }
  }

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
void addModuleQubits(Operation* moduleLike, const uint64_t nqubits,
                     BidirectionalMap& qubits, PatternRewriter& rewriter) {
  const auto& qubit = QubitType::get(moduleLike->getContext());
  for (std::size_t i = 0; i < nqubits; ++i) {
    auto qubitOp = rewriter.create<QubitOp>(moduleLike->getLoc(), qubit, i);
    qubits.set(i, qubitOp.getQubit());
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

  BidirectionalMap qubits{};                     // Holds index <-> qubit.
  llvm::BitVector hotQubits(target->nqubits());  // Currently used qubits.
  std::vector<uint64_t> perm(target->nqubits()); // Holds initial layout.

  initial_mapping_funcs::identity(perm);

  auto result = base->walk([&](ExtractOp extract) {
    rewriter.setInsertionPoint(extract);

    const Operation* successor = extract->getNextNode();

    // TODO: What trait to use here best?
    Operation* moduleLike = extract->getParentOp();
    assert(moduleLike);
    if (!moduleLike->hasAttr(attribute_names::TARGET)) {
      setModuleTarget(moduleLike, rewriter, *target);
      addModuleQubits(moduleLike, target->nqubits(), qubits, rewriter);
    }

    const std::optional<uint64_t> indexAttr = extract.getIndexAttr();
    assert(indexAttr.has_value()); // Assume a static index for now.

    const std::size_t idx = perm[indexAttr.value()];
    const Value dynQ = extract.getOutQubit();     // The dynamic qubit.
    const Value statQ = qubits.getForward()[idx]; // The static qubit.

    if (hotQubits[idx]) {
      return WalkResult(extract.emitOpError()
                        << "extracted the same qubit twice.");
    }

    if (hotQubits.all()) {
      return WalkResult(
          extract.emitOpError()
          << "requires more qubits than the architecture supports");
    }

    hotQubits.set(idx);

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

    Lookahead<2> lookahead(qubits);
    for(const auto& l : lookahead.get()) {
      for(const auto& gate : l.gates) {
        llvm::outs() << gate->getLoc() << '\n';
      }
      llvm::outs() << "------" << '\n';
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