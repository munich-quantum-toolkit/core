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

template <class T1, class T2> struct BidirectionalMap {
  using MapType = llvm::DenseMap<T1, T2>;
  using InvMapType = llvm::DenseMap<T2, T1>;

  // Set bidirectional (T1 <-> T2) mapping.
  void set(const T1 a, const T2 b) {
    map_[a] = b;
    invMap_[b] = a;
  }

  // Get reference of forward (T1 -> T2) map.
  [[nodiscard]] MapType& getForward() { return map_; }

  // Get reference of backward (T1 <- T2) map.
  [[nodiscard]] InvMapType& getBackward() { return invMap_; }

private:
  MapType map_;
  InvMapType invMap_;
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
                     BidirectionalMap<std::size_t, Value>& qubits,
                     PatternRewriter& rewriter) {
  const auto& qubit = QubitType::get(moduleLike->getContext());
  for (std::size_t i = 0; i < nqubits; ++i) {
    auto qubitOp = rewriter.create<QubitOp>(moduleLike->getLoc(), qubit, i);
    qubits.set(i, qubitOp);
  }
}

/// @brief Absorb single-qubit operations on a wire for all qubits in @p front.
llvm::DenseSet<Value> absorbSingleQubitOps(const llvm::DenseSet<Value>& front) {
  llvm::DenseSet<Value> result;

  auto absorb = [](Value q) -> Value {
    while (!q.getUsers().empty()) {
      Operation* nextOnWire = *q.getUsers().begin();
      if (auto reset = dyn_cast<ResetOp>(nextOnWire)) {
        q = reset.getOutQubit();
      } else if (auto unitary = dyn_cast<UnitaryInterface>(nextOnWire)) {
        if (!unitary.getAllCtrlInQubits().empty()) {
          break;
        }
        q = unitary.getOutQubits().front(); // TODO: GPhase Gate.
      } else if (auto measure = dyn_cast<MeasureOp>(nextOnWire)) {
        q = measure.getOutQubit();
      } else { // Insert.
        assert(dyn_cast<InsertOp>(nextOnWire));
        return nullptr;
      }
    }

    return q;
  };

  for (const Value& exec : front) {
    if (auto q = absorb(exec)) {
      result.insert(q);
    }
  }

  return result;
}

LogicalResult spanAndFold(Operation* base,
                          std::unique_ptr<Architecture> target) {
  MLIRContext* ctx = base->getContext();
  PatternRewriter rewriter(ctx);

  // CURRENT ASSUMPTIONS
  // 1) All extract's are BUNDLED.
  // 2) Qubits are extracted ONCE.
  // 3) All used qubits are extracted before any computation.

  llvm::DenseSet<Value> hotQubits{};             // Currently used qubits.
  BidirectionalMap<std::size_t, Value> qubits{}; // Index to qubit & vice versa.
  std::vector<uint64_t> perm(target->nqubits()); // Holds initial mapping.

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

    if (hotQubits.contains(statQ)) {
      return WalkResult(extract.emitOpError()
                        << "extracted the same qubit twice.");
    }

    if (hotQubits.size() == target->nqubits()) {
      return WalkResult(
          extract.emitOpError()
          << "requires more qubits than the architecture supports");
    }

    hotQubits.insert(statQ);
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

    llvm::DenseSet<Value> front;
    for (const auto& pair : qubits.getForward()) {
      front.insert(pair.second);
    }

    while (true) {
      front = absorbSingleQubitOps(front);

      if (front.empty()) {
        break;
      }

      // Find next layer.
      llvm::DenseSet<UnitaryInterface> layer{};
      for (auto ready : front) {
        Operation* user = *(ready.getUsers().begin());
        assert(dyn_cast<UnitaryInterface>(user));

        if (auto unitary = dyn_cast<UnitaryInterface>(user)) {

          // All single qubit operations should have been absorbed. What's
          // left are one-control one-target gates.

          assert(!unitary.getInQubits().empty());
          assert(!unitary.getAllCtrlInQubits().empty());
          assert(unitary.getInQubits().size() == 1);
          assert(unitary.getAllCtrlInQubits().size() == 1);

          auto target = unitary.getInQubits().front();
          auto control = unitary.getAllCtrlInQubits().front();
          if (front.contains(target) && front.contains(control)) {
            layer.insert(unitary);
          }
        }
      }

      for (auto unitary : layer) {
        llvm::outs() << unitary->getLoc() << '\n';

        // Insert 'out's.
        front.insert(unitary.getOutQubits().front());
        front.insert(unitary.getAllCtrlOutQubits().front());
        // Remove 'in's.
        front.erase(unitary.getInQubits().front());
        front.erase(unitary.getAllCtrlInQubits().front());
      }
      llvm::outs() << "-----------\n";
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