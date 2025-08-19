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
#include <iterator>
#include <memory>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
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

struct RoutingState {
  using MapType = llvm::DenseMap<mlir::Value, std::size_t>;
  using InvMapType = llvm::DenseMap<std::size_t, mlir::Value>;

  explicit RoutingState() = default;

  /// @brief Return reference to static qubit pool.
  [[nodiscard]] std::queue<mlir::Value>& pool() { return pool_; }

  /// @brief If possible, retrieve a free static qubit from the static pool.
  [[nodiscard]] std::optional<mlir::Value> getFromPool() {
    if (pool_.empty()) {
      return std::nullopt;
    }

    auto q = pool_.front();
    pool_.pop();
    return q;
  }

  /// @brief Add static qubit to the pool.
  void addToPool(mlir::Value q) { pool().push(q); }

  /// @brief Return the mapping from values to indices.
  [[nodiscard]] MapType& map() { return map_; }

  /// @brief Return the mapping from indices to values.
  [[nodiscard]] InvMapType& invMap() { return invMap_; }

  [[nodiscard]] llvm::DenseSet<mlir::Value>& executables() {
    return executables_;
  }

  void clear() {
    map_.clear();
    invMap_.clear();
    std::queue<mlir::Value>().swap(pool_); // clear queue.
  }

private:
  std::unique_ptr<Architecture> target_;

  llvm::DenseMap<mlir::Value, std::size_t> map_;
  llvm::DenseMap<std::size_t, mlir::Value> invMap_;

  llvm::DenseSet<mlir::Value> executables_;

  std::queue<mlir::Value> pool_;
};

namespace attribute_names {
constexpr std::string TARGET = "mqtopt.target";
} // namespace attribute_names

} // namespace

namespace {
using namespace mlir;

/// @brief Add attributes to module-like parent specifying the architecture.
void setModuleTarget(Operation* moduleLike, PatternRewriter& rewriter,
                     const Architecture& target) {
  const auto nameAttr = rewriter.getStringAttr(target.name());
  moduleLike->setAttr(attribute_names::TARGET, nameAttr);
}

/// @brief Create @p nqubits static qubits in module-like parent.
void addModuleQubits(Operation* moduleLike, const uint64_t nqubits,
                     RoutingState& state, PatternRewriter& rewriter) {
  const auto& qubit = QubitType::get(moduleLike->getContext());
  for (std::size_t i = 0; i < nqubits; ++i) {
    auto qubitOp = rewriter.create<QubitOp>(moduleLike->getLoc(), qubit, i);
    state.addToPool(qubitOp);
  }
}

/// @brief Absorb single qubit gates on a wire for a single qubit @p q.
Value absorb1QGates(Value q) {
  assert(!q.getUsers().empty());

  Operation* nextOnWire = *(q.getUsers().begin());
  auto unitary = dyn_cast<UnitaryInterface>(nextOnWire);

  if (!unitary) {
    return q;
  }

  if (!unitary.getAllCtrlInQubits().empty()) {
    return q;
  }

  return absorb1QGates(unitary.getOutQubits().front());
}

LogicalResult spanAndFold(Operation* base,
                          std::unique_ptr<Architecture> target) {
  MLIRContext* ctx = base->getContext();
  PatternRewriter rewriter(ctx);

  RoutingState state{};

  auto result = base->walk([&](Operation* op) {
    rewriter.setInsertionPoint(op);

    if (auto extract = dyn_cast<ExtractOp>(op)) {
      // TODO: What trait to use here best?
      Operation* moduleLike = op->getParentOp();
      assert(moduleLike);
      if (!moduleLike->hasAttr(attribute_names::TARGET)) {
        setModuleTarget(moduleLike, rewriter, *target);
        addModuleQubits(moduleLike, target->nqubits(), state, rewriter);
      }

      auto qubit = state.getFromPool();
      if (!qubit) {
        return WalkResult(
            extract.emitOpError()
            << "requires one more qubit than the architecture supports");
      }
      state.executables().insert(qubit.value());

      rewriter.replaceAllUsesWith(extract.getOutQubit(), qubit.value());
      rewriter.replaceAllUsesWith(extract.getOutQreg(), extract.getInQreg());

      Operation* successor = extract->getNextNode();
      if (successor && dyn_cast<ExtractOp>(successor)) {
        return WalkResult::advance();
      }

      // Extract phase over.
      // Mapping circuit until inserts arrive.

      while (successor &&
             !dyn_cast<InsertOp>(
                 successor)) { // This is not right. Walking Ops twice.
        auto absorbed = llvm::map_range(state.executables(), absorb1QGates);
        state.executables() =
            llvm::DenseSet<Value>(absorbed.begin(), absorbed.end());

        llvm::DenseSet<UnitaryInterface> layer{};
        for (auto ready : state.executables()) {
          // This could be an insertion, measurement, ...
          Operation* user = *(ready.getUsers().begin());
          if (auto unitary = dyn_cast<UnitaryInterface>(user)) {
            auto target = unitary.getInQubits().front();
            if (auto control = unitary.getAllCtrlInQubits().front()) {
              if (state.executables().contains(target) &&
                  state.executables().contains(control)) {
                layer.insert(unitary);
              }
            }
          }
        }

        llvm::outs() << "LAYER:\n";
        for (auto unitary : layer) {
          llvm::outs() << '\t' << unitary->getLoc() << '\n';

          // Insert 'out's.
          state.executables().insert(unitary.getOutQubits().front());
          state.executables().insert(unitary.getAllCtrlOutQubits().front());
          // Remove 'in's.
          state.executables().erase(unitary.getInQubits().front());
          state.executables().erase(unitary.getAllCtrlInQubits().front());
        }
        llvm::outs() << "-----------\n";

        successor = successor->getNextNode();
      }

    } else if (auto insert = dyn_cast<InsertOp>(op)) {
      return WalkResult::advance();
    } else if (auto dealloc = dyn_cast<DeallocOp>(op)) {
      return WalkResult::advance();
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