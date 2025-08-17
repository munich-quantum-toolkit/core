/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Common/Compat.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <string>
#include <string_view>
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

struct RoutingContext {
  explicit RoutingContext(std::unique_ptr<Architecture> target)
      : target_(std::move(target)) {
    available_.reserve(target_->nqubits());
  }
  /// @brief Return the target architecture as reference.
  [[nodiscard]] Architecture& target() { return *target_; }
  /// @brief Return all available static qubits.
  [[nodiscard]] std::vector<mlir::Value> available() { return available_; }

private:
  std::vector<mlir::Value> available_;
  std::unique_ptr<Architecture> target_;
};

template <typename OpType>
class StatefulOpRewritePattern : public mlir::OpRewritePattern<OpType> {
  using mlir::OpRewritePattern<OpType>::OpRewritePattern;

public:
  StatefulOpRewritePattern(mlir::MLIRContext* context,
                           const std::shared_ptr<RoutingContext>& state)
      : mlir::OpRewritePattern<OpType>(context), state_(state) {}

  /// @brief Return the state object as reference.
  [[nodiscard]] RoutingContext& state() const { return *state_; }

private:
  std::shared_ptr<RoutingContext> state_;
};

namespace attribute_names {
constexpr std::string TARGET_NAME = "mqtopt.target.name";
constexpr std::string TARGET_NQUBITS = "mqtopt.target.nqubits";
} // namespace attribute_names

} // namespace

/// @brief This pattern removes the `allocQubitRegister` operation and assigns
/// static qubits.
struct RewriteAlloc final : StatefulOpRewritePattern<AllocOp> {
  using StatefulOpRewritePattern<AllocOp>::StatefulOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AllocOp op, mlir::PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    assert(module && "Expecting module");

    // At first, we create as many static qubits as the architecture supports.
    // This aligns with qiskit's 'Layout' stage. Moreover, we add module
    // attributes specifying the architecture.
    if (!module->hasAttr(attribute_names::TARGET_NAME)) {
      const auto& target = state().target();

      // Add architecture information to routed module.
      const auto nameAttr = rewriter.getStringAttr(target.name());
      module->setAttr(attribute_names::TARGET_NAME, nameAttr);

      const auto nqubits = target.nqubits();
      const auto nqubitsType = rewriter.getIntegerType(64);
      const auto nqubitsAttr = rewriter.getIntegerAttr(nqubitsType, nqubits);
      module->setAttr(attribute_names::TARGET_NQUBITS, nqubitsAttr);

      // Create and collect static qubits for routed module.
      const auto& qubit = QubitType::get(rewriter.getContext());
      for (std::size_t i = 0; i < target.nqubits(); ++i) {
        auto qubitOp = rewriter.create<QubitOp>(module->getLoc(), qubit, i);
        state().available().push_back(qubitOp);
      }
    }

    // The 'alloc' Op won't be needed anymore.
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

struct RewriteExtract final : StatefulOpRewritePattern<ExtractOp> {
  using StatefulOpRewritePattern<ExtractOp>::StatefulOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ExtractOp op,
                  mlir::PatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RewriteInsert final : StatefulOpRewritePattern<InsertOp> {
  using StatefulOpRewritePattern<InsertOp>::StatefulOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(InsertOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RewriteDealloc final : StatefulOpRewritePattern<DeallocOp> {
  using StatefulOpRewritePattern<DeallocOp>::StatefulOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DeallocOp op,
                  mlir::PatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {
  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    auto target = std::make_unique<Architecture>("iqm-spark", 5);
    auto state = std::make_shared<RoutingContext>(std::move(target));

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteAlloc, RewriteExtract, RewriteInsert, RewriteDealloc>(
        ctx, state);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(APPLY_PATTERNS_GREEDILY(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt