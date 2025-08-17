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
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <cstddef>
#include <memory>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
struct Architecture {
  explicit Architecture(std::size_t nqubits) : nqubits_(nqubits) {}
  [[nodiscard]] constexpr std::size_t nqubits() const { return nqubits_; }

private:
  std::size_t nqubits_;
};

struct RoutingContext {
  explicit RoutingContext(std::unique_ptr<Architecture> target)
      : target_(std::move(target)) {}

  Architecture& target() { return *target_; }

private:
  std::unique_ptr<Architecture> target_;
};
} // namespace

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
 */
struct RoutingPass final : impl::RoutingPassBase<RoutingPass> {

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    // The target architecture.
    RoutingContext state(std::make_unique<Architecture>(5));

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(APPLY_PATTERNS_GREEDILY(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt