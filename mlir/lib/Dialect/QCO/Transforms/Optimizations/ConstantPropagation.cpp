/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

namespace mlir::qco {

#define GEN_PASS_DEF_CONSTANTPROPAGATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/**
 * @brief Quantum constant propagation.
 *
 * Assumes all input qubits start in |0>, propagates the quantum/classical state
 * through the circuit up to a complexity threshold, and removes operations that
 * are superfluous given that state.
 *
 * The analysis is done as an MLIR `DenseForwardDataFlowAnalysis` over a
 * `UnionTable` lattice, with a separate rewrite phase driven by the computed
 * facts.
 */
struct ConstantPropagation final
    : impl::ConstantPropagationBase<ConstantPropagation> {
  using ConstantPropagationBase::ConstantPropagationBase;

  void runOnOperation() override {
    // TODO(mlir/constant-propagation-v2): implement in stages --
    //   1. QuantumState, 2. UnionTable/HybridState, 3.
    //   ConstantPropagationAnalysis,
    //   4. Decisions + Rewriter + driver, 5. pass-level tests.
  }
};

} // namespace

} // namespace mlir::qco
