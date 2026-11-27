/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QIR/Transforms/Passes.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir::qir {

#define GEN_PASS_DEF_QIRCLEANUPPASS
#include "mqt/Dialect/QIR/Transforms/Passes.h.inc"

namespace {

struct QIRCleanupPass final : impl::QIRCleanupPassBase<QIRCleanupPass> {
protected:
  void runOnOperation() override { removeUnusedDeclarations(getOperation()); }
};

} // namespace
} // namespace mlir::qir
