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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

#include <tuple>

namespace mlir::qir {

#define GEN_PASS_DEF_QIRCLEANUPPASS
#include "mqt/Dialect/QIR/Transforms/Passes.h.inc"

namespace {

struct QIRCleanupPass final : impl::QIRCleanupPassBase<QIRCleanupPass> {
protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    // The cleanup canonicalizer skips region simplification, whose dead-value
    // analysis dominates large programs; erasing unreachable blocks suffices.
    IRRewriter rewriter(moduleOp.getContext());
    for (auto function : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
      std::ignore = eraseUnreachableBlocks(rewriter, function.getBody());
    }
    removeUnusedDeclarations(moduleOp);
  }
};

} // namespace
} // namespace mlir::qir
