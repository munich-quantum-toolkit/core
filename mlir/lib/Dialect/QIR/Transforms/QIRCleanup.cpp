/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

namespace mlir::qir {

#define GEN_PASS_DEF_QIRCLEANUPPASS
#include "mlir/Dialect/QIR/Transforms/Passes.h.inc"

[[nodiscard]] static StringRef getCalleeName(LLVM::CallOp callOp) {
  auto calleeAttr = callOp.getCalleeAttr();
  auto flatRef = calleeAttr;
  if (!flatRef) {
    return {};
  }
  return flatRef.getValue();
}

static void dropUnusedExternalDeclarations(ModuleOp module) {
  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<LLVM::LLVMFuncOp>())) {
    if (!funcOp.isExternal()) {
      continue;
    }
    if (!SymbolTable::symbolKnownUseEmpty(funcOp, module)) {
      continue;
    }
    funcOp.erase();
  }
}

namespace {

/// Remove matching allocation-release pairs of qubit arrays.
///
/// Matches an unused
/// `__quantum__rt__qubit_array_allocate`-`__quantum__rt__qubit_array_release`
/// pair on the same stack slot.
struct RemoveDeadQubitArrayPair final : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp releaseCall,
                                PatternRewriter& rewriter) const override {
    if (getCalleeName(releaseCall) != QIR_QUBIT_ARRAY_RELEASE ||
        releaseCall.getNumOperands() < 2) {
      return failure();
    }

    auto allocaOp = releaseCall.getOperand(1).getDefiningOp<LLVM::AllocaOp>();
    if (!allocaOp) {
      return failure();
    }

    LLVM::CallOp allocCall = nullptr;
    for (Operation* user : allocaOp.getResult().getUsers()) {
      auto callOp = dyn_cast<LLVM::CallOp>(user);
      if (!callOp) {
        return failure();
      }

      if (callOp == releaseCall) {
        continue;
      }

      if (getCalleeName(callOp) != QIR_QUBIT_ARRAY_ALLOC ||
          callOp.getNumOperands() < 2 ||
          callOp.getOperand(1) != allocaOp.getResult()) {
        return failure();
      }
      if (allocCall != nullptr) {
        return failure();
      }
      allocCall = callOp;
    }

    if (!allocCall) {
      return failure();
    }

    rewriter.eraseOp(releaseCall);
    rewriter.eraseOp(allocCall);
    if (allocaOp->use_empty()) {
      rewriter.eraseOp(allocaOp);
    }
    return success();
  }
};

/// Clean up QIR.
///
/// Removes dead allocation-release pairs of qubit arrays, drops unused
/// external declarations.
struct QIRCleanupPass final : impl::QIRCleanupPassBase<QIRCleanupPass> {
protected:
  void runOnOperation() override {
    auto module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveDeadQubitArrayPair>(&getContext());

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    dropUnusedExternalDeclarations(module);
  }
};

} // namespace

} // namespace mlir::qir
