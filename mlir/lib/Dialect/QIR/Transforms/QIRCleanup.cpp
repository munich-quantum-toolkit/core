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

[[nodiscard]] static StringAttr getMetadataKey(const Attribute attr) {
  auto pair = dyn_cast<ArrayAttr>(attr);
  if (!pair || pair.size() != 2) {
    return {};
  }
  auto key = dyn_cast<StringAttr>(pair[0]);
  if (!key || !isa<StringAttr>(pair[1])) {
    return {};
  }
  return key;
}

[[nodiscard]] static StringRef getCalleeName(LLVM::CallOp callOp) {
  auto calleeAttr = callOp.getCalleeAttr();
  auto flatRef = dyn_cast_or_null<FlatSymbolRefAttr>(calleeAttr);
  if (!flatRef) {
    return {};
  }
  return flatRef.getValue();
}

[[nodiscard]] static bool moduleHasDynamicQubitRuntimeCalls(ModuleOp module) {
  return llvm::any_of(module.getOps<LLVM::CallOp>(), [](LLVM::CallOp callOp) {
    const auto callee = getCalleeName(callOp);
    return callee == QIR_QUBIT_ALLOC || callee == QIR_QUBIT_ARRAY_ALLOC;
  });
}

[[nodiscard]] static bool moduleHasDynamicResultRuntimeCalls(ModuleOp module) {
  return llvm::any_of(module.getOps<LLVM::CallOp>(), [](LLVM::CallOp callOp) {
    const auto callee = getCalleeName(callOp);
    return callee == QIR_RESULT_ALLOC || callee == QIR_RESULT_ARRAY_ALLOC;
  });
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

static void normalizeQIRMetadata(ModuleOp module) {
  auto main = getMainFunction(module);
  if (!main) {
    return;
  }

  auto passthroughAttr = main->getAttrOfType<ArrayAttr>("passthrough");
  if (!passthroughAttr) {
    return;
  }

  const bool hasDynamicQubit = moduleHasDynamicQubitRuntimeCalls(module);
  const bool hasDynamicResult = moduleHasDynamicResultRuntimeCalls(module);
  if (hasDynamicQubit && hasDynamicResult) {
    return;
  }

  OpBuilder builder(module.getContext());
  SmallVector<Attribute> updatedMetadata;
  updatedMetadata.reserve(passthroughAttr.size());

  for (const auto attr : passthroughAttr) {
    const auto key = getMetadataKey(attr);
    if (!key) {
      updatedMetadata.push_back(attr);
      continue;
    }

    if (key.getValue() == "dynamic_qubit_management" && !hasDynamicQubit) {
      continue;
    }
    if (key.getValue() == "dynamic_result_management" && !hasDynamicResult) {
      continue;
    }

    updatedMetadata.push_back(attr);
  }

  main->setAttr("passthrough", builder.getArrayAttr(updatedMetadata));
}

namespace {

/**
 * @brief Remove matching allocation-release pairs of qubit arrays.
 * @details Matches an unused
 * `__quantum__rt__qubit_array_allocate`-`__quantum__rt__qubit_array_release`
 * pair on the same stack slot.
 */
struct RemoveDeadQubitArrayPair final : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  [[nodiscard]] static bool haveEqualIntegerValues(Value lhs, Value rhs) {
    if (lhs == rhs) {
      return true;
    }
    auto lhsConstant = lhs.getDefiningOp<LLVM::ConstantOp>();
    auto rhsConstant = rhs.getDefiningOp<LLVM::ConstantOp>();
    if (!lhsConstant || !rhsConstant) {
      return false;
    }
    const auto lhsValue = dyn_cast<IntegerAttr>(lhsConstant.getValue());
    const auto rhsValue = dyn_cast<IntegerAttr>(rhsConstant.getValue());
    return lhsValue && rhsValue && lhsValue == rhsValue;
  }

  LogicalResult matchAndRewrite(LLVM::CallOp releaseCall,
                                PatternRewriter& rewriter) const override {
    if (getCalleeName(releaseCall) != QIR_QUBIT_ARRAY_RELEASE) {
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
          callOp.getOperand(1) != allocaOp.getResult() ||
          !callOp.getOperand(2).getDefiningOp<LLVM::ZeroOp>()) {
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
    if (allocCall->getBlock() != releaseCall->getBlock() ||
        !allocCall->isBeforeInBlock(releaseCall) ||
        !haveEqualIntegerValues(allocCall.getOperand(0),
                                releaseCall.getOperand(0)) ||
        !haveEqualIntegerValues(allocCall.getOperand(0),
                                allocaOp.getArraySize())) {
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

/**
 * @brief Clean up QIR.
 * @details Removes dead allocation-release pairs of qubit arrays, drops unused
 * external declarations, and normalizes QIR metadata.
 */
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
    normalizeQIRMetadata(module);
  }
};

} // namespace

} // namespace mlir::qir
