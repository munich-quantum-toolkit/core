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
#include "mqt/Support/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

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

void populateQIRCleanupPipeline(mlir::OpPassManager& pm,
                                mlir::QIRProfile profile) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createMem2Reg());
  pm.addPass(mlir::qir::createQIRCleanupPass());
  pm.addPass(mlir::createRemoveDeadValuesPass());
  pm.addPass(mlir::qir::createQIRSetAttributesAndMetadata({profile}));
}
