/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/Passes.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

void runCanonicalizationPasses(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createRemoveDeadValuesPass());
  if (pm.run(module).failed()) {
    llvm::errs() << "Failed to run canonicalization passes.\n";
  }
}
