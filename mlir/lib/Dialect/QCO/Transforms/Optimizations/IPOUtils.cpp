/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "IPOUtils.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qco {

func::FuncOp copyFunction(func::FuncOp funcOp, StringRef newName) {
  auto newFunc = funcOp.clone();
  newFunc.setName(newName.str());
  return newFunc;
}

} // namespace mlir::qco
