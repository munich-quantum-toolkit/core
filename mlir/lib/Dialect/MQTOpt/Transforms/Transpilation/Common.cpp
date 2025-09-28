/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace mqt::ir::opt {
bool isEntryPoint(mlir::func::FuncOp op) {
  const auto passthroughAttr =
      op->getAttrOfType<mlir::ArrayAttr>(PASSTHROUGH_ATTR);
  if (!passthroughAttr) {
    return false;
  }

  return llvm::any_of(passthroughAttr, [](const mlir::Attribute attr) {
    return isa<mlir::StringAttr>(attr) &&
           cast<mlir::StringAttr>(attr) == ENTRY_POINT_ATTR;
  });
}
} // namespace mqt::ir::opt
