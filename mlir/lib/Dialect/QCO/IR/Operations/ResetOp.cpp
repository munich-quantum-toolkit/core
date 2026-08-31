/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <mlir/IR/OperationSupport.h>

using namespace mlir;
using namespace mlir::qco;

OpFoldResult ResetOp::fold(FoldAdaptor /*adaptor*/) {
  if (getQubitIn().getDefiningOp<AllocOp>()) {
    return getQubitIn();
  }

  return {};
}
