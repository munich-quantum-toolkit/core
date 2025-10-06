/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/IR/FluxDialect.h"

#include "mlir/Dialect/Flux/IR/FluxOps.h"
#include "mlir/Dialect/Flux/IR/FluxTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::flux;

#include "mlir/Dialect/Flux/IR/FluxOpsDialect.cpp.inc"

void FluxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Flux/IR/FluxOps.cpp.inc"
      >();
  registerTypes();
}

void FluxDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Flux/IR/FluxOpsTypes.cpp.inc"
      >();
}
