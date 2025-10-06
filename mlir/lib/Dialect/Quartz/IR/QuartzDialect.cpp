/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include "mlir/Dialect/Quartz/IR/QuartzOps.h"
#include "mlir/Dialect/Quartz/IR/QuartzTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::quartz;

#include "mlir/Dialect/Quartz/IR/QuartzOpsDialect.cpp.inc"

void QuartzDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quartz/IR/QuartzOps.cpp.inc"
      >();
  registerTypes();
}

void QuartzDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Quartz/IR/QuartzOpsTypes.cpp.inc"
      >();
}
