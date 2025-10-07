/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace mlir {
using namespace mlir::quartz;

class QuartzProgramBuilder {
public:
  explicit QuartzProgramBuilder(MLIRContext* context);

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  void initialize();

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  // Dynamic allocation
  Value allocQubit();

  // Static qubit reference
  Value qubit(size_t index);

  /// TODO

  //===--------------------------------------------------------------------===//
  // Finalization
  //===--------------------------------------------------------------------===//

  ModuleOp finalize();
  ModuleOp getModule() const;

private:
  OpBuilder builder;
  ModuleOp module;
  Location loc;
};
} // namespace mlir
