/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qc {

/// Build a reference-semantic loop with early exits, then lift its CFG to SCF.
/// All exit and continuation edges carry the current values of the same slots.
class LoopBuilder {
public:
  LoopBuilder(OpBuilder& builder, Location location, ValueRange initialState);
  ValueRange arguments();
  void enterBody(Value condition, ValueRange state);
  void branch(bool continuing, ValueRange state);
  FailureOr<SmallVector<Value>> finish();

private:
  OpBuilder& builder;
  Location location;
  scf::ExecuteRegionOp regionOp;
  Block* header;
  Block* decision;
  Block* exit;
};

} // namespace mlir::qc
