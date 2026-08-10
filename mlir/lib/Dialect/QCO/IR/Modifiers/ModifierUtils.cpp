/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ModifierUtils.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/RegionUtils.h>

namespace mlir::qco::detail {

LogicalResult verifyModifierBody(Operation* modifierOp, Block& body) {
  if (llvm::any_of(body, [](Operation& operation) {
        return isa<AllocOp, SinkOp, StaticOp, MeasureOp, ResetOp,
                   qtensor::ExtractOp, qtensor::InsertOp>(operation);
      })) {
    return modifierOp->emitOpError(
        "body must not contain non-unitary quantum operations or modify a "
        "quantum register");
  }

  SetVector<Value> captures;
  getUsedValuesDefinedAbove(modifierOp->getRegions(), captures);
  if (llvm::any_of(captures, [](const Value value) {
        return isa<QubitType>(value.getType());
      })) {
    return modifierOp->emitOpError(
        "body must not capture qubits from above; use only its aliased block "
        "arguments");
  }

  return success();
}

} // namespace mlir::qco::detail
