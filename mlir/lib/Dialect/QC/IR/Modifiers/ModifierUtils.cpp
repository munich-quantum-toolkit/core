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

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/RegionUtils.h>

namespace mlir::qc::detail {

LogicalResult verifyModifierBody(Operation* modifierOp, Block& body) {
  const auto hasNonUnitaryOperation =
      body.walk([](Operation* operation) {
            return isa<AllocOp, DeallocOp, StaticOp, MeasureOp, ResetOp,
                       memref::LoadOp, memref::StoreOp>(operation)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted();
  if (hasNonUnitaryOperation) {
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

void getModifierSuccessorRegions(Operation* modifierOp, Region& body,
                                 RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor>& regions) {
  if (point.isParent()) {
    regions.emplace_back(&body, body.getArguments());
    return;
  }
  // QC modifiers have no results, so nothing is forwarded back out.
  regions.emplace_back(modifierOp, modifierOp->getResults());
}

} // namespace mlir::qc::detail
