/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/QCOUtils.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/QCO/IR/QCODialect.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <optional>

namespace mlir::qco {

std::optional<Matrix2x2> composeSingleQubitBodyMatrix(Block& block) {
  Matrix2x2 acc = Matrix2x2::identity();
  Complex global{1.0, 0.0};
  bool found = false;
  for (Operation& op : block.without_terminator()) {
    if (!TypeSwitch<Operation*, bool>(&op)
             .Case<BarrierOp>([](auto) { return true; })
             .Case<GPhaseOp>([&](GPhaseOp gphase) {
               const auto matrix = gphase.getUnitaryMatrix();
               if (!matrix) {
                 return false;
               }
               global *= (*matrix)(0, 0);
               return true;
             })
             .Case<UnitaryOpInterface>([&](UnitaryOpInterface unitary) {
               Matrix2x2 matrix;
               if (!unitary.getUnitaryMatrix2x2(matrix)) {
                 return false;
               }
               acc.premultiplyBy(matrix);
               found = true;
               return true;
             })
             .Default([](Operation* operation) {
               const auto usesQubit = [](Value value) {
                 return isa<QubitType>(value.getType());
               };
               return !llvm::any_of(operation->getOperands(), usesQubit) &&
                      !llvm::any_of(operation->getResults(), usesQubit);
             })) {
      return std::nullopt;
    }
  }
  if (!found && std::abs(global - Complex{1.0, 0.0}) <= utils::TOLERANCE) {
    return std::nullopt;
  }
  acc *= global;
  return acc;
}

} // namespace mlir::qco
