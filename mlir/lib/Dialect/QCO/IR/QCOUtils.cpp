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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/QCO/IR/QCODialect.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <optional>

namespace mlir::qco {

[[nodiscard]] static std::optional<std::size_t>
lookupWireId(const DenseMap<Value, std::size_t>& wireIds, Value wire) {
  if (const auto it = wireIds.find(wire); it != wireIds.end()) {
    return it->second;
  }
  return std::nullopt;
}

static void propagateWireIds(UnitaryOpInterface unitary,
                             DenseMap<Value, std::size_t>& wireIds) {
  for (Value input : unitary.getInputQubits()) {
    if (const auto wire = lookupWireId(wireIds, input)) {
      wireIds[unitary.getOutputForInput(input)] = *wire;
    }
  }
}

[[nodiscard]] static std::optional<DynamicMatrix>
embedUnitaryInBody(UnitaryOpInterface unitary, std::size_t numTargets,
                   const DenseMap<Value, std::size_t>& wireIds) {
  const auto numOpQubits = unitary.getNumQubits();
  if (numOpQubits == 0 || numOpQubits > 2) {
    return std::nullopt;
  }

  if (numOpQubits == 1) {
    const auto wire = lookupWireId(wireIds, unitary.getInputQubit(0));
    if (!wire.has_value()) {
      return std::nullopt;
    }
    if (numTargets == 1) {
      const auto matrix = unitary.getUnitaryMatrix<DynamicMatrix>();
      if (!matrix || matrix->rows() != Matrix2x2::K_ROWS) {
        return std::nullopt;
      }
      return *matrix;
    }
    const auto matrix = unitary.getUnitaryMatrix<Matrix2x2>();
    if (!matrix) {
      return std::nullopt;
    }
    return matrix->embedInNqubit(numTargets, *wire);
  }

  const auto q0 = lookupWireId(wireIds, unitary.getInputQubit(0));
  const auto q1 = lookupWireId(wireIds, unitary.getInputQubit(1));
  if (!q0.has_value() || !q1.has_value()) {
    return std::nullopt;
  }
  if (numTargets == 2 && *q0 == 0 && *q1 == 1) {
    const auto matrix = unitary.getUnitaryMatrix<DynamicMatrix>();
    if (!matrix || matrix->rows() != Matrix4x4::K_ROWS) {
      return std::nullopt;
    }
    return *matrix;
  }
  const auto matrix = unitary.getUnitaryMatrix<Matrix4x4>();
  if (!matrix) {
    return std::nullopt;
  }
  return matrix->embedInNqubit(numTargets, *q0, *q1);
}

std::optional<DynamicMatrix> composeNTargetBodyMatrix(Block& block,
                                                      std::size_t numTargets) {
  if (numTargets == 0 || numTargets > kMaxModifierTargetQubits ||
      block.getNumArguments() != numTargets) {
    return std::nullopt;
  }

  std::optional<DynamicMatrix> acc;
  Complex global{1.0, 0.0};
  bool found = false;

  DenseMap<Value, std::size_t> wireIds;
  for (std::size_t i = 0; i < numTargets; ++i) {
    wireIds[block.getArgument(i)] = i;
  }

  for (Operation& op : block.without_terminator()) {
    const bool handled =
        TypeSwitch<Operation*, bool>(&op)
            .Case<BarrierOp>([&](BarrierOp barrier) {
              propagateWireIds(barrier, wireIds);
              return true;
            })
            .Case<GPhaseOp>([&](GPhaseOp gphase) {
              const auto matrix = gphase.getUnitaryMatrix();
              if (!matrix) {
                return false;
              }
              global *= matrix->value;
              found = true;
              return true;
            })
            .Case<UnitaryOpInterface>([&](UnitaryOpInterface unitary) {
              if (!unitary.hasCompileTimeKnownUnitaryMatrix()) {
                return false;
              }
              std::optional<DynamicMatrix> step =
                  embedUnitaryInBody(unitary, numTargets, wireIds);
              if (!step.has_value()) {
                return false;
              }
              if (acc.has_value()) {
                acc->premultiplyBy(std::move(*step));
              } else {
                acc = std::move(step);
              }
              found = true;
              propagateWireIds(unitary, wireIds);
              return true;
            })
            .Default([&](Operation* unknown) {
              const auto usesQubit = [](Value value) {
                return isa<QubitType>(value.getType());
              };
              return !llvm::any_of(unknown->getOperands(), usesQubit) &&
                     !llvm::any_of(unknown->getResults(), usesQubit);
            });

    if (!handled) {
      return std::nullopt;
    }
  }

  if (!found) {
    return std::nullopt;
  }
  if (!acc.has_value()) {
    acc =
        DynamicMatrix::identity(static_cast<std::int64_t>(1ULL << numTargets));
  }
  *acc *= global;
  return acc;
}

} // namespace mlir::qco
