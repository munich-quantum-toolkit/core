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
#include <cstdint>
#include <optional>

namespace mlir::qco {

/// Returns the wire index for @p wire in @p wireIds, or `std::nullopt` if
/// untracked.
[[nodiscard]] static std::optional<size_t>
lookupWireId(const DenseMap<Value, size_t>& wireIds, Value wire) {
  if (const auto it = wireIds.find(wire); it != wireIds.end()) {
    return it->second;
  }
  return std::nullopt;
}

/// Propagates wire indices from unitary inputs to outputs via @p wireIds.
static void propagateWireIds(UnitaryOpInterface unitary,
                             DenseMap<Value, size_t>& wireIds) {
  for (Value input : unitary.getInputQubits()) {
    if (const auto wire = lookupWireId(wireIds, input)) {
      wireIds[unitary.getOutputForInput(input)] = *wire;
    }
  }
}

/// Embeds a compile-time 1Q/2Q @p unitary into @p acc on @p numTargets modifier
/// wires using @p wireIds.
static bool applyUnitaryInBody(UnitaryOpInterface unitary, size_t numTargets,
                               const DenseMap<Value, size_t>& wireIds,
                               DynamicMatrix& acc) {
  const auto numOpQubits = unitary.getNumQubits();
  if (numOpQubits == 0 || numOpQubits > 2) {
    return false;
  }

  if (numOpQubits == 1) {
    const auto wire = lookupWireId(wireIds, unitary.getInputQubit(0));
    if (!wire.has_value()) {
      return false;
    }
    acc.premultiplyByEmbedded1Q(*unitary.getUnitaryMatrix<Matrix2x2>(),
                                numTargets, *wire);
    return true;
  }

  const auto q0 = lookupWireId(wireIds, unitary.getInputQubit(0));
  const auto q1 = lookupWireId(wireIds, unitary.getInputQubit(1));
  if (!q0.has_value() || !q1.has_value()) {
    return false;
  }
  Matrix4x4 gate = *unitary.getUnitaryMatrix<Matrix4x4>();
  if (numTargets == 2) {
    gate = gate.reorderForQubits(*q0, *q1);
  }
  acc.premultiplyByEmbedded2Q(gate, numTargets, numTargets == 2 ? 0 : *q0,
                              numTargets == 2 ? 1 : *q1);
  return true;
}

std::optional<DynamicMatrix> composeBodyMatrix(Block& block,
                                               size_t numTargets) {
  if (numTargets == 0 || numTargets > kMaxModifierTargetQubits ||
      block.getNumArguments() != numTargets) {
    return std::nullopt;
  }

  std::optional<DynamicMatrix> acc;
  Complex global{1.0, 0.0};
  bool found = false;

  DenseMap<Value, size_t> wireIds;
  for (size_t i = 0; i < numTargets; ++i) {
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
              if (!acc.has_value()) {
                acc = DynamicMatrix::identity(
                    static_cast<int64_t>(1ULL << numTargets));
              }
              if (!applyUnitaryInBody(unitary, numTargets, wireIds, *acc)) {
                return false;
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
    acc = DynamicMatrix::identity(static_cast<int64_t>(1ULL << numTargets));
  }
  *acc *= global;
  return acc;
}

} // namespace mlir::qco
