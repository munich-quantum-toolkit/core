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
  for (auto [input, output] :
       llvm::zip_equal(unitary.getInputQubits(), unitary.getOutputQubits())) {
    if (const auto wire = lookupWireId(wireIds, input)) {
      wireIds[output] = *wire;
    }
  }
}

/// Returns the @p unitary embedded on @p numTargets modifier wires using @p
/// wireIds.
[[nodiscard]] static std::optional<DynamicMatrix>
embedUnitaryInBody(UnitaryOpInterface unitary, size_t numTargets,
                   const DenseMap<Value, size_t>& wireIds) {
  const auto numOpQubits = unitary.getNumQubits();
  if (numOpQubits == 0 || numOpQubits > 2) {
    return std::nullopt;
  }

  if (numOpQubits == 1) {
    const auto wire = lookupWireId(wireIds, unitary.getInputQubit(0));
    if (!wire.has_value()) {
      return std::nullopt;
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
  const auto matrix = unitary.getUnitaryMatrix<Matrix4x4>();
  if (!matrix) {
    return std::nullopt;
  }
  return matrix->embedInNqubit(numTargets, *q0, *q1);
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
              auto embedded = embedUnitaryInBody(unitary, numTargets, wireIds);
              if (!embedded.has_value()) {
                return false;
              }
              if (!acc.has_value()) {
                acc.swap(embedded);
              } else {
                acc->premultiplyBy(*embedded);
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
