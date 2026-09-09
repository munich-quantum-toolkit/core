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
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MQT/IR/MQTDialect.h>
#include <mlir/Dialect/QCO/IR/QCODialect.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::qco {

[[nodiscard]] static LogicalResult verifyLinearValue(Value value) {
  if (!isLinearQubitType(value.getType()) || value.hasOneUse()) {
    return success();
  }
  return emitError(value.getLoc())
         << "expected linear QCO value to have exactly one use, but found "
         << value.getNumUses();
}

LogicalResult verifyLinearity(Operation* root) {
  func::FuncOp entryPoint;
  if (auto moduleOp = dyn_cast<ModuleOp>(root)) {
    entryPoint = mqt::getEntryPoint(moduleOp);
  }

  DenseSet<uint64_t> staticIndices;
  const auto walkResult = root->walk([&](Operation* op) {
    if (auto staticOp = dyn_cast<StaticOp>(op)) {
      if (entryPoint &&
          (entryPoint.isDeclaration() ||
           staticOp->getBlock() != &entryPoint.getBody().front())) {
        staticOp.emitError()
            << "expected static qubits in the entry block of program entry "
               "function @"
            << entryPoint.getSymName();
        return WalkResult::interrupt();
      }
      if (!staticIndices.insert(staticOp.getIndex()).second) {
        staticOp.emitError()
            << "expected each static qubit index to identify one linear "
               "value, but found duplicate index "
            << staticOp.getIndex();
        return WalkResult::interrupt();
      }
    }
    for (auto result : op->getResults()) {
      if (failed(verifyLinearValue(result))) {
        return WalkResult::interrupt();
      }
    }
    for (Region& region : op->getRegions()) {
      for (Block& block : region) {
        for (auto argument : block.getArguments()) {
          if (failed(verifyLinearValue(argument))) {
            return WalkResult::interrupt();
          }
        }
      }
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

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

/// Embed the first unitary, then premultiply subsequent gates in place.
[[nodiscard]] static bool
premultiplyUnitaryInBody(UnitaryOpInterface unitary, size_t numTargets,
                         const DenseMap<Value, size_t>& wireIds,
                         std::optional<DynamicMatrix>& acc) {
  const auto numOpQubits = unitary.getNumQubits();
  if (numOpQubits == 0 || numOpQubits > 2) {
    if (numOpQubits != numTargets ||
        !llvm::all_of(
            llvm::enumerate(unitary.getInputQubits()), [&](auto entry) {
              return lookupWireId(wireIds, entry.value()) == entry.index();
            })) {
      return false;
    }
    auto matrix = unitary.getUnitaryMatrix<DynamicMatrix>();
    if (!matrix) {
      return false;
    }
    if (acc) {
      acc->premultiplyBy(*matrix);
    } else {
      acc.swap(matrix);
    }
    return true;
  }

  if (numOpQubits == 1) {
    const auto wire = lookupWireId(wireIds, unitary.getInputQubit(0));
    if (!wire.has_value()) {
      return false;
    }
    const auto matrix = unitary.getUnitaryMatrix<Matrix2x2>();
    if (!matrix) {
      return false;
    }
    if (acc) {
      acc->premultiplyByEmbedded1Q(*matrix, numTargets, *wire);
    } else {
      acc = matrix->embedInNqubit(numTargets, *wire);
    }
    return true;
  }

  const auto q0 = lookupWireId(wireIds, unitary.getInputQubit(0));
  const auto q1 = lookupWireId(wireIds, unitary.getInputQubit(1));
  if (!q0.has_value() || !q1.has_value()) {
    return false;
  }
  const auto matrix = unitary.getUnitaryMatrix<Matrix4x4>();
  if (!matrix) {
    return false;
  }
  if (!acc) {
    acc = matrix->embedInNqubit(numTargets, *q0, *q1);
  } else if (numTargets == 2) {
    acc->premultiplyByEmbedded2Q(matrix->reorderForQubits(*q0, *q1), 2, 0, 1);
  } else {
    acc->premultiplyByEmbedded2Q(*matrix, numTargets, *q0, *q1);
  }
  return true;
}

std::optional<DynamicMatrix> composeBodyMatrix(Block& block,
                                               size_t numTargets) {
  if (numTargets > kMaxModifierTargetQubits ||
      block.getNumArguments() != numTargets) {
    return std::nullopt;
  }

  std::optional<DynamicMatrix> acc;
  Complex global{1.0, 0.0};

  DenseMap<Value, size_t> wireIds;
  for (size_t i = 0; i < numTargets; ++i) {
    wireIds[block.getArgument(i)] = i;
  }

  for (Operation& op : block.without_terminator()) {
    const bool handled =
        TypeSwitch<Operation*, bool>(&op)
            .Case([&](BarrierOp barrier) {
              propagateWireIds(barrier, wireIds);
              return true;
            })
            .Case([&](GPhaseOp gphase) {
              const auto matrix = gphase.getUnitaryMatrix();
              if (!matrix) {
                return false;
              }
              global *= matrix->value;
              return true;
            })
            .Case([&](UnitaryOpInterface unitary) {
              if (!premultiplyUnitaryInBody(unitary, numTargets, wireIds,
                                            acc)) {
                return false;
              }
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

  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getTargets().size() != numTargets ||
      !llvm::all_of(llvm::enumerate(yield.getTargets()), [&](auto entry) {
        return lookupWireId(wireIds, entry.value()) == entry.index();
      })) {
    return std::nullopt;
  }
  if (!acc.has_value()) {
    acc = DynamicMatrix::identity(static_cast<int64_t>(1ULL << numTargets));
  }
  *acc *= global;
  return acc;
}

} // namespace mlir::qco
