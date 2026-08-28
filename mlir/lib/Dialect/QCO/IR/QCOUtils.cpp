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

#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MQT/IR/MQTDialect.h>
#include <mlir/Dialect/QCO/IR/QCODialect.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>

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
  SmallVector<Operation*> operations{root};
  for (size_t next = 0; next < operations.size(); ++next) {
    auto* op = operations[next];
    if (auto staticOp = dyn_cast<StaticOp>(op)) {
      if (entryPoint &&
          (entryPoint.isDeclaration() ||
           staticOp->getBlock() != &entryPoint.getBody().front())) {
        staticOp.emitError()
            << "expected static qubits in the entry block of program entry "
               "function @"
            << entryPoint.getSymName();
        return failure();
      }
      if (!staticIndices.insert(staticOp.getIndex()).second) {
        staticOp.emitError()
            << "expected each static qubit index to identify one linear "
               "value, but found duplicate index "
            << staticOp.getIndex();
        return failure();
      }
    }
    for (auto result : op->getResults()) {
      if (failed(verifyLinearValue(result))) {
        return failure();
      }
    }
    for (Region& region : op->getRegions()) {
      for (Block& block : region) {
        for (auto argument : block.getArguments()) {
          if (failed(verifyLinearValue(argument))) {
            return failure();
          }
        }
        for (auto& nestedOp : block) {
          operations.push_back(&nestedOp);
        }
      }
    }
  }
  return success();
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

bool hasComposableBodyMatrix(Block& block, size_t numTargets) {
  if (!isModifierMatrixSizeSupported(numTargets) ||
      block.getNumArguments() != numTargets ||
      block.getTerminator()->getNumOperands() != numTargets) {
    return false;
  }

  if (auto sole = mqt::getSoleBodyUnitary<UnitaryOpInterface>(block);
      sole && sole.getNumQubits() > 2) {
    if (sole.getNumQubits() != numTargets ||
        !sole.hasCompileTimeKnownUnitaryMatrix()) {
      return false;
    }
    const auto inputsMatch =
        llvm::all_of(llvm::enumerate(sole.getInputQubits()), [&](auto indexed) {
          return indexed.value() == block.getArgument(indexed.index());
        });
    const auto outputsMatch = llvm::all_of(
        llvm::zip_equal(sole.getOutputQubits(),
                        block.getTerminator()->getOperands()),
        [](auto pair) { return std::get<0>(pair) == std::get<1>(pair); });
    return inputsMatch && outputsMatch;
  }

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
            .Case<GPhaseOp>([](GPhaseOp gphase) {
              return cast<UnitaryOpInterface>(gphase.getOperation())
                  .hasCompileTimeKnownUnitaryMatrix();
            })
            .Case<UnitaryOpInterface>([&](UnitaryOpInterface unitary) {
              if (unitary.getNumQubits() == 0 || unitary.getNumQubits() > 2 ||
                  !unitary.hasCompileTimeKnownUnitaryMatrix() ||
                  llvm::any_of(unitary.getInputQubits(), [&](Value input) {
                    return !wireIds.contains(input);
                  })) {
                return false;
              }
              propagateWireIds(unitary, wireIds);
              return true;
            })
            .Default([&](Operation* unknown) {
              const auto usesQubit = [](Value value) {
                return isLinearQubitType(value.getType());
              };
              return !mqt::containsUnitaryOperation<UnitaryOpInterface>(
                         unknown) &&
                     !llvm::any_of(unknown->getOperands(), usesQubit) &&
                     !llvm::any_of(unknown->getResults(), usesQubit);
            });
    if (!handled) {
      return false;
    }
  }

  for (auto [index, yielded] :
       llvm::enumerate(block.getTerminator()->getOperands())) {
    const auto wire = lookupWireId(wireIds, yielded);
    if (!wire.has_value() || *wire != index) {
      return false;
    }
  }
  return true;
}

std::optional<DynamicMatrix> composeBodyMatrix(Block& block,
                                               size_t numTargets) {
  if (!hasComposableBodyMatrix(block, numTargets)) {
    return std::nullopt;
  }

  if (auto sole = mqt::getSoleBodyUnitary<UnitaryOpInterface>(block);
      sole && sole.getNumQubits() > 2 && sole.getNumQubits() == numTargets) {
    const auto inputsMatch =
        llvm::all_of(llvm::enumerate(sole.getInputQubits()), [&](auto indexed) {
          return indexed.value() == block.getArgument(indexed.index());
        });
    const auto outputsMatch = llvm::all_of(
        llvm::zip_equal(sole.getOutputQubits(),
                        block.getTerminator()->getOperands()),
        [](auto pair) { return std::get<0>(pair) == std::get<1>(pair); });
    if (!inputsMatch || !outputsMatch) {
      return std::nullopt;
    }
    auto matrix = sole.getUnitaryMatrix<DynamicMatrix>();
    const auto expectedDim = static_cast<int64_t>(1ULL << numTargets);
    if (!matrix || matrix->rows() != expectedDim ||
        matrix->cols() != expectedDim) {
      return std::nullopt;
    }
    return matrix;
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
              propagateWireIds(unitary, wireIds);
              return true;
            })
            .Default([&](Operation* unknown) {
              const auto usesQubit = [](Value value) {
                return isLinearQubitType(value.getType());
              };
              return !mqt::containsUnitaryOperation<UnitaryOpInterface>(
                         unknown) &&
                     !llvm::any_of(unknown->getOperands(), usesQubit) &&
                     !llvm::any_of(unknown->getResults(), usesQubit);
            });

    if (!handled) {
      return std::nullopt;
    }
  }

  for (auto [index, yielded] :
       llvm::enumerate(block.getTerminator()->getOperands())) {
    const auto wire = lookupWireId(wireIds, yielded);
    if (!wire.has_value() || *wire != index) {
      return std::nullopt;
    }
  }
  if (!acc.has_value()) {
    acc = DynamicMatrix::identity(static_cast<int64_t>(1ULL << numTargets));
  }
  *acc *= global;
  return acc;
}

} // namespace mlir::qco
