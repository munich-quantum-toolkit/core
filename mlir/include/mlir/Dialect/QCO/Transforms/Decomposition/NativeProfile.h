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

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>

namespace mlir::qco::decomposition {

/**
 * @brief Native gate kinds that may appear in a two-qubit synthesis menu.
 */
enum class NativeGateKind : std::uint8_t {
  U,
  X,
  Sx,
  Rz,
  Rx,
  Ry,
  R,
  Cx,
  Cz,
  Rzz,
};

/**
 * @brief Single-qubit emission strategy resolved from a native-gate menu.
 */
enum class SingleQubitMode : std::uint8_t {
  ZSXX,     ///< `RZ` / `SX` / `X` via ZYZ decomposition.
  U3,       ///< Generic `U(theta, phi, lambda)`.
  R,        ///< `R(theta, phi)` chain (`Rx`/`Ry` as `R`).
  AxisPair, ///< Two fixed rotation axes (see @ref AxisPair).
};

/**
 * @brief Rotation-axis pair for @ref SingleQubitMode::AxisPair emitters.
 */
enum class AxisPair : std::uint8_t {
  RxRz,
  RxRy,
  RyRz,
};

/**
 * @brief Entangling basis gate for two-qubit Weyl synthesis.
 */
enum class EntanglerBasis : std::uint8_t {
  None,
  Cx,
  Cz,
};

struct SingleQubitEmitterSpec {
  SingleQubitMode mode = SingleQubitMode::U3;
  AxisPair axisPair = AxisPair::RxRz;
  bool supportsDirectRx = false;
};

/**
 * @brief Resolved native-gate menu for two-qubit Weyl synthesis.
 */
struct NativeProfileSpec {
  bool allowRzz = false;
  llvm::DenseSet<NativeGateKind> allowedGates;
  llvm::SmallVector<SingleQubitEmitterSpec> singleQubitEmitters;
  llvm::SmallVector<EntanglerBasis> entanglerBases;
};

/**
 * @brief Euler basis used to emit single-qubit factors for @p emitter.
 */
[[nodiscard]] EulerBasis
emitterEulerBasis(const SingleQubitEmitterSpec& emitter);

/**
 * @brief Parses a comma-separated native-gate menu (e.g. `"u,cx,rzz"`).
 */
[[nodiscard]] std::optional<NativeProfileSpec>
parseNativeSpec(StringRef nativeGates);

/**
 * @brief Synthesizes a composed two-qubit unitary as gates in @p spec.
 */
[[nodiscard]] LogicalResult
synthesizeUnitary2QWeyl(OpBuilder& builder, Location loc, Value qubit0,
                        Value qubit1, const Matrix4x4& target,
                        const NativeProfileSpec& spec, Value& outQubit0,
                        Value& outQubit1);

/**
 * @brief Number of entangling basis gates required to synthesize @p target.
 *
 * @return Entangler count for @p spec, or `std::nullopt` if synthesis fails.
 */
[[nodiscard]] std::optional<std::uint8_t>
twoQubitEntanglerCount(const Matrix4x4& target, const NativeProfileSpec& spec);

/**
 * @brief Maps a compile-time single-qubit op to its native-menu gate kind.
 *
 * Returns `std::nullopt` for non-primitive or unsupported ops.
 */
[[nodiscard]] std::optional<NativeGateKind>
nativeGateKindFor(UnitaryOpInterface op);

/**
 * @brief Returns true when @p op is already on the resolved single-qubit menu.
 *
 * `BarrierOp` and `GPhaseOp` are always allowed.
 */
[[nodiscard]] bool allowsSingleQubitOp(UnitaryOpInterface op,
                                       const NativeProfileSpec& spec);

/**
 * @brief Entangler basis for a 1-control, 1-target `CtrlOp` with `X`/`Z` body.
 */
[[nodiscard]] std::optional<EntanglerBasis>
entanglerBasisForSingleTargetCtrl(CtrlOp ctrl);

/** @brief Returns true when @p spec lists @p basis as an entangler. */
[[nodiscard]] bool profileAllowsEntangler(const NativeProfileSpec& spec,
                                          EntanglerBasis basis);

/**
 * @brief Returns true when a 1-control, 1-target `CtrlOp` is on @p spec.
 */
[[nodiscard]] bool allowsSingleTargetCtrl(CtrlOp ctrl,
                                          const NativeProfileSpec& spec);

/**
 * @brief Returns true when a bare two-qubit op (currently `RZZ`) is on @p spec.
 */
[[nodiscard]] bool allowsBareTwoQubitOp(Operation* op,
                                        const NativeProfileSpec& spec);

/**
 * @brief Returns true when @p op is a native two-qubit gate under @p spec.
 */
[[nodiscard]] bool allowsTwoQubitOp(Operation* op,
                                    const NativeProfileSpec& spec);

/**
 * @brief Fills @p matrix with the 4x4 unitary for a two-qubit block op.
 *
 * Handles single-target `CtrlOp` shells and generic two-qubit unitaries.
 * Returns false for barriers, global phase, and unsupported shapes.
 */
[[nodiscard]] bool assignTwoQubitOpMatrix(Operation* op, Matrix4x4& matrix);

} // namespace mlir::qco::decomposition
