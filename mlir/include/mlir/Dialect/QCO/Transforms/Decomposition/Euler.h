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

#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::qco::decomposition {

/**
 * @brief Native gate sets for single-qubit Euler synthesis.
 */
enum class EulerBasis : std::uint8_t {
  ZYZ = 0,  ///< `RZ(phi) * RY(theta) * RZ(lambda)`.
  ZXZ = 1,  ///< `RZ(phi) * RX(theta) * RZ(lambda)`.
  XZX = 2,  ///< `RX(phi) * RZ(theta) * RX(lambda)`.
  XYX = 3,  ///< `RX(phi) * RY(theta) * RX(lambda)`.
  U = 4,    ///< `U(theta, phi, lambda)`.
  ZSXX = 5, ///< `RZ` / `SX` / `X` synthesis via ZYZ decomposition.
};

/**
 * @brief Parses a basis name (e.g. `zyz`, `zsxx`; case-insensitive).
 *
 * @param basis The basis name.
 * @return The parsed basis, or `std::nullopt` if unrecognized.
 */
[[nodiscard]] std::optional<EulerBasis> parseEulerBasis(StringRef basis);

/**
 * @brief Synthesizes a composed single-qubit unitary as gates in @p basis.
 *
 * Returns `std::nullopt` when @p hasNonBasisGate is false and resynthesis
 * would not shorten a run of @p runSize gates; otherwise emits gates
 * (including `qco.gphase` when needed).
 *
 * @param builder Builder for the emitted operations.
 * @param loc Location for the emitted operations.
 * @param qubit Input qubit value.
 * @param composed Composed unitary to synthesize.
 * @param runSize Number of gates in the run.
 * @param hasNonBasisGate Whether the run contains a gate outside @p basis.
 * @param basis The target Euler basis.
 * @return The synthesized qubit, or `std::nullopt` if synthesis is skipped.
 */
[[nodiscard]] std::optional<Value>
synthesizeUnitary1QEuler(OpBuilder& builder, Location loc, Value qubit,
                         const Matrix2x2& composed, std::size_t runSize,
                         bool hasNonBasisGate, EulerBasis basis);

} // namespace mlir::qco::decomposition
