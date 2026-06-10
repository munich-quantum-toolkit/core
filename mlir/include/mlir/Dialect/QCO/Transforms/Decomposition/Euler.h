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
 * @brief Euler angles `(theta, phi, lambda)` and global phase for a 2x2
 * unitary.
 */
struct EulerAngles {
  double theta = 0.0;
  double phi = 0.0;
  double lambda = 0.0;
  double phase = 0.0;
};

/**
 * @brief Native gate sets for single-qubit Euler synthesis.
 */
enum class EulerBasis : std::uint8_t {
  ZYZ = 0,  ///< `RZ(phi) * RY(theta) * RZ(lambda)`.
  ZXZ = 1,  ///< `RZ(phi) * RX(theta) * RZ(lambda)`.
  XZX = 2,  ///< `RX(phi) * RZ(theta) * RX(lambda)`.
  XYX = 3,  ///< `RX(phi) * RY(theta) * RX(lambda)`.
  U = 4,    ///< `U(theta, phi, lambda)`.
  ZSXX = 5, ///< `RZ` / `SX` / `X` chain equivalent to ZYZ.
};

/**
 * @brief Extracts Euler parameters from single-qubit unitary matrices.
 */
class EulerDecomposition {
  friend Value synthesizeUnitary1QEuler(OpBuilder& builder, Location loc,
                                        Value qubit,
                                        const Matrix2x2& targetMatrix,
                                        EulerBasis basis);

public:
  /**
   * @brief Extracts `(theta, phi, lambda, phase)` for KAK and `U` bases.
   *
   * Does not support `EulerBasis::ZSXX`; use `synthesizeUnitary1QEuler` or
   * `synthesisGateCount` instead.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @param basis The target Euler basis.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles anglesFromUnitary(const Matrix2x2& matrix,
                                                     EulerBasis basis);

private:
  /**
   * @brief Extracts parameters for `RZ(phi) * RY(theta) * RZ(lambda)`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsZYZ(const Matrix2x2& matrix);

  /**
   * @brief Extracts parameters for `U(theta, phi, lambda)`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsU(const Matrix2x2& matrix);

  /**
   * @brief Extracts parameters for `RZ(phi) * RX(theta) * RZ(lambda)`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsZXZ(const Matrix2x2& matrix);

  /**
   * @brief Extracts parameters for `RX(phi) * RY(theta) * RX(lambda)`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsXYX(const Matrix2x2& matrix);

  /**
   * @brief Extracts parameters for `RX(phi) * RZ(theta) * RX(lambda)`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsXZX(const Matrix2x2& matrix);
};

/**
 * @brief Parses a basis name (e.g. `zyz`, `zsxx`; case-insensitive).
 *
 * @param basis The basis name.
 * @return The parsed basis, or `std::nullopt` if unrecognized.
 */
[[nodiscard]] std::optional<EulerBasis> parseEulerBasis(StringRef basis);

/**
 * @brief Synthesizes `targetMatrix` as gates in `basis`.
 *
 * Emits `qco.gphase` when needed so the result matches exactly, not only up to
 * global phase.
 *
 * @param builder Builder for the emitted operations.
 * @param loc Location for the emitted operations.
 * @param qubit Input qubit value.
 * @param targetMatrix The single-qubit unitary to synthesize.
 * @param basis The target Euler basis.
 * @return The output qubit value.
 */
[[nodiscard]] Value synthesizeUnitary1QEuler(OpBuilder& builder, Location loc,
                                             Value qubit,
                                             const Matrix2x2& targetMatrix,
                                             EulerBasis basis);

/**
 * @brief Number of basis gates `synthesizeUnitary1QEuler` would emit.
 *
 * Excludes `qco.gphase`. Used by the fuse pass to detect overlong in-basis
 * runs.
 *
 * @param targetMatrix The single-qubit unitary that would be synthesized.
 * @param basis The target Euler basis.
 * @return The gate count (1 for `U`, 3 for KAK bases, 3 or 5 for `ZSXX`).
 */
[[nodiscard]] std::size_t synthesisGateCount(const Matrix2x2& targetMatrix,
                                             EulerBasis basis);

} // namespace mlir::qco::decomposition
