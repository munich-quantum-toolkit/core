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

#include <Eigen/Core>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>

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
  ZYZ = 0, ///< `RZ(phi) * RY(theta) * RZ(lambda)`.
  ZXZ = 1, ///< `RZ(phi) * RX(theta) * RZ(lambda)`.
  XZX = 2, ///< `RX(phi) * RZ(theta) * RX(lambda)`.
  XYX = 3, ///< `RX(phi) * RY(theta) * RX(lambda)`.
  U = 4,   ///< `U(theta, phi, lambda)`.
  ZSXX =
      5, ///< ZYZ-equivalent chain over `RZ`, `SX`, and `X` (see `paramsPSX`).
};

/**
 * @brief Extracts Euler parameters from single-qubit unitary matrices.
 */
class EulerDecomposition {
  friend Value synthesizeUnitary1QEuler(OpBuilder& builder, Location loc,
                                        Value qubit,
                                        const Eigen::Matrix2cd& targetMatrix,
                                        EulerBasis basis);

public:
  /**
   * @brief Extracts `(theta, phi, lambda, phase)` for the requested basis.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @param basis The target Euler basis.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles
  anglesFromUnitary(const Eigen::Matrix2cd& matrix, EulerBasis basis);

private:
  /**
   * @brief Extracts parameters for `RZ(phi) * RY(theta) * RZ(lambda)`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsZYZ(const Eigen::Matrix2cd& matrix);

  /**
   * @brief Extracts parameters for a `U(theta, phi, lambda)` factorization.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsU(const Eigen::Matrix2cd& matrix);

  /**
   * @brief Extracts parameters for `RZ(phi) * RX(theta) * RZ(lambda)`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsZXZ(const Eigen::Matrix2cd& matrix);

  /**
   * @brief Extracts parameters for `RX(phi) * RY(theta) * RX(lambda)`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsXYX(const Eigen::Matrix2cd& matrix);

  /**
   * @brief Extracts parameters for `RX(phi) * RZ(theta) * RX(lambda)`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsXZX(const Eigen::Matrix2cd& matrix);

  /**
   * @brief Extracts ZYZ-equivalent angles and global phase for `ZSXX`
   * synthesis.
   *
   * Returns the same `(theta, phi, lambda)` as `paramsZYZ`; `phase` includes
   * the offset to the native `RZ`/`SX`/`X` chain emitted for `ZSXX`.
   *
   * @param matrix The single-qubit unitary to decompose.
   * @return The extracted Euler angles and global phase.
   */
  [[nodiscard]] static EulerAngles paramsPSX(const Eigen::Matrix2cd& matrix);
};

/**
 * @brief Parses a user-facing basis string (e.g. "zyz", "zsxx").
 *
 * @param basis The basis name (case-insensitive).
 * @return The parsed Euler basis, or `std::nullopt` if unrecognized.
 */
[[nodiscard]] std::optional<EulerBasis> parseEulerBasis(StringRef basis);

/**
 * @brief Emits gates reconstructing `targetMatrix` in the given basis.
 *
 * Includes a global phase (`qco.gphase`) when needed for an exact match.
 *
 * @param builder Builder used to create the operations.
 * @param loc Source location for the created operations.
 * @param qubit Input qubit value.
 * @param targetMatrix The single-qubit unitary to synthesize.
 * @param basis The target Euler basis.
 * @return The transformed qubit value.
 */
[[nodiscard]] Value
synthesizeUnitary1QEuler(OpBuilder& builder, Location loc, Value qubit,
                         const Eigen::Matrix2cd& targetMatrix,
                         EulerBasis basis);

} // namespace mlir::qco::decomposition
