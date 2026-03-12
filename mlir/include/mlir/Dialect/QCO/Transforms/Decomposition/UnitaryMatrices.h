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

#include "Gate.h"

#include <Eigen/Core>

namespace mlir::qco::decomposition {

inline constexpr double SQRT2 = 1.414213562373095048801688724209698079L;
inline constexpr double FRAC1_SQRT2 =
    0.707106781186547524400844362104849039284835937688474036588L;

[[nodiscard]] Eigen::Matrix2cd uMatrix(double lambda, double phi, double theta);

[[nodiscard]] Eigen::Matrix2cd u2Matrix(double lambda, double phi);

[[nodiscard]] Eigen::Matrix2cd rxMatrix(double theta);

[[nodiscard]] Eigen::Matrix2cd ryMatrix(double theta);

[[nodiscard]] Eigen::Matrix2cd rzMatrix(double theta);

[[nodiscard]] Eigen::Matrix4cd rxxMatrix(double theta);

[[nodiscard]] Eigen::Matrix4cd ryyMatrix(double theta);

[[nodiscard]] Eigen::Matrix4cd rzzMatrix(double theta);

[[nodiscard]] Eigen::Matrix2cd pMatrix(double lambda);

inline const Eigen::Matrix4cd SWAP_GATE{
    {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}};
inline const Eigen::Matrix2cd H_GATE{{FRAC1_SQRT2, FRAC1_SQRT2},
                                     {FRAC1_SQRT2, -FRAC1_SQRT2}};
inline const Eigen::Matrix2cd IPZ{{{0, 1}, 0}, {0, {0, -1}}};
inline const Eigen::Matrix2cd IPY{{0, 1}, {-1, 0}};
inline const Eigen::Matrix2cd IPX{{0, {0, 1}}, {{0, 1}, 0}};

[[nodiscard]] Eigen::Matrix4cd
expandToTwoQubits(const Eigen::Matrix2cd& singleQubitMatrix, QubitId qubitId);

[[nodiscard]] Eigen::Matrix4cd
fixTwoQubitMatrixQubitOrder(const Eigen::Matrix4cd& twoQubitMatrix,
                            const llvm::SmallVector<QubitId, 2>& qubitIds);

[[nodiscard]] Eigen::Matrix2cd getSingleQubitMatrix(const Gate& gate);

// TODO: remove? only used for verification of circuit and in unittests
[[nodiscard]] Eigen::Matrix4cd getTwoQubitMatrix(const Gate& gate);

} // namespace mlir::qco::decomposition
