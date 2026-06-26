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

#include <array>
#include <optional>

namespace mlir::qco {

/**
 * @brief Computes the eigendecomposition of a real symmetric `4x4` matrix.
 *
 * Uses Householder tridiagonalization (EISPACK `tred2`) followed by implicit
 * QL iteration (`tql2`) on the tridiagonal form. Adapted from John Burkardt's
 * MIT-licensed EISPACK C port (`tred2`, `tql2`):
 * https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
 * Original Fortran: https://netlib.org/eispack/tred2.f,
 * https://netlib.org/eispack/tql2.f
 *
 * @pre @p symmetric is real and symmetric: `symmetric[i,j] == symmetric[j,i]`
 * for all `i, j`. Only the lower triangle (including the diagonal) is read,
 * but supplying a non-symmetric matrix yields undefined numerical results.
 *
 * @param symmetric Row-major real symmetric `4x4` matrix.
 * @return Ascending eigenvalues and matching eigenvectors (as columns).
 */
[[nodiscard]] SymmetricEigen4
decomposeSymmetricEigen4(const std::array<double, 16>& symmetric);

/**
 * @brief Computes the eigendecomposition of a real symmetric `4x4` matrix.
 *
 * @copydoc decomposeSymmetricEigen4(const std::array<double, 16>&)
 *
 * @param matrix Source matrix; only @ref Matrix4x4::realPart is used.
 * @pre Entries are real (imaginary parts must be negligible). The real parts
 * must form a symmetric matrix; imaginary parts are ignored.
 */
[[nodiscard]] SymmetricEigen4 decomposeSymmetricEigen4(const Matrix4x4& matrix);

/**
 * @brief Computes the eigendecomposition of a `1x1` dynamic matrix.
 *
 * @param matrix Source matrix (must be `1x1`).
 * @return The single eigenpair.
 */
[[nodiscard]] std::optional<ComplexEigen>
decomposeComplexEigen1x1(const DynamicMatrix& matrix);

/**
 * @brief Computes the eigendecomposition of a `2x2` complex matrix.
 *
 * Uses a closed-form formula for `2x2` matrices.
 *
 * @param matrix Source matrix.
 * @return Eigenpairs, or `std::nullopt` if the closed-form solver produces
 * non-finite eigenvalues.
 */
[[nodiscard]] std::optional<ComplexEigen2>
decomposeComplexEigen2(const Matrix2x2& matrix);

/**
 * @brief Computes the eigendecomposition of a `4x4` complex matrix.
 *
 * Uses stack-specialized EISPACK `corth` / `comqr2` for fixed `n = 4`
 * (complex Hessenberg reduction and QR eigenanalysis). `pythag` and `csroot`
 * follow John Burkardt's MIT-licensed EISPACK C port; `cdiv`, `corth`, and
 * `comqr2` follow NETLIB EISPACK Fortran
 * (https://netlib.org/eispack/cdiv.f, https://netlib.org/eispack/corth.f,
 * https://netlib.org/eispack/comqr2.f). See also
 * https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
 *
 * @param matrix Source matrix.
 * @return Eigenpairs, or `std::nullopt` if the solver does not converge.
 */
[[nodiscard]] std::optional<ComplexEigen4>
decomposeComplexEigen4(const Matrix4x4& matrix);

/**
 * @brief Computes the eigendecomposition of a square dynamic matrix.
 *
 * Uses EISPACK `corth` followed by `comqr2` for dimensions other than `1`, `2`,
 * and `4` (which have specialized paths in @ref DynamicMatrix::complexEigen).
 * `pythag` and `csroot` follow John Burkardt's MIT-licensed EISPACK C port;
 * `cdiv`, `corth`, and `comqr2` follow NETLIB EISPACK Fortran
 * (https://netlib.org/eispack/cdiv.f, https://netlib.org/eispack/corth.f,
 * https://netlib.org/eispack/comqr2.f). See also
 * https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
 *
 * @param matrix Square source matrix.
 * @return Eigenpairs, or `std::nullopt` if the matrix is not square, its
 * dimension exceeds `INT_MAX`, or the solver does not converge.
 */
[[nodiscard]] std::optional<ComplexEigen>
decomposeComplexEigenDynamic(const DynamicMatrix& matrix);

/**
 * @brief Lifts a fixed `2x2` eigendecomposition to @ref ComplexEigen.
 *
 * @param eigen2 Fixed-size eigenpairs from @ref decomposeComplexEigen2.
 * @return Dynamic-matrix eigenvector storage with the same eigenvalues.
 */
[[nodiscard]] ComplexEigen fromComplexEigen(const ComplexEigen2& eigen2);

/**
 * @brief Lifts a fixed `4x4` eigendecomposition to @ref ComplexEigen.
 *
 * @param eigen4 Fixed-size eigenpairs from @ref decomposeComplexEigen4.
 * @return Dynamic-matrix eigenvector storage with the same eigenvalues.
 */
[[nodiscard]] ComplexEigen fromComplexEigen(const ComplexEigen4& eigen4);

} // namespace mlir::qco
