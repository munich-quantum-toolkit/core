/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"

#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/Support/ErrorHandling.h>

#include <cmath>
#include <complex>
#include <numbers>

namespace mlir::qco::helpers {

bool isUnitaryMatrix(const Matrix2x2& matrix, double tolerance) {
  return (matrix.adjoint() * matrix).isIdentity(tolerance);
}

bool isUnitaryMatrix(const Matrix4x4& matrix, double tolerance) {
  return (matrix.adjoint() * matrix).isIdentity(tolerance);
}

double remEuclid(double a, double b) {
  if (b == 0.0) {
    llvm::reportFatalInternalError(
        "remEuclid expects non-zero divisor; callers like mod2pi pass positive "
        "constants");
  }
  auto r = std::fmod(a, b);
  return (r < 0.0) ? r + std::abs(b) : r;
}

double mod2pi(double angle, double angleZeroEpsilon) {
  // remEuclid() isn't exactly the same as Python's % operator, but
  // because the RHS here is a constant and positive it is effectively
  // equivalent for this case
  auto wrapped = remEuclid(angle + std::numbers::pi, 2 * std::numbers::pi) -
                 std::numbers::pi;
  if (std::abs(wrapped - std::numbers::pi) < angleZeroEpsilon) {
    // Canonicalize the upper endpoint back to -pi so callers always receive a
    // half-open interval [-pi, pi).
    return -std::numbers::pi;
  }
  return wrapped;
}

double traceToFidelity(const std::complex<double>& x) {
  // Average two-qubit process fidelity given the Hilbert-Schmidt overlap
  // `x = tr(U_target^dag * U_actual)`. For a 4x4 unitary the general formula is
  // `F_avg = (d + |tr|^2) / (d * (d + 1))` with `d = 4`, which reduces to the
  // `(4 + |x|^2) / 20` expression below. See e.g. Horodecki/Nielsen.
  auto xAbs = std::abs(x);
  return (4.0 + (xAbs * xAbs)) / 20.0;
}

std::complex<double> globalPhaseFactor(double globalPhase) {
  return std::exp(std::complex<double>{0, 1} * globalPhase);
}

} // namespace mlir::qco::helpers
