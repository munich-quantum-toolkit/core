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

#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"

#include <llvm/Support/ErrorHandling.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>

namespace mlir::qco::helpers {

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

std::size_t getComplexity(decomposition::GateKind type,
                          std::size_t numOfQubits) {
  if (type == decomposition::GateKind::GPhase) {
    return 0;
  }
  if (numOfQubits > 1) {
    // Multi-qubit operations dominate the heuristic cost model.
    constexpr std::size_t multiQubitFactor = 10;
    return (numOfQubits - 1) * multiQubitFactor;
  }
  return 1;
}

std::complex<double> globalPhaseFactor(double globalPhase) {
  return std::exp(std::complex<double>{0, 1} * globalPhase);
}

} // namespace mlir::qco::helpers
