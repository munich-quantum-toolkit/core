/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Decomposition/Helpers.h"

#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <llvm/Support/Casting.h>
#include <numbers>
#include <stdexcept>

namespace mlir::qco::helpers {

qc::OpType getQcType(UnitaryOpInterface op) {
  try {
    auto type = op.getBaseSymbol();
    if (type == "ctrl") {
      type = llvm::cast<CtrlOp>(op).getBodyUnitary().getBaseSymbol();
    }
    return qc::opTypeFromString(type.str());
  } catch (const std::invalid_argument& /*exception*/) {
    return qc::OpType::None;
  }
}

double remEuclid(double a, double b) {
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
    return -std::numbers::pi;
  }
  return wrapped;
}

double traceToFidelity(const std::complex<double>& x) {
  auto xAbs = std::abs(x);
  return (4.0 + xAbs * xAbs) / 20.0;
}

std::size_t getComplexity(qc::OpType type, std::size_t numOfQubits) {
  if (numOfQubits > 1) {
    constexpr std::size_t multiQubitFactor = 10;
    return (numOfQubits - 1) * multiQubitFactor;
  }
  if (type == qc::GPhase) {
    return 0;
  }
  return 1;
}

std::complex<double> globalPhaseFactor(double globalPhase) {
  return std::exp(std::complex<double>{0, 1} * globalPhase);
}

} // namespace mlir::qco::helpers
