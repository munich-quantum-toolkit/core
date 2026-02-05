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

#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <Eigen/Core>        // NOLINT(misc-include-cleaner)
#include <Eigen/Eigenvalues> // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <optional>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct> // NOLINT(misc-include-cleaner)

namespace mlir::qco {
using fp = qc::fp;
using qfp = std::complex<fp>;
// NOLINTBEGIN(misc-include-cleaner)
using matrix2x2 = Eigen::Matrix2<qfp>;
using matrix4x4 = Eigen::Matrix4<qfp>;
using rmatrix4x4 = Eigen::Matrix4<fp>;
using diagonal4x4 = Eigen::Vector<qfp, 4>;
using rdiagonal4x4 = Eigen::Vector<fp, 4>;
// NOLINTEND(misc-include-cleaner)

constexpr qfp C_ZERO{0., 0.};
constexpr qfp C_ONE{1., 0.};
constexpr qfp C_M_ONE{-1., 0.};
constexpr qfp IM{0., 1.};
constexpr qfp M_IM{0., -1.};

} // namespace mlir::qco

namespace mlir::qco::helpers {

[[nodiscard]] inline qc::OpType getQcType(UnitaryOpInterface op) {
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

// NOLINTBEGIN(misc-include-cleaner)
template <typename T, int N, int M>
[[nodiscard]] inline auto selfAdjointEvd(Eigen::Matrix<T, N, M> a) {
  Eigen::SelfAdjointEigenSolver<decltype(a)> s;
  s.compute(a); // TODO: computeDirect is faster
  auto vecs = s.eigenvectors().eval();
  auto vals = s.eigenvalues();
  return std::make_pair(vecs, vals);
}

template <typename T, int N, int M>
[[nodiscard]] bool isUnitaryMatrix(const Eigen::Matrix<T, N, M>& matrix,
                                   fp tolerance = 1e-13) {
  return (matrix.transpose().conjugate() * matrix).isIdentity(tolerance);
}
// NOLINTEND(misc-include-cleaner)

[[nodiscard]] inline fp remEuclid(fp a, fp b) {
  auto r = std::fmod(a, b);
  return (r < 0.0) ? r + std::abs(b) : r;
}

// Wrap angle into interval [-π,π). If within atol of the endpoint, clamp
// to -π
[[nodiscard]] inline fp mod2pi(fp angle, fp angleZeroEpsilon = 1e-13) {
  // remEuclid() isn't exactly the same as Python's % operator, but
  // because the RHS here is a constant and positive it is effectively
  // equivalent for this case
  auto wrapped = remEuclid(angle + qc::PI, qc::TAU) - qc::PI;
  if (std::abs(wrapped - qc::PI) < angleZeroEpsilon) {
    return -qc::PI;
  }
  return wrapped;
}

[[nodiscard]] inline fp traceToFidelity(const qfp& x) {
  auto xAbs = std::abs(x);
  return (4.0 + xAbs * xAbs) / 20.0;
}

[[nodiscard]] inline std::size_t getComplexity(qc::OpType type,
                                               std::size_t numOfQubits) {
  if (numOfQubits > 1) {
    constexpr std::size_t multiQubitFactor = 10;
    return (numOfQubits - 1) * multiQubitFactor;
  }
  if (type == qc::GPhase) {
    return 0;
  }
  return 1;
}

} // namespace mlir::qco::helpers
