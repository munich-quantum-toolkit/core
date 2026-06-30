/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::qco::decomposition {

using namespace std::complex_literals;

static constexpr double PI = std::numbers::pi;
static constexpr double PI_OVER_4 = PI / 4.0;
static constexpr double INV_SQRT2 = 1.0 / std::numbers::sqrt2;

static constexpr Matrix2x2 K12_R_ARR = Matrix2x2::fromElements(
    1i * INV_SQRT2, INV_SQRT2, -INV_SQRT2, -1i * INV_SQRT2);
static constexpr Matrix2x2 K12_L_ARR =
    Matrix2x2::fromElements(Complex{0.5, 0.5}, Complex{0.5, 0.5},
                            Complex{-0.5, 0.5}, Complex{0.5, -0.5});
static constexpr Matrix2x2 K22_L =
    Matrix2x2::fromElements(INV_SQRT2, -INV_SQRT2, INV_SQRT2, INV_SQRT2);
static constexpr Matrix2x2 K22_R = Matrix2x2::fromElements(0, 1, -1, 0);

static double remEuclid(const double a, const double b) {
  if (b == 0.0) {
    llvm::reportFatalInternalError("remEuclid expects non-zero divisor");
  }
  const auto r = std::fmod(a, b);
  return (r < 0.0) ? r + std::abs(b) : r;
}

static bool relativeEq(double lhs, double rhs, double epsilon,
                       double maxRelative) {
  if (lhs == rhs) {
    return true;
  }
  if (std::isinf(lhs) || std::isinf(rhs)) {
    return false;
  }
  const auto absDiff = std::abs(lhs - rhs);
  if (absDiff <= epsilon) {
    return true;
  }
  const auto absLhs = std::abs(lhs);
  const auto absRhs = std::abs(rhs);
  if (absRhs > absLhs) {
    return absDiff <= absRhs * maxRelative;
  }
  return absDiff <= absLhs * maxRelative;
}

static double traceToFidelity(const Complex& trace) {
  const auto traceAbs = std::abs(trace);
  return (4.0 + (traceAbs * traceAbs)) / 20.0;
}

//===----------------------------------------------------------------------===//
// TwoQubitBasisDecomposer
//===----------------------------------------------------------------------===//

TwoQubitBasisDecomposer
TwoQubitBasisDecomposer::create(const Matrix4x4& basisMatrix,
                                double basisFidelity) {
  if (!std::isfinite(basisFidelity) || basisFidelity < 0.0 ||
      basisFidelity > 1.0) {
    llvm::reportFatalInternalError(llvm::formatv(
        "TwoQubitBasisDecomposer: basisFidelity must be finite and in [0, 1] "
        "(got {0})",
        basisFidelity));
  }

  const auto basisWeyl =
      TwoQubitWeylDecomposition::create(basisMatrix, WEYL_DEFAULT_FIDELITY);
  const auto isSuperControlled =
      relativeEq(basisWeyl.a(), PI_OVER_4, WEYL_DIAGONALIZATION_TOLERANCE,
                 WEYL_SUPER_CONTROLLED_MAX_RELATIVE) &&
      relativeEq(basisWeyl.c(), 0.0, WEYL_DIAGONALIZATION_TOLERANCE,
                 WEYL_SUPER_CONTROLLED_MAX_RELATIVE);

  const auto b = basisWeyl.b();
  const Complex expMinusB = std::exp(-1i * b);
  const Complex expPlusB = std::exp(1i * b);
  const Complex expMinus2B = expMinusB * expMinusB;
  const Complex expPlus2B = expPlusB * expPlusB;
  const double cos2B = expPlus2B.real();
  const double sin2B = expPlus2B.imag();

  Complex temp{0.5, -0.5};
  const Matrix2x2 k11l =
      Matrix2x2::fromElements(temp * (-1i * expMinusB), temp * expMinusB,
                              temp * (-1i * expPlusB), temp * -expPlusB);
  const Matrix2x2 k11r = Matrix2x2::fromElements(
      INV_SQRT2 * (1i * expMinusB), INV_SQRT2 * -expMinusB,
      INV_SQRT2 * expPlusB, INV_SQRT2 * (-1i * expPlusB));
  const Matrix2x2 k32lK21l = Matrix2x2::fromElements(
      INV_SQRT2 * Complex{1., cos2B}, INV_SQRT2 * (1i * sin2B),
      INV_SQRT2 * (1i * sin2B), INV_SQRT2 * Complex{1., -cos2B});
  temp = Complex{0.5, 0.5};
  const Matrix2x2 k21r =
      Matrix2x2::fromElements(temp * (-1i * expMinus2B), temp * expMinus2B,
                              temp * (1i * expPlus2B), temp * expPlus2B);
  const Matrix2x2 k31l =
      Matrix2x2::fromElements(INV_SQRT2 * expMinusB, INV_SQRT2 * expMinusB,
                              INV_SQRT2 * -expPlusB, INV_SQRT2 * expPlusB);
  const Matrix2x2 k31r =
      Matrix2x2::fromElements(1i * expPlusB, 0, 0, -1i * expMinusB);
  const Matrix2x2 k32r = Matrix2x2::fromElements(
      temp * expPlusB, temp * -expMinusB, temp * (-1i * expPlusB),
      temp * (-1i * expMinusB));

  const auto k1lDagger = basisWeyl.k1l().adjoint();
  const auto k1rDagger = basisWeyl.k1r().adjoint();
  const auto k2lDagger = basisWeyl.k2l().adjoint();
  const auto k2rDagger = basisWeyl.k2r().adjoint();

  const Matrix2x2 k11lK1lDagger = k11l * k1lDagger;
  const Matrix2x2 k11rK1rDagger = k11r * k1rDagger;
  const Matrix2x2 k2lDaggerK12l = k2lDagger * K12_L_ARR;
  const Matrix2x2 k2rDaggerK12r = k2rDagger * K12_R_ARR;
  const Matrix2x2 iPauliZ = Complex{0.0, 1.0} * ZOp::getUnitaryMatrix();

  TwoQubitBasisDecomposer decomposer;
  decomposer.basisFidelity = basisFidelity;
  decomposer.basisWeyl = basisWeyl;
  decomposer.isSuperControlled = isSuperControlled;
  decomposer.smb = TwoQubitBasisDecomposer::SmbPrecomputed{
      .u0l = k31l * k1lDagger,
      .u0r = k31r * k1rDagger,
      .u1l = k2lDagger * k32lK21l * k1lDagger,
      .u1ra = k2rDagger * k32r,
      .u1rb = k21r * k1rDagger,
      .u2la = k2lDagger * K22_L,
      .u2lb = k11lK1lDagger,
      .u2ra = k2rDagger * K22_R,
      .u2rb = k11rK1rDagger,
      .u3l = k2lDaggerK12l,
      .u3r = k2rDaggerK12r,
      .q0l = K12_L_ARR.adjoint() * k1lDagger,
      .q0r = K12_R_ARR.adjoint() * iPauliZ * k1rDagger,
      .q1la = k2lDagger * k11l.adjoint(),
      .q1ra = k2rDagger * iPauliZ * k11r.adjoint(),
  };
  return decomposer;
}

std::optional<TwoQubitNativeDecomposition>
TwoQubitBasisDecomposer::decomposeTarget(
    const Matrix4x4& targetUnitary,
    const std::optional<std::uint8_t> numBasisGateUses) const {
  const auto targetWeyl =
      TwoQubitWeylDecomposition::create(targetUnitary, WEYL_DEFAULT_FIDELITY);
  return twoQubitDecompose(targetWeyl, numBasisGateUses);
}

std::optional<TwoQubitNativeDecomposition>
TwoQubitBasisDecomposer::twoQubitDecompose(
    const TwoQubitWeylDecomposition& targetDecomposition,
    std::optional<std::uint8_t> numBasisGateUses) const {
  const auto traceValues = traces(targetDecomposition);

  std::uint8_t bestNbasis;
  if (numBasisGateUses) {
    bestNbasis = *numBasisGateUses;
  } else {
    auto bestValue = std::numeric_limits<double>::lowest();
    auto bestIndex = -1;
    double fidelityPower = 1.0;
    for (int i = 0; std::cmp_less(i, traceValues.size()); ++i) {
      const auto value = traceToFidelity(traceValues[i]) * fidelityPower;
      fidelityPower *= basisFidelity;
      if (std::isnan(value)) {
        continue;
      }
      if (value > bestValue) {
        bestIndex = i;
        bestValue = value;
      }
    }
    if (bestIndex < 0) {
      llvm::reportFatalInternalError("Unable to select basis-gate count: all "
                                     "candidate fidelities are NaN");
    }
    bestNbasis = static_cast<std::uint8_t>(bestIndex);
  }
  if (bestNbasis > 1 && !isSuperControlled) {
    return std::nullopt;
  }

  SmallVector<Matrix2x2, 8> factors;
  switch (bestNbasis) {
  case 0:
    factors = decomp0(targetDecomposition);
    break;
  case 1:
    factors = decomp1(targetDecomposition);
    break;
  case 2:
    factors = decomp2Supercontrolled(targetDecomposition);
    break;
  case 3:
    factors = decomp3Supercontrolled(targetDecomposition);
    break;
  default:
    return std::nullopt;
  }

  double globalPhase = targetDecomposition.globalPhase();
  globalPhase -= bestNbasis * basisWeyl.globalPhase();
  if (bestNbasis == 2) {
    globalPhase += PI;
  }
  globalPhase = remEuclid(globalPhase, 2.0 * PI);

  return TwoQubitNativeDecomposition{
      .numBasisUses = bestNbasis,
      .singleQubitFactors = std::move(factors),
      .globalPhase = globalPhase,
  };
}

SmallVector<Matrix2x2, 8>
TwoQubitBasisDecomposer::decomp0(const TwoQubitWeylDecomposition& target) {
  return SmallVector<Matrix2x2, 8>{
      target.k1r() * target.k2r(),
      target.k1l() * target.k2l(),
  };
}

SmallVector<Matrix2x2, 8> TwoQubitBasisDecomposer::decomp1(
    const TwoQubitWeylDecomposition& target) const {
  return SmallVector<Matrix2x2, 8>{
      basisWeyl.k2r().adjoint() * target.k2r(),
      basisWeyl.k2l().adjoint() * target.k2l(),
      target.k1r() * basisWeyl.k1r().adjoint(),
      target.k1l() * basisWeyl.k1l().adjoint(),
  };
}

SmallVector<Matrix2x2, 8> TwoQubitBasisDecomposer::decomp2Supercontrolled(
    const TwoQubitWeylDecomposition& target) const {
  if (!isSuperControlled) {
    llvm::reportFatalInternalError(
        "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
        "- no guarantee for exact decomposition with two basis gates");
  }
  return SmallVector<Matrix2x2, 8>{
      smb.u3r * target.k2r(),
      smb.u3l * target.k2l(),
      smb.q1ra * RZOp::unitaryMatrix(2. * target.b()) * smb.u2rb,
      smb.q1la * RZOp::unitaryMatrix(-2. * target.a()) * smb.u2lb,
      target.k1r() * smb.q0r,
      target.k1l() * smb.q0l,
  };
}

SmallVector<Matrix2x2, 8> TwoQubitBasisDecomposer::decomp3Supercontrolled(
    const TwoQubitWeylDecomposition& target) const {
  if (!isSuperControlled) {
    llvm::reportFatalInternalError(
        "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
        "- no guarantee for exact decomposition with three basis gates");
  }
  return SmallVector<Matrix2x2, 8>{
      smb.u3r * target.k2r(),
      smb.u3l * target.k2l(),
      smb.u2ra * RZOp::unitaryMatrix(2. * target.b()) * smb.u2rb,
      smb.u2la * RZOp::unitaryMatrix(-2. * target.a()) * smb.u2lb,
      smb.u1ra * RZOp::unitaryMatrix(-2. * target.c()) * smb.u1rb,
      smb.u1l,
      target.k1r() * smb.u0r,
      target.k1l() * smb.u0l,
  };
}

std::array<std::complex<double>, 4>
TwoQubitBasisDecomposer::traces(const TwoQubitWeylDecomposition& target) const {
  return {
      4. * std::complex<double>{std::cos(target.a()) * std::cos(target.b()) *
                                    std::cos(target.c()),
                                std::sin(target.a()) * std::sin(target.b()) *
                                    std::sin(target.c())},
      4. * std::complex<double>{std::cos(PI_OVER_4 - target.a()) *
                                    std::cos(basisWeyl.b() - target.b()) *
                                    std::cos(target.c()),
                                std::sin(PI_OVER_4 - target.a()) *
                                    std::sin(basisWeyl.b() - target.b()) *
                                    std::sin(target.c())},
      std::complex<double>{4. * std::cos(target.c()), 0.},
      std::complex<double>{4., 0.},
  };
}

std::optional<TwoQubitNativeDecomposition>
decomposeTwoQubitWithBasis(const Matrix4x4& target,
                           const Matrix4x4& basisMatrix,
                           const double basisFidelity,
                           const std::optional<std::uint8_t> numBasisUses) {
  const auto decomposer =
      TwoQubitBasisDecomposer::create(basisMatrix, basisFidelity);
  return decomposer.decomposeTarget(target, numBasisUses);
}

} // namespace mlir::qco::decomposition
