/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeGateset.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <tuple>
#include <utility>

namespace mlir::qco::decomposition {

using namespace std::complex_literals;

namespace {

enum class Specialization : std::uint8_t {
  General,
  IdEquiv,
  SWAPEquiv,
  PartialSWAPEquiv,
  PartialSWAPFlipEquiv,
  ControlledEquiv,
  MirrorControlledEquiv,
  FSimaabEquiv,
  FSimabbEquiv,
  FSimabmbEquiv,
};

struct ChamberState {
  std::array<double, 3> cs{};
  Matrix2x2 k1l;
  Matrix2x2 k1r;
  Matrix2x2 k2l;
  Matrix2x2 k2r;
  double globalPhase{};
  double a{};
  double b{};
  double c{};
};

} // namespace

static constexpr double PI = std::numbers::pi;
static constexpr double PI_OVER_4 = PI / 4.0;

static constexpr Matrix2x2 I_PAULI_X = Matrix2x2::fromElements(0, 1i, 1i, 0);
static constexpr Matrix2x2 I_PAULI_Y = Matrix2x2::fromElements(0, 1, -1, 0);
static constexpr Matrix2x2 I_PAULI_Z = Matrix2x2::fromElements(1i, 0, 0, -1i);

static constexpr Matrix4x4 MAGIC_BASIS_NON_NORMALIZED =
    Matrix4x4::fromElements( //
        1, 1i, 0, 0,         //
        0, 0, 1i, 1,         //
        0, 0, 1i, -1,        //
        1, -1i, 0, 0);
static constexpr Matrix4x4 MAGIC_BASIS_NON_NORMALIZED_DAGGER =
    Matrix4x4::fromElements(                          //
        0.5, 0, 0, 0.5,                               //
        Complex{0.0, -0.5}, 0, 0, Complex{0.0, 0.5},  //
        0, Complex{0.0, -0.5}, Complex{0.0, -0.5}, 0, //
        0, 0.5, -0.5, 0);

static Matrix4x4 magicBasisTransform(const Matrix4x4& unitary,
                                     bool outOfMagicBasis) {
  if (outOfMagicBasis) {
    return MAGIC_BASIS_NON_NORMALIZED_DAGGER * unitary *
           MAGIC_BASIS_NON_NORMALIZED;
  }
  return MAGIC_BASIS_NON_NORMALIZED * unitary *
         MAGIC_BASIS_NON_NORMALIZED_DAGGER;
}

static double closestPartialSwap(double a, double b, double c) {
  const auto m = (a + b + c) / 3.;
  const auto am = a - m;
  const auto bm = b - m;
  const auto cm = c - m;
  const auto ab = a - b;
  const auto bc = b - c;
  const auto ca = c - a;
  return m + (am * bm * cm * (6. + (ab * ab) + (bc * bc) + (ca * ca)) / 18.);
}

/** @brief Uniform sample in `(0, 1]` from `std::mt19937`. */
static double uniformOpenUnit(std::mt19937& rng) {
  return (static_cast<double>(rng()) + 0.5) /
         (static_cast<double>(std::mt19937::max()) + 1.0);
}

/** @brief Standard-normal sample via Box-Muller. */
static double normalSample(std::mt19937& rng) {
  const double u1 = uniformOpenUnit(rng);
  const double u2 = uniformOpenUnit(rng);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
}

static std::pair<Matrix4x4, std::array<Complex, 4>>
diagonalizeComplexSymmetric(const Matrix4x4& m,
                            double precision = WEYL_DIAGONALIZATION_TOLERANCE) {
  auto state = std::mt19937{2023};

  const auto mReal = m.realPart();
  const auto mImag = m.imagPart();

  double bestErr = std::numeric_limits<double>::max();
  constexpr auto maxDiagonalizationAttempts = 100;
  for (int i = 0; i < maxDiagonalizationAttempts; ++i) {
    double randA{};
    double randB{};
    // Fixed perturbation coefficients for the first diagonalization attempt,
    // carried over from Qiskit's two-qubit Weyl decomposition (legacy Python
    // RNG values). The loop usually succeeds on this trial; hard-coding them
    // keeps the common path independent of any RNG.
    if (i == 0) {
      randA = 1.2602066112249388;
      randB = 0.22317849046722027;
    } else {
      randA = normalSample(state);
      randB = normalSample(state);
    }
    std::array<double, 16> m2Real{};
    for (std::size_t k = 0; k < m2Real.size(); ++k) {
      m2Real[k] = (randA * mReal[k]) + (randB * mImag[k]);
    }
    const Matrix4x4 p = Matrix4x4::fromRealRowMajor(m2Real)
                            .symmetricEigenDecomposition()
                            .eigenvectors;
    const std::array<Complex, 4> d = (p.transpose() * m * p).diagonal();

    const auto compare = p * Matrix4x4::fromDiagonal(d) * p.transpose();
    double err = 0.0;
    for (std::size_t r = 0; r < 4; ++r) {
      for (std::size_t cc = 0; cc < 4; ++cc) {
        err = std::max(err, std::abs(compare(r, cc) - m(r, cc)));
      }
    }
    bestErr = std::min(bestErr, err);
    if (compare.isApprox(m, precision)) {
      assert((p.transpose() * p).isIdentity(WEYL_DIAGONALIZATION_TOLERANCE));
      assert(std::abs(Matrix4x4::fromDiagonal(d).determinant() - 1.0) <
             WEYL_DIAGONALIZATION_TOLERANCE);
      return {p, d};
    }
  }
  llvm::reportFatalInternalError(llvm::formatv(
      "TwoQubitWeylDecomposition: failed to diagonalize M2 ({0} iterations). "
      "best error = {1:e}, precision = {2:e}",
      maxDiagonalizationAttempts, bestErr, precision));
}

static std::tuple<Matrix2x2, Matrix2x2, double>
decomposeTwoQubitProductGate(const Matrix4x4& specialUnitary) {
  Matrix2x2 r =
      Matrix2x2::fromElements(specialUnitary(0, 0), specialUnitary(0, 1),
                              specialUnitary(1, 0), specialUnitary(1, 1));
  auto detR = r.determinant();
  if (std::abs(detR) < 0.1) {
    r = Matrix2x2::fromElements(specialUnitary(2, 0), specialUnitary(2, 1),
                                specialUnitary(3, 0), specialUnitary(3, 1));
    detR = r.determinant();
  }
  if (std::abs(detR) < 0.1) {
    llvm::reportFatalInternalError(
        "decomposeTwoQubitProductGate: unable to decompose: det_r < 0.1");
  }
  r *= (1.0 / std::sqrt(detR));
  const Matrix2x2 rTConj = r.adjoint();

  const Matrix4x4 temp =
      specialUnitary * Matrix4x4::kron(Matrix2x2::identity(), rTConj);

  Matrix2x2 l =
      Matrix2x2::fromElements(temp(0, 0), temp(0, 2), temp(2, 0), temp(2, 2));
  auto detL = l.determinant();
  if (std::abs(detL) < 0.9) {
    llvm::reportFatalInternalError(
        "decomposeTwoQubitProductGate: unable to decompose: detL < 0.9");
  }
  l *= (1.0 / std::sqrt(detL));
  const auto phase = std::arg(detL) / 2.;

  return {l, r, phase};
}

static std::complex<double> getTrace(double a, double b, double c, double ap,
                                     double bp, double cp) {
  const auto da = a - ap;
  const auto db = b - bp;
  const auto dc = c - cp;
  return 4. * std::complex<double>{std::cos(da) * std::cos(db) * std::cos(dc),
                                   std::sin(da) * std::sin(db) * std::sin(dc)};
}

static Specialization
bestSpecialization(const TwoQubitWeylDecomposition& decomposition,
                   const std::optional<double>& requestedFidelity) {
  auto isClose = [&](double ap, double bp, double cp) -> bool {
    const auto tr = getTrace(decomposition.a(), decomposition.b(),
                             decomposition.c(), ap, bp, cp);
    if (requestedFidelity) {
      return traceToFidelity(tr) >= *requestedFidelity;
    }
    return false;
  };

  if (isClose(0., 0., 0.)) {
    return Specialization::IdEquiv;
  }
  if (isClose(PI_OVER_4, PI_OVER_4, PI_OVER_4) ||
      isClose(PI_OVER_4, PI_OVER_4, -PI_OVER_4)) {
    return Specialization::SWAPEquiv;
  }
  if (const auto closestAbc = closestPartialSwap(
          decomposition.a(), decomposition.b(), decomposition.c());
      isClose(closestAbc, closestAbc, closestAbc)) {
    return Specialization::PartialSWAPEquiv;
  }
  if (const auto closestAbMinusC = closestPartialSwap(
          decomposition.a(), decomposition.b(), -decomposition.c());
      isClose(closestAbMinusC, closestAbMinusC, -closestAbMinusC)) {
    return Specialization::PartialSWAPFlipEquiv;
  }
  if (isClose(decomposition.a(), 0., 0.)) {
    return Specialization::ControlledEquiv;
  }
  if (isClose(PI_OVER_4, PI_OVER_4, decomposition.c())) {
    return Specialization::MirrorControlledEquiv;
  }
  if (isClose((decomposition.a() + decomposition.b()) / 2.,
              (decomposition.a() + decomposition.b()) / 2.,
              decomposition.c())) {
    return Specialization::FSimaabEquiv;
  }
  if (isClose(decomposition.a(), (decomposition.b() + decomposition.c()) / 2.,
              (decomposition.b() + decomposition.c()) / 2.)) {
    return Specialization::FSimabbEquiv;
  }
  if (isClose(decomposition.a(), (decomposition.b() - decomposition.c()) / 2.,
              (decomposition.c() - decomposition.b()) / 2.)) {
    return Specialization::FSimabmbEquiv;
  }
  return Specialization::General;
}

static std::pair<Matrix4x4, double> projectToSU4(const Matrix4x4& unitary) {
  auto u = unitary;
  const auto detU = u.determinant();
  u *= std::pow(detU, -0.25);
  return {u, std::arg(detU) / 4.0};
}

static std::tuple<Matrix4x4, Matrix4x4, std::array<double, 3>,
                  std::array<double, 4>>
computeOrderedWeylCoordinates(const Matrix4x4& u) {
  const auto uP = magicBasisTransform(u, /*outOfMagicBasis=*/true);
  const Matrix4x4 m2 = uP.transpose() * uP;
  auto [p, d] = diagonalizeComplexSymmetric(m2);

  std::array<double, 4> dReal{};
  for (std::size_t i = 0; i < d.size(); ++i) {
    dReal[i] = -std::arg(d[i]) / 2.0;
  }
  dReal[3] = -dReal[0] - dReal[1] - dReal[2];

  std::array<double, 3> cs{};
  for (std::size_t i = 0; i < cs.size(); ++i) {
    cs[i] = remEuclid((dReal[i] + dReal[3]) / 2.0, 2.0 * PI);
  }

  // Sort coordinates by min(x mod pi/2, pi/2 - x mod pi/2).
  std::array<double, 3> cstemp{};
  for (std::size_t i = 0; i < cs.size(); ++i) {
    const auto tmp = remEuclid(cs[i], PI / 2.0);
    cstemp[i] = std::min(tmp, (PI / 2.0) - tmp);
  }
  std::array<std::size_t, 3> order{0, 1, 2};
  std::ranges::stable_sort(
      order, [&](auto a, auto b) { return cstemp[a] < cstemp[b]; });
  order = {order[1], order[2], order[0]};
  cs = {cs[order[0]], cs[order[1]], cs[order[2]]};
  {
    const std::array<double, 3> reordered{dReal[order[0]], dReal[order[1]],
                                          dReal[order[2]]};
    dReal[0] = reordered[0];
    dReal[1] = reordered[1];
    dReal[2] = reordered[2];
  }

  const Matrix4x4 pOrig = p;
  for (std::size_t i = 0; i < order.size(); ++i) {
    p.setColumn(i, pOrig.column(order[i]));
  }
  if (p.determinant().real() < 0.0) {
    auto lastColumn = p.column(3);
    for (auto& entry : lastColumn) {
      entry = -entry;
    }
    p.setColumn(3, lastColumn);
  }
  assert(std::abs(p.determinant() - 1.0) < WEYL_DIAGONALIZATION_TOLERANCE);

  return {uP, p, cs, dReal};
}

static ChamberState buildChamberState(const Matrix4x4& u, const Matrix4x4& uP,
                                      Matrix4x4 p, std::array<double, 3> cs,
                                      const std::array<double, 4>& dReal,
                                      double globalPhase) {
  const Matrix4x4 temp =
      Matrix4x4::fromDiagonal(std::exp(1i * dReal[0]), std::exp(1i * dReal[1]),
                              std::exp(1i * dReal[2]), std::exp(1i * dReal[3]));

  Matrix4x4 k1 = uP * p * temp;
  assert((k1.transpose() * k1).isIdentity(WEYL_TOLERANCE));
  assert(k1.determinant().real() > 0.0);
  k1 = magicBasisTransform(k1, /*outOfMagicBasis=*/false);

  Matrix4x4 k2 = p.adjoint();
  assert((k2.transpose() * k2).isIdentity(WEYL_TOLERANCE));
  assert(k2.determinant().real() > 0.0);
  k2 = magicBasisTransform(k2, /*outOfMagicBasis=*/false);

  assert((k1 *
          magicBasisTransform(Matrix4x4::fromDiagonal(std::exp(-1i * dReal[0]),
                                                      std::exp(-1i * dReal[1]),
                                                      std::exp(-1i * dReal[2]),
                                                      std::exp(-1i * dReal[3])),
                              /*outOfMagicBasis=*/false) *
          k2)
             .isApprox(u, WEYL_TOLERANCE));

  auto [k1l, k1r, phaseL] = decomposeTwoQubitProductGate(k1);
  auto [k2l, k2r, phaseR] = decomposeTwoQubitProductGate(k2);
  assert(Matrix4x4::kron(k1l, k1r).isApprox(k1, WEYL_TOLERANCE));
  assert(Matrix4x4::kron(k2l, k2r).isApprox(k2, WEYL_TOLERANCE));
  globalPhase += phaseL + phaseR;

  if (cs[0] > (PI / 2.0)) {
    cs[0] -= 3.0 * (PI / 2.0);
    k1l = k1l * I_PAULI_Y;
    k1r = k1r * I_PAULI_Y;
    globalPhase += (PI / 2.0);
  }
  if (cs[1] > (PI / 2.0)) {
    cs[1] -= 3.0 * (PI / 2.0);
    k1l = k1l * I_PAULI_X;
    k1r = k1r * I_PAULI_X;
    globalPhase += (PI / 2.0);
  }
  auto conjs = 0;
  if (cs[0] > PI_OVER_4) {
    cs[0] = (PI / 2.0) - cs[0];
    k1l = k1l * I_PAULI_Y;
    k2r = I_PAULI_Y * k2r;
    conjs += 1;
    globalPhase -= (PI / 2.0);
  }
  if (cs[1] > PI_OVER_4) {
    cs[1] = (PI / 2.0) - cs[1];
    k1l = k1l * I_PAULI_X;
    k2r = I_PAULI_X * k2r;
    conjs += 1;
    globalPhase += (PI / 2.0);
    if (conjs == 1) {
      globalPhase -= PI;
    }
  }
  if (cs[2] > (PI / 2.0)) {
    cs[2] -= 3.0 * (PI / 2.0);
    k1l = k1l * I_PAULI_Z;
    k1r = k1r * I_PAULI_Z;
    globalPhase += (PI / 2.0);
    if (conjs == 1) {
      globalPhase -= PI;
    }
  }
  if (conjs == 1) {
    cs[2] = (PI / 2.0) - cs[2];
    k1l = k1l * I_PAULI_Z;
    k2r = I_PAULI_Z * k2r;
    globalPhase += (PI / 2.0);
  }
  if (cs[2] > PI_OVER_4) {
    cs[2] -= (PI / 2.0);
    k1l = k1l * I_PAULI_Z;
    k1r = k1r * I_PAULI_Z;
    globalPhase -= (PI / 2.0);
  }

  ChamberState chamber;
  chamber.cs = cs;
  chamber.k1l = k1l;
  chamber.k1r = k1r;
  chamber.k2l = k2l;
  chamber.k2r = k2r;
  chamber.globalPhase = globalPhase;
  chamber.a = cs[1];
  chamber.b = cs[0];
  chamber.c = cs[2];
  return chamber;
}

//===----------------------------------------------------------------------===//
// TwoQubitWeylDecomposition
//===----------------------------------------------------------------------===//

void TwoQubitWeylDecomposition::finalizeSpecializationPhase(
    bool flippedFromOriginal, double preSpecializationA,
    double preSpecializationB, double preSpecializationC,
    const std::optional<double>& fidelity) {
  const auto trace =
      flippedFromOriginal
          ? getTrace((PI / 2.0) - preSpecializationA, preSpecializationB,
                     -preSpecializationC, a_, b_, c_)
          : getTrace(preSpecializationA, preSpecializationB, preSpecializationC,
                     a_, b_, c_);
  const double calculatedFidelity = traceToFidelity(trace);
  if (fidelity &&
      calculatedFidelity + WEYL_DIAGONALIZATION_TOLERANCE < *fidelity) {
    llvm::reportFatalInternalError(llvm::formatv(
        "TwoQubitWeylDecomposition: Calculated fidelity of "
        "specialization is worse than requested fidelity ({0:F4} vs {1:F4})!",
        calculatedFidelity, *fidelity));
  }
  globalPhase_ += std::arg(trace);
}

TwoQubitWeylDecomposition
TwoQubitWeylDecomposition::create(const Matrix4x4& unitaryMatrix,
                                  std::optional<double> fidelity) {
  if (fidelity &&
      (!std::isfinite(*fidelity) || *fidelity < 0.0 || *fidelity > 1.0)) {
    llvm::reportFatalInternalError(llvm::formatv(
        "TwoQubitWeylDecomposition: fidelity must be finite and in [0, 1] "
        "(got {0})",
        *fidelity));
  }

  const auto [u, globalPhase0] = projectToSU4(unitaryMatrix);
  auto [uP, p, cs, dReal] = computeOrderedWeylCoordinates(u);
  const auto chamber = buildChamberState(u, uP, p, cs, dReal, globalPhase0);
  TwoQubitWeylDecomposition decomposition;
  decomposition.a_ = chamber.a;
  decomposition.b_ = chamber.b;
  decomposition.c_ = chamber.c;
  decomposition.globalPhase_ = chamber.globalPhase;
  decomposition.k1l_ = chamber.k1l;
  decomposition.k2l_ = chamber.k2l;
  decomposition.k1r_ = chamber.k1r;
  decomposition.k2r_ = chamber.k2r;

  assert(decomposition.unitaryMatrix().isApprox(unitaryMatrix, WEYL_TOLERANCE));

  const bool flippedFromOriginal = decomposition.applySpecialization(fidelity);
  decomposition.finalizeSpecializationPhase(flippedFromOriginal, chamber.a,
                                            chamber.b, chamber.c, fidelity);

  return decomposition;
}

Matrix4x4 TwoQubitWeylDecomposition::unitaryMatrix() const {
  return Matrix4x4::kron(k1l_, k1r_) * getCanonicalMatrix() *
         Matrix4x4::kron(k2l_, k2r_) * std::polar(1.0, globalPhase_);
}

Matrix4x4 unitaryMatrix(const TwoQubitNativeDecomposition& decomposition,
                        const Matrix4x4& basisGate) {
  const auto requiredFactors =
      singleQubitFactorCount(decomposition.numBasisUses);
  if (decomposition.singleQubitFactors.size() < requiredFactors) {
    llvm::reportFatalInternalError(llvm::formatv(
        "unitaryMatrix: expected at least {0} single-qubit factors for "
        "numBasisUses = {1}, got {2}",
        requiredFactors, decomposition.numBasisUses,
        decomposition.singleQubitFactors.size()));
  }
  const auto& factors = decomposition.singleQubitFactors;
  const auto layer = [&](const std::size_t i) {
    return Matrix4x4::kron(factors[(2 * i) + 1], factors[2 * i]);
  };
  Matrix4x4 matrix = layer(0);
  for (std::uint8_t i = 0; i < decomposition.numBasisUses; ++i) {
    matrix = basisGate * matrix;
    matrix = layer(static_cast<std::size_t>(i) + 1) * matrix;
  }
  return matrix * std::polar(1.0, decomposition.globalPhase);
}

Matrix4x4 TwoQubitWeylDecomposition::getCanonicalMatrix(double a, double b,
                                                        double c) {
  const auto zero = Complex{0.0, 0.0};
  const auto expPlusC = std::exp(Complex{0.0, c});
  const auto expMinusC = std::exp(Complex{0.0, -c});
  const auto cosAMinusB = std::cos(a - b);
  const auto cosAPlusB = std::cos(a + b);
  const auto iSinAMinusB = Complex{0.0, 1.0} * std::sin(a - b);
  const auto iSinAPlusB = Complex{0.0, 1.0} * std::sin(a + b);

  // Closed form of RZZ(-2c) * RYY(-2b) * RXX(-2a) = exp(-i(a XX + b YY + c
  // ZZ)).
  return Matrix4x4::fromElements(
      cosAMinusB * expPlusC, zero, zero, iSinAMinusB * expPlusC, //
      zero, cosAPlusB * expMinusC, iSinAPlusB * expMinusC, zero, //
      zero, iSinAPlusB * expMinusC, cosAPlusB * expMinusC, zero, //
      iSinAMinusB * expPlusC, zero, zero, cosAMinusB * expPlusC);
}

bool TwoQubitWeylDecomposition::applySpecialization(
    const std::optional<double>& requestedFidelity) {
  bool flippedFromOriginal = false;
  const auto newSpecialization = bestSpecialization(*this, requestedFidelity);
  if (newSpecialization == Specialization::General) {
    return flippedFromOriginal;
  }

  switch (newSpecialization) {
  case Specialization::IdEquiv:
    a_ = 0.;
    b_ = 0.;
    c_ = 0.;
    k1l_ = k1l_ * k2l_;
    k2l_ = Matrix2x2::identity();
    k1r_ = k1r_ * k2r_;
    k2r_ = Matrix2x2::identity();
    break;
  case Specialization::SWAPEquiv:
    if (c_ > 0.) {
      k1l_ = k1l_ * k2r_;
      k1r_ = k1r_ * k2l_;
      k2l_ = Matrix2x2::identity();
      k2r_ = Matrix2x2::identity();
    } else {
      flippedFromOriginal = true;
      globalPhase_ += (PI / 2.0);
      k1l_ = k1l_ * I_PAULI_Z * k2r_;
      k1r_ = k1r_ * I_PAULI_Z * k2l_;
      k2l_ = Matrix2x2::identity();
      k2r_ = Matrix2x2::identity();
    }
    a_ = PI_OVER_4;
    b_ = PI_OVER_4;
    c_ = PI_OVER_4;
    break;
  case Specialization::PartialSWAPEquiv: {
    const auto closest = closestPartialSwap(a_, b_, c_);
    const auto k2lDagger = k2l_.adjoint();
    a_ = closest;
    b_ = closest;
    c_ = closest;
    k1l_ = k1l_ * k2l_;
    k1r_ = k1r_ * k2l_;
    k2r_ = k2lDagger * k2r_;
    k2l_ = Matrix2x2::identity();
    break;
  }
  case Specialization::PartialSWAPFlipEquiv: {
    const auto closest = closestPartialSwap(a_, b_, -c_);
    const auto k2lDagger = k2l_.adjoint();
    a_ = closest;
    b_ = closest;
    c_ = -closest;
    k1l_ = k1l_ * k2l_;
    k1r_ = k1r_ * I_PAULI_Z * k2l_ * I_PAULI_Z;
    k2r_ = I_PAULI_Z * k2lDagger * I_PAULI_Z * k2r_;
    k2l_ = Matrix2x2::identity();
    break;
  }
  case Specialization::ControlledEquiv: {
    const auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, EulerBasis::XYX);
    const auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
        anglesFromUnitary(k2r_, EulerBasis::XYX);
    b_ = 0.;
    c_ = 0.;
    globalPhase_ = globalPhase_ + k2lphase + k2rphase;
    k1l_ = k1l_ * RXOp::unitaryMatrix(k2lphi);
    k2l_ = RYOp::unitaryMatrix(k2ltheta) * RXOp::unitaryMatrix(k2llambda);
    k1r_ = k1r_ * RXOp::unitaryMatrix(k2rphi);
    k2r_ = RYOp::unitaryMatrix(k2rtheta) * RXOp::unitaryMatrix(k2rlambda);
    break;
  }
  case Specialization::MirrorControlledEquiv: {
    const auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, EulerBasis::ZYZ);
    const auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
        anglesFromUnitary(k2r_, EulerBasis::ZYZ);
    a_ = PI_OVER_4;
    b_ = PI_OVER_4;
    globalPhase_ = globalPhase_ + k2lphase + k2rphase;
    k1l_ = k1l_ * RZOp::unitaryMatrix(k2rphi);
    k2l_ = RYOp::unitaryMatrix(k2ltheta) * RZOp::unitaryMatrix(k2llambda);
    k1r_ = k1r_ * RZOp::unitaryMatrix(k2lphi);
    k2r_ = RYOp::unitaryMatrix(k2rtheta) * RZOp::unitaryMatrix(k2rlambda);
    break;
  }
  case Specialization::FSimaabEquiv: {
    const auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, EulerBasis::ZYZ);
    const auto ab = (a_ + b_) / 2.;
    a_ = ab;
    b_ = ab;
    globalPhase_ += k2lphase;
    k1l_ = k1l_ * RZOp::unitaryMatrix(k2lphi);
    k2l_ = RYOp::unitaryMatrix(k2ltheta) * RZOp::unitaryMatrix(k2llambda);
    k1r_ = k1r_ * RZOp::unitaryMatrix(k2lphi);
    k2r_ = RZOp::unitaryMatrix(-k2lphi) * k2r_;
    break;
  }
  case Specialization::FSimabbEquiv: {
    const auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, EulerBasis::XYX);
    const auto bc = (b_ + c_) / 2.;
    b_ = bc;
    c_ = bc;
    globalPhase_ += k2lphase;
    k1l_ = k1l_ * RXOp::unitaryMatrix(k2lphi);
    k2l_ = RYOp::unitaryMatrix(k2ltheta) * RXOp::unitaryMatrix(k2llambda);
    k1r_ = k1r_ * RXOp::unitaryMatrix(k2lphi);
    k2r_ = RXOp::unitaryMatrix(-k2lphi) * k2r_;
    break;
  }
  case Specialization::FSimabmbEquiv: {
    const auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, EulerBasis::XYX);
    const auto bc = (b_ - c_) / 2.;
    b_ = bc;
    c_ = -bc;
    globalPhase_ += k2lphase;
    k1l_ = k1l_ * RXOp::unitaryMatrix(k2lphi);
    k2l_ = RYOp::unitaryMatrix(k2ltheta) * RXOp::unitaryMatrix(k2llambda);
    k1r_ = k1r_ * I_PAULI_Z * RXOp::unitaryMatrix(k2lphi) * I_PAULI_Z;
    k2r_ = I_PAULI_Z * RXOp::unitaryMatrix(-k2lphi) * I_PAULI_Z * k2r_;
    break;
  }
  case Specialization::General:
    llvm_unreachable("unreachable specialization");
  }
  return flippedFromOriginal;
}

LogicalResult synthesizeUnitary2QWeyl(OpBuilder& builder, Location loc,
                                      Value qubit0, Value qubit1,
                                      const Matrix4x4& target,
                                      const NativeGateset& spec,
                                      Value& outQubit0, Value& outQubit1) {
  const auto native = spec.decomposeTarget(target);
  if (!native || !spec.eulerBasis) {
    return failure();
  }

  emitGPhaseIfNeeded(builder, loc, native->globalPhase);

  Value wire0 = qubit0;
  Value wire1 = qubit1;
  const auto& factors = native->singleQubitFactors;
  const std::uint8_t numBasisUses = native->numBasisUses;
  const std::size_t requiredFactors = singleQubitFactorCount(numBasisUses);
  if (factors.size() != requiredFactors) {
    llvm::reportFatalInternalError(llvm::formatv(
        "synthesizeUnitary2QWeyl: expected {0} single-qubit factors for "
        "numBasisUses = {1}, got {2}",
        requiredFactors, numBasisUses, factors.size()));
  }
  const bool emitCz = spec.entangler == NativeGateKind::CZ;
  const auto emitFactor = [&](Value& wire, std::size_t index) {
    const auto synthesized = synthesizeUnitary1QEuler(
        builder, loc, wire, factors[index], /*runSize=*/0,
        /*hasNonBasisGate=*/true, *spec.eulerBasis);
    if (!synthesized) {
      llvm::reportFatalInternalError(llvm::formatv(
          "synthesizeUnitary2QWeyl: euler synthesis failed for factor index "
          "{0} (layer {1}, qubit {2})",
          index, index / 2, (index % 2 == 0) ? 1 : 0));
    }
    wire = *synthesized;
  };
  const auto emitEntangler = [&]() {
    auto ctrlOp =
        CtrlOp::create(builder, loc, wire0, wire1, [&](Value targetQubit) {
          if (emitCz) {
            return ZOp::create(builder, loc, targetQubit).getOutputQubit(0);
          }
          return XOp::create(builder, loc, targetQubit).getOutputQubit(0);
        });
    wire0 = ctrlOp.getOutputControl(0);
    wire1 = ctrlOp.getOutputTarget(0);
  };

  for (std::uint8_t layer = 0; layer <= numBasisUses; ++layer) {
    emitFactor(wire1, static_cast<std::size_t>(2 * layer));
    emitFactor(wire0, static_cast<std::size_t>((2 * layer) + 1));
    if (layer < numBasisUses) {
      emitEntangler();
    }
  }

  outQubit0 = wire0;
  outQubit1 = wire1;
  return success();
}

} // namespace mlir::qco::decomposition
