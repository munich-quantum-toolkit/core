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

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Support/LLVM.h>

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

} // namespace

static constexpr auto DIAGONALIZATION_PRECISION = WEYL_TOLERANCE / 10;

/** Non-negative remainder of @p a modulo @p b (always in `[0, |b|)`). */
static double remEuclid(const double a, const double b) {
  if (b == 0.0) {
    llvm::reportFatalInternalError("remEuclid expects non-zero divisor");
  }
  const auto r = std::fmod(a, b);
  return (r < 0.0) ? r + std::abs(b) : r;
}

/** Maps `|tr(U^dag V)|` to average two-qubit gate fidelity. */
static double traceToFidelity(const Complex& trace) {
  const auto traceAbs = std::abs(trace);
  return (4.0 + (traceAbs * traceAbs)) / 20.0;
}

static constexpr double INV_SQRT2 = 1.0 / std::numbers::sqrt2;

static Matrix4x4 magicBasisTransform(const Matrix4x4& unitary,
                                     bool outOfMagicBasis) {
  const Matrix4x4 bNonNormalized = Matrix4x4::fromElements( //
      1, 1i, 0, 0,                                          //
      0, 0, 1i, 1,                                          //
      0, 0, 1i, -1,                                         //
      1, -1i, 0, 0);
  const Matrix4x4 bNonNormalizedDagger = Matrix4x4::fromElements( //
      0.5, 0, 0, 0.5,                                             //
      Complex{0.0, -0.5}, 0, 0, Complex{0.0, 0.5},                //
      0, Complex{0.0, -0.5}, Complex{0.0, -0.5}, 0,               //
      0, 0.5, -0.5, 0);
  if (outOfMagicBasis) {
    return bNonNormalizedDagger * unitary * bNonNormalized;
  }
  return bNonNormalized * unitary * bNonNormalizedDagger;
}

static double closestPartialSwap(double a, double b, double c) {
  auto m = (a + b + c) / 3.;
  auto [am, bm, cm] = std::array{a - m, b - m, c - m};
  auto [ab, bc, ca] = std::array{a - b, b - c, c - a};
  return m + (am * bm * cm * (6. + (ab * ab) + (bc * bc) + (ca * ca)) / 18.);
}

static std::pair<Matrix4x4, std::array<Complex, 4>>
diagonalizeComplexSymmetric(const Matrix4x4& m, double precision) {
  auto state = std::mt19937{2023};
  std::normal_distribution<double> dist;

  const auto mReal = m.realPart();
  const auto mImag = m.imagPart();

  double bestErr = 1e300;
  constexpr auto maxDiagonalizationAttempts = 100;
  for (int i = 0; i < maxDiagonalizationAttempts; ++i) {
    double randA{};
    double randB{};
    // Fixed perturbation coefficients for the first diagonalization attempt,
    // carried over from Qiskit's two-qubit Weyl decomposition (legacy Python
    // RNG values). The loop usually succeeds on this trial; fixing randA/randB
    // keeps behavior deterministic while later attempts sample the
    // distribution.
    if (i == 0) {
      randA = 1.2602066112249388;
      randB = 0.22317849046722027;
    } else {
      randA = dist(state);
      randB = dist(state);
    }
    std::array<double, 16> m2Real{};
    for (std::size_t k = 0; k < m2Real.size(); ++k) {
      m2Real[k] = (randA * mReal[k]) + (randB * mImag[k]);
    }
    const Matrix4x4 p = Matrix4x4::symmetricEigen4(m2Real).eigenvectors;
    const std::array<Complex, 4> d = (p.transpose() * m * p).diagonal();

    const auto compare = p * Matrix4x4::fromDiagonal(d) * p.transpose();
    {
      double err = 0.0;
      for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t cc = 0; cc < 4; ++cc) {
          err = std::max(err, std::abs(compare(r, cc) - m(r, cc)));
        }
      }
      bestErr = std::min(bestErr, err);
    }
    if (compare.isApprox(m, precision)) {
      assert((p.transpose() * p).isIdentity(WEYL_TOLERANCE));
      assert(std::abs(Matrix4x4::fromDiagonal(d).determinant() - 1.0) <
             WEYL_TOLERANCE);
      return std::make_pair(p, d);
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

  Matrix4x4 temp =
      specialUnitary * Matrix4x4::kron(Matrix2x2::identity(), rTConj);

  Matrix2x2 l =
      Matrix2x2::fromElements(temp(0, 0), temp(0, 2), temp(2, 0), temp(2, 2));
  auto detL = l.determinant();
  if (std::abs(detL) < 0.9) {
    llvm::reportFatalInternalError(
        "decomposeTwoQubitProductGate: unable to decompose: detL < 0.9");
  }
  l *= (1.0 / std::sqrt(detL));
  auto phase = std::arg(detL) / 2.;

  return {l, r, phase};
}

static std::complex<double> getTrace(double a, double b, double c, double ap,
                                     double bp, double cp) {
  auto da = a - ap;
  auto db = b - bp;
  auto dc = c - cp;
  return 4. * std::complex<double>{std::cos(da) * std::cos(db) * std::cos(dc),
                                   std::sin(da) * std::sin(db) * std::sin(dc)};
}

static Specialization
bestSpecialization(const TwoQubitWeylDecomposition& decomposition,
                   const std::optional<double>& requestedFidelity) {
  auto isClose = [&](double ap, double bp, double cp) -> bool {
    auto tr = getTrace(decomposition.a(), decomposition.b(), decomposition.c(),
                       ap, bp, cp);
    if (requestedFidelity) {
      return traceToFidelity(tr) >= *requestedFidelity;
    }
    return false;
  };

  auto closestAbc = closestPartialSwap(decomposition.a(), decomposition.b(),
                                       decomposition.c());
  auto closestAbMinusC = closestPartialSwap(
      decomposition.a(), decomposition.b(), -decomposition.c());

  if (isClose(0., 0., 0.)) {
    return Specialization::IdEquiv;
  }
  if (isClose((std::numbers::pi / 4.0), (std::numbers::pi / 4.0),
              (std::numbers::pi / 4.0)) ||
      isClose((std::numbers::pi / 4.0), (std::numbers::pi / 4.0),
              -(std::numbers::pi / 4.0))) {
    return Specialization::SWAPEquiv;
  }
  if (isClose(closestAbc, closestAbc, closestAbc)) {
    return Specialization::PartialSWAPEquiv;
  }
  if (isClose(closestAbMinusC, closestAbMinusC, -closestAbMinusC)) {
    return Specialization::PartialSWAPFlipEquiv;
  }
  if (isClose(decomposition.a(), 0., 0.)) {
    return Specialization::ControlledEquiv;
  }
  if (isClose((std::numbers::pi / 4.0), (std::numbers::pi / 4.0),
              decomposition.c())) {
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

static bool relativeEq(double lhs, double rhs, double epsilon,
                       double maxRelative) {
  if (lhs == rhs) {
    return true;
  }
  if (std::isinf(lhs) || std::isinf(rhs)) {
    return false;
  }
  auto absDiff = std::abs(lhs - rhs);
  if (absDiff <= epsilon) {
    return true;
  }
  auto absLhs = std::abs(lhs);
  auto absRhs = std::abs(rhs);
  if (absRhs > absLhs) {
    return absDiff <= absRhs * maxRelative;
  }
  return absDiff <= absLhs * maxRelative;
}

TwoQubitWeylDecomposition
TwoQubitWeylDecomposition::create(const Matrix4x4& unitaryMatrix,
                                  std::optional<double> fidelity) {
  auto u = unitaryMatrix;
  auto detU = u.determinant();
  // Project into SU(4): det^{-1/4} removes global phase; arg(det)/4 is tracked.
  auto detPow = std::pow(detU, -0.25);
  u *= detPow;
  auto globalPhase = std::arg(detU) / 4.;

  auto detNormalized = u.determinant();
  if (std::abs(detNormalized - Complex{1.0, 0.0}) > WEYL_TOLERANCE &&
      std::abs(detNormalized) > WEYL_TOLERANCE) {
    u *= std::pow(detNormalized, -0.25);
  }

  auto uP = magicBasisTransform(u, /*outOfMagicBasis=*/true);
  const Matrix4x4 m2 = uP.transpose() * uP;

  auto [p, d] = diagonalizeComplexSymmetric(m2, DIAGONALIZATION_PRECISION);

  // Map eigenvalue phases to Weyl coordinates in [0, 2*pi).
  constexpr double pi = std::numbers::pi;
  std::array<double, 4> dReal{};
  for (std::size_t i = 0; i < d.size(); ++i) {
    dReal[i] = -std::arg(d[i]) / 2.0;
  }
  dReal[3] = -dReal[0] - dReal[1] - dReal[2];
  std::array<double, 3> cs{};
  for (std::size_t i = 0; i < cs.size(); ++i) {
    cs[i] = remEuclid((dReal[i] + dReal[3]) / 2.0, 2.0 * pi);
  }

  // Sort coordinates by min(x mod pi/2, pi/2 - x mod pi/2).
  std::array<double, 3> cstemp{};
  for (std::size_t i = 0; i < cs.size(); ++i) {
    const auto tmp = remEuclid(cs[i], pi / 2.0);
    cstemp[i] = std::min(tmp, (pi / 2.0) - tmp);
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
  assert(std::abs(p.determinant() - 1.0) < WEYL_TOLERANCE);

  std::array<Complex, 4> tempDiag{};
  for (std::size_t k = 0; k < tempDiag.size(); ++k) {
    tempDiag[k] = std::exp(1i * dReal[k]);
  }
  const Matrix4x4 temp = Matrix4x4::fromDiagonal(tempDiag);

  Matrix4x4 k1 = uP * p * temp;
  // Orthogonality checks use WEYL_TOLERANCE (diagonalization residual scale).
  assert((k1.transpose() * k1).isIdentity(WEYL_TOLERANCE));
  assert(k1.determinant().real() > 0.0);
  k1 = magicBasisTransform(k1, /*outOfMagicBasis=*/false);

  Matrix4x4 k2 = p.adjoint();
  assert((k2.transpose() * k2).isIdentity(WEYL_TOLERANCE));
  assert(k2.determinant().real() > 0.0);
  k2 = magicBasisTransform(k2, /*outOfMagicBasis=*/false);

  std::array<Complex, 4> tempConjDiag{};
  for (std::size_t k = 0; k < tempConjDiag.size(); ++k) {
    tempConjDiag[k] = std::conj(tempDiag[k]);
  }
  assert((k1 *
          magicBasisTransform(Matrix4x4::fromDiagonal(tempConjDiag),
                              /*outOfMagicBasis=*/false) *
          k2)
             .isApprox(u, WEYL_TOLERANCE));

  auto [K1l, K1r, phaseL] = decomposeTwoQubitProductGate(k1);
  auto [K2l, K2r, phaseR] = decomposeTwoQubitProductGate(k2);
  assert(Matrix4x4::kron(K1l, K1r).isApprox(k1, WEYL_TOLERANCE));
  assert(Matrix4x4::kron(K2l, K2r).isApprox(k2, WEYL_TOLERANCE));
  globalPhase += phaseL + phaseR;

  // Map into the Weyl chamber.
  if (cs[0] > (pi / 2.0)) {
    cs[0] -= 3.0 * (pi / 2.0);
    K1l = K1l * iPauliY();
    K1r = K1r * iPauliY();
    globalPhase += (pi / 2.0);
  }
  if (cs[1] > (pi / 2.0)) {
    cs[1] -= 3.0 * (pi / 2.0);
    K1l = K1l * iPauliX();
    K1r = K1r * iPauliX();
    globalPhase += (pi / 2.0);
  }
  auto conjs = 0;
  if (cs[0] > (pi / 4.0)) {
    cs[0] = (pi / 2.0) - cs[0];
    K1l = K1l * iPauliY();
    K2r = iPauliY() * K2r;
    conjs += 1;
    globalPhase -= (pi / 2.0);
  }
  if (cs[1] > (pi / 4.0)) {
    cs[1] = (pi / 2.0) - cs[1];
    K1l = K1l * iPauliX();
    K2r = iPauliX() * K2r;
    conjs += 1;
    globalPhase += (pi / 2.0);
    if (conjs == 1) {
      globalPhase -= pi;
    }
  }
  if (cs[2] > (pi / 2.0)) {
    cs[2] -= 3.0 * (pi / 2.0);
    K1l = K1l * iPauliZ();
    K1r = K1r * iPauliZ();
    globalPhase += (pi / 2.0);
    if (conjs == 1) {
      globalPhase -= pi;
    }
  }
  if (conjs == 1) {
    cs[2] = (pi / 2.0) - cs[2];
    K1l = K1l * iPauliZ();
    K2r = iPauliZ() * K2r;
    globalPhase += (pi / 2.0);
  }
  if (cs[2] > (pi / 4.0)) {
    cs[2] -= (pi / 2.0);
    K1l = K1l * iPauliZ();
    K1r = K1r * iPauliZ();
    globalPhase -= (pi / 2.0);
  }

  // Bind Weyl coordinates to the canonical gate.
  auto [a, b, c] = std::tie(cs[1], cs[0], cs[2]);

  TwoQubitWeylDecomposition decomposition;
  decomposition.a_ = a;
  decomposition.b_ = b;
  decomposition.c_ = c;
  decomposition.globalPhase_ = globalPhase;
  decomposition.k1l_ = K1l;
  decomposition.k2l_ = K2l;
  decomposition.k1r_ = K1r;
  decomposition.k2r_ = K2r;

  assert((Matrix4x4::kron(K1l, K1r) * decomposition.getCanonicalMatrix() *
          Matrix4x4::kron(K2l, K2r) * std::exp(Complex{0.0, 1.0} * globalPhase))
             .isApprox(unitaryMatrix, WEYL_TOLERANCE));

  auto flippedFromOriginal = decomposition.applySpecialization(fidelity);

  auto getTraceValue = [&]() {
    if (flippedFromOriginal) {
      return getTrace((pi / 2.0) - a, b, -c, decomposition.a_, decomposition.b_,
                      decomposition.c_);
    }
    return getTrace(a, b, c, decomposition.a_, decomposition.b_,
                    decomposition.c_);
  };
  const auto trace = getTraceValue();
  const double calculatedFidelity = traceToFidelity(trace);
  if (fidelity && calculatedFidelity + 1.0e-13 < *fidelity) {
    llvm::reportFatalInternalError(llvm::formatv(
        "TwoQubitWeylDecomposition: Calculated fidelity of "
        "specialization is worse than requested fidelity ({0:F4} vs {1:F4})!",
        calculatedFidelity, *fidelity));
  }
  decomposition.globalPhase_ += std::arg(trace);

  assert((Matrix4x4::kron(decomposition.k1l_, decomposition.k1r_) *
          decomposition.getCanonicalMatrix() *
          Matrix4x4::kron(decomposition.k2l_, decomposition.k2r_) *
          std::exp(Complex{0.0, 1.0} * decomposition.globalPhase_))
             .isApprox(unitaryMatrix, WEYL_TOLERANCE));

  return decomposition;
}

Matrix4x4 TwoQubitWeylDecomposition::getCanonicalMatrix(double a, double b,
                                                        double c) {
  // U_d(a,b,c) = exp(i(a XX + b YY + c ZZ)); `-2*a/b/c` matches RXX/RYY/RZZ
  // gate convention; product order matches Qiskit (RZZ · RYY · RXX).
  const auto xx = rxxMatrix(-2.0 * a);
  const auto yy = ryyMatrix(-2.0 * b);
  const auto zz = rzzMatrix(-2.0 * c);
  return zz * yy * xx;
}

bool TwoQubitWeylDecomposition::applySpecialization(
    const std::optional<double>& requestedFidelity) {
  bool flippedFromOriginal = false;
  const auto newSpecialization = bestSpecialization(*this, requestedFidelity);
  if (newSpecialization == Specialization::General) {
    return flippedFromOriginal;
  }

  if (newSpecialization == Specialization::IdEquiv) {
    a_ = 0.;
    b_ = 0.;
    c_ = 0.;
    k1l_ = k1l_ * k2l_;
    k2l_ = Matrix2x2::identity();
    k1r_ = k1r_ * k2r_;
    k2r_ = Matrix2x2::identity();
  } else if (newSpecialization == Specialization::SWAPEquiv) {
    if (c_ > 0.) {
      k1l_ = k1l_ * k2r_;
      k1r_ = k1r_ * k2l_;
      k2l_ = Matrix2x2::identity();
      k2r_ = Matrix2x2::identity();
    } else {
      flippedFromOriginal = true;

      globalPhase_ += (std::numbers::pi / 2.0);
      k1l_ = k1l_ * iPauliZ() * k2r_;
      k1r_ = k1r_ * iPauliZ() * k2l_;
      k2l_ = Matrix2x2::identity();
      k2r_ = Matrix2x2::identity();
    }
    a_ = (std::numbers::pi / 4.0);
    b_ = (std::numbers::pi / 4.0);
    c_ = (std::numbers::pi / 4.0);
  } else if (newSpecialization == Specialization::PartialSWAPEquiv) {
    auto closest = closestPartialSwap(a_, b_, c_);
    auto k2lDagger = k2l_.adjoint();

    a_ = closest;
    b_ = closest;
    c_ = closest;
    k1l_ = k1l_ * k2l_;
    k1r_ = k1r_ * k2l_;
    k2r_ = k2lDagger * k2r_;
    k2l_ = Matrix2x2::identity();
  } else if (newSpecialization == Specialization::PartialSWAPFlipEquiv) {
    auto closest = closestPartialSwap(a_, b_, -c_);
    auto k2lDagger = k2l_.adjoint();

    a_ = closest;
    b_ = closest;
    c_ = -closest;
    k1l_ = k1l_ * k2l_;
    k1r_ = k1r_ * iPauliZ() * k2l_ * iPauliZ();
    k2r_ = iPauliZ() * k2lDagger * iPauliZ() * k2r_;
    k2l_ = Matrix2x2::identity();
  } else if (newSpecialization == Specialization::ControlledEquiv) {
    const EulerBasis eulerBasis = EulerBasis::XYX;
    const auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, eulerBasis);
    const auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
        anglesFromUnitary(k2r_, EulerBasis::XYX);
    b_ = 0.;
    c_ = 0.;
    globalPhase_ = globalPhase_ + k2lphase + k2rphase;
    k1l_ = k1l_ * rxMatrix(k2lphi);
    k2l_ = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
    k1r_ = k1r_ * rxMatrix(k2rphi);
    k2r_ = ryMatrix(k2rtheta) * rxMatrix(k2rlambda);
  } else if (newSpecialization == Specialization::MirrorControlledEquiv) {
    const auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, EulerBasis::ZYZ);
    const auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
        anglesFromUnitary(k2r_, EulerBasis::ZYZ);
    a_ = (std::numbers::pi / 4.0);
    b_ = (std::numbers::pi / 4.0);
    globalPhase_ = globalPhase_ + k2lphase + k2rphase;
    k1l_ = k1l_ * rzMatrix(k2rphi);
    k2l_ = ryMatrix(k2ltheta) * rzMatrix(k2llambda);
    k1r_ = k1r_ * rzMatrix(k2lphi);
    k2r_ = ryMatrix(k2rtheta) * rzMatrix(k2rlambda);
  } else if (newSpecialization == Specialization::FSimaabEquiv) {
    auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, EulerBasis::ZYZ);
    auto ab = (a_ + b_) / 2.;

    a_ = ab;
    b_ = ab;
    globalPhase_ = globalPhase_ + k2lphase;
    k1l_ = k1l_ * rzMatrix(k2lphi);
    k2l_ = ryMatrix(k2ltheta) * rzMatrix(k2llambda);
    k1r_ = k1r_ * rzMatrix(k2lphi);
    k2r_ = rzMatrix(-k2lphi) * k2r_;
  } else if (newSpecialization == Specialization::FSimabbEquiv) {
    auto eulerBasis = EulerBasis::XYX;
    auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, eulerBasis);
    auto bc = (b_ + c_) / 2.;

    b_ = bc;
    c_ = bc;
    globalPhase_ = globalPhase_ + k2lphase;
    k1l_ = k1l_ * rxMatrix(k2lphi);
    k2l_ = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
    k1r_ = k1r_ * rxMatrix(k2lphi);
    k2r_ = rxMatrix(-k2lphi) * k2r_;
  } else if (newSpecialization == Specialization::FSimabmbEquiv) {
    auto eulerBasis = EulerBasis::XYX;
    auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, eulerBasis);
    auto bc = (b_ - c_) / 2.;

    b_ = bc;
    c_ = -bc;
    globalPhase_ = globalPhase_ + k2lphase;
    k1l_ = k1l_ * rxMatrix(k2lphi);
    k2l_ = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
    k1r_ = k1r_ * iPauliZ() * rxMatrix(k2lphi) * iPauliZ();
    k2r_ = iPauliZ() * rxMatrix(-k2lphi) * iPauliZ() * k2r_;
  } else {
    llvm::reportFatalInternalError(
        "Unknown specialization for Weyl decomposition!");
  }
  return flippedFromOriginal;
}

TwoQubitBasisDecomposer
TwoQubitBasisDecomposer::create(const Matrix4x4& basisMatrix,
                                double basisFidelity) {
  const Matrix2x2 k12RArr = Matrix2x2::fromElements(
      1i * INV_SQRT2, INV_SQRT2, -INV_SQRT2, -1i * INV_SQRT2);
  const Matrix2x2 k12LArr =
      Matrix2x2::fromElements(Complex{0.5, 0.5}, Complex{0.5, 0.5},
                              Complex{-0.5, 0.5}, Complex{0.5, -0.5});

  const auto basisDecomposer =
      TwoQubitWeylDecomposition::create(basisMatrix, basisFidelity);
  const auto isSuperControlled =
      relativeEq(basisDecomposer.a(), std::numbers::pi / 4.0, 1e-13, 1e-09) &&
      relativeEq(basisDecomposer.c(), 0.0, 1e-13, 1e-09);

  auto b = basisDecomposer.b();
  Complex temp{0.5, -0.5};
  const Matrix2x2 k11l = Matrix2x2::fromElements(
      temp * (-1i * std::exp(-1i * b)), temp * std::exp(-1i * b),
      temp * (-1i * std::exp(1i * b)), temp * -std::exp(1i * b));
  const Matrix2x2 k11r = Matrix2x2::fromElements(
      INV_SQRT2 * (1i * std::exp(-1i * b)), INV_SQRT2 * -std::exp(-1i * b),
      INV_SQRT2 * std::exp(1i * b), INV_SQRT2 * (-1i * std::exp(1i * b)));
  const Matrix2x2 k32lK21l = Matrix2x2::fromElements(
      INV_SQRT2 * Complex{1., std::cos(2. * b)},
      INV_SQRT2 * (1i * std::sin(2. * b)), INV_SQRT2 * (1i * std::sin(2. * b)),
      INV_SQRT2 * Complex{1., -std::cos(2. * b)});
  temp = Complex{0.5, 0.5};
  const Matrix2x2 k21r = Matrix2x2::fromElements(
      temp * (-1i * std::exp(-2i * b)), temp * std::exp(-2i * b),
      temp * (1i * std::exp(2i * b)), temp * std::exp(2i * b));
  const Matrix2x2 k22l =
      Matrix2x2::fromElements(INV_SQRT2, -INV_SQRT2, INV_SQRT2, INV_SQRT2);
  const Matrix2x2 k22r = Matrix2x2::fromElements(0, 1, -1, 0);
  const Matrix2x2 k31l = Matrix2x2::fromElements(
      INV_SQRT2 * std::exp(-1i * b), INV_SQRT2 * std::exp(-1i * b),
      INV_SQRT2 * -std::exp(1i * b), INV_SQRT2 * std::exp(1i * b));
  const Matrix2x2 k31r = Matrix2x2::fromElements(1i * std::exp(1i * b), 0, 0,
                                                 -1i * std::exp(-1i * b));
  const Matrix2x2 k32r = Matrix2x2::fromElements(
      temp * std::exp(1i * b), temp * -std::exp(-1i * b),
      temp * (-1i * std::exp(1i * b)), temp * (-1i * std::exp(-1i * b)));
  auto k1lDagger = basisDecomposer.k1l().adjoint();
  auto k1rDagger = basisDecomposer.k1r().adjoint();
  auto k2lDagger = basisDecomposer.k2l().adjoint();
  auto k2rDagger = basisDecomposer.k2r().adjoint();
  auto u0l = k31l * k1lDagger;
  auto u0r = k31r * k1rDagger;
  auto u1l = k2lDagger * k32lK21l * k1lDagger;
  auto u1ra = k2rDagger * k32r;
  auto u1rb = k21r * k1rDagger;
  auto u2la = k2lDagger * k22l;
  auto u2lb = k11l * k1lDagger;
  auto u2ra = k2rDagger * k22r;
  auto u2rb = k11r * k1rDagger;
  auto u3l = k2lDagger * k12LArr;
  auto u3r = k2rDagger * k12RArr;
  auto q0l = k12LArr.adjoint() * k1lDagger;
  auto q0r = k12RArr.adjoint() * iPauliZ() * k1rDagger;
  auto q1la = k2lDagger * k11l.adjoint();
  auto q1lb = k11l * k1lDagger;
  auto q1ra = k2rDagger * iPauliZ() * k11r.adjoint();
  auto q1rb = k11r * k1rDagger;
  auto q2l = k2lDagger * k12LArr;
  auto q2r = k2rDagger * k12RArr;

  TwoQubitBasisDecomposer decomposer;
  decomposer.basisFidelity = basisFidelity;
  decomposer.basisDecomposer = basisDecomposer;
  decomposer.isSuperControlled = isSuperControlled;
  decomposer.smb.u0l = u0l;
  decomposer.smb.u0r = u0r;
  decomposer.smb.u1l = u1l;
  decomposer.smb.u1ra = u1ra;
  decomposer.smb.u1rb = u1rb;
  decomposer.smb.u2la = u2la;
  decomposer.smb.u2lb = u2lb;
  decomposer.smb.u2ra = u2ra;
  decomposer.smb.u2rb = u2rb;
  decomposer.smb.u3l = u3l;
  decomposer.smb.u3r = u3r;
  decomposer.smb.q0l = q0l;
  decomposer.smb.q0r = q0r;
  decomposer.smb.q1la = q1la;
  decomposer.smb.q1lb = q1lb;
  decomposer.smb.q1ra = q1ra;
  decomposer.smb.q1rb = q1rb;
  decomposer.smb.q2l = q2l;
  decomposer.smb.q2r = q2r;
  return decomposer;
}

std::optional<TwoQubitNativeDecomposition>
TwoQubitBasisDecomposer::twoQubitDecompose(
    const TwoQubitWeylDecomposition& targetDecomposition,
    std::optional<std::uint8_t> numBasisGateUses) const {
  auto traces = this->traces(targetDecomposition);
  auto getDefaultNbasis = [&]() -> std::uint8_t {
    // Maximize traceToFidelity(traces[i]) * basisFidelity^i over i in {0..3}.
    auto bestValue = std::numeric_limits<double>::lowest();
    auto bestIndex = -1;
    for (int i = 0; std::cmp_less(i, traces.size()); ++i) {
      auto value = traceToFidelity(traces[i]) * std::pow(basisFidelity, i);
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
    return static_cast<std::uint8_t>(bestIndex);
  };
  auto bestNbasis = numBasisGateUses.value_or(getDefaultNbasis());
  if (bestNbasis > 1 && !isSuperControlled) {
    // More than one basis gate requires a super-controlled basis (e.g. CX).
    return std::nullopt;
  }
  auto chooseDecomposition = [&]() {
    if (bestNbasis == 0) {
      return decomp0(targetDecomposition);
    }
    if (bestNbasis == 1) {
      return decomp1(targetDecomposition);
    }
    if (bestNbasis == 2) {
      return decomp2Supercontrolled(targetDecomposition);
    }
    if (bestNbasis == 3) {
      return decomp3Supercontrolled(targetDecomposition);
    }
    llvm::reportFatalInternalError(
        "Invalid number of basis gates to use in basis decomposition (" +
        llvm::Twine(bestNbasis) + ")!");
    llvm_unreachable("");
  };
  SmallVector<Matrix2x2, 8> factors = chooseDecomposition();

  double globalPhase = targetDecomposition.globalPhase();
  globalPhase -= bestNbasis * basisDecomposer.globalPhase();
  if (bestNbasis == 2) {
    // Two-basis template is exact up to a pi global-phase offset.
    globalPhase += std::numbers::pi;
  }
  globalPhase = remEuclid(globalPhase, 2.0 * std::numbers::pi);

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
  // One basis gate; reliable only when the target is in the Weyl chamber.
  return SmallVector<Matrix2x2, 8>{
      basisDecomposer.k2r().adjoint() * target.k2r(),
      basisDecomposer.k2l().adjoint() * target.k2l(),
      target.k1r() * basisDecomposer.k1r().adjoint(),
      target.k1l() * basisDecomposer.k1l().adjoint(),
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
      smb.q2r * target.k2r(),
      smb.q2l * target.k2l(),
      smb.q1ra * rzMatrix(2. * target.b()) * smb.q1rb,
      smb.q1la * rzMatrix(-2. * target.a()) * smb.q1lb,
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
      smb.u2ra * rzMatrix(2. * target.b()) * smb.u2rb,
      smb.u2la * rzMatrix(-2. * target.a()) * smb.u2lb,
      smb.u1ra * rzMatrix(-2. * target.c()) * smb.u1rb,
      smb.u1l,
      target.k1r() * smb.u0r,
      target.k1l() * smb.u0l,
  };
}

std::array<std::complex<double>, 4>
TwoQubitBasisDecomposer::traces(const TwoQubitWeylDecomposition& target) const {
  // Hilbert-Schmidt traces for 0..3 basis uses; index i == numBasisUses.
  // i=0: identity; i=1: one basis gate; i=2: two uses; i=3: exact (trace=4).
  return {
      4. * std::complex<double>{std::cos(target.a()) * std::cos(target.b()) *
                                    std::cos(target.c()),
                                std::sin(target.a()) * std::sin(target.b()) *
                                    std::sin(target.c())},
      4. *
          std::complex<double>{std::cos((std::numbers::pi / 4.0) - target.a()) *
                                   std::cos(basisDecomposer.b() - target.b()) *
                                   std::cos(target.c()),
                               std::sin((std::numbers::pi / 4.0) - target.a()) *
                                   std::sin(basisDecomposer.b() - target.b()) *
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
  const auto weyl = TwoQubitWeylDecomposition::create(target, std::nullopt);
  return decomposer.twoQubitDecompose(weyl, numBasisUses);
}

} // namespace mlir::qco::decomposition
