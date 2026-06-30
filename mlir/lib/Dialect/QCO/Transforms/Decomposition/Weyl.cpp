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

static double remEuclid(const double a, const double b) {
  if (b == 0.0) {
    llvm::reportFatalInternalError("remEuclid expects non-zero divisor");
  }
  const auto r = std::fmod(a, b);
  return (r < 0.0) ? r + std::abs(b) : r;
}

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
  const auto [am, bm, cm] = std::array{a - m, b - m, c - m};
  const auto [ab, bc, ca] = std::array{a - b, b - c, c - a};
  return m + (am * bm * cm * (6. + (ab * ab) + (bc * bc) + (ca * ca)) / 18.);
}

static std::pair<Matrix4x4, std::array<Complex, 4>>
diagonalizeComplexSymmetric(const Matrix4x4& m,
                            double precision = WEYL_DIAGONALIZATION_TOLERANCE) {
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

  const auto closestAbc = closestPartialSwap(
      decomposition.a(), decomposition.b(), decomposition.c());
  const auto closestAbMinusC = closestPartialSwap(
      decomposition.a(), decomposition.b(), -decomposition.c());

  if (isClose(0., 0., 0.)) {
    return Specialization::IdEquiv;
  }
  if (isClose(PI_OVER_4, PI_OVER_4, PI_OVER_4) ||
      isClose(PI_OVER_4, PI_OVER_4, -PI_OVER_4)) {
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
  std::array<Complex, 4> tempDiag{};
  for (std::size_t k = 0; k < tempDiag.size(); ++k) {
    tempDiag[k] = std::exp(1i * dReal[k]);
  }
  const Matrix4x4 temp = Matrix4x4::fromDiagonal(tempDiag);

  Matrix4x4 k1 = uP * p * temp;
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

  auto [k1l, k1r, phaseL] = decomposeTwoQubitProductGate(k1);
  auto [k2l, k2r, phaseR] = decomposeTwoQubitProductGate(k2);
  assert(Matrix4x4::kron(k1l, k1r).isApprox(k1, WEYL_TOLERANCE));
  assert(Matrix4x4::kron(k2l, k2r).isApprox(k2, WEYL_TOLERANCE));
  globalPhase += phaseL + phaseR;

  if (cs[0] > (PI / 2.0)) {
    cs[0] -= 3.0 * (PI / 2.0);
    k1l = k1l * Complex{0.0, 1.0} * YOp::getUnitaryMatrix();
    k1r = k1r * Complex{0.0, 1.0} * YOp::getUnitaryMatrix();
    globalPhase += (PI / 2.0);
  }
  if (cs[1] > (PI / 2.0)) {
    cs[1] -= 3.0 * (PI / 2.0);
    k1l = k1l * Complex{0.0, 1.0} * XOp::getUnitaryMatrix();
    k1r = k1r * Complex{0.0, 1.0} * XOp::getUnitaryMatrix();
    globalPhase += (PI / 2.0);
  }
  auto conjs = 0;
  if (cs[0] > PI_OVER_4) {
    cs[0] = (PI / 2.0) - cs[0];
    k1l = k1l * Complex{0.0, 1.0} * YOp::getUnitaryMatrix();
    k2r = Complex{0.0, 1.0} * YOp::getUnitaryMatrix() * k2r;
    conjs += 1;
    globalPhase -= (PI / 2.0);
  }
  if (cs[1] > PI_OVER_4) {
    cs[1] = (PI / 2.0) - cs[1];
    k1l = k1l * Complex{0.0, 1.0} * XOp::getUnitaryMatrix();
    k2r = Complex{0.0, 1.0} * XOp::getUnitaryMatrix() * k2r;
    conjs += 1;
    globalPhase += (PI / 2.0);
    if (conjs == 1) {
      globalPhase -= PI;
    }
  }
  if (cs[2] > (PI / 2.0)) {
    cs[2] -= 3.0 * (PI / 2.0);
    k1l = k1l * Complex{0.0, 1.0} * ZOp::getUnitaryMatrix();
    k1r = k1r * Complex{0.0, 1.0} * ZOp::getUnitaryMatrix();
    globalPhase += (PI / 2.0);
    if (conjs == 1) {
      globalPhase -= PI;
    }
  }
  if (conjs == 1) {
    cs[2] = (PI / 2.0) - cs[2];
    k1l = k1l * Complex{0.0, 1.0} * ZOp::getUnitaryMatrix();
    k2r = Complex{0.0, 1.0} * ZOp::getUnitaryMatrix() * k2r;
    globalPhase += (PI / 2.0);
  }
  if (cs[2] > PI_OVER_4) {
    cs[2] -= (PI / 2.0);
    k1l = k1l * Complex{0.0, 1.0} * ZOp::getUnitaryMatrix();
    k1r = k1r * Complex{0.0, 1.0} * ZOp::getUnitaryMatrix();
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

  assert((Matrix4x4::kron(decomposition.k1l_, decomposition.k1r_) *
          decomposition.getCanonicalMatrix() *
          Matrix4x4::kron(decomposition.k2l_, decomposition.k2r_) *
          std::exp(Complex{0.0, 1.0} * decomposition.globalPhase_))
             .isApprox(unitaryMatrix, WEYL_TOLERANCE));

  const bool flippedFromOriginal = decomposition.applySpecialization(fidelity);
  decomposition.finalizeSpecializationPhase(flippedFromOriginal, chamber.a,
                                            chamber.b, chamber.c, fidelity);

  const auto reconstructed =
      Matrix4x4::kron(decomposition.k1l_, decomposition.k1r_) *
      decomposition.getCanonicalMatrix() *
      Matrix4x4::kron(decomposition.k2l_, decomposition.k2r_) *
      std::exp(Complex{0.0, 1.0} * decomposition.globalPhase_);
  if (!reconstructed.isApprox(unitaryMatrix, WEYL_TOLERANCE)) {
    llvm::reportFatalInternalError(
        "TwoQubitWeylDecomposition: failed to reconstruct unitary after "
        "specialization");
  }

  return decomposition;
}

Matrix4x4 TwoQubitWeylDecomposition::getCanonicalMatrix(double a, double b,
                                                        double c) {
  return RZZOp::unitaryMatrix(-2.0 * c) * RYYOp::unitaryMatrix(-2.0 * b) *
         RXXOp::unitaryMatrix(-2.0 * a);
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
      k1l_ = k1l_ * Complex{0.0, 1.0} * ZOp::getUnitaryMatrix() * k2r_;
      k1r_ = k1r_ * Complex{0.0, 1.0} * ZOp::getUnitaryMatrix() * k2l_;
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
    k1r_ = k1r_ * Complex{0.0, 1.0} * ZOp::getUnitaryMatrix() * k2l_ *
           Complex{0.0, 1.0} * ZOp::getUnitaryMatrix();
    k2r_ = Complex{0.0, 1.0} * ZOp::getUnitaryMatrix() * k2lDagger *
           Complex{0.0, 1.0} * ZOp::getUnitaryMatrix() * k2r_;
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
    k1r_ = k1r_ * Complex{0.0, 1.0} * ZOp::getUnitaryMatrix() *
           RXOp::unitaryMatrix(k2lphi) * Complex{0.0, 1.0} *
           ZOp::getUnitaryMatrix();
    k2r_ = Complex{0.0, 1.0} * ZOp::getUnitaryMatrix() *
           RXOp::unitaryMatrix(-k2lphi) * Complex{0.0, 1.0} *
           ZOp::getUnitaryMatrix() * k2r_;
    break;
  }
  case Specialization::General:
    llvm_unreachable("unreachable specialization");
  }
  return flippedFromOriginal;
}

} // namespace mlir::qco::decomposition
