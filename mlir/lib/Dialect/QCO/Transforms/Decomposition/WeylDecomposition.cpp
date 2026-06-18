/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/WeylDecomposition.h"

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"
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
#include <numbers>
#include <optional>
#include <random>
#include <tuple>
#include <utility>

namespace mlir::qco::decomposition {

using namespace std::complex_literals;

TwoQubitWeylDecomposition
TwoQubitWeylDecomposition::create(const Matrix4x4& unitaryMatrix,
                                  std::optional<double> fidelity) {
  auto u = unitaryMatrix;
  auto detU = u.determinant();
  // Project into SU(4) by dividing out the fourth root of det(U): for a 4x4
  // unitary, |det(U)| == 1 so `det^{-1/4}` both enforces det == 1 and removes
  // the global phase. The extracted phase is tracked separately in
  // `globalPhase` (quarter of arg(det) to match the fourth-root choice) so the
  // caller can reconstruct the original matrix exactly if needed.
  auto detPow = std::pow(detU, -0.25);
  u *= detPow; // remove global phase from unitary matrix
  auto globalPhase = std::arg(detU) / 4.;

  // Numerical drift can still leave tiny determinant errors after root
  // normalization. Re-normalize once more instead of aborting.
  auto detNormalized = u.determinant();
  if (std::abs(detNormalized - Complex{1.0, 0.0}) > SANITY_CHECK_PRECISION &&
      std::abs(detNormalized) > SANITY_CHECK_PRECISION) {
    u *= std::pow(detNormalized, -0.25);
  }

  // transform unitary matrix to magic basis; this enables two properties:
  // 1. if uP ∈ SO(4), V = A ⊗ B (SO(4) → SU(2) ⊗ SU(2))
  // 2. magic basis diagonalizes canonical gate, allowing calculation of
  //    canonical gate parameters later on
  auto uP = magicBasisTransform(u, MagicBasisTransform::OutOf);
  const Matrix4x4 m2 = uP.transpose() * uP;

  // diagonalization yields eigenvectors (p) and eigenvalues (d);
  // p is used to calculate K1/K2 (and thus the single-qubit gates
  // surrounding the canonical gate); d is used to determine the Weyl
  // coordinates and thus the parameters of the canonical gate
  auto [p, d] = diagonalizeComplexSymmetric(m2, DIAGONALIZATION_PRECISION);

  // extract Weyl coordinates from eigenvalues, map to [0, 2*pi)
  constexpr double pi = std::numbers::pi;
  std::array<double, 4> dReal{};
  for (std::size_t i = 0; i < d.size(); ++i) {
    dReal[i] = -std::arg(d[i]) / 2.0;
  }
  dReal[3] = -dReal[0] - dReal[1] - dReal[2];
  std::array<double, 3> cs{};
  for (std::size_t i = 0; i < cs.size(); ++i) {
    cs[i] = helpers::remEuclid((dReal[i] + dReal[3]) / 2.0, 2.0 * pi);
  }

  // Reorder coordinates according to min(a, pi/2 - a) with
  // a = x mod pi/2 for each Weyl coordinate x
  std::array<double, 3> cstemp{};
  for (std::size_t i = 0; i < cs.size(); ++i) {
    const auto tmp = helpers::remEuclid(cs[i], pi / 2.0);
    cstemp[i] = std::min(tmp, (pi / 2.0) - tmp);
  }
  std::array<std::size_t, 3> order{0, 1, 2};
  std::stable_sort(order.begin(), order.end(),
                   [&](auto a, auto b) { return cstemp[a] < cstemp[b]; });
  order = {order[1], order[2], order[0]};
  cs = {cs[order[0]], cs[order[1]], cs[order[2]]};
  {
    const std::array<double, 3> reordered{dReal[order[0]], dReal[order[1]],
                                          dReal[order[2]]};
    dReal[0] = reordered[0];
    dReal[1] = reordered[1];
    dReal[2] = reordered[2];
  }

  // update eigenvectors (columns of p) according to new order of
  // weyl coordinates
  const Matrix4x4 pOrig = p;
  for (std::size_t i = 0; i < order.size(); ++i) {
    p.setColumn(i, pOrig.column(order[i]));
  }
  // apply correction for determinant if necessary
  if (p.determinant().real() < 0.0) {
    auto lastColumn = p.column(3);
    for (auto& entry : lastColumn) {
      entry = -entry;
    }
    p.setColumn(3, lastColumn);
  }
  assert(std::abs(p.determinant() - 1.0) < SANITY_CHECK_PRECISION);

  // re-create complex eigenvalue matrix; this matrix contains the
  // parameters of the canonical gate which is later used in the
  // verification. Since the matrix is diagonal, the matrix exponential is
  // equivalent to the element-wise exponential function.
  std::array<Complex, 4> tempDiag{};
  for (std::size_t k = 0; k < tempDiag.size(); ++k) {
    tempDiag[k] = std::exp(1i * dReal[k]);
  }
  const Matrix4x4 temp = Matrix4x4::fromDiagonal(tempDiag);

  // combined matrix k1 of 1q gates after canonical gate
  Matrix4x4 k1 = uP * p * temp;
  // k1 must be orthogonal; the tolerance matches the iterative diagonalization
  // residual rather than the (much tighter) default matrix tolerance.
  assert((k1.transpose() * k1).isIdentity(SANITY_CHECK_PRECISION));
  assert(k1.determinant().real() > 0.0);
  k1 = magicBasisTransform(k1, MagicBasisTransform::Into);

  // combined matrix k2 of 1q gates before canonical gate
  Matrix4x4 k2 = p.adjoint();
  // k2 must be orthogonal; see the tolerance note on the k1 check above.
  assert((k2.transpose() * k2).isIdentity(SANITY_CHECK_PRECISION));
  assert(k2.determinant().real() > 0.0);
  k2 = magicBasisTransform(k2, MagicBasisTransform::Into);

  // ensure k1 and k2 are correct (when combined with the canonical gate
  // parameters in-between, they are equivalent to u)
  std::array<Complex, 4> tempConjDiag{};
  for (std::size_t k = 0; k < tempConjDiag.size(); ++k) {
    tempConjDiag[k] = std::conj(tempDiag[k]);
  }
  assert((k1 *
          magicBasisTransform(Matrix4x4::fromDiagonal(tempConjDiag),
                              MagicBasisTransform::Into) *
          k2)
             .isApprox(u, SANITY_CHECK_PRECISION));

  // calculate k1 = K1l ⊗ K1r
  auto [K1l, K1r, phaseL] = decomposeTwoQubitProductGate(k1);
  // decompose k2 = K2l ⊗ K2r
  auto [K2l, K2r, phaseR] = decomposeTwoQubitProductGate(k2);
  assert(kron(K1l, K1r).isApprox(k1, SANITY_CHECK_PRECISION));
  assert(kron(K2l, K2r).isApprox(k2, SANITY_CHECK_PRECISION));
  // accumulate global phase
  globalPhase += phaseL + phaseR;

  // Flip into Weyl chamber
  if (cs[0] > (pi / 2.0)) {
    cs[0] -= 3.0 * (pi / 2.0);
    K1l = K1l * ipy();
    K1r = K1r * ipy();
    globalPhase += (pi / 2.0);
  }
  if (cs[1] > (pi / 2.0)) {
    cs[1] -= 3.0 * (pi / 2.0);
    K1l = K1l * ipx();
    K1r = K1r * ipx();
    globalPhase += (pi / 2.0);
  }
  auto conjs = 0;
  if (cs[0] > (pi / 4.0)) {
    cs[0] = (pi / 2.0) - cs[0];
    K1l = K1l * ipy();
    K2r = ipy() * K2r;
    conjs += 1;
    globalPhase -= (pi / 2.0);
  }
  if (cs[1] > (pi / 4.0)) {
    cs[1] = (pi / 2.0) - cs[1];
    K1l = K1l * ipx();
    K2r = ipx() * K2r;
    conjs += 1;
    globalPhase += (pi / 2.0);
    if (conjs == 1) {
      globalPhase -= pi;
    }
  }
  if (cs[2] > (pi / 2.0)) {
    cs[2] -= 3.0 * (pi / 2.0);
    K1l = K1l * ipz();
    K1r = K1r * ipz();
    globalPhase += (pi / 2.0);
    if (conjs == 1) {
      globalPhase -= pi;
    }
  }
  if (conjs == 1) {
    cs[2] = (pi / 2.0) - cs[2];
    K1l = K1l * ipz();
    K2r = ipz() * K2r;
    globalPhase += (pi / 2.0);
  }
  if (cs[2] > (pi / 4.0)) {
    cs[2] -= (pi / 2.0);
    K1l = K1l * ipz();
    K1r = K1r * ipz();
    globalPhase -= (pi / 2.0);
  }

  // bind weyl coordinates as parameters of canonical gate
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
  decomposition.specialization = Specialization::General;
  decomposition.requestedFidelity = fidelity;

  // make sure decomposition is equal to input
  assert((kron(K1l, K1r) * decomposition.getCanonicalMatrix() * kron(K2l, K2r) *
          helpers::globalPhaseFactor(globalPhase))
             .isApprox(unitaryMatrix, SANITY_CHECK_PRECISION));

  // determine actual specialization of canonical gate so that the 1q
  // matrices can potentially be simplified
  auto flippedFromOriginal = decomposition.applySpecialization();

  auto getTrace = [&]() {
    if (flippedFromOriginal) {
      return TwoQubitWeylDecomposition::getTrace(
          (pi / 2.0) - a, b, -c, decomposition.a_, decomposition.b_,
          decomposition.c_);
    }
    return TwoQubitWeylDecomposition::getTrace(
        a, b, c, decomposition.a_, decomposition.b_, decomposition.c_);
  };
  // use trace to calculate fidelity of applied specialization and
  // adjust global phase
  auto trace = getTrace();
  const double calculatedFidelity = helpers::traceToFidelity(trace);
  // final check if specialization is close enough to the original matrix to
  // satisfy the requested fidelity; since no forced specialization is
  // allowed, this should never fail
  if (decomposition.requestedFidelity &&
      calculatedFidelity + 1.0e-13 < *decomposition.requestedFidelity) {
    llvm::reportFatalInternalError(llvm::formatv(
        "TwoQubitWeylDecomposition: Calculated fidelity of "
        "specialization is worse than requested fidelity ({0:F4} vs {1:F4})!",
        calculatedFidelity, *decomposition.requestedFidelity));
  }
  decomposition.globalPhase_ += std::arg(trace);

  // final check if decomposition is still valid after specialization
  assert((kron(decomposition.k1l_, decomposition.k1r_) *
          decomposition.getCanonicalMatrix() *
          kron(decomposition.k2l_, decomposition.k2r_) *
          helpers::globalPhaseFactor(decomposition.globalPhase_))
             .isApprox(unitaryMatrix, SANITY_CHECK_PRECISION));

  return decomposition;
}

Matrix4x4 TwoQubitWeylDecomposition::getCanonicalMatrix(double a, double b,
                                                        double c) {
  // Canonical gate `U_d(a, b, c) = exp(i * (a*XX + b*YY + c*ZZ))`. XX/YY/ZZ
  // commute pairwise, so any product order is equivalent; the order below is
  // chosen to match common Qiskit/QuantumFlow references. The negated rotation
  // angles (`-2 * a`, ...) compensate for the `RXX/RYY/RZZ` convention
  // `exp(-i * theta/2 * XX)`, so that the factored angles sum back to the
  // intended `+a`, `+b`, `+c`.
  const auto xx = rxxMatrix(-2.0 * a);
  const auto yy = ryyMatrix(-2.0 * b);
  const auto zz = rzzMatrix(-2.0 * c);
  return zz * yy * xx;
}

Matrix4x4
TwoQubitWeylDecomposition::magicBasisTransform(const Matrix4x4& unitary,
                                               MagicBasisTransform direction) {
  // Makhlin "magic basis" transform. Conjugating a 2-qubit unitary by
  // `bNonNormalized` maps SU(2) x SU(2) factors onto SO(4) and diagonalizes
  // the canonical (Weyl) gate. The matrices are stored unnormalized: the
  // `1/2` pre-factor that would normally appear in `B^dagger` is absorbed
  // into `bNonNormalizedDagger` directly so the product `Bd * B == I`
  // without an extra scalar.
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
  if (direction == MagicBasisTransform::OutOf) {
    return bNonNormalizedDagger * unitary * bNonNormalized;
  }
  if (direction == MagicBasisTransform::Into) {
    return bNonNormalized * unitary * bNonNormalizedDagger;
  }
  llvm::reportFatalInternalError("Unknown MagicBasisTransform direction!");
}

double TwoQubitWeylDecomposition::closestPartialSwap(double a, double b,
                                                     double c) {
  auto m = (a + b + c) / 3.;
  auto [am, bm, cm] = std::array{a - m, b - m, c - m};
  auto [ab, bc, ca] = std::array{a - b, b - c, c - a};
  return m + (am * bm * cm * (6. + (ab * ab) + (bc * bc) + (ca * ca)) / 18.);
}

std::pair<Matrix4x4, std::array<Complex, 4>>
TwoQubitWeylDecomposition::diagonalizeComplexSymmetric(const Matrix4x4& m,
                                                       double precision) {
  // We can't use raw `eig` directly because it isn't guaranteed to give
  // us real or orthogonal eigenvectors. Instead, since `M` is
  // complex-symmetric,
  //   M = A + iB
  // for real-symmetric `A` and `B`, and as
  //   M^+ @ M2 = A^2 + B^2 + i [A, B] = 1
  // we must have `A` and `B` commute, and consequently they are
  // simultaneously diagonalizable. Mixing them together _should_ account
  // for any degeneracy problems, but it's not guaranteed, so we repeat it
  // a little bit.  The fixed seed is to make failures deterministic; the
  // value is not important.
  auto state = std::mt19937{2023};
  std::normal_distribution<double> dist;

  const auto mReal = m.realPart();
  const auto mImag = m.imagPart();

  double bestErr = 1e300;
  constexpr auto maxDiagonalizationAttempts = 100;
  for (int i = 0; i < maxDiagonalizationAttempts; ++i) {
    double randA{};
    double randB{};
    // For debugging the algorithm use the same RNG values as the
    // Qiskit implementation for the first random trial.
    // In most cases this loop only executes a single iteration and
    // using the same rng values rules out possible RNG differences
    // as the root cause of a test failure
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
    const Matrix4x4 p = jacobiSymmetricEigen(m2Real).eigenvectors;
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
      // p are the eigenvectors which are decomposed into the
      // single-qubit gates surrounding the canonical gate
      // d is the sqrt of the eigenvalues that are used to determine the
      // weyl coordinates and thus the parameters of the canonical gate
      // check that p is in SO(4)
      assert((p.transpose() * p).isIdentity(SANITY_CHECK_PRECISION));
      // make sure determinant of eigenvalues is 1.0
      assert(std::abs(Matrix4x4::fromDiagonal(d).determinant() - 1.0) <
             SANITY_CHECK_PRECISION);
      return std::make_pair(p, d);
    }
  }
  llvm::reportFatalInternalError(llvm::formatv(
      "TwoQubitWeylDecomposition: failed to diagonalize M2 ({0} iterations). "
      "best error = {1:e}, precision = {2:e}",
      maxDiagonalizationAttempts, bestErr, precision));
}

std::tuple<Matrix2x2, Matrix2x2, double>
TwoQubitWeylDecomposition::decomposeTwoQubitProductGate(
    const Matrix4x4& specialUnitary) {
  // for alternative approaches, see
  // pennylane's math.decomposition.su2su2_to_tensor_products
  // or quantumflow.kronecker_decomposition

  // first quadrant
  Matrix2x2 r =
      Matrix2x2::fromElements(specialUnitary(0, 0), specialUnitary(0, 1),
                              specialUnitary(1, 0), specialUnitary(1, 1));
  auto detR = r.determinant();
  if (std::abs(detR) < 0.1) {
    // third quadrant
    r = Matrix2x2::fromElements(specialUnitary(2, 0), specialUnitary(2, 1),
                                specialUnitary(3, 0), specialUnitary(3, 1));
    detR = r.determinant();
  }
  if (std::abs(detR) < 0.1) {
    llvm::reportFatalInternalError(
        "decomposeTwoQubitProductGate: unable to decompose: det_r < 0.1");
  }
  r *= (1.0 / std::sqrt(detR));
  // transpose with complex conjugate of each element
  const Matrix2x2 rTConj = r.adjoint();

  Matrix4x4 temp = specialUnitary * kron(Matrix2x2::identity(), rTConj);

  // [[a, b, c, d],
  //  [e, f, g, h], => [[a, c],
  //  [i, j, k, l],     [i, k]]
  //  [m, n, o, p]]
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

std::complex<double> TwoQubitWeylDecomposition::getTrace(double a, double b,
                                                         double c, double ap,
                                                         double bp, double cp) {
  // Closed-form Hilbert-Schmidt overlap `tr(U_d(a,b,c)^dag * U_d(ap,bp,cp))`
  // between two canonical (Weyl) gates, expressed in terms of the coordinate
  // differences. Feeding the result into `traceToFidelity` gives the average
  // two-qubit gate fidelity between the two canonical gates, which
  // `bestSpecialization` uses to rank candidate specializations.
  // Reference: Zhang et al., "Geometric theory of nonlocal two-qubit
  // operations", Phys. Rev. A 67, 042313 (2003), Eq. (20).
  auto da = a - ap;
  auto db = b - bp;
  auto dc = c - cp;
  return 4. * std::complex<double>{std::cos(da) * std::cos(db) * std::cos(dc),
                                   std::sin(da) * std::sin(db) * std::sin(dc)};
}

TwoQubitWeylDecomposition::Specialization
TwoQubitWeylDecomposition::bestSpecialization() const {
  auto isClose = [this](double ap, double bp, double cp) -> bool {
    auto tr = getTrace(a_, b_, c_, ap, bp, cp);
    if (requestedFidelity) {
      return helpers::traceToFidelity(tr) >= *requestedFidelity;
    }
    return false;
  };

  auto closestAbc = closestPartialSwap(a_, b_, c_);
  auto closestAbMinusC = closestPartialSwap(a_, b_, -c_);

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
  if (isClose(a_, 0., 0.)) {
    return Specialization::ControlledEquiv;
  }
  if (isClose((std::numbers::pi / 4.0), (std::numbers::pi / 4.0), c_)) {
    return Specialization::MirrorControlledEquiv;
  }
  if (isClose((a_ + b_) / 2., (a_ + b_) / 2., c_)) {
    return Specialization::FSimaabEquiv;
  }
  if (isClose(a_, (b_ + c_) / 2., (b_ + c_) / 2.)) {
    return Specialization::FSimabbEquiv;
  }
  if (isClose(a_, (b_ - c_) / 2., (c_ - b_) / 2.)) {
    return Specialization::FSimabmbEquiv;
  }
  return Specialization::General;
}

bool TwoQubitWeylDecomposition::applySpecialization() {
  if (specialization != Specialization::General) {
    llvm::reportFatalInternalError(
        "Application of specialization only works on "
        "general Weyl decompositions!");
  }
  bool flippedFromOriginal = false;
  auto newSpecialization = bestSpecialization();
  if (newSpecialization == Specialization::General) {
    // U has no special symmetry.
    //
    // This gate binds all 6 possible parameters, so there is no need to
    // make the single-qubit pre-/post-gates canonical.
    return flippedFromOriginal;
  }
  specialization = newSpecialization;

  if (newSpecialization == Specialization::IdEquiv) {
    // :math:`U \sim U_d(0,0,0)`
    // Thus, :math:`\sim Id`
    //
    // This gate binds 0 parameters, we make it canonical by setting:
    //
    // :math:`K2_l = Id` , :math:`K2_r = Id`.
    a_ = 0.;
    b_ = 0.;
    c_ = 0.;
    // unmodified global phase
    k1l_ = k1l_ * k2l_;
    k2l_ = Matrix2x2::identity();
    k1r_ = k1r_ * k2r_;
    k2r_ = Matrix2x2::identity();
  } else if (newSpecialization == Specialization::SWAPEquiv) {
    // :math:`U \sim U_d(\pi/4, \pi/4, \pi/4) \sim U(\pi/4, \pi/4, -\pi/4)`
    // Thus, :math:`U \sim \text{SWAP}`
    //
    // This gate binds 0 parameters, we make it canonical by setting:
    //
    // :math:`K2_l = Id` , :math:`K2_r = Id`.
    if (c_ > 0.) {
      // unmodified global phase
      k1l_ = k1l_ * k2r_;
      k1r_ = k1r_ * k2l_;
      k2l_ = Matrix2x2::identity();
      k2r_ = Matrix2x2::identity();
    } else {
      flippedFromOriginal = true;

      globalPhase_ += (std::numbers::pi / 2.0);
      k1l_ = k1l_ * ipz() * k2r_;
      k1r_ = k1r_ * ipz() * k2l_;
      k2l_ = Matrix2x2::identity();
      k2r_ = Matrix2x2::identity();
    }
    a_ = (std::numbers::pi / 4.0);
    b_ = (std::numbers::pi / 4.0);
    c_ = (std::numbers::pi / 4.0);
  } else if (newSpecialization == Specialization::PartialSWAPEquiv) {
    // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, \alpha\pi/4)`
    // Thus, :math:`U \sim \text{SWAP}^\alpha`
    //
    // This gate binds 3 parameters, we make it canonical by setting:
    //
    // :math:`K2_l = Id`.
    auto closest = closestPartialSwap(a_, b_, c_);
    auto k2lDagger = k2l_.adjoint();

    a_ = closest;
    b_ = closest;
    c_ = closest;
    // unmodified global phase
    k1l_ = k1l_ * k2l_;
    k1r_ = k1r_ * k2l_;
    k2r_ = k2lDagger * k2r_;
    k2l_ = Matrix2x2::identity();
  } else if (newSpecialization == Specialization::PartialSWAPFlipEquiv) {
    // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, -\alpha\pi/4)`
    // Thus, :math:`U \sim \text{SWAP}^\alpha`
    //
    // (a non-equivalent root of SWAP from the TwoQubitWeylPartialSWAPEquiv
    // similar to how :math:`x = (\pm \sqrt(x))^2`)
    //
    // This gate binds 3 parameters, we make it canonical by setting:
    //
    // :math:`K2_l = Id`
    auto closest = closestPartialSwap(a_, b_, -c_);
    auto k2lDagger = k2l_.adjoint();

    a_ = closest;
    b_ = closest;
    c_ = -closest;
    // unmodified global phase
    k1l_ = k1l_ * k2l_;
    k1r_ = k1r_ * ipz() * k2l_ * ipz();
    k2r_ = ipz() * k2lDagger * ipz() * k2r_;
    k2l_ = Matrix2x2::identity();
  } else if (newSpecialization == Specialization::ControlledEquiv) {
    // :math:`U \sim U_d(\alpha, 0, 0)`
    // Thus, :math:`U \sim \text{Ctrl-U}`
    //
    // This gate binds 4 parameters, we make it canonical by setting:
    //
    // :math:`K2_l = Ry(\theta_l) Rx(\lambda_l)`
    // :math:`K2_r = Ry(\theta_r) Rx(\lambda_r)`
    const EulerBasis eulerBasis = EulerBasis::XYX;
    const auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, eulerBasis);
    const auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
        anglesFromUnitary(k2r_, EulerBasis::XYX);
    // unmodified parameter a
    b_ = 0.;
    c_ = 0.;
    globalPhase_ = globalPhase_ + k2lphase + k2rphase;
    k1l_ = k1l_ * rxMatrix(k2lphi);
    k2l_ = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
    k1r_ = k1r_ * rxMatrix(k2rphi);
    k2r_ = ryMatrix(k2rtheta) * rxMatrix(k2rlambda);
  } else if (newSpecialization == Specialization::MirrorControlledEquiv) {
    // :math:`U \sim U_d(\pi/4, \pi/4, \alpha)`
    // Thus, :math:`U \sim \text{SWAP} \cdot \text{Ctrl-U}`
    //
    // This gate binds 4 parameters, we make it canonical by setting:
    //
    // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`
    // :math:`K2_r = Ry(\theta_r)\cdot Rz(\lambda_r)`
    const auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, EulerBasis::ZYZ);
    const auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
        anglesFromUnitary(k2r_, EulerBasis::ZYZ);
    a_ = (std::numbers::pi / 4.0);
    b_ = (std::numbers::pi / 4.0);
    // unmodified parameter c
    globalPhase_ = globalPhase_ + k2lphase + k2rphase;
    k1l_ = k1l_ * rzMatrix(k2rphi);
    k2l_ = ryMatrix(k2ltheta) * rzMatrix(k2llambda);
    k1r_ = k1r_ * rzMatrix(k2lphi);
    k2r_ = ryMatrix(k2rtheta) * rzMatrix(k2rlambda);
  } else if (newSpecialization == Specialization::FSimaabEquiv) {
    // :math:`U \sim U_d(\alpha, \alpha, \beta), \alpha \geq |\beta|`
    //
    // This gate binds 5 parameters, we make it canonical by setting:
    //
    // :math:`K2_l = Ry(\theta_l) \cdot Rz(\lambda_l)`.
    auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, EulerBasis::ZYZ);
    auto ab = (a_ + b_) / 2.;

    a_ = ab;
    b_ = ab;
    // unmodified parameter c
    globalPhase_ = globalPhase_ + k2lphase;
    k1l_ = k1l_ * rzMatrix(k2lphi);
    k2l_ = ryMatrix(k2ltheta) * rzMatrix(k2llambda);
    k1r_ = k1r_ * rzMatrix(k2lphi);
    k2r_ = rzMatrix(-k2lphi) * k2r_;
  } else if (newSpecialization == Specialization::FSimabbEquiv) {
    // :math:`U \sim U_d(\alpha, \beta, \beta), \alpha \geq \beta \geq 0`
    //
    // This gate binds 5 parameters, we make it canonical by setting:
    //
    // :math:`K2_l = Ry(\theta_l) \cdot Rx(\lambda_l)`
    auto eulerBasis = EulerBasis::XYX;
    auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, eulerBasis);
    auto bc = (b_ + c_) / 2.;

    // unmodified parameter a
    b_ = bc;
    c_ = bc;
    globalPhase_ = globalPhase_ + k2lphase;
    k1l_ = k1l_ * rxMatrix(k2lphi);
    k2l_ = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
    k1r_ = k1r_ * rxMatrix(k2lphi);
    k2r_ = rxMatrix(-k2lphi) * k2r_;
  } else if (newSpecialization == Specialization::FSimabmbEquiv) {
    // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
    //
    // This gate binds 5 parameters, we make it canonical by setting:
    //
    // :math:`K2_l = Ry(\theta_l) \cdot Rx(\lambda_l)`
    auto eulerBasis = EulerBasis::XYX;
    auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
        anglesFromUnitary(k2l_, eulerBasis);
    auto bc = (b_ - c_) / 2.;

    // unmodified parameter a
    b_ = bc;
    c_ = -bc;
    globalPhase_ = globalPhase_ + k2lphase;
    k1l_ = k1l_ * rxMatrix(k2lphi);
    k2l_ = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
    k1r_ = k1r_ * ipz() * rxMatrix(k2lphi) * ipz();
    k2r_ = ipz() * rxMatrix(-k2lphi) * ipz() * k2r_;
  } else {
    llvm::reportFatalInternalError(
        "Unknown specialization for Weyl decomposition!");
  }
  return flippedFromOriginal;
}

} // namespace mlir::qco::decomposition
