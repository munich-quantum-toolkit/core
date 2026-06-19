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

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
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

using mlir::qco::Complex;
using mlir::qco::Matrix2x2;
using mlir::qco::Matrix4x4;

static constexpr double SANITY_CHECK_PRECISION = 1e-12;

[[nodiscard]] static bool isUnitaryMatrix(const Matrix2x2& matrix,
                                          double tolerance = 1e-12) {
  return (matrix.adjoint() * matrix).isIdentity(tolerance);
}

[[nodiscard]] static double remEuclid(double a, double b) {
  if (b == 0.0) {
    llvm::reportFatalInternalError("remEuclid expects non-zero divisor");
  }
  const auto r = std::fmod(a, b);
  return (r < 0.0) ? r + std::abs(b) : r;
}

[[nodiscard]] static double traceToFidelity(const std::complex<double>& x) {
  const auto xAbs = std::abs(x);
  return (4.0 + (xAbs * xAbs)) / 20.0;
}

[[nodiscard]] static std::complex<double>
globalPhaseFactor(double globalPhase) {
  return std::exp(std::complex<double>{0, 1} * globalPhase);
}

[[nodiscard]] static Matrix4x4 rxxMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const Complex misin{0., -std::sin(theta / 2.)};
  return Matrix4x4::fromElements(cosTheta, 0, 0, misin, //
                                 0, cosTheta, misin, 0, //
                                 0, misin, cosTheta, 0, //
                                 misin, 0, 0, cosTheta);
}

[[nodiscard]] static Matrix4x4 ryyMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const Complex isin{0., std::sin(theta / 2.)};
  const Complex misin{0., -std::sin(theta / 2.)};
  return Matrix4x4::fromElements(cosTheta, 0, 0, isin,  //
                                 0, cosTheta, misin, 0, //
                                 0, misin, cosTheta, 0, //
                                 isin, 0, 0, cosTheta);
}

[[nodiscard]] static Matrix4x4 rzzMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);
  const Complex em{cosTheta, -sinTheta};
  const Complex ep{cosTheta, sinTheta};
  return Matrix4x4::fromElements(em, 0, 0, 0, //
                                 0, ep, 0, 0, //
                                 0, 0, ep, 0, //
                                 0, 0, 0, em);
}

[[nodiscard]] static const Matrix4x4& swapGate() {
  static const Matrix4x4 MATRIX = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                          0, 0, 1, 0, //
                                                          0, 1, 0, 0, //
                                                          0, 0, 0, 1);
  return MATRIX;
}

[[nodiscard]] static const Matrix4x4& cxGate01() {
  static const Matrix4x4 MATRIX = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                          0, 1, 0, 0, //
                                                          0, 0, 0, 1, //
                                                          0, 0, 1, 0);
  return MATRIX;
}

[[nodiscard]] static const Matrix4x4& cxGate10() {
  static const Matrix4x4 MATRIX = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                          0, 0, 0, 1, //
                                                          0, 0, 1, 0, //
                                                          0, 1, 0, 0);
  return MATRIX;
}

[[nodiscard]] static const Matrix4x4& czGate() {
  static const Matrix4x4 MATRIX = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                          0, 1, 0, 0, //
                                                          0, 0, 1, 0, //
                                                          0, 0, 0, -1);
  return MATRIX;
}

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
    cs[i] = remEuclid((dReal[i] + dReal[3]) / 2.0, 2.0 * pi);
  }

  // Reorder coordinates according to min(a, pi/2 - a) with
  // a = x mod pi/2 for each Weyl coordinate x
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
          globalPhaseFactor(globalPhase))
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
  const double calculatedFidelity = traceToFidelity(trace);
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
          globalPhaseFactor(decomposition.globalPhase_))
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
      return traceToFidelity(tr) >= *requestedFidelity;
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

TwoQubitBasisDecomposer
TwoQubitBasisDecomposer::create(const Matrix4x4& basisMatrix,
                                double basisFidelity) {
  const Matrix2x2 k12RArr = Matrix2x2::fromElements(
      1i * FRAC1_SQRT2, FRAC1_SQRT2, -FRAC1_SQRT2, -1i * FRAC1_SQRT2);
  const Matrix2x2 k12LArr =
      Matrix2x2::fromElements(Complex{0.5, 0.5}, Complex{0.5, 0.5},
                              Complex{-0.5, 0.5}, Complex{0.5, -0.5});

  // The Shende-Markov-Bullock 3-CX sandwich (and its 1/2-CX reductions) used
  // below is derived for a basis CX whose 4x4 matrix is the Qiskit/LSB form
  // `[[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]`, i.e. "control on the LSB
  // factor, target on the MSB factor" of the tensor product. MQT's wider
  // convention places operand 0 on the MSB factor, so the CX/CZ matrix for
  // control-on-wire-0 gives the SWAP-conjugate
  // `[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]`.
  //
  // Because `SWAP * C(a,b,c) * SWAP = C(a,b,c)` but
  // `SWAP * (K1l ⊗ K1r) * SWAP = (K1r ⊗ K1l)`, feeding the MSB matrix directly
  // into the Weyl decomposer would swap the roles of `k1l`/`k1r` (and `k2l`/
  // `k2r`) relative to the hard-coded constants above. To keep the SMB algebra
  // self-consistent we SWAP-conjugate the basis matrix here (restoring the
  // Qiskit/LSB 4x4) and then absorb the resulting "left/right" relabeling at
  // the emission boundary in `decomp{0,1,2,3}` below. This reproduces the
  // pre-flip gate counts without having to re-derive every SMB constant for
  // the MSB basis -- the two routes are algebraically equivalent.
  const Matrix4x4 basisMatrixLsb = swapGate() * basisMatrix * swapGate();
  const auto basisDecomposer =
      TwoQubitWeylDecomposition::create(basisMatrixLsb, basisFidelity);
  const auto isSuperControlled =
      relativeEq(basisDecomposer.a(), std::numbers::pi / 4.0, 1e-13, 1e-09) &&
      relativeEq(basisDecomposer.c(), 0.0, 1e-13, 1e-09);

  // Create some useful matrices U1, U2, U3 are equivalent to the basis,
  // expand as Ui = Ki1.Ubasis.Ki2
  auto b = basisDecomposer.b();
  Complex temp{0.5, -0.5};
  const Matrix2x2 k11l = Matrix2x2::fromElements(
      temp * (-1i * std::exp(-1i * b)), temp * std::exp(-1i * b),
      temp * (-1i * std::exp(1i * b)), temp * -std::exp(1i * b));
  const Matrix2x2 k11r = Matrix2x2::fromElements(
      FRAC1_SQRT2 * (1i * std::exp(-1i * b)), FRAC1_SQRT2 * -std::exp(-1i * b),
      FRAC1_SQRT2 * std::exp(1i * b), FRAC1_SQRT2 * (-1i * std::exp(1i * b)));
  const Matrix2x2 k32lK21l =
      Matrix2x2::fromElements(FRAC1_SQRT2 * Complex{1., std::cos(2. * b)},
                              FRAC1_SQRT2 * (1i * std::sin(2. * b)),
                              FRAC1_SQRT2 * (1i * std::sin(2. * b)),
                              FRAC1_SQRT2 * Complex{1., -std::cos(2. * b)});
  temp = Complex{0.5, 0.5};
  const Matrix2x2 k21r = Matrix2x2::fromElements(
      temp * (-1i * std::exp(-2i * b)), temp * std::exp(-2i * b),
      temp * (1i * std::exp(2i * b)), temp * std::exp(2i * b));
  const Matrix2x2 k22l = Matrix2x2::fromElements(FRAC1_SQRT2, -FRAC1_SQRT2,
                                                 FRAC1_SQRT2, FRAC1_SQRT2);
  const Matrix2x2 k22r = Matrix2x2::fromElements(0, 1, -1, 0);
  const Matrix2x2 k31l = Matrix2x2::fromElements(
      FRAC1_SQRT2 * std::exp(-1i * b), FRAC1_SQRT2 * std::exp(-1i * b),
      FRAC1_SQRT2 * -std::exp(1i * b), FRAC1_SQRT2 * std::exp(1i * b));
  const Matrix2x2 k31r = Matrix2x2::fromElements(1i * std::exp(1i * b), 0, 0,
                                                 -1i * std::exp(-1i * b));
  const Matrix2x2 k32r = Matrix2x2::fromElements(
      temp * std::exp(1i * b), temp * -std::exp(-1i * b),
      temp * (-1i * std::exp(1i * b)), temp * (-1i * std::exp(-1i * b)));
  auto k1lDagger = basisDecomposer.k1l().adjoint();
  auto k1rDagger = basisDecomposer.k1r().adjoint();
  auto k2lDagger = basisDecomposer.k2l().adjoint();
  auto k2rDagger = basisDecomposer.k2r().adjoint();
  // Pre-build the fixed parts of the matrices used in 3-part decomposition
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
  // Pre-build the fixed parts of the matrices used in the 2-part decomposition
  auto q0l = k12LArr.adjoint() * k1lDagger;
  auto q0r = k12RArr.adjoint() * ipz() * k1rDagger;
  auto q1la = k2lDagger * k11l.adjoint();
  auto q1lb = k11l * k1lDagger;
  auto q1ra = k2rDagger * ipz() * k11r.adjoint();
  auto q1rb = k11r * k1rDagger;
  auto q2l = k2lDagger * k12LArr;
  auto q2r = k2rDagger * k12RArr;

  return TwoQubitBasisDecomposer{
      basisFidelity,
      basisDecomposer,
      isSuperControlled,
      u0l,
      u0r,
      u1l,
      u1ra,
      u1rb,
      u2la,
      u2lb,
      u2ra,
      u2rb,
      u3l,
      u3r,
      q0l,
      q0r,
      q1la,
      q1lb,
      q1ra,
      q1rb,
      q2l,
      q2r,
  };
}

std::optional<TwoQubitNativeDecomposition>
TwoQubitBasisDecomposer::twoQubitDecompose(
    const TwoQubitWeylDecomposition& targetDecomposition,
    std::optional<std::uint8_t> numBasisGateUses) const {
  auto traces = this->traces(targetDecomposition);
  auto getDefaultNbasis = [&]() -> std::uint8_t {
    // Pick the number of basis gate uses `i ∈ {0, 1, 2, 3}` that maximizes
    //   expected_fidelity(i) = traceToFidelity(traces[i]) * basisFidelity^i.
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
  // number of basis gates that need to be used in the decomposition
  auto bestNbasis = numBasisGateUses.value_or(getDefaultNbasis());
  if (bestNbasis > 1 && !isSuperControlled) {
    // cannot reliably decompose with more than one basis gate and a
    // non-super-controlled basis gate
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
  TwoQubitLocalUnitaryList factors = chooseDecomposition();
#ifndef NDEBUG
  for (const auto& factor : factors) {
    assert(isUnitaryMatrix(factor));
  }
#endif

  double globalPhase = targetDecomposition.globalPhase();
  globalPhase -= bestNbasis * basisDecomposer.globalPhase();
  if (bestNbasis == 2) {
    // The two-basis (2x CX/CZ) template in `decomp2Supercontrolled` produces
    // a sequence whose global phase is off by `pi` relative to the target;
    // compensate here so the emitted sequence reproduces the target unitary
    // exactly, not just up to sign.
    globalPhase += std::numbers::pi;
  }
  // large global phases can be generated by the decomposition, thus limit
  // it to [0, +2*pi)
  globalPhase = remEuclid(globalPhase, 2.0 * std::numbers::pi);

  return TwoQubitNativeDecomposition{
      .numBasisUses = bestNbasis,
      .singleQubitFactors = std::move(factors),
      .globalPhase = globalPhase,
  };
}

// Ported SMB helpers assume Qiskit Weyl k-factor layout; QCO 4x4 input order
// swaps l/r vs that port. Swap k1l<->k1r and k2l<->k2r when reading ``target``,
// and swap adjacent pairs in each return vector so the emission boundary maps
// matrices to the same wires as the upstream decomposer. ``decomp0`` cancels to
// the unswapped formula.
TwoQubitLocalUnitaryList
TwoQubitBasisDecomposer::decomp0(const TwoQubitWeylDecomposition& target) {
  return TwoQubitLocalUnitaryList{
      target.k1r() * target.k2r(),
      target.k1l() * target.k2l(),
  };
}

TwoQubitLocalUnitaryList TwoQubitBasisDecomposer::decomp1(
    const TwoQubitWeylDecomposition& target) const {
  // may not work for z != 0 and c != 0 (not always in Weyl chamber)
  return TwoQubitLocalUnitaryList{
      basisDecomposer.k2l().adjoint() * target.k2r(),
      basisDecomposer.k2r().adjoint() * target.k2l(),
      target.k1r() * basisDecomposer.k1l().adjoint(),
      target.k1l() * basisDecomposer.k1r().adjoint(),
  };
}

TwoQubitLocalUnitaryList TwoQubitBasisDecomposer::decomp2Supercontrolled(
    const TwoQubitWeylDecomposition& target) const {
  if (!isSuperControlled) {
    llvm::reportFatalInternalError(
        "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
        "- no guarantee for exact decomposition with two basis gates");
  }
  return TwoQubitLocalUnitaryList{
      q2l * target.k2r(),
      q2r * target.k2l(),
      q1la * rzMatrix(-2. * target.a()) * q1lb,
      q1ra * rzMatrix(2. * target.b()) * q1rb,
      target.k1r() * q0l,
      target.k1l() * q0r,
  };
}

TwoQubitLocalUnitaryList TwoQubitBasisDecomposer::decomp3Supercontrolled(
    const TwoQubitWeylDecomposition& target) const {
  if (!isSuperControlled) {
    llvm::reportFatalInternalError(
        "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
        "- no guarantee for exact decomposition with three basis gates");
  }
  return TwoQubitLocalUnitaryList{
      u3l * target.k2r(),
      u3r * target.k2l(),
      u2la * rzMatrix(-2. * target.a()) * u2lb,
      u2ra * rzMatrix(2. * target.b()) * u2rb,
      u1l,
      u1ra * rzMatrix(-2. * target.c()) * u1rb,
      target.k1r() * u0l,
      target.k1l() * u0r,
  };
}

std::array<std::complex<double>, 4>
TwoQubitBasisDecomposer::traces(const TwoQubitWeylDecomposition& target) const {
  // Returns the Hilbert-Schmidt traces between the target canonical gate and
  // the best candidate reachable with `0, 1, 2, 3` uses of the basis gate,
  // respectively. Fed into `traceToFidelity` by `getDefaultNbasis` to pick
  // the best basis-gate count. The closed-form expressions specialize
  // `TwoQubitWeylDecomposition::getTrace(a, b, c, ap, bp, cp)` for:
  //   i == 0: no basis gate       (ap == bp == cp == 0)
  //   i == 1: one basis use       (ap == pi/4, bp == basis.b, cp == 0)
  //   i == 2: two basis uses      (ap == 0, bp == 0, cp == -target.c)
  //   i == 3: three basis uses    (target reachable exactly -> trace == 4)
  // so the array has length 4 and is indexed by the number of basis uses.
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

bool TwoQubitBasisDecomposer::relativeEq(double lhs, double rhs, double epsilon,
                                         double maxRelative) {
  // Handle same infinities
  if (lhs == rhs) {
    return true;
  }

  // Handle remaining infinities
  if (std::isinf(lhs) || std::isinf(rhs)) {
    return false;
  }

  auto absDiff = std::abs(lhs - rhs);

  // For when the numbers are really close together
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

//===----------------------------------------------------------------------===//
// Native-spec parsing and two-qubit synthesis
//===----------------------------------------------------------------------===//

static constexpr double PI = std::numbers::pi;

[[nodiscard]] static std::optional<NativeGateKind>
parseGateToken(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<NativeGateKind>>(name)
      .Case("u", NativeGateKind::U)
      .Case("x", NativeGateKind::X)
      .Case("sx", NativeGateKind::Sx)
      .Cases("rz", "p", NativeGateKind::Rz)
      .Case("rx", NativeGateKind::Rx)
      .Case("ry", NativeGateKind::Ry)
      .Case("r", NativeGateKind::R)
      .Case("cx", NativeGateKind::Cx)
      .Case("cz", NativeGateKind::Cz)
      .Case("rzz", NativeGateKind::Rzz)
      .Default(std::nullopt);
}

[[nodiscard]] static std::optional<llvm::DenseSet<NativeGateKind>>
parseGateSet(llvm::StringRef nativeGates) {
  llvm::DenseSet<NativeGateKind> gates;
  llvm::SmallVector<llvm::StringRef> parts;
  nativeGates.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef part : parts) {
    const auto token = part.trim().lower();
    if (token.empty()) {
      continue;
    }
    const auto gate = parseGateToken(token);
    if (!gate) {
      return std::nullopt;
    }
    gates.insert(*gate);
  }
  return gates;
}

[[nodiscard]] static SingleQubitEmitterSpec
makeEmitterSpec(SingleQubitMode mode, AxisPair axisPair = AxisPair::RxRz,
                bool supportsDirectRx = false) {
  return {
      .mode = mode, .axisPair = axisPair, .supportsDirectRx = supportsDirectRx};
}

static void
addEmitterIfAbsent(llvm::SmallVectorImpl<SingleQubitEmitterSpec>& emitters,
                   SingleQubitMode mode, AxisPair axisPair = AxisPair::RxRz,
                   bool supportsDirectRx = false) {
  const bool present = llvm::any_of(emitters, [&](const auto& e) {
    return e.mode == mode && e.axisPair == axisPair &&
           e.supportsDirectRx == supportsDirectRx;
  });
  if (!present) {
    emitters.push_back(makeEmitterSpec(mode, axisPair, supportsDirectRx));
  }
}

[[nodiscard]] static llvm::SmallVector<NativeGateKind, 4>
allowedGatesForEmitter(const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX: {
    llvm::SmallVector<NativeGateKind, 4> gates{
        NativeGateKind::X, NativeGateKind::Sx, NativeGateKind::Rz};
    if (emitter.supportsDirectRx) {
      gates.push_back(NativeGateKind::Rx);
    }
    return gates;
  }
  case SingleQubitMode::U3:
    return {NativeGateKind::U};
  case SingleQubitMode::R:
    return {NativeGateKind::R};
  case SingleQubitMode::AxisPair:
    switch (emitter.axisPair) {
    case AxisPair::RxRz:
      return {NativeGateKind::Rx, NativeGateKind::Rz};
    case AxisPair::RxRy:
      return {NativeGateKind::Rx, NativeGateKind::Ry};
    case AxisPair::RyRz:
      return {NativeGateKind::Ry, NativeGateKind::Rz};
    }
    break;
  }
  llvm_unreachable("unknown single-qubit mode");
}

[[nodiscard]] static llvm::SmallVector<NativeGateKind, 2>
allowedGatesForEntangler(EntanglerBasis entangler) {
  switch (entangler) {
  case EntanglerBasis::None:
    return {};
  case EntanglerBasis::Cx:
    return {NativeGateKind::Cx};
  case EntanglerBasis::Cz:
    return {NativeGateKind::Cz};
  }
  llvm_unreachable("unknown entangler basis");
}

static void populateAllowedGates(NativeProfileSpec& spec) {
  spec.allowedGates.clear();
  for (const auto& emitter : spec.singleQubitEmitters) {
    const auto allowed = allowedGatesForEmitter(emitter);
    spec.allowedGates.insert(allowed.begin(), allowed.end());
  }
  for (const auto entangler : spec.entanglerBases) {
    const auto allowed = allowedGatesForEntangler(entangler);
    spec.allowedGates.insert(allowed.begin(), allowed.end());
  }
  if (spec.allowRzz) {
    spec.allowedGates.insert(NativeGateKind::Rzz);
  }
}

[[nodiscard]] static std::optional<EntanglerBasis>
selectEntangler(const NativeProfileSpec& spec) {
  if (llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cx)) {
    return EntanglerBasis::Cx;
  }
  if (llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cz)) {
    return EntanglerBasis::Cz;
  }
  return std::nullopt;
}

[[nodiscard]] static Matrix4x4 entanglerMatrix(EntanglerBasis entangler) {
  return entangler == EntanglerBasis::Cz ? czGate() : cxGate01();
}

[[nodiscard]] static std::optional<TwoQubitNativeDecomposition>
decomposeWithEntangler(const Matrix4x4& target, EntanglerBasis entangler) {
  auto decomposer =
      TwoQubitBasisDecomposer::create(entanglerMatrix(entangler), 1.0);
  auto weyl = TwoQubitWeylDecomposition::create(target, std::nullopt);
  return decomposer.twoQubitDecompose(weyl, std::nullopt);
}

static void emitGPhaseIfNonTrivial(OpBuilder& builder, Location loc,
                                   double phase) {
  constexpr double epsilon = 1e-12;
  if (std::abs(phase) > epsilon) {
    GPhaseOp::create(builder, loc, phase);
  }
}

[[nodiscard]] static Value emitSingleQubitMatrix(OpBuilder& builder,
                                                 Location loc, Value inQubit,
                                                 const Matrix2x2& matrix,
                                                 EulerBasis basis) {
  return *synthesizeUnitary1QEuler(builder, loc, inQubit, matrix,
                                   /*runSize=*/0, /*hasNonBasisGate=*/true,
                                   basis);
}

EulerBasis emitterEulerBasis(const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return EulerBasis::ZSXX;
  case SingleQubitMode::U3:
    return EulerBasis::U;
  case SingleQubitMode::R:
    return EulerBasis::R;
  case SingleQubitMode::AxisPair:
    switch (emitter.axisPair) {
    case AxisPair::RxRz:
      return EulerBasis::XZX;
    case AxisPair::RxRy:
      return EulerBasis::XYX;
    case AxisPair::RyRz:
      return EulerBasis::ZYZ;
    }
    break;
  }
  llvm_unreachable("unknown single-qubit mode");
}

std::optional<NativeProfileSpec> parseNativeSpec(llvm::StringRef nativeGates) {
  const auto gates = parseGateSet(nativeGates);
  if (!gates || gates->empty()) {
    return std::nullopt;
  }
  const auto has = [&](NativeGateKind kind) { return gates->contains(kind); };

  NativeProfileSpec spec;

  if (has(NativeGateKind::U)) {
    addEmitterIfAbsent(spec.singleQubitEmitters, SingleQubitMode::U3);
  }
  const bool hasXSxRz = has(NativeGateKind::X) && has(NativeGateKind::Sx) &&
                        has(NativeGateKind::Rz);
  if (hasXSxRz) {
    addEmitterIfAbsent(spec.singleQubitEmitters, SingleQubitMode::ZSXX,
                       AxisPair::RxRz,
                       /*supportsDirectRx=*/has(NativeGateKind::Rx));
  }
  if (has(NativeGateKind::R)) {
    addEmitterIfAbsent(spec.singleQubitEmitters, SingleQubitMode::R);
  }
  struct AxisPairRule {
    AxisPair axis;
    NativeGateKind left;
    NativeGateKind right;
  };
  for (const auto& rule : {
           AxisPairRule{.axis = AxisPair::RxRz,
                        .left = NativeGateKind::Rx,
                        .right = NativeGateKind::Rz},
           AxisPairRule{.axis = AxisPair::RxRy,
                        .left = NativeGateKind::Rx,
                        .right = NativeGateKind::Ry},
           AxisPairRule{.axis = AxisPair::RyRz,
                        .left = NativeGateKind::Ry,
                        .right = NativeGateKind::Rz},
       }) {
    if (has(rule.left) && has(rule.right)) {
      addEmitterIfAbsent(spec.singleQubitEmitters, SingleQubitMode::AxisPair,
                         rule.axis);
    }
  }
  if (spec.singleQubitEmitters.empty()) {
    return std::nullopt;
  }

  if (has(NativeGateKind::Cx)) {
    spec.entanglerBases.push_back(EntanglerBasis::Cx);
  }
  if (has(NativeGateKind::Cz)) {
    spec.entanglerBases.push_back(EntanglerBasis::Cz);
  }
  spec.allowRzz = has(NativeGateKind::Rzz);

  populateAllowedGates(spec);
  return spec;
}

LogicalResult synthesizeUnitary2QWeyl(OpBuilder& builder, Location loc,
                                      Value qubit0, Value qubit1,
                                      const Matrix4x4& target,
                                      const NativeProfileSpec& spec,
                                      Value& outQubit0, Value& outQubit1) {
  const auto entangler = selectEntangler(spec);
  if (!entangler) {
    return failure();
  }
  const auto native = decomposeWithEntangler(target, *entangler);
  if (!native) {
    return failure();
  }
  const auto basis = emitterEulerBasis(spec.singleQubitEmitters.front());

  emitGPhaseIfNonTrivial(builder, loc, native->globalPhase);

  Value wire0 = qubit0;
  Value wire1 = qubit1;
  const auto& factors = native->singleQubitFactors;
  const std::uint8_t numBasisUses = native->numBasisUses;
  const auto emitFactor = [&](Value& wire, std::size_t index) {
    wire = emitSingleQubitMatrix(builder, loc, wire, factors[index], basis);
  };
  const auto emitEntangler = [&]() {
    auto ctrlOp = CtrlOp::create(
        builder, loc, ValueRange{wire0}, ValueRange{wire1},
        [&](ValueRange targetArgs) -> llvm::SmallVector<Value> {
          if (*entangler == EntanglerBasis::Cz) {
            return {ZOp::create(builder, loc, targetArgs[0]).getOutputQubit(0)};
          }
          return {XOp::create(builder, loc, targetArgs[0]).getOutputQubit(0)};
        });
    wire0 = ctrlOp.getOutputControl(0);
    wire1 = ctrlOp.getOutputTarget(0);
  };

  for (std::uint8_t i = 0; i < numBasisUses; ++i) {
    emitFactor(wire1, static_cast<std::size_t>(2 * i));
    emitFactor(wire0, static_cast<std::size_t>((2 * i) + 1));
    emitEntangler();
  }
  emitFactor(wire1, static_cast<std::size_t>(2 * numBasisUses));
  emitFactor(wire0, static_cast<std::size_t>((2 * numBasisUses) + 1));

  outQubit0 = wire0;
  outQubit1 = wire1;
  return success();
}

std::optional<std::uint8_t>
twoQubitEntanglerCount(const Matrix4x4& target, const NativeProfileSpec& spec) {
  const auto entangler = selectEntangler(spec);
  if (!entangler) {
    return std::nullopt;
  }
  const auto native = decomposeWithEntangler(target, *entangler);
  if (!native) {
    return std::nullopt;
  }
  return native->numBasisUses;
}

Matrix2x2 rxMatrix(double theta) {
  const auto halfTheta = theta / 2.;
  const Complex cos{std::cos(halfTheta), 0.};
  const Complex isin{0., -std::sin(halfTheta)};
  return Matrix2x2::fromElements(cos, isin, isin, cos);
}

Matrix2x2 ryMatrix(double theta) {
  const auto halfTheta = theta / 2.;
  const Complex cos{std::cos(halfTheta), 0.};
  const Complex sin{std::sin(halfTheta), 0.};
  return Matrix2x2::fromElements(cos, -sin, sin, cos);
}

Matrix2x2 rzMatrix(double theta) {
  return Matrix2x2::fromElements(
      Complex{std::cos(theta / 2.), -std::sin(theta / 2.)}, 0., 0.,
      Complex{std::cos(theta / 2.), std::sin(theta / 2.)});
}

const Matrix2x2& ipz() {
  static const Matrix2x2 MATRIX =
      Matrix2x2::fromElements(Complex{0, 1}, 0, 0, Complex{0, -1});
  return MATRIX;
}

const Matrix2x2& ipy() {
  static const Matrix2x2 MATRIX = Matrix2x2::fromElements(0, 1, -1, 0);
  return MATRIX;
}

const Matrix2x2& ipx() {
  static const Matrix2x2 MATRIX =
      Matrix2x2::fromElements(0, Complex{0, 1}, Complex{0, 1}, 0);
  return MATRIX;
}

} // namespace mlir::qco::decomposition
