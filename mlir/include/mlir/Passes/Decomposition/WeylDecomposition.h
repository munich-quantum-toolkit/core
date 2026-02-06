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

#include "EulerBasis.h"
#include "EulerDecomposition.h"
#include "Helpers.h"
#include "UnitaryMatrices.h"
#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"

#include <Eigen/Core> // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <random>
#include <tuple>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>

namespace mlir::qco::decomposition {
/**
 * Allowed deviation for internal assert statements which ensure the correctness
 * of the decompositions.
 */
constexpr double SANITY_CHECK_PRECISION = 1e-12;

/**
 * Weyl decomposition of a 2-qubit unitary matrix (4x4).
 * The result consists of four 2x2 1-qubit matrices (k1l, k2l, k1r, k2r) and
 * three parameters for a canonical gate (a, b, c). The matrices can then be
 * decomposed using a single-qubit decomposition into e.g. rotation gates and
 * the canonical gate is RXX(-2 * a), RYY(-2 * b), RZZ (-2 * c).
 */
class TwoQubitWeylDecomposition {
public:
  /**
   * Create Weyl decomposition.
   *
   * @param unitaryMatrix Matrix of the two-qubit operation/series to be
   *                      decomposed.
   * @param fidelity Tolerance to assume a specialization which is used to
   *                 reduce the number of parameters required by the canonical
   *                 gate and thus potentially decreasing the number of basis
   *                 gates.
   */
  static TwoQubitWeylDecomposition create(const Eigen::Matrix4cd& unitaryMatrix,
                                          std::optional<double> fidelity) {
    auto u = unitaryMatrix;
    auto detU = u.determinant();
    auto detPow = std::pow(detU, -0.25);
    u *= detPow; // remove global phase from unitary matrix
    auto globalPhase = std::arg(detU) / 4.;

    // this should have normalized determinant of u, so that u ∈ SU(4)
    assert(std::abs(u.determinant() - 1.0) < SANITY_CHECK_PRECISION);

    // transform unitary matrix to magic basis; this enables two properties:
    // 1. if uP ∈ SO(4), V = A ⊗ B (SO(4) → SU(2) ⊗ SU(2))
    // 2. magic basis diagonalizes canonical gate, allowing calculation of
    //    canonical gate parameters later on
    auto uP = magicBasisTransform(u, MagicBasisTransform::OutOf);
    const Eigen::Matrix4cd m2 = uP.transpose() * uP;

    // diagonalization yields eigenvectors (p) and eigenvalues (d);
    // p is used to calculate K1/K2 (and thus the single-qubit gates
    // surrounding the canonical gate); d is is used to determine the weyl
    // coordinates and thus the parameters of the canonical gate
    // TODO: it may be possible to lower the precision
    auto [p, d] = diagonalizeComplexSymmetric(m2, 1e-13);

    // extract Weyl coordinates from eigenvalues, map to [0, 2*pi)
    // NOLINTNEXTLINE(misc-include-cleaner)
    Eigen::Vector3d cs;
    Eigen::Vector4d dReal = -1.0 * d.cwiseArg() / 2.0;
    dReal(3) = -dReal(0) - dReal(1) - dReal(2);
    for (int i = 0; i < cs.size(); ++i) {
      assert(i < dReal.size());
      cs[i] = helpers::remEuclid((dReal(i) + dReal(3)) / 2.0, qc::TAU);
    }

    // re-order coordinates and according to min(a, pi/2 - a) with
    // a = x mod pi/2 for each weyl coordinate x
    decltype(cs) cstemp;
    llvm::transform(cs, cstemp.begin(), [](auto&& x) {
      auto tmp = helpers::remEuclid(x, qc::PI_2);
      return std::min(tmp, qc::PI_2 - tmp);
    });
    std::array<int, 3> order{0, 1, 2};
    llvm::stable_sort(order,
                      [&](auto a, auto b) { return cstemp[a] < cstemp[b]; });
    std::tie(order[0], order[1], order[2]) =
        std::tuple{order[1], order[2], order[0]};
    std::tie(cs[0], cs[1], cs[2]) =
        std::tuple{cs[order[0]], cs[order[1]], cs[order[2]]};
    std::tie(dReal(0), dReal(1), dReal(2)) =
        std::tuple{dReal(order[0]), dReal(order[1]), dReal(order[2])};

    // update eigenvectors (columns of p) according to new order of
    // weyl coordinates
    Eigen::Matrix4cd pOrig = p;
    for (int i = 0; std::cmp_less(i, order.size()); ++i) {
      p.col(i) = pOrig.col(order[i]);
    }
    // apply correction for determinant if necessary
    if (p.determinant().real() < 0.0) {
      auto lastColumnIndex = p.cols() - 1;
      p.col(lastColumnIndex) *= -1.0;
    }
    assert(std::abs(p.determinant() - 1.0) < SANITY_CHECK_PRECISION);

    // re-create complex eigenvalue matrix; this matrix contains the
    // parameters of the canonical gate which is later used in the
    // verification
    Eigen::Matrix4cd temp = dReal.asDiagonal();
    temp *= std::complex<double>{0, 1};
    // since the matrix is diagonal, matrix exponential is equivalent to
    // element-wise exponential function
    temp.diagonal() = temp.diagonal().array().exp().matrix();

    // combined matrix k1 of 1q gates after canonical gate
    Eigen::Matrix4cd k1 = uP * p * temp;
    assert((k1.transpose() * k1).isIdentity()); // k1 must be orthogonal
    assert(k1.determinant().real() > 0.0);
    k1 = magicBasisTransform(k1, MagicBasisTransform::Into);

    // combined matrix k2 of 1q gates before canonical gate
    Eigen::Matrix4cd k2 = p.transpose().conjugate();
    assert((k2.transpose() * k2).isIdentity()); // k2 must be orthogonal
    assert(k2.determinant().real() > 0.0);
    k2 = magicBasisTransform(k2, MagicBasisTransform::Into);

    // ensure k1 and k2 are correct (when combined with the canonical gate
    // parameters in-between, they are equivalent to u)
    assert(
        (k1 * magicBasisTransform(temp.conjugate(), MagicBasisTransform::Into) *
         k2)
            .isApprox(u, SANITY_CHECK_PRECISION));

    // calculate k1 = K1l ⊗ K1r
    auto [K1l, K1r, phase_l] = decomposeTwoQubitProductGate(k1);
    // decompose k2 = K2l ⊗ K2r
    auto [K2l, K2r, phase_r] = decomposeTwoQubitProductGate(k2);
    assert(
        Eigen::kroneckerProduct(K1l, K1r).isApprox(k1, SANITY_CHECK_PRECISION));
    assert(
        Eigen::kroneckerProduct(K2l, K2r).isApprox(k2, SANITY_CHECK_PRECISION));
    // accumulate global phase
    globalPhase += phase_l + phase_r;

    // Flip into Weyl chamber
    if (cs[0] > qc::PI_2) {
      cs[0] -= 3.0 * qc::PI_2;
      K1l = K1l * IPY;
      K1r = K1r * IPY;
      globalPhase += qc::PI_2;
    }
    if (cs[1] > qc::PI_2) {
      cs[1] -= 3.0 * qc::PI_2;
      K1l = K1l * IPX;
      K1r = K1r * IPX;
      globalPhase += qc::PI_2;
    }
    auto conjs = 0;
    if (cs[0] > qc::PI_4) {
      cs[0] = qc::PI_2 - cs[0];
      K1l = K1l * IPY;
      K2r = IPY * K2r;
      conjs += 1;
      globalPhase -= qc::PI_2;
    }
    if (cs[1] > qc::PI_4) {
      cs[1] = qc::PI_2 - cs[1];
      K1l = K1l * IPX;
      K2r = IPX * K2r;
      conjs += 1;
      globalPhase += qc::PI_2;
      if (conjs == 1) {
        globalPhase -= qc::PI;
      }
    }
    if (cs[2] > qc::PI_2) {
      cs[2] -= 3.0 * qc::PI_2;
      K1l = K1l * IPZ;
      K1r = K1r * IPZ;
      globalPhase += qc::PI_2;
      if (conjs == 1) {
        globalPhase -= qc::PI;
      }
    }
    if (conjs == 1) {
      cs[2] = qc::PI_2 - cs[2];
      K1l = K1l * IPZ;
      K2r = IPZ * K2r;
      globalPhase += qc::PI_2;
    }
    if (cs[2] > qc::PI_4) {
      cs[2] -= qc::PI_2;
      K1l = K1l * IPZ;
      K1r = K1r * IPZ;
      globalPhase -= qc::PI_2;
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
    decomposition.defaultEulerBasis = EulerBasis::ZYZ;
    decomposition.requestedFidelity = fidelity;
    // will be calculated if a specialization is used; set to -1 for now
    decomposition.calculatedFidelity = -1.0;
    decomposition.unitaryMatrix = unitaryMatrix;

    // make sure decomposition is equal to input
    assert((Eigen::kroneckerProduct(K1l, K1r) *
            decomposition.getCanonicalMatrix() *
            Eigen::kroneckerProduct(K2l, K2r) *
            helpers::globalPhaseFactor(globalPhase))
               .isApprox(unitaryMatrix, SANITY_CHECK_PRECISION));

    // determine actual specialization of canonical gate so that the 1q
    // matrices can potentially be simplified
    auto flippedFromOriginal = decomposition.applySpecialization();

    auto getTrace = [&]() {
      if (flippedFromOriginal) {
        return TwoQubitWeylDecomposition::getTrace(
            qc::PI_2 - a, b, -c, decomposition.a_, decomposition.b_,
            decomposition.c_);
      }
      return TwoQubitWeylDecomposition::getTrace(
          a, b, c, decomposition.a_, decomposition.b_, decomposition.c_);
    };
    // use trace to calculate fidelity of applied specialization and
    // adjust global phase
    auto trace = getTrace();
    decomposition.calculatedFidelity = helpers::traceToFidelity(trace);
    // final check if specialization is close enough to the original matrix to
    // satisfy the requested fidelity; since no forced specialization is
    // allowed, this should never fail
    if (decomposition.requestedFidelity &&
        decomposition.calculatedFidelity + 1.0e-13 <
            *decomposition.requestedFidelity) {
      llvm::reportFatalInternalError(llvm::formatv(
          "TwoQubitWeylDecomposition: Calculated fidelity of "
          "specialization is worse than requested fidelity ({0:F4} vs {1:F4})!",
          decomposition.calculatedFidelity, *decomposition.requestedFidelity));
    }
    decomposition.globalPhase_ += std::arg(trace);

    // final check if decomposition is still valid after specialization
    assert((Eigen::kroneckerProduct(decomposition.k1l_, decomposition.k1r_) *
            decomposition.getCanonicalMatrix() *
            Eigen::kroneckerProduct(decomposition.k2l_, decomposition.k2r_) *
            helpers::globalPhaseFactor(decomposition.globalPhase_))
               .isApprox(unitaryMatrix, SANITY_CHECK_PRECISION));

    return decomposition;
  }

  TwoQubitWeylDecomposition(const TwoQubitWeylDecomposition&) = default;
  TwoQubitWeylDecomposition(TwoQubitWeylDecomposition&&) = default;
  TwoQubitWeylDecomposition&
  operator=(const TwoQubitWeylDecomposition&) = default;
  TwoQubitWeylDecomposition& operator=(TwoQubitWeylDecomposition&&) = default;

  /**
   * Calculate matrix of canonical gate based on its parameters a, b, c.
   */
  [[nodiscard]] Eigen::Matrix4cd getCanonicalMatrix() const {
    return getCanonicalMatrix(a_, b_, c_);
  }

  /**
   * First parameter of canonical gate.
   *
   * @note must be multiplied by -2.0 for rotation angle of RXX gate
   */
  [[nodiscard]] double a() const { return a_; }
  /**
   * First parameter of canonical gate.
   *
   * @note must be multiplied by -2.0 for rotation angle of RYY gate
   */
  [[nodiscard]] double b() const { return b_; }
  /**
   * First parameter of canonical gate.
   *
   * @note must be multiplied by -2.0 for rotation angle of RZZ gate
   */
  [[nodiscard]] double c() const { return c_; }
  /**
   * Necessary global phase adjustment after applying decomposition.
   */
  [[nodiscard]] double globalPhase() const { return globalPhase_; }

  /**
   * "Left" qubit after canonical gate.
   *
   * q1 - k2r - C -  k1r  -
   *            A
   * q0 - k2l - N - *k1l* -
   */
  [[nodiscard]] const Eigen::Matrix2cd& k1l() const { return k1l_; }
  /**
   * "Left" qubit before canonical gate.
   *
   * q1 -  k2r  - C - k1r -
   *              A
   * q0 - *k2l* - N - k1l -
   */
  [[nodiscard]] const Eigen::Matrix2cd& k2l() const { return k2l_; }
  /**
   * "Right" qubit after canonical gate.
   *
   * q1 - k2r - C - *k1r* -
   *            A
   * q0 - k2l - N -  k1l  -
   */
  [[nodiscard]] const Eigen::Matrix2cd& k1r() const { return k1r_; }
  /**
   * "Right" qubit before canonical gate.
   *
   * q1 - *k2r* - C - k1r -
   *              A
   * q0 -  k2l  - N - k1l -
   */
  [[nodiscard]] const Eigen::Matrix2cd& k2r() const { return k2r_; }

  /**
   * Calculate matrix of canonical gate based on given parameters a, b, c.
   */
  [[nodiscard]] static Eigen::Matrix4cd getCanonicalMatrix(double a, double b,
                                                           double c) {
    auto xx = getTwoQubitMatrix({
        .type = qc::RXX,
        .parameter = {-2.0 * a},
        .qubitId = {0, 1},
    });
    auto yy = getTwoQubitMatrix({
        .type = qc::RYY,
        .parameter = {-2.0 * b},
        .qubitId = {0, 1},
    });
    auto zz = getTwoQubitMatrix({
        .type = qc::RZZ,
        .parameter = {-2.0 * c},
        .qubitId = {0, 1},
    });
    return zz * yy * xx;
  }

protected:
  enum class Specialization : std::uint8_t {
    General,               // canonical gate has no special symmetry.
    IdEquiv,               // canonical gate is identity.
    SWAPEquiv,             // canonical gate is SWAP.
    PartialSWAPEquiv,      // canonical gate is partial SWAP.
    PartialSWAPFlipEquiv,  // canonical gate is flipped partial SWAP.
    ControlledEquiv,       // canonical gate is a controlled gate.
    MirrorControlledEquiv, // canonical gate is swap + controlled gate.

    // These next 3 gates use the definition of fSim from eq (1) in:
    // https://arxiv.org/pdf/2001.08343.pdf
    FSimaabEquiv,  // parameters a=b & a!=c
    FSimabbEquiv,  // parameters a!=b & b=c
    FSimabmbEquiv, // parameters a!=b!=c & -b=c
  };

  enum class MagicBasisTransform : std::uint8_t {
    Into,
    OutOf,
  };

  TwoQubitWeylDecomposition() = default;

  static Eigen::Matrix4cd magicBasisTransform(const Eigen::Matrix4cd& unitary,
                                              MagicBasisTransform direction) {
    using namespace std::complex_literals;
    const Eigen::Matrix4cd bNonNormalized{
        {1, 1i, 0, 0},
        {0, 0, 1i, 1},
        {0, 0, 1i, -1},
        {1, -1i, 0, 0},
    };

    const Eigen::Matrix4cd bNonNormalizedDagger{
        {0.5, 0, 0, 0.5},
        {-0.5i, 0, 0, 0.5i},
        {0, -0.5i, -0.5i, 0},
        {0, 0.5, -0.5, 0},
    };
    if (direction == MagicBasisTransform::OutOf) {
      return bNonNormalizedDagger * unitary * bNonNormalized;
    }
    if (direction == MagicBasisTransform::Into) {
      return bNonNormalized * unitary * bNonNormalizedDagger;
    }
    llvm::reportFatalInternalError("Unknown MagicBasisTransform direction!");
  }

  static double closestPartialSwap(double a, double b, double c) {
    auto m = (a + b + c) / 3.;
    auto [am, bm, cm] = std::array{a - m, b - m, c - m};
    auto [ab, bc, ca] = std::array{a - b, b - c, c - a};
    return m + (am * bm * cm * (6. + ab * ab + bc * bc + ca * ca) / 18.);
  }

  /**
   * Diagonalize given complex symmetric matrix M into (P, d) using a
   * randomized algorithm.
   * This approach is used in both qiskit and quantumflow.
   *
   * P is the matrix of real or orthogonal eigenvectors of M with P ∈ SO(4)
   * d is a vector containing sqrt(eigenvalues) of M with unit-magnitude
   * elements (for each element, complex magnitude is 1.0).
   * D is d as a diagonal matrix.
   *
   * M = P * D * P^T
   *
   * @return pair of (P, D.diagonal())
   */
  [[nodiscard]] static std::pair<Eigen::Matrix4cd, Eigen::Vector4cd>
  diagonalizeComplexSymmetric(const Eigen::Matrix4cd& m, double precision) {
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

    for (int i = 0; i < 100; ++i) {
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
      const Eigen::Matrix4d m2Real = randA * m.real() + randB * m.imag();
      auto&& pReal = helpers::selfAdjointEvd(m2Real).first;
      const Eigen::Matrix4cd p = pReal;
      auto&& d = (p.transpose() * m * p).diagonal();

      auto&& compare = p * d.asDiagonal() * p.transpose();
      if (compare.isApprox(m, precision)) {
        // p are the eigenvectors which are decomposed into the
        // single-qubit gates surrounding the canonical gate
        // d is the sqrt of the eigenvalues that are used to determine the
        // weyl coordinates and thus the parameters of the canonical gate
        // check that p is in SO(4)
        assert((p.transpose() * p).isIdentity(SANITY_CHECK_PRECISION));
        // make sure determinant of eigenvalues is 1.0
        assert(std::abs(Eigen::Matrix4cd{d.asDiagonal()}.determinant() - 1.0) <
               SANITY_CHECK_PRECISION);
        return std::make_pair(p, d);
      }
    }
    llvm::reportFatalInternalError(
        "TwoQubitWeylDecomposition: failed to diagonalize M2.");
  }

  /**
   * Decompose a special unitary matrix C that is the combination of two
   * single-qubit gates A and B into its single-qubit matrices.
   *
   * C = A ⊗ B
   *
   * @param specialUnitary Special unitary matrix C
   *
   * @return single-qubit matrices A and B and the required
   *         global phase adjustment
   */
  static std::tuple<Eigen::Matrix2cd, Eigen::Matrix2cd, double>
  decomposeTwoQubitProductGate(const Eigen::Matrix4cd& specialUnitary) {
    // for alternative approaches, see
    // pennylane's math.decomposition.su2su2_to_tensor_products
    // or quantumflow.kronecker_decomposition

    // first quadrant
    Eigen::Matrix2cd r{{specialUnitary(0, 0), specialUnitary(0, 1)},
                       {specialUnitary(1, 0), specialUnitary(1, 1)}};
    auto detR = r.determinant();
    if (std::abs(detR) < 0.1) {
      // third quadrant
      r = Eigen::Matrix2cd{{specialUnitary(2, 0), specialUnitary(2, 1)},
                           {specialUnitary(3, 0), specialUnitary(3, 1)}};
      detR = r.determinant();
    }
    if (std::abs(detR) < 0.1) {
      llvm::reportFatalInternalError(
          "decompose_two_qubit_product_gate: unable to decompose: det_r < 0.1");
    }
    r /= std::sqrt(detR);
    // transpose with complex conjugate of each element
    const Eigen::Matrix2cd rTConj = r.transpose().conjugate();

    Eigen::Matrix4cd temp =
        Eigen::kroneckerProduct(Eigen::Matrix2cd::Identity(), rTConj);
    temp = specialUnitary * temp;

    // [[a, b, c, d],
    //  [e, f, g, h], => [[a, c],
    //  [i, j, k, l],     [i, k]]
    //  [m, n, o, p]]
    Eigen::Matrix2cd l{{temp(0, 0), temp(0, 2)}, {temp(2, 0), temp(2, 2)}};
    auto detL = l.determinant();
    if (std::abs(detL) < 0.9) {
      llvm::reportFatalInternalError(
          "decompose_two_qubit_product_gate: unable to decompose: detL < 0.9");
    }
    l /= std::sqrt(detL);
    auto phase = std::arg(detL) / 2.;

    return {l, r, phase};
  }

  /**
   * Calculate trace of two sets of parameters for the canonical gate.
   * The trace has been defined in: https://arxiv.org/abs/1811.12926
   */
  [[nodiscard]] static std::complex<double>
  getTrace(double a, double b, double c, double ap, double bp, double cp) {
    auto da = a - ap;
    auto db = b - bp;
    auto dc = c - cp;
    return 4. *
           std::complex<double>{std::cos(da) * std::cos(db) * std::cos(dc),
                                std::sin(da) * std::sin(db) * std::sin(dc)};
  }

  /**
   * Choose the best specialization for the for the canonical gate.
   * This will use the requestedFidelity to determine if a specialization is
   * close enough to the actual canonical gate matrix.
   */
  [[nodiscard]] Specialization bestSpecialization() const {
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
    if (isClose(qc::PI_4, qc::PI_4, qc::PI_4) ||
        isClose(qc::PI_4, qc::PI_4, -qc::PI_4)) {
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
    if (isClose(qc::PI_4, qc::PI_4, c_)) {
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

  /**
   * @return true if the specialization flipped the original decomposition
   */
  bool applySpecialization() {
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
      k2l_ = Eigen::Matrix2cd::Identity();
      k1r_ = k1r_ * k2r_;
      k2r_ = Eigen::Matrix2cd::Identity();
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
        k2l_ = Eigen::Matrix2cd::Identity();
        k2r_ = Eigen::Matrix2cd::Identity();
      } else {
        flippedFromOriginal = true;

        globalPhase_ += qc::PI_2;
        k1l_ = k1l_ * IPZ * k2r_;
        k1r_ = k1r_ * IPZ * k2l_;
        k2l_ = Eigen::Matrix2cd::Identity();
        k2r_ = Eigen::Matrix2cd::Identity();
      }
      a_ = qc::PI_4;
      b_ = qc::PI_4;
      c_ = qc::PI_4;
    } else if (newSpecialization == Specialization::PartialSWAPEquiv) {
      // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, \alpha\pi/4)`
      // Thus, :math:`U \sim \text{SWAP}^\alpha`
      //
      // This gate binds 3 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Id`.
      auto closest = closestPartialSwap(a_, b_, c_);
      auto k2lDagger = k2l_.transpose().conjugate();

      a_ = closest;
      b_ = closest;
      c_ = closest;
      // unmodified global phase
      k1l_ = k1l_ * k2l_;
      k1r_ = k1r_ * k2l_;
      k2r_ = k2lDagger * k2r_;
      k2l_ = Eigen::Matrix2cd::Identity();
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
      auto k2lDagger = k2l_.transpose().conjugate();

      a_ = closest;
      b_ = closest;
      c_ = -closest;
      // unmodified global phase
      k1l_ = k1l_ * k2l_;
      k1r_ = k1r_ * IPZ * k2l_ * IPZ;
      k2r_ = IPZ * k2lDagger * IPZ * k2r_;
      k2l_ = Eigen::Matrix2cd::Identity();
    } else if (newSpecialization == Specialization::ControlledEquiv) {
      // :math:`U \sim U_d(\alpha, 0, 0)`
      // Thus, :math:`U \sim \text{Ctrl-U}`
      //
      // This gate binds 4 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Ry(\theta_l) Rx(\lambda_l)`
      // :math:`K2_r = Ry(\theta_r) Rx(\lambda_r)`
      auto eulerBasis = EulerBasis::XYX;
      auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
          EulerDecomposition::anglesFromUnitary(k2l_, eulerBasis);
      auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
          EulerDecomposition::anglesFromUnitary(k2r_, eulerBasis);

      // unmodified parameter a
      b_ = 0.;
      c_ = 0.;
      globalPhase_ = globalPhase_ + k2lphase + k2rphase;
      k1l_ = k1l_ * rxMatrix(k2lphi);
      k2l_ = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
      k1r_ = k1r_ * rxMatrix(k2rphi);
      k2r_ = ryMatrix(k2rtheta) * rxMatrix(k2rlambda);
      defaultEulerBasis = eulerBasis;
    } else if (newSpecialization == Specialization::MirrorControlledEquiv) {
      // :math:`U \sim U_d(\pi/4, \pi/4, \alpha)`
      // Thus, :math:`U \sim \text{SWAP} \cdot \text{Ctrl-U}`
      //
      // This gate binds 4 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`
      // :math:`K2_r = Ry(\theta_r)\cdot Rz(\lambda_r)`
      auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
          EulerDecomposition::anglesFromUnitary(k2l_, EulerBasis::ZYZ);
      auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
          EulerDecomposition::anglesFromUnitary(k2r_, EulerBasis::ZYZ);

      a_ = qc::PI_4;
      b_ = qc::PI_4;
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
      // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`.
      auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
          EulerDecomposition::anglesFromUnitary(k2l_, EulerBasis::ZYZ);
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
      // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
      //
      // This gate binds 5 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
      auto eulerBasis = EulerBasis::XYX;
      auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
          EulerDecomposition::anglesFromUnitary(k2l_, eulerBasis);
      auto bc = (b_ + c_) / 2.;

      // unmodified parameter a
      b_ = bc;
      c_ = bc;
      globalPhase_ = globalPhase_ + k2lphase;
      k1l_ = k1l_ * rxMatrix(k2lphi);
      k2l_ = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
      k1r_ = k1r_ * rxMatrix(k2lphi);
      k2r_ = rxMatrix(-k2lphi) * k2r_;
      defaultEulerBasis = eulerBasis;
    } else if (newSpecialization == Specialization::FSimabmbEquiv) {
      // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
      //
      // This gate binds 5 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
      auto eulerBasis = EulerBasis::XYX;
      auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
          EulerDecomposition::anglesFromUnitary(k2l_, eulerBasis);
      auto bc = (b_ - c_) / 2.;

      // unmodified parameter a
      b_ = bc;
      c_ = -bc;
      globalPhase_ = globalPhase_ + k2lphase;
      k1l_ = k1l_ * rxMatrix(k2lphi);
      k2l_ = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
      k1r_ = k1r_ * IPZ * rxMatrix(k2lphi) * IPZ;
      k2r_ = IPZ * rxMatrix(-k2lphi) * IPZ * k2r_;
      defaultEulerBasis = eulerBasis;
    } else {
      llvm::reportFatalInternalError(
          "Unknown specialization for Weyl decomposition!");
    }
    return flippedFromOriginal;
  }

private:
  // a, b, c are the parameters of the canonical gate (CAN)
  double a_{}; // rotation of RXX gate in CAN (must be taken times -2.0)
  double b_{}; // rotation of RYY gate in CAN (must be taken times -2.0)
  double c_{}; // rotation of RZZ gate in CAN (must be taken times -2.0)
  double globalPhase_{}; // global phase adjustment
  /**
   * q1 - k2r - C - k1r -
   *            A
   * q0 - k2l - N - k1l -
   */
  Eigen::Matrix2cd k1l_; // "left" qubit after canonical gate
  Eigen::Matrix2cd k2l_; // "left" qubit before canonical gate
  Eigen::Matrix2cd k1r_; // "right" qubit after canonical gate
  Eigen::Matrix2cd k2r_; // "right" qubit before canonical gate
  Specialization specialization{
      Specialization::General}; // detected symmetries in the matrix
  EulerBasis defaultEulerBasis{
      EulerBasis::U3};            // recommended euler basis for k1l/k2l/k1r/k2r
  std::optional<double>           // desired fidelity;
      requestedFidelity;          // if set to std::nullopt, no automatic
                                  // specialization will be applied
  double calculatedFidelity{};    // actual fidelity of decomposition
  Eigen::Matrix4cd unitaryMatrix; // original matrix for this decomposition
};
} // namespace mlir::qco::decomposition
