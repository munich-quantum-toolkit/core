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
constexpr fp SANITY_CHECK_PRECISION = 1e-12;

/**
 * Weyl decomposition of a 2-qubit unitary matrix (4x4).
 * The result consists of four 2x2 1-qubit matrices (k1l, k2l, k1r, k2r) and
 * three parameters for a canonical gate (a, b, c). The matrices can then be
 * decomposed using a single-qubit decomposition into e.g. rotation gates and
 * the canonical gate is RXX(-2 * a), RYY(-2 * b), RZZ (-2 * c).
 */
struct TwoQubitWeylDecomposition {
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

  // a, b, c are the parameters of the canonical gate (CAN)
  fp a;           // rotation of RXX gate in CAN (must be taken times -2.0)
  fp b;           // rotation of RYY gate in CAN (must be taken times -2.0)
  fp c;           // rotation of RZZ gate in CAN (must be taken times -2.0)
  fp globalPhase; // global phase adjustment
  /**
   * q1 - k2r - C - k1r -
   *            A
   * q0 - k2l - N - k1l -
   */
  matrix2x2 k1l;                 // "left" qubit after canonical gate
  matrix2x2 k2l;                 // "left" qubit before canonical gate
  matrix2x2 k1r;                 // "right" qubit after canonical gate
  matrix2x2 k2r;                 // "right" qubit before canonical gate
  Specialization specialization; // detected symmetries in the matrix
  EulerBasis defaultEulerBasis;  // recommended euler basis for k1l/k2l/k1r/k2r
  std::optional<fp> requestedFidelity; // desired fidelity;
                                       // if set to std::nullopt, no automatic
                                       // specialization will be applied
  fp calculatedFidelity;               // actual fidelity of decomposition
  matrix4x4 unitaryMatrix;             // original matrix for this decomposition

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
  static TwoQubitWeylDecomposition create(const matrix4x4& unitaryMatrix,
                                          std::optional<fp> fidelity) {
    auto u = unitaryMatrix;
    auto detU = u.determinant();
    auto detPow = std::pow(detU, static_cast<fp>(-0.25));
    u *= detPow; // remove global phase from unitary matrix
    auto globalPhase = std::arg(detU) / 4.;

    // this should have normalized determinant of u, so that u ∈ SU(4)
    assert(std::abs(u.determinant() - 1.0) < SANITY_CHECK_PRECISION);

    // transform unitary matrix to magic basis; this enables two properties:
    // 1. if uP ∈ SO(4), V = A ⊗ B (SO(4) → SU(2) ⊗ SU(2))
    // 2. magic basis diagonalizes canonical gate, allowing calculation of
    //    canonical gate parameters later on
    auto uP = magicBasisTransform(u, MagicBasisTransform::OutOf);
    const matrix4x4 m2 = uP.transpose() * uP;

    // diagonalization yields eigenvectors (p) and eigenvalues (d);
    // p is used to calculate K1/K2 (and thus the single-qubit gates
    // surrounding the canonical gate); d is is used to determine the weyl
    // coordinates and thus the parameters of the canonical gate
    // TODO: it may be possible to lower the precision
    auto [p, d] = diagonalizeComplexSymmetric(m2, 1e-13);

    // extract Weyl coordinates from eigenvalues, map to [0, 2*pi)
    // NOLINTNEXTLINE(misc-include-cleaner)
    Eigen::Vector<fp, 3> cs;
    rdiagonal4x4 dReal = -1.0 * d.cwiseArg() / 2.0;
    dReal(3) = -dReal(0) - dReal(1) - dReal(2);
    for (int i = 0; i < static_cast<int>(cs.size()); ++i) {
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
    matrix4x4 pOrig = p;
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
    matrix4x4 temp = dReal.asDiagonal();
    temp *= IM;
    // since the matrix is diagonal, matrix exponental is equivalent to
    // element-wise exponential function
    temp.diagonal() = temp.diagonal().array().exp().matrix();

    // combined matrix k1 of 1q gates after canonical gate
    matrix4x4 k1 = uP * p * temp;
    assert((k1.transpose() * k1).isIdentity()); // k1 must be orthogonal
    assert(k1.determinant().real() > 0.0);
    k1 = magicBasisTransform(k1, MagicBasisTransform::Into);

    // combined matrix k2 of 1q gates before canonical gate
    matrix4x4 k2 = p.transpose().conjugate();
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

    TwoQubitWeylDecomposition decomposition{
        .a = a,
        .b = b,
        .c = c,
        .globalPhase = globalPhase,
        .k1l = K1l,
        .k2l = K2l,
        .k1r = K1r,
        .k2r = K2r,
        .specialization = Specialization::General,
        .defaultEulerBasis = EulerBasis::ZYZ,
        .requestedFidelity = fidelity,
        // will be calculated if a specialization is used, set to -1 for now
        .calculatedFidelity = -1.0,
        .unitaryMatrix = unitaryMatrix,
    };
    // make sure decomposition is equal to input
    assert((Eigen::kroneckerProduct(K1l, K1r) *
            decomposition.getCanonicalMatrix() *
            Eigen::kroneckerProduct(K2l, K2r) * std::exp(IM * globalPhase))
               .isApprox(unitaryMatrix, SANITY_CHECK_PRECISION));

    // determine actual specialization of canonical gate so that the 1q
    // matrices can potentially be simplified
    auto flippedFromOriginal = decomposition.applySpecialization();

    auto getTrace = [&]() {
      if (flippedFromOriginal) {
        return TwoQubitWeylDecomposition::getTrace(
            qc::PI_2 - a, b, -c, decomposition.a, decomposition.b,
            decomposition.c);
      }
      return TwoQubitWeylDecomposition::getTrace(
          a, b, c, decomposition.a, decomposition.b, decomposition.c);
    };
    // use trace to calculate fidelity of applied specialization and
    // adjust global phase
    auto trace = getTrace();
    decomposition.calculatedFidelity = helpers::traceToFidelity(trace);
    // final check if specialization is close enough to the original matrix to
    // satisfy the requested fidelity; since no forced specialization is
    // allowed, this should never fail
    if (decomposition.requestedFidelity) {
      if (decomposition.calculatedFidelity + 1.0e-13 <
          *decomposition.requestedFidelity) {
        throw std::runtime_error{
            "TwoQubitWeylDecomposition: Calculated fidelity of "
            "specialization is worse than requested fidelity!"};
      }
    }
    decomposition.globalPhase += std::arg(trace);

    // final check if decomposition is still valid after specialization
    assert((Eigen::kroneckerProduct(decomposition.k1l, decomposition.k1r) *
            decomposition.getCanonicalMatrix() *
            Eigen::kroneckerProduct(decomposition.k2l, decomposition.k2r) *
            std::exp(IM * decomposition.globalPhase))
               .isApprox(unitaryMatrix, SANITY_CHECK_PRECISION));

    return decomposition;
  }

  /**
   * Calculate matrix of canonical gate based on its parameters a, b, c.
   */
  [[nodiscard]] matrix4x4 getCanonicalMatrix() const {
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
  static constexpr fp SANITY_CHECK_PRECISION = 1e-12;

  // https://docs.rs/faer/latest/faer/mat/generic/struct.Mat.html#method.self_adjoint_eigen
  template <typename T> static auto selfAdjointEigenLower(T&& a) {
    auto [U, S] = helpers::selfAdjointEvd(std::forward<T>(a));

    return std::make_pair(U, S);
  }

  enum class MagicBasisTransform : std::uint8_t {
    Into,
    OutOf,
  };

  static matrix4x4 magicBasisTransform(const matrix4x4& unitary,
                                       MagicBasisTransform direction) {
    const matrix4x4 bNonNormalized{
        {C_ONE, IM, C_ZERO, C_ZERO},
        {C_ZERO, C_ZERO, IM, C_ONE},
        {C_ZERO, C_ZERO, IM, C_M_ONE},
        {C_ONE, M_IM, C_ZERO, C_ZERO},
    };

    const matrix4x4 bNonNormalizedDagger{
        {qfp(0.5, 0.), C_ZERO, C_ZERO, qfp(0.5, 0.)},
        {qfp(0., -0.5), C_ZERO, C_ZERO, qfp(0., 0.5)},
        {C_ZERO, qfp(0., -0.5), qfp(0., -0.5), C_ZERO},
        {C_ZERO, qfp(0.5, 0.), qfp(-0.5, 0.), C_ZERO},
    };
    if (direction == MagicBasisTransform::OutOf) {
      return bNonNormalizedDagger * unitary * bNonNormalized;
    }
    if (direction == MagicBasisTransform::Into) {
      return bNonNormalized * unitary * bNonNormalizedDagger;
    }
    throw std::logic_error{"Unknown MagicBasisTransform direction!"};
  }

  static fp closestPartialSwap(fp a, fp b, fp c) {
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
  [[nodiscard]] static std::pair<matrix4x4, diagonal4x4>
  diagonalizeComplexSymmetric(const matrix4x4& m, fp precision) {
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
    std::normal_distribution<fp> dist;

    for (int i = 0; i < 100; ++i) {
      fp randA{};
      fp randB{};
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
      const rmatrix4x4 m2Real = randA * m.real() + randB * m.imag();
      const rmatrix4x4 pReal = selfAdjointEigenLower(m2Real).first;
      const matrix4x4 p = pReal;
      const diagonal4x4 d = (p.transpose() * m * p).diagonal();

      const matrix4x4 diagD = d.asDiagonal();

      const matrix4x4 compare = p * diagD * p.transpose();
      if (compare.isApprox(m, precision)) {
        // p are the eigenvectors which are decomposed into the
        // single-qubit gates surrounding the canonical gate
        // d is the sqrt of the eigenvalues that are used to determine the
        // weyl coordinates and thus the parameters of the canonical gate
        // check that p is in SO(4)
        assert((p.transpose() * p).isIdentity(SANITY_CHECK_PRECISION));
        // make sure determinant of eigenvalues is 1.0
        assert(std::abs(matrix4x4{d.asDiagonal()}.determinant() - 1.0) <
               SANITY_CHECK_PRECISION);
        return std::make_pair(p, d);
      }
    }
    throw std::runtime_error{
        "TwoQubitWeylDecomposition: failed to diagonalize M2."};
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
  static std::tuple<matrix2x2, matrix2x2, fp>
  decomposeTwoQubitProductGate(const matrix4x4& specialUnitary) {
    // for alternative approaches, see
    // pennylane's math.decomposition.su2su2_to_tensor_products
    // or quantumflow.kronecker_decomposition

    // first quadrant
    matrix2x2 r{{specialUnitary(0, 0), specialUnitary(0, 1)},
                {specialUnitary(1, 0), specialUnitary(1, 1)}};
    auto detR = r.determinant();
    if (std::abs(detR) < 0.1) {
      // third quadrant
      r = matrix2x2{{specialUnitary(2, 0), specialUnitary(2, 1)},
                    {specialUnitary(3, 0), specialUnitary(3, 1)}};
      detR = r.determinant();
    }
    if (std::abs(detR) < 0.1) {
      throw std::runtime_error{
          "decompose_two_qubit_product_gate: unable to decompose: det_r < 0.1"};
    }
    r /= std::sqrt(detR);
    // transpose with complex conjugate of each element
    const matrix2x2 rTConj = r.transpose().conjugate();

    matrix4x4 temp = Eigen::kroneckerProduct(matrix2x2::Identity(), rTConj);
    temp = specialUnitary * temp;

    // [[a, b, c, d],
    //  [e, f, g, h], => [[a, c],
    //  [i, j, k, l],     [i, k]]
    //  [m, n, o, p]]
    matrix2x2 l{{temp(0, 0), temp(0, 2)}, {temp(2, 0), temp(2, 2)}};
    auto detL = l.determinant();
    if (std::abs(detL) < 0.9) {
      throw std::runtime_error{
          "decompose_two_qubit_product_gate: unable to decompose: detL < 0.9"};
    }
    l /= std::sqrt(detL);
    auto phase = std::arg(detL) / 2.;

    return {l, r, phase};
  }

  /**
   * Calculate trace of two sets of parameters for the canonical gate.
   * The trace has been defined in: https://arxiv.org/abs/1811.12926
   */
  [[nodiscard]] static qfp getTrace(fp a, fp b, fp c, fp ap, fp bp, fp cp) {
    auto da = a - ap;
    auto db = b - bp;
    auto dc = c - cp;
    return static_cast<fp>(4.) *
           qfp(std::cos(da) * std::cos(db) * std::cos(dc),
               std::sin(da) * std::sin(db) * std::sin(dc));
  }

  /**
   * Choose the best specialization for the for the canonical gate.
   * This will use the requestedFidelity to determine if a specialization is
   * close enough to the actual canonical gate matrix.
   */
  [[nodiscard]] Specialization bestSpecialization() const {
    auto isClose = [this](fp ap, fp bp, fp cp) -> bool {
      auto tr = getTrace(a, b, c, ap, bp, cp);
      if (requestedFidelity) {
        return helpers::traceToFidelity(tr) >= *requestedFidelity;
      }
      return false;
    };

    auto closestAbc = closestPartialSwap(a, b, c);
    auto closestAbMinusC = closestPartialSwap(a, b, -c);

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
    if (isClose(a, 0., 0.)) {
      return Specialization::ControlledEquiv;
    }
    if (isClose(qc::PI_4, qc::PI_4, c)) {
      return Specialization::MirrorControlledEquiv;
    }
    if (isClose((a + b) / 2., (a + b) / 2., c)) {
      return Specialization::FSimaabEquiv;
    }
    if (isClose(a, (b + c) / 2., (b + c) / 2.)) {
      return Specialization::FSimabbEquiv;
    }
    if (isClose(a, (b - c) / 2., (c - b) / 2.)) {
      return Specialization::FSimabmbEquiv;
    }
    return Specialization::General;
  }

  /**
   * @return true if the specialization flipped the original decomposition
   */
  bool applySpecialization() {
    if (specialization != Specialization::General) {
      throw std::logic_error{"Application of specialization only works on "
                             "general decomposition!"};
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
      a = 0.;
      b = 0.;
      c = 0.;
      // unmodified global phase
      k1l = k1l * k2l;
      k2l = matrix2x2::Identity();
      k1r = k1r * k2r;
      k2r = matrix2x2::Identity();
    } else if (newSpecialization == Specialization::SWAPEquiv) {
      // :math:`U \sim U_d(\pi/4, \pi/4, \pi/4) \sim U(\pi/4, \pi/4, -\pi/4)`
      // Thus, :math:`U \sim \text{SWAP}`
      //
      // This gate binds 0 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Id` , :math:`K2_r = Id`.
      if (c > 0.) {
        // unmodified global phase
        k1l = k1l * k2r;
        k1r = k1r * k2l;
        k2l = matrix2x2::Identity();
        k2r = matrix2x2::Identity();
      } else {
        flippedFromOriginal = true;

        globalPhase += qc::PI_2;
        k1l = k1l * IPZ * k2r;
        k1r = k1r * IPZ * k2l;
        k2l = matrix2x2::Identity();
        k2r = matrix2x2::Identity();
      }
      a = qc::PI_4;
      b = qc::PI_4;
      c = qc::PI_4;
    } else if (newSpecialization == Specialization::PartialSWAPEquiv) {
      // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, \alpha\pi/4)`
      // Thus, :math:`U \sim \text{SWAP}^\alpha`
      //
      // This gate binds 3 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Id`.
      auto closest = closestPartialSwap(a, b, c);
      auto k2lDagger = k2l.transpose().conjugate();

      a = closest;
      b = closest;
      c = closest;
      // unmodified global phase
      k1l = k1l * k2l;
      k1r = k1r * k2l;
      k2r = k2lDagger * k2r;
      k2l = matrix2x2::Identity();
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
      auto closest = closestPartialSwap(a, b, -c);
      auto k2lDagger = k2l.transpose().conjugate();

      a = closest;
      b = closest;
      c = -closest;
      // unmodified global phase
      k1l = k1l * k2l;
      k1r = k1r * IPZ * k2l * IPZ;
      k2r = IPZ * k2lDagger * IPZ * k2r;
      k2l = matrix2x2::Identity();
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
          EulerDecomposition::anglesFromUnitary(k2l, eulerBasis);
      auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
          EulerDecomposition::anglesFromUnitary(k2r, eulerBasis);

      // unmodified parameter a
      b = 0.;
      c = 0.;
      globalPhase = globalPhase + k2lphase + k2rphase;
      k1l = k1l * rxMatrix(k2lphi);
      k2l = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
      k1r = k1r * rxMatrix(k2rphi);
      k2r = ryMatrix(k2rtheta) * rxMatrix(k2rlambda);
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
          EulerDecomposition::anglesFromUnitary(k2l, EulerBasis::ZYZ);
      auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
          EulerDecomposition::anglesFromUnitary(k2r, EulerBasis::ZYZ);

      a = qc::PI_4;
      b = qc::PI_4;
      // unmodified parameter c
      globalPhase = globalPhase + k2lphase + k2rphase;
      k1l = k1l * rzMatrix(k2rphi);
      k2l = ryMatrix(k2ltheta) * rzMatrix(k2llambda);
      k1r = k1r * rzMatrix(k2lphi);
      k2r = ryMatrix(k2rtheta) * rzMatrix(k2rlambda);
    } else if (newSpecialization == Specialization::FSimaabEquiv) {
      // :math:`U \sim U_d(\alpha, \alpha, \beta), \alpha \geq |\beta|`
      //
      // This gate binds 5 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`.
      auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
          EulerDecomposition::anglesFromUnitary(k2l, EulerBasis::ZYZ);
      auto ab = (a + b) / 2.;

      a = ab;
      b = ab;
      // unmodified parameter c
      globalPhase = globalPhase + k2lphase;
      k1l = k1l * rzMatrix(k2lphi);
      k2l = ryMatrix(k2ltheta) * rzMatrix(k2llambda);
      k1r = k1r * rzMatrix(k2lphi);
      k2r = rzMatrix(-k2lphi) * k2r;
    } else if (newSpecialization == Specialization::FSimabbEquiv) {
      // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
      //
      // This gate binds 5 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
      auto eulerBasis = EulerBasis::XYX;
      auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
          EulerDecomposition::anglesFromUnitary(k2l, eulerBasis);
      auto bc = (b + c) / 2.;

      // unmodified parameter a
      b = bc;
      c = bc;
      globalPhase = globalPhase + k2lphase;
      k1l = k1l * rxMatrix(k2lphi);
      k2l = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
      k1r = k1r * rxMatrix(k2lphi);
      k2r = rxMatrix(-k2lphi) * k2r;
      defaultEulerBasis = eulerBasis;
    } else if (newSpecialization == Specialization::FSimabmbEquiv) {
      // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
      //
      // This gate binds 5 parameters, we make it canonical by setting:
      //
      // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
      auto eulerBasis = EulerBasis::XYX;
      auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
          EulerDecomposition::anglesFromUnitary(k2l, eulerBasis);
      auto bc = (b - c) / 2.;

      // unmodified parameter a
      b = bc;
      c = -bc;
      globalPhase = globalPhase + k2lphase;
      k1l = k1l * rxMatrix(k2lphi);
      k2l = ryMatrix(k2ltheta) * rxMatrix(k2llambda);
      k1r = k1r * IPZ * rxMatrix(k2lphi) * IPZ;
      k2r = IPZ * rxMatrix(-k2lphi) * IPZ * k2r;
      defaultEulerBasis = eulerBasis;
    } else {
      throw std::logic_error{"Unknown specialization"};
    }
    return flippedFromOriginal;
  }
};
} // namespace mlir::qco::decomposition
