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

#include <Eigen/Core> // NOLINT(misc-include-cleaner)
#include <cassert>
#include <complex>
#include <cstdint>
#include <optional>
#include <tuple>
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
                                          std::optional<double> fidelity);

  ~TwoQubitWeylDecomposition() = default;
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
   * Second parameter of canonical gate.
   *
   * @note must be multiplied by -2.0 for rotation angle of RYY gate
   */
  [[nodiscard]] double b() const { return b_; }
  /**
   * Third parameter of canonical gate.
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
                                                           double c);

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

  /**
   * Threshold for imprecision in computation of diagonalization.
   */
  static constexpr auto DIAGONALIZATION_PRECISION = 1e-13;

  TwoQubitWeylDecomposition() = default;

  [[nodiscard]] static Eigen::Matrix4cd
  magicBasisTransform(const Eigen::Matrix4cd& unitary,
                      MagicBasisTransform direction);

  [[nodiscard]] static double closestPartialSwap(double a, double b, double c);

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
  diagonalizeComplexSymmetric(const Eigen::Matrix4cd& m, double precision);

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
  decomposeTwoQubitProductGate(const Eigen::Matrix4cd& specialUnitary);

  /**
   * Calculate trace of two sets of parameters for the canonical gate.
   * The trace has been defined in: https://arxiv.org/abs/1811.12926
   */
  [[nodiscard]] static std::complex<double>
  getTrace(double a, double b, double c, double ap, double bp, double cp);

  /**
   * Choose the best specialization for the canonical gate.
   * This will use the requestedFidelity to determine if a specialization is
   * close enough to the actual canonical gate matrix.
   */
  [[nodiscard]] Specialization bestSpecialization() const;

  /**
   * @return true if the specialization flipped the original decomposition
   */
  bool applySpecialization();

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
