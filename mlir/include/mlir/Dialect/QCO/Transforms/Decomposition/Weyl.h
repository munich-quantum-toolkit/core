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

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <complex>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::qco::decomposition {

/**
 * @brief Native gate kinds that may appear in a two-qubit synthesis menu.
 */
enum class NativeGateKind : std::uint8_t {
  U,
  X,
  Sx,
  Rz,
  Rx,
  Ry,
  R,
  Cx,
  Cz,
  Rzz,
};

/**
 * @brief Single-qubit emission strategy resolved from a native-gate menu.
 */
enum class SingleQubitMode : std::uint8_t {
  ZSXX,     ///< `RZ` / `SX` / `X` via ZYZ decomposition.
  U3,       ///< Generic `U(theta, phi, lambda)`.
  R,        ///< `R(theta, phi)` chain (`Rx`/`Ry` as `R`).
  AxisPair, ///< Two fixed rotation axes (see @ref AxisPair).
};

/**
 * @brief Rotation-axis pair for @ref SingleQubitMode::AxisPair emitters.
 */
enum class AxisPair : std::uint8_t {
  RxRz,
  RxRy,
  RyRz,
};

/**
 * @brief Entangling basis gate for two-qubit Weyl synthesis.
 */
enum class EntanglerBasis : std::uint8_t {
  None,
  Cx,
  Cz,
};

/// Resolved single-qubit emitter entry in a @ref NativeProfileSpec.
struct SingleQubitEmitterSpec {
  SingleQubitMode mode = SingleQubitMode::U3;
  AxisPair axisPair = AxisPair::RxRz;
  bool supportsDirectRx = false;
};

/**
 * @brief Fully resolved native-gate menu for two-qubit Weyl synthesis.
 *
 * Produced by @ref parseNativeSpec from a comma-separated gate list such as
 * `"x,sx,rz,cx"`.
 */
struct NativeProfileSpec {
  bool allowRzz = false;
  llvm::DenseSet<NativeGateKind> allowedGates;
  llvm::SmallVector<SingleQubitEmitterSpec> singleQubitEmitters;
  llvm::SmallVector<EntanglerBasis> entanglerBases;
};

/**
 * @brief Weyl decomposition of a 2-qubit unitary matrix (4x4).
 *
 * The result consists of four 2x2 single-qubit matrices (`k1l`, `k2l`,
 * `k1r`, `k2r`) and three parameters for a canonical gate (`a`, `b`, `c`).
 * The canonical gate is `RXX(-2 * a) * RYY(-2 * b) * RZZ(-2 * c)`.
 *
 * @note Adapted from TwoQubitWeylDecomposition in the IBM Qiskit framework.
 *       (C) Copyright IBM 2023
 *
 *       This code is licensed under the Apache License, Version 2.0. You may
 *       obtain a copy of this license in the LICENSE.txt file in the root
 *       directory of this source tree or at
 *       https://www.apache.org/licenses/LICENSE-2.0.
 *
 *       Any modifications or derivative works of this code must retain this
 *       copyright notice, and modified files need to carry a notice
 *       indicating that they have been altered from the originals.
 */
class TwoQubitWeylDecomposition {
public:
  /**
   * @brief Create Weyl decomposition.
   *
   * @param unitaryMatrix Matrix of the two-qubit operation/series to be
   *                      decomposed.
   * @param fidelity Tolerance to assume a specialization which is used to
   *                 reduce the number of parameters required by the canonical
   *                 gate and thus potentially decreasing the number of basis
   *                 gates.
   */
  static TwoQubitWeylDecomposition create(const Matrix4x4& unitaryMatrix,
                                          std::optional<double> fidelity);

  ~TwoQubitWeylDecomposition() = default;
  TwoQubitWeylDecomposition(const TwoQubitWeylDecomposition&) = default;
  TwoQubitWeylDecomposition(TwoQubitWeylDecomposition&&) = default;
  TwoQubitWeylDecomposition&
  operator=(const TwoQubitWeylDecomposition&) = default;
  TwoQubitWeylDecomposition& operator=(TwoQubitWeylDecomposition&&) = default;

  /** @brief Matrix of the canonical gate from its parameters `a`, `b`, `c`. */
  [[nodiscard]] Matrix4x4 getCanonicalMatrix() const {
    return getCanonicalMatrix(a_, b_, c_);
  }

  /**
   * @brief First parameter of the canonical gate.
   *
   * @note Multiply by `-2.0` for the `RXX` rotation angle.
   */
  [[nodiscard]] double a() const { return a_; }
  /**
   * @brief Second parameter of the canonical gate.
   *
   * @note Multiply by `-2.0` for the `RYY` rotation angle.
   */
  [[nodiscard]] double b() const { return b_; }
  /**
   * @brief Third parameter of the canonical gate.
   *
   * @note Multiply by `-2.0` for the `RZZ` rotation angle.
   */
  [[nodiscard]] double c() const { return c_; }
  /** @brief Global phase adjustment after applying the decomposition. */
  [[nodiscard]] double globalPhase() const { return globalPhase_; }

  /**
   * @brief Left single-qubit factor after the canonical gate.
   *
   * ```
   * q1 - k2r - C -  k1r  -
   *            A
   * q0 - k2l - N - *k1l* -
   * ```
   */
  [[nodiscard]] const Matrix2x2& k1l() const { return k1l_; }
  /**
   * @brief Left single-qubit factor before the canonical gate.
   *
   * ```
   * q1 -  k2r  - C - k1r -
   *              A
   * q0 - *k2l* - N - k1l -
   * ```
   */
  [[nodiscard]] const Matrix2x2& k2l() const { return k2l_; }
  /**
   * @brief Right single-qubit factor after the canonical gate.
   *
   * ```
   * q1 - k2r - C - *k1r* -
   *            A
   * q0 - k2l - N -  k1l  -
   * ```
   */
  [[nodiscard]] const Matrix2x2& k1r() const { return k1r_; }
  /**
   * @brief Right single-qubit factor before the canonical gate.
   *
   * ```
   * q1 - *k2r* - C - k1r -
   *              A
   * q0 -  k2l  - N - k1l -
   * ```
   */
  [[nodiscard]] const Matrix2x2& k2r() const { return k2r_; }

  /** @brief Canonical gate matrix for parameters `a`, `b`, `c`. */
  [[nodiscard]] static Matrix4x4 getCanonicalMatrix(double a, double b,
                                                    double c);

protected:
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

  enum class MagicBasisTransform : std::uint8_t {
    Into,
    OutOf,
  };

  static constexpr auto DIAGONALIZATION_PRECISION = 1e-13;

  TwoQubitWeylDecomposition() = default;

  [[nodiscard]] static Matrix4x4
  magicBasisTransform(const Matrix4x4& unitary, MagicBasisTransform direction);

  [[nodiscard]] static double closestPartialSwap(double a, double b, double c);

  [[nodiscard]] static std::pair<Matrix4x4, std::array<Complex, 4>>
  diagonalizeComplexSymmetric(const Matrix4x4& m, double precision);

  static std::tuple<Matrix2x2, Matrix2x2, double>
  decomposeTwoQubitProductGate(const Matrix4x4& specialUnitary);

  [[nodiscard]] static std::complex<double>
  getTrace(double a, double b, double c, double ap, double bp, double cp);

  [[nodiscard]] Specialization bestSpecialization() const;

  bool applySpecialization();

private:
  double a_{};
  double b_{};
  double c_{};
  double globalPhase_{};
  Matrix2x2 k1l_;
  Matrix2x2 k2l_;
  Matrix2x2 k1r_;
  Matrix2x2 k2r_;
  Specialization specialization{Specialization::General};
  std::optional<double> requestedFidelity;
};

using TwoQubitLocalUnitaryList = llvm::SmallVector<Matrix2x2, 8>;

/**
 * @brief Result of a two-qubit basis decomposition as single-qubit factors and
 *        entangler uses.
 *
 * Factors are stored in emission order. For `i` in `[0, numBasisUses)` the
 * pair `(singleQubitFactors[2*i], singleQubitFactors[2*i + 1])` is applied to
 * qubits `1` and `0` respectively, followed by one entangler. The final pair
 * `(singleQubitFactors[2*numBasisUses], singleQubitFactors[2*numBasisUses+1])`
 * is applied after the last entangler. The list therefore has length
 * `2 * (numBasisUses + 1)`.
 */
struct TwoQubitNativeDecomposition {
  /// Number of basis-gate (entangler) uses.
  std::uint8_t numBasisUses = 0;
  /// Single-qubit factors in emission order (see struct comment).
  TwoQubitLocalUnitaryList singleQubitFactors;
  /// Residual global phase (radians) not represented by factors/entanglers.
  double globalPhase = 0.0;
};

/**
 * @brief Decomposer initialized with a two-qubit basis gate for canonical-gate
 *        (RXX+RYY+RZZ) synthesis.
 *
 * @note Adapted from TwoQubitBasisDecomposer in the IBM Qiskit framework.
 *       (C) Copyright IBM 2023
 *
 *       This code is licensed under the Apache License, Version 2.0. You may
 *       obtain a copy of this license in the LICENSE.txt file in the root
 *       directory of this source tree or at
 *       https://www.apache.org/licenses/LICENSE-2.0.
 *
 *       Any modifications or derivative works of this code must retain this
 *       copyright notice, and modified files need to carry a notice
 *       indicating that they have been altered from the originals.
 */
class TwoQubitBasisDecomposer {
public:
  /**
   * @brief Create decomposer for the specified entangler matrix.
   *
   * The entangler appears between 0 and 3 times in each decomposition. The 4x4
   * matrix must be in MQT operand order (qubit 0 = MSB).
   */
  [[nodiscard]] static TwoQubitBasisDecomposer
  create(const Matrix4x4& basisMatrix, double basisFidelity);

  /**
   * @brief Perform decomposition using this decomposer's basis gate.
   *
   * @param targetDecomposition Prepared Weyl decomposition of the unitary to
   *                            decompose.
   * @param numBasisGateUses Force a given number of basis gates; when unset,
   *                         the optimal count is selected from the
   *                         Hilbert-Schmidt traces.
   * @return Single-qubit factors and entangler count, or `std::nullopt` when
   *         more than one basis gate would be required but the basis gate is
   *         not super-controlled.
   */
  [[nodiscard]] std::optional<TwoQubitNativeDecomposition>
  twoQubitDecompose(const TwoQubitWeylDecomposition& targetDecomposition,
                    std::optional<std::uint8_t> numBasisGateUses) const;

protected:
  // NOLINTBEGIN(modernize-pass-by-value)
  /** @brief Constructs decomposer instance. */
  TwoQubitBasisDecomposer(
      double basisFidelity, const TwoQubitWeylDecomposition& basisDecomposer,
      bool isSuperControlled, const Matrix2x2& u0l, const Matrix2x2& u0r,
      const Matrix2x2& u1l, const Matrix2x2& u1ra, const Matrix2x2& u1rb,
      const Matrix2x2& u2la, const Matrix2x2& u2lb, const Matrix2x2& u2ra,
      const Matrix2x2& u2rb, const Matrix2x2& u3l, const Matrix2x2& u3r,
      const Matrix2x2& q0l, const Matrix2x2& q0r, const Matrix2x2& q1la,
      const Matrix2x2& q1lb, const Matrix2x2& q1ra, const Matrix2x2& q1rb,
      const Matrix2x2& q2l, const Matrix2x2& q2r)
      : basisFidelity{basisFidelity}, basisDecomposer{basisDecomposer},
        isSuperControlled{isSuperControlled}, u0l{u0l}, u0r{u0r}, u1l{u1l},
        u1ra{u1ra}, u1rb{u1rb}, u2la{u2la}, u2lb{u2lb}, u2ra{u2ra}, u2rb{u2rb},
        u3l{u3l}, u3r{u3r}, q0l{q0l}, q0r{q0r}, q1la{q1la}, q1lb{q1lb},
        q1ra{q1ra}, q1rb{q1rb}, q2l{q2l}, q2r{q2r} {}
  // NOLINTEND(modernize-pass-by-value)

  /**
   * @brief Decomposition with 0 basis gates.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 0 uses of the basis gate.
   * Result :math:`U_r` has trace:
   *
   * .. math::
   *
   *     \Big\vert\text{Tr}(U_r\cdot U_\text{target}^{\dag})\Big\vert =
   *     4\Big\vert (\cos(x)\cos(y)\cos(z)+ j \sin(x)\sin(y)\sin(z)\Big\vert
   *
   * which is optimal for all targets and bases.
   */
  [[nodiscard]] static TwoQubitLocalUnitaryList
  decomp0(const TwoQubitWeylDecomposition& target);

  /**
   * @brief Decomposition with 1 basis gate.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 1 use of the basis gate
   * :math:`\sim U_d(a, b, c)`. Result :math:`U_r` has trace:
   *
   * .. math::
   *
   *     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^{\dag})\Big\vert =
   *     4\Big\vert \cos(x-a)\cos(y-b)\cos(z-c) + j
   *     \sin(x-a)\sin(y-b)\sin(z-c)\Big\vert
   *
   * which is optimal for all targets and bases with ``z==0`` or ``c==0``.
   */
  [[nodiscard]] TwoQubitLocalUnitaryList
  decomp1(const TwoQubitWeylDecomposition& target) const;

  /**
   * @brief Decomposition with 2 basis gates.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 2 uses of the basis gate.
   *
   * For supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b, result
   * :math:`U_r` has trace
   *
   * .. math::
   *
   *     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^\dag) \Big\vert =
   * 4\cos(z)
   *
   * which is the optimal approximation for basis of CNOT-class
   * :math:`\sim U_d(\pi/4, 0, 0)` or DCNOT-class
   * :math:`\sim U_d(\pi/4, \pi/4, 0)` and any target. It may be sub-optimal
   * for :math:`b \neq 0` (i.e. there exists an exact decomposition for any
   * target using :math:`B \sim U_d(\pi/4, \pi/8, 0)`, but it may not be this
   * decomposition). This is an exact decomposition for supercontrolled basis
   * and target :math:`\sim U_d(x, y, 0)`. No guarantees for
   * non-supercontrolled basis.
   */
  [[nodiscard]] TwoQubitLocalUnitaryList
  decomp2Supercontrolled(const TwoQubitWeylDecomposition& target) const;

  /**
   * @brief Decomposition with 3 basis gates.
   *
   * Exact decomposition for supercontrolled basis :math:`\sim U_d(\pi/4, b,
   * 0)`, all b, and any target. No guarantees for non-supercontrolled basis.
   */
  [[nodiscard]] TwoQubitLocalUnitaryList
  decomp3Supercontrolled(const TwoQubitWeylDecomposition& target) const;

  /**
   * @brief Traces for canonical-gate parameter combinations of target and
   *        basis.
   *
   * Used to determine the smallest number of basis gates needed for an
   * equivalent canonical gate.
   */
  [[nodiscard]] std::array<std::complex<double>, 4>
  traces(const TwoQubitWeylDecomposition& target) const;

  [[nodiscard]] static bool relativeEq(double lhs, double rhs, double epsilon,
                                       double maxRelative);

private:
  double basisFidelity;
  TwoQubitWeylDecomposition basisDecomposer;
  bool isSuperControlled;
  Matrix2x2 u0l;
  Matrix2x2 u0r;
  Matrix2x2 u1l;
  Matrix2x2 u1ra;
  Matrix2x2 u1rb;
  Matrix2x2 u2la;
  Matrix2x2 u2lb;
  Matrix2x2 u2ra;
  Matrix2x2 u2rb;
  Matrix2x2 u3l;
  Matrix2x2 u3r;
  Matrix2x2 q0l;
  Matrix2x2 q0r;
  Matrix2x2 q1la;
  Matrix2x2 q1lb;
  Matrix2x2 q1ra;
  Matrix2x2 q1rb;
  Matrix2x2 q2l;
  Matrix2x2 q2r;
};

/**
 * @brief Euler basis used to emit single-qubit factors for @p emitter.
 */
[[nodiscard]] EulerBasis
emitterEulerBasis(const SingleQubitEmitterSpec& emitter);

/**
 * @brief Parses a comma-separated native-gate menu (e.g. `"u,cx,rzz"`).
 *
 * @param nativeGates Comma-separated gate names (case-insensitive).
 * @return The resolved profile, or `std::nullopt` if any token is unknown or
 *         no legal single-qubit emitter can be derived.
 */
[[nodiscard]] std::optional<NativeProfileSpec>
parseNativeSpec(StringRef nativeGates);

/**
 * @brief Synthesizes a composed two-qubit unitary as gates in @p spec.
 *
 * Emits single-qubit factors from the first resolved emitter in @p spec and
 * entanglers from the preferred basis (`CX` over `CZ`). Returns `failure()`
 * when @p spec has no entangler or the Weyl decomposition cannot be realized.
 *
 * @param builder Builder for the emitted operations.
 * @param loc Location for the emitted operations.
 * @param qubit0 First qubit (control wire for entanglers).
 * @param qubit1 Second qubit (target wire for entanglers).
 * @param target Composed unitary to synthesize (MQT operand order).
 * @param spec Resolved native-gate menu.
 * @param outQubit0 Output value for qubit 0 after synthesis.
 * @param outQubit1 Output value for qubit 1 after synthesis.
 * @return `success()` when gates were emitted, `failure()` otherwise.
 */
[[nodiscard]] LogicalResult
synthesizeUnitary2QWeyl(OpBuilder& builder, Location loc, Value qubit0,
                        Value qubit1, const Matrix4x4& target,
                        const NativeProfileSpec& spec, Value& outQubit0,
                        Value& outQubit1);

/**
 * @brief Number of entangling basis gates required to synthesize @p target.
 *
 * @return Entangler count for @p spec, or `std::nullopt` if synthesis fails.
 */
[[nodiscard]] std::optional<std::uint8_t>
twoQubitEntanglerCount(const Matrix4x4& target, const NativeProfileSpec& spec);

/// Common constant `1/sqrt(2)` used by gate-matrix factories.
inline constexpr double FRAC1_SQRT2 =
    0.707106781186547524400844362104849039284835937688474036588L;

/// Axis rotations `exp(-i theta/2 * sigma_{x,y,z})`.
[[nodiscard]] Matrix2x2 rxMatrix(double theta);
[[nodiscard]] Matrix2x2 ryMatrix(double theta);
[[nodiscard]] Matrix2x2 rzMatrix(double theta);
/// `i * sigma_z`; useful when factoring Pauli rotations out of a 2x2.
[[nodiscard]] const Matrix2x2& ipz();
/// `i * sigma_y`.
[[nodiscard]] const Matrix2x2& ipy();
/// `i * sigma_x`.
[[nodiscard]] const Matrix2x2& ipx();
} // namespace mlir::qco::decomposition
