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
#include "GateSequence.h"
#include "Helpers.h"
#include "UnitaryMatrices.h"
#include "WeylDecomposition.h"
#include "ir/Definitions.hpp"

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
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
#include <utility>

namespace mlir::qco::decomposition {

/**
 * Decomposer that must be initialized with a two-qubit basis gate that will
 * be used to generate a circuit equivalent to a canonical gate (RXX+RYY+RZZ).
 */
class TwoQubitBasisDecomposer {
public:
  /**
   * Create decomposer that allows two-qubit decompositions based on the
   * specified basis gate.
   * This basis gate will appear between 0 and 3 times in each decompositions.
   * The order of qubits is relevant and will change the results accordingly.
   * The decomposer cannot handle different basis gates in the same
   * decomposition (different order of the qubits also counts as a different
   * basis gate).
   */
  static TwoQubitBasisDecomposer create(const Gate& basisGate,
                                        fp basisFidelity) {
    auto relativeEq = [](auto&& lhs, auto&& rhs, auto&& epsilon,
                         auto&& maxRelative) {
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
    };
    const matrix2x2 k12RArr{
        {qfp(0., FRAC1_SQRT2), qfp(FRAC1_SQRT2, 0.)},
        {qfp(-FRAC1_SQRT2, 0.), qfp(0., -FRAC1_SQRT2)},
    };
    const matrix2x2 k12LArr{
        {qfp(0.5, 0.5), qfp(0.5, 0.5)},
        {qfp(-0.5, 0.5), qfp(0.5, -0.5)},
    };

    const auto basisDecomposer =
        decomposition::TwoQubitWeylDecomposition::create(
            getTwoQubitMatrix(basisGate), basisFidelity);
    const auto isSuperControlled =
        relativeEq(basisDecomposer.a, qc::PI_4, 1e-13, 1e-09) &&
        relativeEq(basisDecomposer.c, 0.0, 1e-13, 1e-09);

    // Create some useful matrices U1, U2, U3 are equivalent to the basis,
    // expand as Ui = Ki1.Ubasis.Ki2
    auto b = basisDecomposer.b;
    auto temp = qfp(0.5, -0.5);
    const matrix2x2 k11l{
        {temp * (M_IM * std::exp(qfp(0., -b))), temp * std::exp(qfp(0., -b))},
        {temp * (M_IM * std::exp(qfp(0., b))), temp * -std::exp(qfp(0., b))}};
    const matrix2x2 k11r{{FRAC1_SQRT2 * (IM * std::exp(qfp(0., -b))),
                          FRAC1_SQRT2 * -std::exp(qfp(0., -b))},
                         {FRAC1_SQRT2 * std::exp(qfp(0., b)),
                          FRAC1_SQRT2 * (M_IM * std::exp(qfp(0., b)))}};
    const matrix2x2 k32lK21l{{FRAC1_SQRT2 * qfp(1., std::cos(2. * b)),
                              FRAC1_SQRT2 * (IM * std::sin(2. * b))},
                             {FRAC1_SQRT2 * (IM * std::sin(2. * b)),
                              FRAC1_SQRT2 * qfp(1., -std::cos(2. * b))}};
    temp = qfp(0.5, 0.5);
    const matrix2x2 k21r{
        {temp * (M_IM * std::exp(qfp(0., -2. * b))),
         temp * std::exp(qfp(0., -2. * b))},
        {temp * (IM * std::exp(qfp(0., 2. * b))),
         temp * std::exp(qfp(0., 2. * b))},
    };
    const matrix2x2 k22l{
        {qfp(FRAC1_SQRT2, 0.), qfp(-FRAC1_SQRT2, 0.)},
        {qfp(FRAC1_SQRT2, 0.), qfp(FRAC1_SQRT2, 0.)},
    };
    const matrix2x2 k22r{{C_ZERO, C_ONE}, {C_M_ONE, C_ZERO}};
    const matrix2x2 k31l{
        {FRAC1_SQRT2 * std::exp(qfp(0., -b)),
         FRAC1_SQRT2 * std::exp(qfp(0., -b))},
        {FRAC1_SQRT2 * -std::exp(qfp(0., b)),
         FRAC1_SQRT2 * std::exp(qfp(0., b))},
    };
    const matrix2x2 k31r{
        {IM * std::exp(qfp(0., b)), C_ZERO},
        {C_ZERO, M_IM * std::exp(qfp(0., -b))},
    };
    temp = qfp(0.5, 0.5);
    const matrix2x2 k32r{
        {temp * std::exp(qfp(0., b)), temp * -std::exp(qfp(0., -b))},
        {temp * (M_IM * std::exp(qfp(0., b))),
         temp * (M_IM * std::exp(qfp(0., -b)))},
    };
    auto k1lDagger = basisDecomposer.k1l.transpose().conjugate();
    auto k1rDagger = basisDecomposer.k1r.transpose().conjugate();
    auto k2lDagger = basisDecomposer.k2l.transpose().conjugate();
    auto k2rDagger = basisDecomposer.k2r.transpose().conjugate();
    // Pre-build the fixed parts of the matrices used in 3-part
    // decomposition
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
    // Pre-build the fixed parts of the matrices used in the 2-part
    // decomposition
    auto q0l = k12LArr.transpose().conjugate() * k1lDagger;
    auto q0r = k12RArr.transpose().conjugate() * IPZ * k1rDagger;
    auto q1la = k2lDagger * k11l.transpose().conjugate();
    auto q1lb = k11l * k1lDagger;
    auto q1ra = k2rDagger * IPZ * k11r.transpose().conjugate();
    auto q1rb = k11r * k1rDagger;
    auto q2l = k2lDagger * k12LArr;
    auto q2r = k2rDagger * k12RArr;

    return TwoQubitBasisDecomposer{
        basisGate,
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

  /**
   * Perform decomposition using the basis gate of this decomposer.
   *
   * @param targetDecomposition Prepared Weyl decomposition of unitary matrix
   *                            to be decomposed.
   * @param target1qEulerBases List of euler bases that should be tried out to
   *                           find the best one for each euler decomposition.
   *                           All bases will be mixed to get the best overall
   *                           result.
   * @param basisFidelity Fidelity for lowering the number of basis gates
   *                      required
   * @param approximate If true, use basisFidelity or, if std::nullopt, use
   *                    basisFidelity of this decomposer. If false, fidelity
   *                    of 1.0 will be assumed.
   * @param numBasisGateUses Force use of given number of basis gates.
   */
  [[nodiscard]] std::optional<TwoQubitGateSequence> twoQubitDecompose(
      const decomposition::TwoQubitWeylDecomposition& targetDecomposition,
      const llvm::SmallVector<EulerBasis>& target1qEulerBases,
      std::optional<fp> basisFidelity, bool approximate,
      std::optional<std::uint8_t> numBasisGateUses) const {
    auto getBasisFidelity = [&]() {
      if (approximate) {
        return basisFidelity.value_or(this->basisFidelity);
      }
      return static_cast<fp>(1.0);
    };
    fp actualBasisFidelity = getBasisFidelity();
    auto traces = this->traces(targetDecomposition);
    auto getDefaultNbasis = [&]() {
      // determine smallest number of basis gates required to fulfill given
      // basis fidelity constraint
      auto bestValue = std::numeric_limits<fp>::min();
      auto bestIndex = -1;
      for (int i = 0; std::cmp_less(i, traces.size()); ++i) {
        // lower basis fidelity means it becomes easier to use fewer basis gates
        // through a rougher approximation
        auto value = helpers::traceToFidelity(traces[i]) *
                     std::pow(actualBasisFidelity, i);
        if (value > bestValue) {
          bestIndex = i;
          bestValue = value;
        }
      }
      // index in traces equals number of basis gates
      return bestIndex;
    };
    // number of basis gates that need to be used in the decomposition
    auto bestNbasis = numBasisGateUses.value_or(getDefaultNbasis());
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
      throw std::logic_error{"Invalid basis to use"};
    };
    auto decomposition = chooseDecomposition();
    llvm::SmallVector<std::optional<TwoQubitGateSequence>, 8>
        eulerDecompositions;
    for (auto&& decomp : decomposition) {
      assert(helpers::isUnitaryMatrix(decomp));
      auto eulerDecomp = unitaryToGateSequenceInner(decomp, target1qEulerBases,
                                                    0, true, std::nullopt);
      eulerDecompositions.push_back(eulerDecomp);
    }
    TwoQubitGateSequence gates{
        .gates = {},
        .globalPhase = targetDecomposition.globalPhase,
    };
    // Worst case length is 5x 1q gates for each 1q decomposition + 1x 2q
    // gate We might overallocate a bit if the euler basis is different but
    // the worst case is just 16 extra elements with just a String and 2
    // smallvecs each. This is only transient though as the circuit
    // sequences aren't long lived and are just used to create a
    // QuantumCircuit or DAGCircuit when we return to Python space.
    constexpr auto twoQubitSequenceDefaultCapacity = 21;
    gates.gates.reserve(twoQubitSequenceDefaultCapacity);
    gates.globalPhase -= bestNbasis * basisDecomposer.globalPhase;
    if (bestNbasis == 2) {
      gates.globalPhase += qc::PI;
    }

    auto addEulerDecomposition = [&](std::size_t index, QubitId qubitId) {
      if (auto&& eulerDecomp = eulerDecompositions[index]) {
        for (auto&& gate : eulerDecomp->gates) {
          gates.gates.push_back({.type = gate.type,
                                 .parameter = gate.parameter,
                                 .qubitId = {qubitId}});
        }
        gates.globalPhase += eulerDecomp->globalPhase;
      }
    };

    for (std::size_t i = 0; i < bestNbasis; ++i) {
      // add single-qubit decompositions before basis gate
      addEulerDecomposition(2 * i, 0);
      addEulerDecomposition((2 * i) + 1, 1);

      // add basis gate
      gates.gates.push_back(basisGate);
    }

    // add single-qubit decompositions after basis gate
    addEulerDecomposition(2UL * bestNbasis, 0);
    addEulerDecomposition((2UL * bestNbasis) + 1, 1);

    // large global phases can be generated by the decomposition, thus limit
    // it to [0, +2*pi); TODO: can be removed, should be done by something
    // like constant folding
    gates.globalPhase = helpers::remEuclid(gates.globalPhase, qc::TAU);

    return gates;
  }

protected:
  // NOLINTBEGIN(modernize-pass-by-value)
  /**
   * Constructs decomposer instance.
   */
  TwoQubitBasisDecomposer(
      Gate basisGate, fp basisFidelity,
      const decomposition::TwoQubitWeylDecomposition& basisDecomposer,
      bool isSuperControlled, const matrix2x2& u0l, const matrix2x2& u0r,
      const matrix2x2& u1l, const matrix2x2& u1ra, const matrix2x2& u1rb,
      const matrix2x2& u2la, const matrix2x2& u2lb, const matrix2x2& u2ra,
      const matrix2x2& u2rb, const matrix2x2& u3l, const matrix2x2& u3r,
      const matrix2x2& q0l, const matrix2x2& q0r, const matrix2x2& q1la,
      const matrix2x2& q1lb, const matrix2x2& q1ra, const matrix2x2& q1rb,
      const matrix2x2& q2l, const matrix2x2& q2r)
      : basisGate{std::move(basisGate)}, basisFidelity{basisFidelity},
        basisDecomposer{basisDecomposer}, isSuperControlled{isSuperControlled},
        u0l{u0l}, u0r{u0r}, u1l{u1l}, u1ra{u1ra}, u1rb{u1rb}, u2la{u2la},
        u2lb{u2lb}, u2ra{u2ra}, u2rb{u2rb}, u3l{u3l}, u3r{u3r}, q0l{q0l},
        q0r{q0r}, q1la{q1la}, q1lb{q1lb}, q1ra{q1ra}, q1rb{q1rb}, q2l{q2l},
        q2r{q2r} {}
  // NOLINTEND(modernize-pass-by-value)

  /**
   * Calculate decompositions when no basis gate is required.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 0 uses of the
   * basis gate. Result :math:`U_r` has trace:
   *
   * .. math::
   *
   *     \Big\vert\text{Tr}(U_r\cdot U_\text{target}^{\dag})\Big\vert =
   *     4\Big\vert (\cos(x)\cos(y)\cos(z)+ j \sin(x)\sin(y)\sin(z)\Big\vert
   *
   * which is optimal for all targets and bases
   */
  [[nodiscard]] static llvm::SmallVector<matrix2x2>
  decomp0(const decomposition::TwoQubitWeylDecomposition& target) {
    return {
        target.k1r * target.k2r,
        target.k1l * target.k2l,
    };
  }

  /**
   * Calculate decompositions when one basis gate is required.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 1 use of the
   * basis gate math:`\sim U_d(a, b, c)`. Result :math:`U_r` has trace:
   *
   * .. math::
   *
   *     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^{\dag})\Big\vert =
   *     4\Big\vert \cos(x-a)\cos(y-b)\cos(z-c) + j
   *     \sin(x-a)\sin(y-b)\sin(z-c)\Big\vert
   *
   * which is optimal for all targets and bases with ``z==0`` or ``c==0``.
   */
  [[nodiscard]] llvm::SmallVector<matrix2x2>
  decomp1(const decomposition::TwoQubitWeylDecomposition& target) const {
    // may not work for z != 0 and c != 0 (not always in Weyl chamber)
    return {
        basisDecomposer.k2r.transpose().conjugate() * target.k2r,
        basisDecomposer.k2l.transpose().conjugate() * target.k2l,
        target.k1r * basisDecomposer.k1r.transpose().conjugate(),
        target.k1l * basisDecomposer.k1l.transpose().conjugate(),
    };
  }

  /**
   * Calculate decompositions when two basis gates are required.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 2 uses of the
   * basis gate.
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
  [[nodiscard]] llvm::SmallVector<matrix2x2> decomp2Supercontrolled(
      const decomposition::TwoQubitWeylDecomposition& target) const {
    if (!isSuperControlled) {
      llvm::reportFatalInternalError(
          "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
          "- no guarantee for exact decomposition with two basis gates");
    }
    return {
        q2r * target.k2r,
        q2l * target.k2l,
        q1ra * rzMatrix(2. * target.b) * q1rb,
        q1la * rzMatrix(-2. * target.a) * q1lb,
        target.k1r * q0r,
        target.k1l * q0l,
    };
  }

  /**
   * Calculate decompositions when three basis gates are required.
   *
   * Decompose target with 3 uses of the basis.
   *
   * This is an exact decomposition for supercontrolled basis
   * :math:`\sim U_d(\pi/4, b, 0)`, all b, and any target. No guarantees for
   * non-supercontrolled basis.
   */
  [[nodiscard]] llvm::SmallVector<matrix2x2> decomp3Supercontrolled(
      const decomposition::TwoQubitWeylDecomposition& target) const {
    if (!isSuperControlled) {
      llvm::reportFatalInternalError(
          "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
          "- no guarantee for exact decomposition with three basis gates");
    }
    return {
        u3r * target.k2r,
        u3l * target.k2l,
        u2ra * rzMatrix(2. * target.b) * u2rb,
        u2la * rzMatrix(-2. * target.a) * u2lb,
        u1ra * rzMatrix(-2. * target.c) * u1rb,
        u1l,
        target.k1r * u0r,
        target.k1l * u0l,
    };
  }

  /**
   * Calculate traces for a combination of the parameters of the canonical
   * gates of the target and basis decompositions.
   * This can be used to determine the smallest number of basis gates that are
   * necessary to construct an equivalent to the canonical gate.
   */
  [[nodiscard]] std::array<qfp, 4>
  traces(const decomposition::TwoQubitWeylDecomposition& target) const {
    return {
        static_cast<fp>(4.) *
            qfp(std::cos(target.a) * std::cos(target.b) * std::cos(target.c),
                std::sin(target.a) * std::sin(target.b) * std::sin(target.c)),
        static_cast<fp>(4.) *
            qfp(std::cos(qc::PI_4 - target.a) *
                    std::cos(basisDecomposer.b - target.b) * std::cos(target.c),
                std::sin(qc::PI_4 - target.a) *
                    std::sin(basisDecomposer.b - target.b) *
                    std::sin(target.c)),
        qfp(4. * std::cos(target.c), 0.),
        qfp(4., 0.),
    };
  }

  /**
   * Decompose a single-qubit unitary matrix into a single-qubit gate
   * sequence. Multiple euler bases may be specified and the one with the
   * least complexity will be chosen.
   */
  [[nodiscard]] static OneQubitGateSequence unitaryToGateSequenceInner(
      const matrix2x2& unitaryMat,
      const llvm::SmallVector<EulerBasis>& targetBasisList, QubitId /*qubit*/,
      // TODO: add error map here: per qubit a mapping of operation to error
      // value for better calculateError()
      bool simplify, std::optional<fp> atol) {
    auto calculateError = [](const OneQubitGateSequence& sequence) -> fp {
      return static_cast<fp>(sequence.complexity());
    };

    auto minError = std::numeric_limits<fp>::max();
    OneQubitGateSequence bestCircuit;
    for (auto targetBasis : targetBasisList) {
      auto circuit = EulerDecomposition::generateCircuit(
          targetBasis, unitaryMat, simplify, atol);
      // check top-left 2x2 matrix of generated circuit since the circuit
      // operates only on one qubit
      assert((circuit.getUnitaryMatrix().block<2, 2>(0, 0).isApprox(
          unitaryMat, SANITY_CHECK_PRECISION)));
      auto error = calculateError(circuit);
      if (error < minError) {
        bestCircuit = circuit;
        minError = error;
      }
    }
    return bestCircuit;
  }

private:
  // basis gate of this decomposer instance
  Gate basisGate{};
  // fidelity with which the basis gate decomposition has been calculated
  fp basisFidelity;
  // cached decomposition for basis gate
  decomposition::TwoQubitWeylDecomposition basisDecomposer;
  // true if basis gate is super-controlled
  bool isSuperControlled;

  // pre-built components for decomposition with 3 basis gates
  matrix2x2 u0l;
  matrix2x2 u0r;
  matrix2x2 u1l;
  matrix2x2 u1ra;
  matrix2x2 u1rb;
  matrix2x2 u2la;
  matrix2x2 u2lb;
  matrix2x2 u2ra;
  matrix2x2 u2rb;
  matrix2x2 u3l;
  matrix2x2 u3r;

  // pre-built components for decomposition with 2 basis gates
  matrix2x2 q0l;
  matrix2x2 q0r;
  matrix2x2 q1la;
  matrix2x2 q1lb;
  matrix2x2 q1ra;
  matrix2x2 q1rb;
  matrix2x2 q2l;
  matrix2x2 q2r;
};

} // namespace mlir::qco::decomposition
