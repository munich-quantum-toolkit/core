/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Decomposition/BasisDecomposer.h"

#include "ir/Definitions.hpp"
#include "mlir/Passes/Decomposition/EulerBasis.h"
#include "mlir/Passes/Decomposition/EulerDecomposition.h"
#include "mlir/Passes/Decomposition/GateSequence.h"
#include "mlir/Passes/Decomposition/Helpers.h"
#include "mlir/Passes/Decomposition/UnitaryMatrices.h"
#include "mlir/Passes/Decomposition/WeylDecomposition.h"

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
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
TwoQubitBasisDecomposer TwoQubitBasisDecomposer::create(const Gate& basisGate,
                                                        double basisFidelity) {
  using namespace std::complex_literals;

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
  const Eigen::Matrix2cd k12RArr{
      {1i * FRAC1_SQRT2, FRAC1_SQRT2},
      {-FRAC1_SQRT2, -1i * FRAC1_SQRT2},
  };
  const Eigen::Matrix2cd k12LArr{
      {{0.5, 0.5}, {0.5, 0.5}},
      {{-0.5, 0.5}, {0.5, -0.5}},
  };

  const auto basisDecomposer = decomposition::TwoQubitWeylDecomposition::create(
      getTwoQubitMatrix(basisGate), basisFidelity);
  const auto isSuperControlled =
      relativeEq(basisDecomposer.a(), qc::PI_4, 1e-13, 1e-09) &&
      relativeEq(basisDecomposer.c(), 0.0, 1e-13, 1e-09);

  // Create some useful matrices U1, U2, U3 are equivalent to the basis,
  // expand as Ui = Ki1.Ubasis.Ki2
  auto b = basisDecomposer.b();
  std::complex<double> temp{0.5, -0.5};
  const Eigen::Matrix2cd k11l{
      {temp * (-1i * std::exp(-1i * b)), temp * std::exp(-1i * b)},
      {temp * (-1i * std::exp(1i * b)), temp * -std::exp(1i * b)}};
  const Eigen::Matrix2cd k11r{
      {FRAC1_SQRT2 * (1i * std::exp(-1i * b)),
       FRAC1_SQRT2 * -std::exp(-1i * b)},
      {FRAC1_SQRT2 * std::exp(1i * b), FRAC1_SQRT2 * (-1i * std::exp(1i * b))}};
  const Eigen::Matrix2cd k32lK21l{
      {FRAC1_SQRT2 * std::complex<double>{1., std::cos(2. * b)},
       FRAC1_SQRT2 * (1i * std::sin(2. * b))},
      {FRAC1_SQRT2 * (1i * std::sin(2. * b)),
       FRAC1_SQRT2 * std::complex<double>{1., -std::cos(2. * b)}}};
  temp = std::complex<double>{0.5, 0.5};
  const Eigen::Matrix2cd k21r{
      {temp * (-1i * std::exp(-2i * b)), temp * std::exp(-2i * b)},
      {temp * (1i * std::exp(2i * b)), temp * std::exp(2i * b)},
  };
  const Eigen::Matrix2cd k22l{
      {FRAC1_SQRT2, -FRAC1_SQRT2},
      {FRAC1_SQRT2, FRAC1_SQRT2},
  };
  const Eigen::Matrix2cd k22r{{0, 1}, {-1, 0}};
  const Eigen::Matrix2cd k31l{
      {FRAC1_SQRT2 * std::exp(-1i * b), FRAC1_SQRT2 * std::exp(-1i * b)},
      {FRAC1_SQRT2 * -std::exp(1i * b), FRAC1_SQRT2 * std::exp(1i * b)},
  };
  const Eigen::Matrix2cd k31r{
      {1i * std::exp(1i * b), 0},
      {0, -1i * std::exp(-1i * b)},
  };
  temp = std::complex<double>{0.5, 0.5};
  const Eigen::Matrix2cd k32r{
      {temp * std::exp(1i * b), temp * -std::exp(-1i * b)},
      {temp * (-1i * std::exp(1i * b)), temp * (-1i * std::exp(-1i * b))},
  };
  auto k1lDagger = basisDecomposer.k1l().transpose().conjugate();
  auto k1rDagger = basisDecomposer.k1r().transpose().conjugate();
  auto k2lDagger = basisDecomposer.k2l().transpose().conjugate();
  auto k2rDagger = basisDecomposer.k2r().transpose().conjugate();
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

std::optional<TwoQubitGateSequence> TwoQubitBasisDecomposer::twoQubitDecompose(
    const decomposition::TwoQubitWeylDecomposition& targetDecomposition,
    const llvm::SmallVector<EulerBasis>& target1qEulerBases,
    std::optional<double> basisFidelity, bool approximate,
    std::optional<std::uint8_t> numBasisGateUses) const {
  if (target1qEulerBases.empty()) {
    llvm::reportFatalUsageError(
        "Unable to perform two-qubit basis decomposition without at least "
        "one euler basis!");
  }

  auto getBasisFidelity = [&]() {
    if (approximate) {
      return basisFidelity.value_or(this->basisFidelity);
    }
    return 1.0;
  };
  double actualBasisFidelity = getBasisFidelity();
  auto traces = this->traces(targetDecomposition);
  auto getDefaultNbasis = [&]() {
    // determine smallest number of basis gates required to fulfill given
    // basis fidelity constraint
    auto bestValue = std::numeric_limits<double>::lowest();
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
    llvm::reportFatalInternalError(
        "Invalid number of basis gates to use in basis decomposition (" +
        llvm::Twine(bestNbasis) + ")!");
    llvm_unreachable("");
  };
  auto decomposition = chooseDecomposition();
  llvm::SmallVector<std::optional<TwoQubitGateSequence>, 8> eulerDecompositions;
  for (auto&& decomp : decomposition) {
    assert(helpers::isUnitaryMatrix(decomp));
    auto eulerDecomp = unitaryToGateSequence(decomp, target1qEulerBases, 0,
                                             true, std::nullopt);
    eulerDecompositions.push_back(eulerDecomp);
  }
  TwoQubitGateSequence gates{
      .gates = {},
      .globalPhase = targetDecomposition.globalPhase(),
  };
  // Worst case length is 5x 1q gates for each 1q decomposition + 1x 2q
  // gate We might overallocate a bit if the euler basis is different but
  // the worst case is just 16 extra elements with just a String and 2
  // smallvecs each. This is only transient though as the circuit
  // sequences aren't long lived and are just used to create a
  // QuantumCircuit or DAGCircuit when we return to Python space.
  constexpr auto twoQubitSequenceDefaultCapacity = 21;
  gates.gates.reserve(twoQubitSequenceDefaultCapacity);
  gates.globalPhase -= bestNbasis * basisDecomposer.globalPhase();
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

llvm::SmallVector<Eigen::Matrix2cd>
TwoQubitBasisDecomposer::decomp0(const TwoQubitWeylDecomposition& target) {
  return {
      target.k1r() * target.k2r(),
      target.k1l() * target.k2l(),
  };
}

llvm::SmallVector<Eigen::Matrix2cd> TwoQubitBasisDecomposer::decomp1(
    const TwoQubitWeylDecomposition& target) const {
  // may not work for z != 0 and c != 0 (not always in Weyl chamber)
  return {
      basisDecomposer.k2r().transpose().conjugate() * target.k2r(),
      basisDecomposer.k2l().transpose().conjugate() * target.k2l(),
      target.k1r() * basisDecomposer.k1r().transpose().conjugate(),
      target.k1l() * basisDecomposer.k1l().transpose().conjugate(),
  };
}

llvm::SmallVector<Eigen::Matrix2cd>
TwoQubitBasisDecomposer::decomp2Supercontrolled(
    const TwoQubitWeylDecomposition& target) const {
  if (!isSuperControlled) {
    llvm::reportFatalInternalError(
        "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
        "- no guarantee for exact decomposition with two basis gates");
  }
  return {
      q2r * target.k2r(),
      q2l * target.k2l(),
      q1ra * rzMatrix(2. * target.b()) * q1rb,
      q1la * rzMatrix(-2. * target.a()) * q1lb,
      target.k1r() * q0r,
      target.k1l() * q0l,
  };
}

llvm::SmallVector<Eigen::Matrix2cd>
TwoQubitBasisDecomposer::decomp3Supercontrolled(
    const TwoQubitWeylDecomposition& target) const {
  if (!isSuperControlled) {
    llvm::reportFatalInternalError(
        "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
        "- no guarantee for exact decomposition with three basis gates");
  }
  return {
      u3r * target.k2r(),
      u3l * target.k2l(),
      u2ra * rzMatrix(2. * target.b()) * u2rb,
      u2la * rzMatrix(-2. * target.a()) * u2lb,
      u1ra * rzMatrix(-2. * target.c()) * u1rb,
      u1l,
      target.k1r() * u0r,
      target.k1l() * u0l,
  };
}

std::array<std::complex<double>, 4>
TwoQubitBasisDecomposer::traces(const TwoQubitWeylDecomposition& target) const {
  return {
      4. * std::complex<double>{std::cos(target.a()) * std::cos(target.b()) *
                                    std::cos(target.c()),
                                std::sin(target.a()) * std::sin(target.b()) *
                                    std::sin(target.c())},
      4. * std::complex<double>{std::cos(qc::PI_4 - target.a()) *
                                    std::cos(basisDecomposer.b() - target.b()) *
                                    std::cos(target.c()),
                                std::sin(qc::PI_4 - target.a()) *
                                    std::sin(basisDecomposer.b() - target.b()) *
                                    std::sin(target.c())},
      std::complex<double>{4. * std::cos(target.c()), 0.},
      std::complex<double>{4., 0.},
  };
}

OneQubitGateSequence TwoQubitBasisDecomposer::unitaryToGateSequence(
    const Eigen::Matrix2cd& unitaryMat,
    const llvm::SmallVector<EulerBasis>& targetBasisList, QubitId /*qubit*/,
    // TODO: add error map here: per qubit a mapping of
    // operation to error value for better calculateError()
    bool simplify, std::optional<double> atol) {
  assert(!targetBasisList.empty());

  auto calculateError = [](const OneQubitGateSequence& sequence) -> double {
    return static_cast<double>(sequence.complexity());
  };

  auto minError = std::numeric_limits<double>::max();
  OneQubitGateSequence bestCircuit;
  for (auto targetBasis : targetBasisList) {
    auto circuit = EulerDecomposition::generateCircuit(targetBasis, unitaryMat,
                                                       simplify, atol);
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

} // namespace mlir::qco::decomposition
