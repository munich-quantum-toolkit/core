/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Operations.hpp"

#include "dd/Complex.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Package.hpp"
#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dd {
namespace {

GateMatrix uMat(const fp lambda, const fp phi, const fp theta) {
  return GateMatrix{{{std::cos(theta / 2.), 0.},
                     {-std::cos(lambda) * std::sin(theta / 2.),
                      -std::sin(lambda) * std::sin(theta / 2.)},
                     {std::cos(phi) * std::sin(theta / 2.),
                      std::sin(phi) * std::sin(theta / 2.)},
                     {std::cos(lambda + phi) * std::cos(theta / 2.),
                      std::sin(lambda + phi) * std::cos(theta / 2.)}}};
}

GateMatrix u2Mat(const fp lambda, const fp phi) {
  return GateMatrix{
      SQRT2_2,
      {-std::cos(lambda) * SQRT2_2, -std::sin(lambda) * SQRT2_2},
      {std::cos(phi) * SQRT2_2, std::sin(phi) * SQRT2_2},
      {std::cos(lambda + phi) * SQRT2_2, std::sin(lambda + phi) * SQRT2_2}};
}

GateMatrix pMat(const fp lambda) {
  return GateMatrix{1, 0, 0, {std::cos(lambda), std::sin(lambda)}};
}

GateMatrix rxMat(const fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), 0.},
                     {0., -std::sin(lambda / 2.)},
                     {0., -std::sin(lambda / 2.)},
                     {std::cos(lambda / 2.), 0.}}};
}

GateMatrix ryMat(const fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), 0.},
                     {-std::sin(lambda / 2.), 0.},
                     {std::sin(lambda / 2.), 0.},
                     {std::cos(lambda / 2.), 0.}}};
}

GateMatrix rzMat(const fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), -std::sin(lambda / 2.)},
                     0,
                     0,
                     {std::cos(lambda / 2.), std::sin(lambda / 2.)}}};
}

GateMatrix rMat(const fp theta, const fp phi) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);
  const auto sinPhi = std::sin(phi);
  const auto cosPhi = std::cos(phi);
  const std::complex<fp> diag = {cosTheta, 0.};
  const std::complex<fp> m01 = {-sinTheta * sinPhi, -sinTheta * cosPhi};
  const std::complex<fp> m10 = {sinTheta * sinPhi, -sinTheta * cosPhi};

  return GateMatrix{diag, m01, m10, diag};
}

TwoQubitGateMatrix rxxMat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{
      {{cosTheta, 0, 0, {0., -sinTheta}},
       {0, cosTheta, {0., -sinTheta}, 0},
       {0, {0., -sinTheta}, cosTheta, 0},
       {std::complex<fp>{0., -sinTheta}, 0, 0, cosTheta}}};
}

TwoQubitGateMatrix ryyMat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{{{cosTheta, 0, 0, {0., sinTheta}},
                             {0, cosTheta, {0., -sinTheta}, 0},
                             {0, {0., -sinTheta}, cosTheta, 0},
                             {std::complex<fp>{0., sinTheta}, 0, 0, cosTheta}}};
}

TwoQubitGateMatrix rzzMat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{{{std::complex<fp>{cosTheta, -sinTheta}, 0, 0, 0},
                             {0, {cosTheta, sinTheta}, 0, 0},
                             {0, 0, {cosTheta, sinTheta}, 0},
                             {0, 0, 0, {cosTheta, -sinTheta}}}};
}

TwoQubitGateMatrix rzxMat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{{{cosTheta, {0., -sinTheta}, 0, 0},
                             {std::complex<fp>{0., -sinTheta}, cosTheta, 0, 0},
                             {0, 0, cosTheta, {0., sinTheta}},
                             {0, 0, {0., sinTheta}, cosTheta}}};
}

TwoQubitGateMatrix xxMinusYYMat(const fp theta, const fp beta = 0.) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);
  const auto cosBeta = std::cos(beta);
  const auto sinBeta = std::sin(beta);

  return TwoQubitGateMatrix{
      {{cosTheta, 0, 0, {-sinBeta * sinTheta, -cosBeta * sinTheta}},
       {0, 1, 0, 0},
       {0, 0, 1, 0},
       {std::complex<fp>{sinBeta * sinTheta, -cosBeta * sinTheta}, 0, 0,
        cosTheta}}};
}

TwoQubitGateMatrix xxPlusYYMat(const fp theta, const fp beta = 0.) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);
  const auto cosBeta = std::cos(beta);
  const auto sinBeta = std::sin(beta);

  return TwoQubitGateMatrix{
      {{1, 0, 0, 0},
       {0, cosTheta, {sinBeta * sinTheta, -cosBeta * sinTheta}, 0},
       {0, {-sinBeta * sinTheta, -cosBeta * sinTheta}, cosTheta, 0},
       {0, 0, 0, 1}}};
}

GateMatrix singleQubitGateMatrix(const qc::OpType type,
                                 const std::vector<fp>& params) {
  switch (type) {
  case qc::I:
    return {1, 0, 0, 1};
  case qc::H:
    return {SQRT2_2, SQRT2_2, SQRT2_2, -SQRT2_2};
  case qc::X:
    return {0, 1, 1, 0};
  case qc::Y:
    return {0, {0, -1}, {0, 1}, 0};
  case qc::Z:
    return {1, 0, 0, -1};
  case qc::S:
    return {1, 0, 0, {0, 1}};
  case qc::Sdg:
    return {1, 0, 0, {0, -1}};
  case qc::T:
    return {1, 0, 0, {SQRT2_2, SQRT2_2}};
  case qc::Tdg:
    return {1, 0, 0, {SQRT2_2, -SQRT2_2}};
  case qc::SX:
    return {std::complex<fp>{0.5, 0.5}, std::complex<fp>{0.5, -0.5},
            std::complex<fp>{0.5, -0.5}, std::complex<fp>{0.5, 0.5}};
  case qc::SXdg:
    return {std::complex<fp>{0.5, -0.5}, std::complex<fp>{0.5, 0.5},
            std::complex<fp>{0.5, 0.5}, std::complex<fp>{0.5, -0.5}};
  case qc::V:
    return {SQRT2_2, {0., -SQRT2_2}, {0., -SQRT2_2}, SQRT2_2};
  case qc::Vdg:
    return {SQRT2_2, {0., SQRT2_2}, {0., SQRT2_2}, SQRT2_2};
  case qc::U:
    return uMat(params.at(2), params.at(1), params.at(0));
  case qc::U2:
    return u2Mat(params.at(1), params.at(0));
  case qc::P:
    return pMat(params.at(0));
  case qc::RX:
    return rxMat(params.at(0));
  case qc::RY:
    return ryMat(params.at(0));
  case qc::RZ:
    return rzMat(params.at(0));
  case qc::R:
    return rMat(params.at(0), params.at(1));
  default:
    throw std::invalid_argument("Invalid single-qubit gate type");
  }
}

TwoQubitGateMatrix twoQubitGateMatrix(const qc::OpType type,
                                      const std::vector<fp>& params) {
  switch (type) {
  case qc::SWAP:
    return {{{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}};
  case qc::iSWAP:
    return {{{1, 0, 0, 0}, {0, 0, {0, 1}, 0}, {0, {0, 1}, 0, 0}, {0, 0, 0, 1}}};
  case qc::iSWAPdg:
    return {
        {{1, 0, 0, 0}, {0, 0, {0, -1}, 0}, {0, {0, -1}, 0, 0}, {0, 0, 0, 1}}};
  case qc::ECR:
    return {{{0, 0, SQRT2_2, {0, SQRT2_2}},
             {0, 0, {0, SQRT2_2}, SQRT2_2},
             {SQRT2_2, {0, -SQRT2_2}, 0, 0},
             {std::complex<fp>{0., -SQRT2_2}, SQRT2_2, 0, 0}}};
  case qc::DCX:
    return {{{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 1, 0, 0}}};
  case qc::Peres:
    return {{{0, 0, 0, 1}, {0, 0, 1, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}}};
  case qc::Peresdg:
    return {{{0, 0, 1, 0}, {0, 0, 0, 1}, {0, 1, 0, 0}, {1, 0, 0, 0}}};
  case qc::RXX:
    return rxxMat(params.at(0));
  case qc::RYY:
    return ryyMat(params.at(0));
  case qc::RZZ:
    return rzzMat(params.at(0));
  case qc::RZX:
    return rzxMat(params.at(0));
  case qc::XXminusYY:
    return xxMinusYYMat(params.at(0), params.at(1));
  case qc::XXplusYY:
    return xxPlusYYMat(params.at(0), params.at(1));
  default:
    throw std::invalid_argument("Invalid two-qubit gate type");
  }
}

ThreeQubitGateMatrix threeQubitGateMatrix(const qc::OpType type) {
  switch (type) {
  case qc::RCCX: {
    ThreeQubitGateMatrix matrix{};
    for (size_t i = 0; i < THREE_QUBIT_GATE_DIM; ++i) {
      matrix[i][i] = 1.;
    }
    matrix[5][5] = -1.;
    matrix[6][6] = 0.;
    matrix[7][7] = 0.;
    matrix[6][7] = {0., -1.};
    matrix[7][6] = {0., 1.};
    return matrix;
  }
  default:
    throw std::invalid_argument("Invalid three-qubit gate type");
  }
}

MatrixDD makeStandardOperationDD(Package& dd, const qc::OpType type,
                                 const std::vector<fp>& params,
                                 const qc::Controls& controls,
                                 const std::vector<qc::Qubit>& targets) {
  if (qc::isSingleQubitGate(type)) {
    if (targets.size() != 1) {
      throw std::invalid_argument(
          "Expected exactly one target qubit for single-qubit gate");
    }
    return dd.makeGateDD(singleQubitGateMatrix(type, params), controls,
                         targets[0U]);
  }
  if (qc::isTwoQubitGate(type)) {
    if (targets.size() != 2) {
      throw std::invalid_argument(
          "Expected two target qubits for two-qubit gate");
    }
    return dd.makeTwoQubitGateDD(twoQubitGateMatrix(type, params), controls,
                                 targets[0U], targets[1U]);
  }
  if (qc::isThreeQubitGate(type)) {
    if (targets.size() != 3) {
      throw std::invalid_argument(
          "Expected three target qubits for three-qubit gate");
    }
    return dd.makeThreeQubitGateDD(threeQubitGateMatrix(type), controls,
                                   targets[0U], targets[1U], targets[2U]);
  }
  throw std::runtime_error("Unexpected operation type");
}

MatrixDD makeStandardOperationDD(const qc::StandardOperation& op, Package& dd,
                                 const qc::Controls& controls,
                                 const std::vector<qc::Qubit>& targets,
                                 const bool inverse) {
  auto type = op.getType();

  if (!inverse) {
    return makeStandardOperationDD(dd, type, op.getParameter(), controls,
                                   targets);
  }

  // invert the operation
  std::vector<fp> params = op.getParameter();
  std::vector<qc::Qubit> targetQubits = targets;

  switch (type) {
  // operations that are self-inverse do not need any changes
  case qc::I:
  case qc::H:
  case qc::X:
  case qc::Y:
  case qc::Z:
  case qc::SWAP:
  case qc::ECR:
  case qc::RCCX:
    break;
  // operations that have an inverse gate with the same parameters
  case qc::iSWAP:
  case qc::iSWAPdg:
  case qc::Peres:
  case qc::Peresdg:
  case qc::S:
  case qc::Sdg:
  case qc::T:
  case qc::Tdg:
  case qc::V:
  case qc::Vdg:
  case qc::SX:
  case qc::SXdg:
    type = static_cast<qc::OpType>(+type ^ qc::OpTypeInv);
    break;
  // operations that can be inversed by negating the first parameter
  case qc::RXX:
  case qc::RYY:
  case qc::RZZ:
  case qc::RZX:
  case qc::RX:
  case qc::RY:
  case qc::RZ:
  case qc::R:
  case qc::P:
  case qc::XXminusYY:
  case qc::XXplusYY:
    params[0U] = -params[0U];
    break;
  // other special cases
  case qc::DCX:
    if (targetQubits.size() != 2) {
      throw std::runtime_error("Invalid target qubits for DCX");
    }
    // DCX is not self-inverse, but the inverse is just swapping the targets
    std::swap(targetQubits[0], targetQubits[1]);
    break;
  // invert all parameters
  case qc::U:
    // swap [a, b, c] to [a, c, b]
    std::swap(params[1U], params[2U]);
    for (auto& param : params) {
      param = -param;
    }
    break;
  case qc::U2:
    std::swap(params[0U], params[1U]);
    params[0U] = -params[0U] - PI;
    params[1U] = -params[1U] + PI;
    break;

  default:
    std::ostringstream oss{};
    oss << "negation for gate " << op.getName() << " not available!";
    throw std::runtime_error(oss.str());
  }
  return makeStandardOperationDD(dd, type, params, controls, targetQubits);
}

} // namespace

MatrixDD getDD(const qc::Operation& op, Package& dd,
               const qc::Permutation& permutation, const bool inverse) {
  const auto type = op.getType();

  if (type == qc::Barrier) {
    return Package::makeIdent();
  }

  if (type == qc::GPhase) {
    auto phase = op.getParameter()[0U];
    if (inverse) {
      phase = -phase;
    }
    auto id = Package::makeIdent();
    id.w = dd.cn.lookup(std::cos(phase), std::sin(phase));
    return id;
  }

  if (op.isStandardOperation()) {
    const auto& standardOp = dynamic_cast<const qc::StandardOperation&>(op);
    const auto& targets = permutation.apply(standardOp.getTargets());
    const auto& controls = permutation.apply(standardOp.getControls());

    return makeStandardOperationDD(standardOp, dd, controls, targets, inverse);
  }

  if (op.isCompoundOperation()) {
    const auto& compoundOp = dynamic_cast<const qc::CompoundOperation&>(op);
    auto e = Package::makeIdent();
    if (inverse) {
      for (const auto& operation : compoundOp) {
        e = dd.multiply(e, getInverseDD(*operation, dd, permutation));
      }
    } else {
      for (const auto& operation : compoundOp) {
        e = dd.multiply(getDD(*operation, dd, permutation), e);
      }
    }
    return e;
  }

  assert(op.isNonUnitaryOperation());
  throw std::invalid_argument("DD for non-unitary operation not available!");
}

MatrixDD getInverseDD(const qc::Operation& op, Package& dd,
                      const qc::Permutation& permutation) {
  return getDD(op, dd, permutation, true);
}

VectorDD applyUnitaryOperation(const qc::Operation& op, const VectorDD& in,
                               Package& dd,
                               const qc::Permutation& permutation) {
  return dd.applyOperation(getDD(op, dd, permutation), in);
}

MatrixDD applyUnitaryOperation(const qc::Operation& op, const MatrixDD& in,
                               Package& dd, const qc::Permutation& permutation,
                               const bool applyFromLeft) {
  return dd.applyOperation(getDD(op, dd, permutation), in, applyFromLeft);
}

VectorDD applyMeasurement(const qc::NonUnitaryOperation& op, VectorDD in,
                          Package& dd, std::mt19937_64& rng,
                          std::vector<bool>& measurements,
                          const qc::Permutation& permutation) {
  assert(op.getType() == qc::Measure);
  const auto& qubits = permutation.apply(op.getTargets());
  const auto& bits = op.getClassics();
  for (size_t j = 0U; j < qubits.size(); ++j) {
    measurements.at(bits.at(j)) =
        dd.measureOneCollapsing(in, static_cast<dd::Qubit>(qubits.at(j)),
                                rng) == '1';
  }
  return in;
}

VectorDD applyReset(const qc::NonUnitaryOperation& op, VectorDD in, Package& dd,
                    std::mt19937_64& rng, const qc::Permutation& permutation) {
  assert(op.getType() == qc::Reset);
  const auto& qubits = permutation.apply(op.getTargets());
  for (const auto& qubit : qubits) {
    const auto bit =
        dd.measureOneCollapsing(in, static_cast<dd::Qubit>(qubit), rng);
    // apply an X operation whenever the measured result is one
    if (bit == '1') {
      const auto x = qc::StandardOperation(qubit, qc::X);
      in = applyUnitaryOperation(x, in, dd);
    }
  }
  return in;
}

VectorDD applyIfElseOperation(const qc::IfElseOperation& op, const VectorDD& in,
                              Package& dd,
                              const std::vector<bool>& measurements,
                              const qc::Permutation& permutation) {
  const auto& comparisonKind = op.getComparisonKind();

  // determine the actual value from measurements
  std::uint64_t expectedValue = 0U;
  auto actualValue = 0ULL;
  if (const auto& controlRegister = op.getControlRegister();
      controlRegister.has_value()) {
    assert(!op.getControlBit().has_value());
    expectedValue = op.getExpectedValueRegister();
    const auto regStart = controlRegister->getStartIndex();
    const auto regSize = controlRegister->getSize();
    for (std::size_t j = 0; j < regSize; ++j) {
      if (measurements[regStart + j]) {
        actualValue |= 1ULL << j;
      }
    }
  }
  if (const auto& controlBit = op.getControlBit(); controlBit.has_value()) {
    assert(!op.getControlRegister().has_value());
    expectedValue = op.getExpectedValueBit() ? 1U : 0U;
    actualValue = measurements[*controlBit] ? 1U : 0U;
  }

  // check if the actual value matches the expected value according to the
  // comparison kind
  const auto control = [actualValue, expectedValue, comparisonKind]() {
    switch (comparisonKind) {
    case qc::ComparisonKind::Eq:
      return actualValue == expectedValue;
    case qc::ComparisonKind::Neq:
      return actualValue != expectedValue;
    case qc::ComparisonKind::Lt:
      return actualValue < expectedValue;
    case qc::ComparisonKind::Leq:
      return actualValue <= expectedValue;
    case qc::ComparisonKind::Gt:
      return actualValue > expectedValue;
    case qc::ComparisonKind::Geq:
      return actualValue >= expectedValue;
    }
    qc::unreachable();
  }();

  if (!control) {
    auto* elseOp = op.getElseOp();
    if (elseOp == nullptr) {
      return in;
    }
    return applyUnitaryOperation(*elseOp, in, dd, permutation);
  }

  auto* thenOp = op.getThenOp();
  if (thenOp == nullptr) {
    return in;
  }
  return applyUnitaryOperation(*thenOp, in, dd, permutation);
}

bool isExecutableVirtually(const qc::Operation& op) noexcept {
  switch (op.getType()) {
  case qc::I:
  case qc::Barrier:
    return true;
  case qc::SWAP:
    return !op.isControlled();
  default:
    return false;
  }
}

void applyVirtualOperation(const qc::Operation& op,
                           qc::Permutation& permutation) noexcept {
  // SWAP gates can be executed virtually by changing the permutation
  if (op.getType() == qc::SWAP) {
    const auto& targets = op.getTargets();
    std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
  }
}

VectorDD applyGlobalPhase(VectorDD& in, const fp& phase, Package& dd) {
  in.w = dd.cn.lookup(in.w * ComplexValue{std::polar(1.0, phase)});

  return in;
}

} // namespace dd
