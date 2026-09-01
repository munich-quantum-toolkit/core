/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/GateMatrixDefinitions.hpp"

#include "dd/DDDefinitions.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
using namespace dd;

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

/**
 * @brief Computes the matrix representation of the R(θ, φ) gate.
 * @param theta The rotation angle θ.
 * @param phi The rotation axis angle φ.
 * @return The gate matrix for the R(θ, φ) rotation.
 *
 * @details The R(θ, φ) gate is defined as R(θ, φ) = exp(-i*θ/2*(cos(φ)X +
 * sin(φ)Y)), which results in the matrix:
 * [[cos(θ/2), -i*e^(-iφ)*sin(θ/2)],
 *  [-i*e^(iφ)*sin(θ/2), cos(θ/2)]]
 */
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
} // namespace

namespace dd {

GateMatrix opToSingleQubitGateMatrix(const GateType t,
                                     const std::vector<fp>& params) {
  switch (t) {
  case GateType::I:
    return {1, 0, 0, 1};
  case GateType::H:
    return {SQRT2_2, SQRT2_2, SQRT2_2, -SQRT2_2};
  case GateType::X:
    return {0, 1, 1, 0};
  case GateType::Y:
    return {0, {0, -1}, {0, 1}, 0};
  case GateType::Z:
    return {1, 0, 0, -1};
  case GateType::S:
    return {1, 0, 0, {0, 1}};
  case GateType::Sdg:
    return {1, 0, 0, {0, -1}};
  case GateType::T:
    return {1, 0, 0, {SQRT2_2, SQRT2_2}};
  case GateType::Tdg:
    return {1, 0, 0, {SQRT2_2, -SQRT2_2}};
  case GateType::SX:
    return {std::complex<fp>{0.5, 0.5}, std::complex<fp>{0.5, -0.5},
            std::complex<fp>{0.5, -0.5}, std::complex<fp>{0.5, 0.5}};
  case GateType::SXdg:
    return {std::complex<fp>{0.5, -0.5}, std::complex<fp>{0.5, 0.5},
            std::complex<fp>{0.5, 0.5}, std::complex<fp>{0.5, -0.5}};
  case GateType::U:
    // shuffle parameters to match semantics of parameter <-> matrix from
    // getGateDD
    return uMat(params.at(2), params.at(1), params.at(0));
  case GateType::U2:
    // swap parameters to match semantics of parameter <-> matrix from
    // getGateDD
    return u2Mat(params.at(1), params.at(0));
  case GateType::P:
    return pMat(params.at(0));
  case GateType::RX:
    return rxMat(params.at(0));
  case GateType::RY:
    return ryMat(params.at(0));
  case GateType::RZ:
    return rzMat(params.at(0));
  case GateType::R:
    return rMat(params.at(0), params.at(1));
  default:
    throw std::invalid_argument("Invalid single-qubit gate type");
  }
}

TwoQubitGateMatrix opToTwoQubitGateMatrix(const GateType t,
                                          const std::vector<fp>& params) {
  switch (t) {
  case GateType::SWAP:
    return {{{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}};
  case GateType::iSWAP:
    return {{{1, 0, 0, 0}, {0, 0, {0, 1}, 0}, {0, {0, 1}, 0, 0}, {0, 0, 0, 1}}};
  case GateType::ECR:
    return {{{0, 0, SQRT2_2, {0, SQRT2_2}},
             {0, 0, {0, SQRT2_2}, SQRT2_2},
             {SQRT2_2, {0, -SQRT2_2}, 0, 0},
             {std::complex<fp>{0., -SQRT2_2}, SQRT2_2, 0, 0}}};
  case GateType::DCX:
    return {{{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 1, 0, 0}}};
  case GateType::RXX:
    return rxxMat(params.at(0));
  case GateType::RYY:
    return ryyMat(params.at(0));
  case GateType::RZZ:
    return rzzMat(params.at(0));
  case GateType::RZX:
    return rzxMat(params.at(0));
  case GateType::XXminusYY:
    return xxMinusYYMat(params.at(0), params.at(1));
  case GateType::XXplusYY:
    return xxPlusYYMat(params.at(0), params.at(1));
  default:
    throw std::invalid_argument("Invalid two-qubit gate type");
  }
}

ThreeQubitGateMatrix
opToThreeQubitGateMatrix(const GateType t, const std::vector<fp>& /*params*/) {
  switch (t) {
  case GateType::RCCX: {
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
} // namespace dd
