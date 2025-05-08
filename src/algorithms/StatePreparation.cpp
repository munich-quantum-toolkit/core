/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/StatePreparation.hpp"

#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <tuple>
#include <vector>

namespace qc {
using Matrix = std::vector<std::vector<double>>;

namespace {
template <typename T>
[[nodiscard]] auto twoNorm(const std::vector<T>& vec) -> double {
  double norm = 0;
  for (auto elem : vec) {
    norm += std::norm(elem);
  }
  return sqrt(norm);
}

template <typename T>
[[nodiscard]] auto isNormalized(const std::vector<T>& vec, const double eps)
    -> bool {
  return std::abs(1 - twoNorm(vec)) < eps;
}

[[nodiscard]] auto kroneckerProduct(const Matrix& matrixA,
                                    const Matrix& matrixB) -> Matrix {
  size_t const rowA = matrixA.size();
  size_t const rowB = matrixB.size();
  size_t const colA = matrixA[0].size();
  size_t const colB = matrixB[0].size();
  // initialize size
  Matrix newMatrix{(rowA * rowB), std::vector<double>(colA * colB, 0)};
  // code taken from RosettaCode slightly adapted
  for (size_t i = 0; i < rowA; ++i) {
    // k loops till rowB
    for (size_t j = 0; j < colA; ++j) {
      // j loops till colA
      for (size_t k = 0; k < rowB; ++k) {
        // l loops till colB
        for (size_t l = 0; l < colB; ++l) {
          // Each element of matrix A is
          // multiplied by whole Matrix B
          // resp and stored as Matrix C
          newMatrix[(i * rowB) + k][(j * colB) + l] =
              matrixA[i][j] * matrixB[k][l];
        }
      }
    }
  }
  return newMatrix;
}

[[nodiscard]] auto createIdentity(const size_t size) -> Matrix {
  auto identity{std::vector(size, std::vector<double>(size, 0))};
  for (size_t i = 0; i < size; ++i) {
    identity[i][i] = 1;
  }
  return identity;
}

[[nodiscard]] auto matrixVectorProd(const Matrix& matrix,
                                    const std::vector<double>& vector)
    -> std::vector<double> {
  std::vector<double> result;
  result.reserve(matrix.size());
  for (const auto& matrixVec : matrix) {
    double sum{0};
    for (size_t i = 0; i < matrixVec.size(); ++i) {
      sum += matrixVec[i] * vector[i];
    }
    result.emplace_back(sum);
  }
  return result;
}

/**
 * @brief recursive implementation that returns multiplexer circuit
 * @param targetGate : Ry or Rz gate to apply to target qubit, multiplexed
 * over all other "select" qubits
 * @param angles : list of rotation angles to apply Ry and Rz
 * @param lastCnot : add last CNOT if true
 * @return multiplexer circuit as QuantumComputation
 */
[[nodiscard]] auto multiplex(OpType targetGate, std::vector<double> angles,
                             const bool lastCnot) -> QuantumComputation {
  size_t const listLen = angles.size();
  auto const localNumQubits = static_cast<Qubit>(
      std::floor(std::log2(static_cast<double>(listLen))) + 1);
  QuantumComputation multiplexer{localNumQubits};
  // recursion base case
  if (localNumQubits == 1) {
    multiplexer.emplace_back<StandardOperation>(0, targetGate,
                                                std::vector{angles[0]});
    return multiplexer;
  }

  Matrix const matrix{std::vector{0.5, 0.5}, std::vector{0.5, -0.5}};
  Matrix const identity = createIdentity(1 << (localNumQubits - 2));
  Matrix const angleWeights = kroneckerProduct(matrix, identity);

  angles = matrixVectorProd(angleWeights, angles);

  std::vector<double> const angles1{
      std::make_move_iterator(angles.begin()),
      std::make_move_iterator(angles.begin() +
                              static_cast<int64_t>(listLen) / 2)};
  QuantumComputation multiplex1 = multiplex(targetGate, angles1, false);

  // append multiplex1 to multiplexer
  multiplexer.emplace_back<Operation>(multiplex1.asOperation());
  // flips the LSB qubit, control on MSB
  multiplexer.cx(localNumQubits - 1, 0);

  std::vector<double> const angles2{std::make_move_iterator(angles.begin()) +
                                        static_cast<int64_t>(listLen) / 2,
                                    std::make_move_iterator(angles.end())};
  QuantumComputation multiplex2 = multiplex(targetGate, angles2, false);

  // extra efficiency by reversing (!= inverting) second multiplex
  if (listLen > 1) {
    multiplex2.reverse();
    multiplexer.emplace_back<Operation>(multiplex2.asOperation());
  } else {
    multiplexer.emplace_back<Operation>(multiplex2.asOperation());
  }

  if (lastCnot) {
    multiplexer.cx(localNumQubits - 1, 0);
  }

  CircuitOptimizer::flattenOperations(multiplexer);
  return multiplexer;
}

[[nodiscard]] auto blochAngles(std::complex<double> const complexA,
                               std::complex<double> const complexB,
                               const double eps)
    -> std::tuple<std::complex<double>, double, double> {
  double theta{0};
  double phi{0};
  double finalT{0};
  double const magA = std::abs(complexA);
  double const magB = std::abs(complexB);
  double const finalR = sqrt(pow(magA, 2) + pow(magB, 2));
  if (finalR > eps) {
    theta = 2 * acos(magA / finalR);
    double const aAngle = std::arg(complexA);
    double const bAngle = std::arg(complexB);
    finalT = aAngle + bAngle;
    phi = bAngle - aAngle;
  }
  return {std::polar(finalR, finalT / 2.), theta, phi};
}

// works out Ry and Rz rotation angles used to disentangle LSB qubit
// rotations make up block diagonal matrix U
[[nodiscard]] auto
rotationsToDisentangle(const std::vector<std::complex<double>>& amplitudes,
                       const double eps)
    -> std::tuple<std::vector<std::complex<double>>, std::vector<double>,
                  std::vector<double>> {
  size_t const amplitudesHalf = amplitudes.size() / 2;
  std::vector<std::complex<double>> remainingVector;
  std::vector<double> thetas;
  std::vector<double> phis;

  remainingVector.reserve(amplitudesHalf);
  thetas.reserve(amplitudesHalf);
  phis.reserve(amplitudesHalf);

  for (size_t i = 0; i < amplitudesHalf; ++i) {
    auto [remains, theta, phi] =
        blochAngles(amplitudes[2 * i], amplitudes[(2 * i) + 1], eps);
    remainingVector.emplace_back(remains);
    // minus sign because we move it to zero
    thetas.emplace_back(-theta);
    phis.emplace_back(-phi);
  }
  return {remainingVector, thetas, phis};
}

// creates circuit that takes desired vector to zero
[[nodiscard]] auto
gatesToUncompute(std::vector<std::complex<double>>& amplitudes,
                 const size_t numQubits, const double eps)
    -> QuantumComputation {
  QuantumComputation disentangler{numQubits};
  for (size_t i = 0; i < numQubits; ++i) {
    // rotations to disentangle LSB
    auto [remainingParams, thetas, phis] =
        rotationsToDisentangle(amplitudes, eps);
    amplitudes = remainingParams;
    // perform required rotations
    bool addLastCnot = true;
    double const phisNorm = twoNorm(phis);
    double const thetasNorm = twoNorm(thetas);
    if (phisNorm > eps && thetasNorm > eps) {
      addLastCnot = false;
    }
    if (phisNorm > eps) {
      // call multiplex with RZGate
      QuantumComputation rzMultiplexer = multiplex(RZ, phis, addLastCnot);
      // append rzMultiplexer to disentangler, but it should only attach on
      // qubits i-numQubits, thus "i" is added to the local qubit indices
      for (const auto& op : rzMultiplexer) {
        for (auto& target : op->getTargets()) {
          target += static_cast<Qubit>(i);
        }
        // as there were some systematic compiler errors with accessing the
        // qubit directly the controls are collected and then newly set
        Controls newControls;
        for (const auto& control : op->getControls()) {
          newControls.emplace(control.qubit + static_cast<Qubit>(i));
        }
        op->setControls(newControls);
      }
      disentangler.emplace_back<Operation>(rzMultiplexer.asOperation());
    }
    if (thetasNorm > eps) {
      // call multiplex with RYGate
      QuantumComputation ryMultiplexer = multiplex(RY, thetas, addLastCnot);
      // append reversed ry_multiplexer to disentangler, but it should only
      // attach on qubits i-numQubits, thus "i" is added to the local qubit
      // indices
      ryMultiplexer.reverse();
      for (const auto& op : ryMultiplexer) {
        for (auto& target : op->getTargets()) {
          target += static_cast<Qubit>(i);
        }
        // as there were some systematic compiler errors with accessing the
        // qubit directly the controls are collected and then newly set
        Controls newControls;
        for (const auto& control : op->getControls()) {
          newControls.emplace(control.qubit + static_cast<Qubit>(i));
        }
        op->setControls(newControls);
      }
      disentangler.emplace_back<Operation>(ryMultiplexer.asOperation());
    }
  }
  // adjust global phase according to the last e^(it)
  double const arg = std::arg(std::accumulate(
      amplitudes.begin(), amplitudes.end(), std::complex<double>(0, 0)));
  if (std::abs(arg) > eps) {
    disentangler.gphase(arg);
  }
  return disentangler;
}
} // namespace

auto createStatePreparationCircuit(
    const std::vector<std::complex<double>>& amplitudes, const double eps)
    -> QuantumComputation {

  auto amplitudesCopy = amplitudes;

  if (!isNormalized(amplitudes, eps)) {
    throw std::invalid_argument{
        "Using State Preparation with Amplitudes that are not normalized"};
  }

  // check if the number of elements in the vector is a power of two
  if (amplitudes.empty() ||
      (amplitudes.size() & (amplitudes.size() - 1)) != 0) {
    throw std::invalid_argument{
        "Using State Preparation with vector size that is not a power of 2"};
  }
  const auto numQubits = static_cast<size_t>(std::log2(amplitudes.size()));
  QuantumComputation toZeroCircuit =
      gatesToUncompute(amplitudesCopy, numQubits, eps);

  // invert circuit
  toZeroCircuit.invert();

  return toZeroCircuit;
}
} // namespace qc
