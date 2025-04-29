/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/Grover.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>

namespace qc {
auto appendGroverInitialization(QuantumComputation& qc) -> void {
  const auto nDataQubits = static_cast<Qubit>(qc.getNqubits() - 1);
  qc.x(nDataQubits);
  for (Qubit i = 0; i < nDataQubits; ++i) {
    qc.h(i);
  }
}

auto appendGroverOracle(QuantumComputation& qc,
                        const GroverBitString& targetValue) -> void {
  const auto nDataQubits = static_cast<Qubit>(qc.getNqubits() - 1);
  Controls controls{};
  for (Qubit i = 0; i < nDataQubits; ++i) {
    controls.emplace(i, targetValue.test(i) ? Control::Type::Pos
                                            : Control::Type::Neg);
  }
  qc.mcz(controls, nDataQubits);
}

auto appendGroverDiffusion(QuantumComputation& qc) -> void {
  const auto nDataQubits = static_cast<Qubit>(qc.getNqubits() - 1);
  for (Qubit i = 0; i < nDataQubits; ++i) {
    qc.h(i);
  }
  for (Qubit i = 0; i < nDataQubits; ++i) {
    qc.x(i);
  }
  Controls controls{};
  for (Qubit j = 1; j < nDataQubits; ++j) {
    controls.emplace(j);
  }
  qc.mcz(controls, 0);
  for (auto i = static_cast<std::make_signed_t<Qubit>>(nDataQubits - 1); i >= 0;
       --i) {
    qc.x(static_cast<Qubit>(i));
  }
  for (auto i = static_cast<std::make_signed_t<Qubit>>(nDataQubits - 1); i >= 0;
       --i) {
    qc.h(static_cast<Qubit>(i));
  }
}

auto computeNumberOfIterations(const Qubit nq) -> std::size_t {
  if (nq <= 2) {
    return 1;
  }
  if (nq % 2 == 1) {
    return static_cast<std::size_t>(std::round(
        PI_4 * std::pow(2., (static_cast<double>(nq + 1) / 2.) - 1.) *
        std::sqrt(2)));
  }
  return static_cast<std::size_t>(
      std::round(PI_4 * std::pow(2., static_cast<double>(nq) / 2.)));
}

namespace {
[[nodiscard]] auto generateTargetValue(const std::size_t nDataQubits,
                                       std::mt19937_64& mt) -> GroverBitString {
  GroverBitString targetValue;
  std::bernoulli_distribution distribution{};
  for (std::size_t i = 0; i < nDataQubits; i++) {
    if (distribution(mt)) {
      targetValue.set(i);
    }
  }
  return targetValue;
}

[[nodiscard]] auto getGroverName(const GroverBitString& s, const Qubit nq)
    -> std::string {
  auto expected = s.to_string();
  std::reverse(expected.begin(), expected.end());
  while (expected.length() > nq) {
    expected.pop_back();
  }
  std::reverse(expected.begin(), expected.end());
  return "grover_" + std::to_string(nq) + "_" + expected;
}

auto constructGroverCircuit(QuantumComputation& qc, const Qubit nq,
                            const GroverBitString& targetValue) {
  qc.setName(getGroverName(targetValue, nq));
  qc.addQubitRegister(nq, "q");
  qc.addQubitRegister(1, "flag");
  qc.addClassicalRegister(nq);

  // create initial superposition
  appendGroverInitialization(qc);

  // apply Grover iterations
  const auto iterations = computeNumberOfIterations(nq);
  for (std::size_t j = 0; j < iterations; ++j) {
    appendGroverOracle(qc, targetValue);
    appendGroverDiffusion(qc);
  }

  // measure the resulting state
  for (Qubit i = 0; i < nq; ++i) {
    qc.measure(i, i);
  }
}
} // namespace

auto createGrover(const Qubit nq, const std::size_t seed)
    -> QuantumComputation {
  auto qc = QuantumComputation(0, 0, seed);
  const auto targetValue = generateTargetValue(nq, qc.getGenerator());
  constructGroverCircuit(qc, nq, targetValue);
  return qc;
}

auto createGrover(const Qubit nq, const GroverBitString& targetValue)
    -> QuantumComputation {
  auto qc = QuantumComputation();
  constructGroverCircuit(qc, nq, targetValue);
  return qc;
}

} // namespace qc
