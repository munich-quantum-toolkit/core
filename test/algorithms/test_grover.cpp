/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/Grover.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumberUniqueTable.hpp"
#include "dd/Simulation.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>

namespace {

[[nodiscard]] auto makePalindromicTarget(const qc::Qubit nqubits,
                                         const std::size_t pattern)
    -> std::string {
  std::string target(nqubits, '0');
  for (std::size_t i = 0; i < (nqubits + 1U) / 2U; ++i) {
    if ((pattern + i) % 3U == 1U) {
      continue;
    }
    target[i] = '1';
    target[nqubits - 1U - i] = '1';
  }
  return target;
}

class Grover
    : public testing::TestWithParam<std::tuple<qc::Qubit, std::size_t>> {
protected:
  void TearDown() override {
    dd->garbageCollect(true);
    EXPECT_EQ(dd->cn.realCount(), dd::immortals::size());
  }

  void SetUp() override {
    std::tie(nqubits, seed) = GetParam();
    dd = std::make_unique<dd::Package>(nqubits + 1);
    expected = makePalindromicTarget(nqubits, seed);
    targetValue = qc::GroverBitString(expected);
    qc = qc::createGrover(nqubits, targetValue);

    /// Exercise the seeded overload without deriving an oracle from its name.
    static_cast<void>(qc::createGrover(nqubits, seed));
  }

  qc::Qubit nqubits = 0;
  std::size_t seed = 0;
  std::unique_ptr<dd::Package> dd;
  qc::QuantumComputation qc;
  dd::MatrixDD func{};
  std::string expected;
  qc::GroverBitString targetValue;
};

} // namespace

constexpr qc::Qubit GROVER_MAX_QUBITS = 15;
constexpr std::size_t GROVER_NUM_SEEDS = 5;
constexpr dd::fp GROVER_GOAL_PROBABILITY = 0.9;

INSTANTIATE_TEST_SUITE_P(
    Grover, Grover,
    testing::Combine(
        testing::Range(static_cast<qc::Qubit>(2), GROVER_MAX_QUBITS + 1, 3),
        testing::Range(static_cast<std::size_t>(0), GROVER_NUM_SEEDS)),
    [](const testing::TestParamInfo<Grover::ParamType>& inf) {
      const auto nqubits = std::get<0>(inf.param);
      const auto seed = std::get<1>(inf.param);
      std::stringstream ss{};
      ss << nqubits + 1;
      if (nqubits == 0) {
        ss << "_qubit_";
      } else {
        ss << "_qubits_";
      }
      ss << seed;
      return ss.str();
    });

TEST_P(Grover, Functionality) {
  auto x = '1' + expected;
  std::ranges::reverse(x);
  std::ranges::replace(x, '1', '2');

  qc::QuantumComputation groverIteration(qc.getNqubits());
  qc::appendGroverOracle(groverIteration, targetValue);
  qc::appendGroverDiffusion(groverIteration);

  const auto iteration = buildFunctionality(groverIteration, *dd);

  auto e = iteration;
  dd->incRef(e);
  const auto iterations = qc::computeNumberOfIterations(nqubits);
  for (std::size_t i = 0U; i < iterations - 1U; ++i) {
    e = dd->applyOperation(iteration, e);
  }

  qc::QuantumComputation setup(qc.getNqubits());
  qc::appendGroverInitialization(setup);
  const auto g = buildFunctionality(setup, *dd);
  const auto f = dd->multiply(e, g);
  dd->incRef(f);
  dd->decRef(e);
  dd->decRef(g);
  func = f;

  dd->decRef(iteration);

  const auto c = func.getValueByPath(qc.getNqubits(), x);
  const auto prob = std::norm(c);
  EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);

  dd->decRef(func);
}

TEST_P(Grover, FunctionalityRecursive) {
  auto x = '1' + expected;
  std::ranges::reverse(x);
  std::ranges::replace(x, '1', '2');

  qc::QuantumComputation groverIteration(qc.getNqubits());
  qc::appendGroverOracle(groverIteration, targetValue);
  qc::appendGroverDiffusion(groverIteration);

  const auto iter = buildFunctionalityRecursive(groverIteration, *dd);
  auto e = iter;
  const auto iterations = qc::computeNumberOfIterations(nqubits);
  const std::bitset<128U> iterBits(iterations);
  const auto msb = static_cast<std::size_t>(std::floor(std::log2(iterations)));
  auto f = iter;
  dd->incRef(f);
  bool zero = !iterBits[0U];
  for (std::size_t j = 1U; j <= msb; ++j) {
    auto tmp = dd->multiply(f, f);
    dd->incRef(tmp);
    dd->decRef(f);
    f = tmp;
    if (iterBits[j]) {
      if (zero) {
        dd->incRef(f);
        dd->decRef(e);
        e = f;
        zero = false;
      } else {
        e = dd->applyOperation(f, e);
      }
    }
  }
  dd->decRef(f);

  // apply state preparation setup
  qc::QuantumComputation statePrep(qc.getNqubits());
  qc::appendGroverInitialization(statePrep);
  const auto s = buildFunctionality(statePrep, *dd);
  func = dd->multiply(e, s);
  dd->incRef(func);
  dd->decRef(s);
  dd->decRef(e);

  const auto c = func.getValueByPath(qc.getNqubits(), x);
  const auto prob = std::norm(c);
  EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);

  dd->decRef(func);
}

TEST_P(Grover, Simulation) {
  constexpr std::size_t shots = 1024;
  const auto measurements = dd::sample(qc, shots);
  const auto result = measurements.find(expected);
  const auto correctShots = result == measurements.end() ? 0U : result->second;
  const auto probability =
      static_cast<double>(correctShots) / static_cast<double>(shots);

  EXPECT_GE(probability, GROVER_GOAL_PROBABILITY);
}
