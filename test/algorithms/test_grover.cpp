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
#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumberUniqueTable.hpp"
#include "dd/Simulation.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <ranges>
#include <sstream>
#include <tuple>

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
    qc = qc::createGrover(nqubits, seed);
    qc.printStatistics(std::cout);

    // parse expected result from circuit name
    const auto& name = qc.getName();
    expected = name.substr(name.find_last_of('_') + 1);
    targetValue = qc::GroverBitString(expected);
  }

  qc::Qubit nqubits = 0;
  std::size_t seed = 0;
  std::unique_ptr<dd::Package> dd;
  qc::QuantumComputation qc;
  dd::MatrixDD func{};
  std::string expected;
  qc::GroverBitString targetValue;
};

constexpr qc::Qubit GROVER_MAX_QUBITS = 15;
constexpr std::size_t GROVER_NUM_SEEDS = 5;
constexpr dd::fp GROVER_ACCURACY = 1e-2;
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

  // amplitude of the searched-for entry should be 1
  const auto c = func.getValueByPath(qc.getNqubits(), x);
  EXPECT_NEAR(std::abs(c.real()), 1, GROVER_ACCURACY);
  EXPECT_NEAR(std::abs(c.imag()), 0, GROVER_ACCURACY);
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

  // amplitude of the searched-for entry should be 1
  const auto c = func.getValueByPath(qc.getNqubits(), x);
  EXPECT_NEAR(std::abs(c.real()), 1, GROVER_ACCURACY);
  EXPECT_NEAR(std::abs(c.imag()), 0, GROVER_ACCURACY);
  const auto prob = std::norm(c);
  EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);

  dd->decRef(func);
}

TEST_P(Grover, Simulation) {
  constexpr std::size_t shots = 1024;
  const auto measurements = dd::sample(qc, shots);
  ASSERT_TRUE(measurements.contains(expected));
  const auto correctShots = measurements.at(expected);
  const auto probability =
      static_cast<double>(correctShots) / static_cast<double>(shots);

  EXPECT_GE(probability, GROVER_GOAL_PROBABILITY);
}
