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
#include <sstream>
#include <tuple>

class Grover
    : public testing::TestWithParam<std::tuple<qc::Qubit, std::size_t>> {
protected:
  void TearDown() override {
    dd->garbageCollect(true);
    // number of complex table entries after clean-up should equal 0
    EXPECT_EQ(dd->cn.realCount(), 0);
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
  std::reverse(x.begin(), x.end());
  std::replace(x.begin(), x.end(), '1', '2');

  qc::QuantumComputation groverSetup(qc.getNqubits());
  qc::appendGroverInitialization(groverSetup);

  qc::QuantumComputation groverIteration(qc.getNqubits());
  qc::appendGroverOracle(groverIteration, targetValue);
  qc::appendGroverDiffusion(groverIteration);

  const auto setup = buildFunctionality(groverSetup, *dd);
  const auto iterationOp = buildFunctionality(groverIteration, *dd);
  const auto iterations = qc::computeNumberOfIterations(nqubits);

  auto iteration = iterationOp;
  for (std::size_t i = 0U; i < iterations - 1U; ++i) {
    const auto next = dd->multiply(iterationOp, iteration);
    dd->track(next);
    dd->untrack(iteration); // This will automatically untrack the iterationOp.
    dd->garbageCollect();
    iteration = next;
  }

  const auto groverFull = dd->multiply(iteration, setup);
  dd->track(groverFull);

  // Amplitude of the searched-for entry should be 1
  const auto c = groverFull.getValueByPath(qc.getNqubits(), x);
  const auto prob = std::norm(c);

  EXPECT_NEAR(std::abs(c.real()), 1, GROVER_ACCURACY);
  EXPECT_NEAR(std::abs(c.imag()), 0, GROVER_ACCURACY);
  EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);

  dd->untrack(setup);
  dd->untrack(iteration);
  dd->untrack(groverFull);
}

TEST_P(Grover, FunctionalityRecursive) {
  auto x = '1' + expected;
  std::reverse(x.begin(), x.end());
  std::replace(x.begin(), x.end(), '1', '2');

  qc::QuantumComputation groverSetup(qc.getNqubits());
  qc::appendGroverInitialization(groverSetup);

  qc::QuantumComputation groverIteration(qc.getNqubits());
  qc::appendGroverOracle(groverIteration, targetValue);
  qc::appendGroverDiffusion(groverIteration);

  const auto setup = buildFunctionalityRecursive(groverSetup, *dd);
  const auto iterationOp = buildFunctionalityRecursive(groverIteration, *dd);
  const auto iterations = qc::computeNumberOfIterations(nqubits);

  auto iteration = iterationOp;
  for (std::size_t i = 0U; i < iterations - 1U; ++i) {
    const auto next = dd->multiply(iterationOp, iteration);
    dd->track(next);
    dd->untrack(iteration); // This will automatically untrack the iterationOp.
    dd->garbageCollect();
    iteration = next;
  }

  const auto groverFull = dd->multiply(iteration, setup);
  dd->track(groverFull);

  // amplitude of the searched-for entry should be 1
  const auto c = groverFull.getValueByPath(qc.getNqubits(), x);
  const auto prob = std::norm(c);

  EXPECT_NEAR(std::abs(c.real()), 1, GROVER_ACCURACY);
  EXPECT_NEAR(std::abs(c.imag()), 0, GROVER_ACCURACY);
  EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);

  dd->untrack(setup);
  dd->untrack(iteration);
  dd->untrack(groverFull);
}

TEST_P(Grover, Simulation) {
  constexpr std::size_t shots = 1024;
  const auto measurements = dd::sample(qc, shots);
  ASSERT_TRUE(measurements.find(expected) != measurements.end());
  const auto correctShots = measurements.at(expected);
  const auto probability =
      static_cast<double>(correctShots) / static_cast<double>(shots);

  EXPECT_GE(probability, GROVER_GOAL_PROBABILITY);
}
