/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/RandomCliffordCircuit.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/Definitions.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>

class RandomClifford : public testing::TestWithParam<qc::Qubit> {
protected:
  void TearDown() override {}
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    RandomClifford, RandomClifford, testing::Range<qc::Qubit>(1U, 9U),
    [](const testing::TestParamInfo<RandomClifford::ParamType>& inf) {
      // Generate names for test cases
      const auto nqubits = inf.param;
      std::stringstream ss{};
      ss << static_cast<std::size_t>(nqubits) << "_qubits";
      return ss.str();
    });

TEST_P(RandomClifford, simulate) {
  const auto nq = GetParam();
  constexpr auto numReps = 16U;

  const auto dd = std::make_unique<dd::Package>(nq);
  for (size_t i = 0; i < numReps; ++i) {
    auto qc =
        qc::createRandomCliffordCircuit(nq, static_cast<std::size_t>(nq) * nq);
    auto in = dd->makeZeroState(nq);
    ASSERT_NO_THROW({ dd::simulate(qc, in, *dd); });
    qc.printStatistics(std::cout);
  }
}

TEST_P(RandomClifford, buildFunctionality) {
  const auto nq = GetParam();

  const auto dd = std::make_unique<dd::Package>(nq);
  const auto qc = qc::createRandomCliffordCircuit(
      nq, static_cast<std::size_t>(nq) * nq, 12345);
  ASSERT_NO_THROW({ dd::buildFunctionality(qc, *dd); });
  qc.printStatistics(std::cout);
}
