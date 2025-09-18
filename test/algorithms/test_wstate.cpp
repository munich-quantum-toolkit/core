/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/WState.hpp"
#include "dd/Simulation.hpp"
#include "ir/Definitions.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class WState : public testing::TestWithParam<qc::Qubit> {};

namespace {
std::vector<std::string> generateWStateStrings(const std::size_t length) {
  std::vector<std::string> result;
  result.reserve(length);
  for (std::size_t i = 0U; i < length; ++i) {
    auto binaryString = std::string(length, '0');
    binaryString[i] = '1';
    result.emplace_back(binaryString);
  }
  return result;
}
} // namespace

INSTANTIATE_TEST_SUITE_P(
    WState, WState, testing::Range<qc::Qubit>(1U, 128U, 7U),
    [](const testing::TestParamInfo<WState::ParamType>& inf) {
      // Generate names for test cases
      const auto nqubits = inf.param;
      std::stringstream ss{};
      ss << nqubits << "_qubits";
      return ss.str();
    });

TEST_P(WState, FunctionTest) {
  const auto nq = GetParam();
  const auto qc = qc::createWState(nq);
  constexpr std::size_t shots = 4096U;
  const auto measurements = dd::sample(qc, shots);
  for (const auto& result : generateWStateStrings(nq)) {
    EXPECT_TRUE(measurements.contains(result));
  }
}
