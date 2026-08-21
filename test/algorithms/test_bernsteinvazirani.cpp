/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/BernsteinVazirani.hpp"
#include "dd/Simulation.hpp"
#include "ir/QuantumComputation.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <sstream>
#include <string>

namespace {

class BernsteinVazirani : public testing::TestWithParam<std::string> {
protected:
  void TearDown() override {}
  void SetUp() override {}
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    BernsteinVazirani, BernsteinVazirani,
    testing::Values("0", "11", "101", "1001", "1000001", "10011001",
                    "1010000101", "101100001101", "10010000001001",
                    (std::string{"1"} + std::string(39U, '0') + "1")),
    [](const testing::TestParamInfo<BernsteinVazirani::ParamType>& inf) {
      std::stringstream ss{};
      ss << "case_" << inf.index << "_" << inf.param.size() << "_bits";
      return ss.str();
    });

TEST_P(BernsteinVazirani, FunctionTest) {
  // get hidden bitstring
  const auto& expected = GetParam();
  const auto s = qc::BVBitString(expected);

  // construct Bernstein Vazirani circuit
  const auto qc = qc::createBernsteinVazirani(s);

  // simulate the circuit
  constexpr std::size_t shots = 1024;
  const auto measurements = dd::sample(qc, shots);

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements.at(expected), shots);
}

TEST_P(BernsteinVazirani, FunctionTestDynamic) {
  // get hidden bitstring
  const auto& expected = GetParam();
  const auto s = qc::BVBitString(expected);

  // construct Bernstein Vazirani circuit
  const auto qc = qc::createIterativeBernsteinVazirani(s);

  // simulate the circuit
  constexpr std::size_t shots = 1024;
  const auto measurements = dd::sample(qc, shots);

  // expect to obtain the hidden bitstring with certainty
  EXPECT_EQ(measurements.at(expected), shots);
}

TEST_F(BernsteinVazirani, LargeCircuitConstruction) {
  constexpr std::size_t nq = 127;
  constexpr std::size_t seed = 0;
  [[maybe_unused]] const auto qc = qc::createBernsteinVazirani(nq, seed);
}

TEST_F(BernsteinVazirani, DynamicCircuitConstruction) {
  constexpr std::size_t nq = 127;
  constexpr std::size_t seed = 0;
  [[maybe_unused]] const auto qc =
      qc::createIterativeBernsteinVazirani(nq, seed);
}

TEST_P(BernsteinVazirani, OrdinaryAndIterativeResultsAgree) {
  // get hidden bitstring
  const auto s = qc::BVBitString(GetParam());
  const auto bv = qc::createBernsteinVazirani(s);
  const auto dbv = qc::createIterativeBernsteinVazirani(s);

  constexpr std::size_t shots = 128U;
  constexpr std::size_t seed = 7U;
  EXPECT_EQ(dd::sample(bv, shots, seed), dd::sample(dbv, shots, seed));
}
