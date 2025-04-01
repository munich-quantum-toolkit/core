/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/StatePreparation.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/QuantumComputation.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

constexpr double EPS = 1e-10;

class StatePreparation
    : public testing::TestWithParam<std::vector<std::complex<double>>> {
protected:
  std::vector<std::complex<double>> amplitudes;

  void TearDown() override {}
  void SetUp() override { amplitudes = GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(
    StatePreparation, StatePreparation,
    testing::Values(
        std::vector<std::complex<double>>{std::complex{1 / std::sqrt(2)},
                                          std::complex{-1 / std::sqrt(2)}},
        std::vector<std::complex<double>>{
            std::complex<double>{1 / std::sqrt(2)},
            std::complex<double>{0, -1 / std::sqrt(2)}},
        std::vector<std::complex<double>>{0, std::complex{1 / std::sqrt(2)},
                                          std::complex{-1 / std::sqrt(2)}, 0},
        std::vector<std::complex<double>>{
            std::complex<double>{1 / std::sqrt(13)},
            std::complex<double>{-1 / std::sqrt(13)},
            std::complex<double>{1 / std::sqrt(13), -1 / std::sqrt(13)},
            std::complex<double>{0, 3 / std::sqrt(13)}},
        std::vector<std::complex<double>>{
            std::complex<double>{1. / 4}, std::complex<double>{1. / 4},
            std::complex<double>{1. / 4}, std::complex<double>{1. / 4},
            std::complex<double>{1. / 4}, std::complex<double>{1. / 4},
            std::complex<double>{1. / 4}, std::complex<double>{3. / 4}},
        std::vector<std::complex<double>>{
            std::complex<double>{1. / 4}, std::complex<double>{0, 1. / 4},
            std::complex<double>{1. / 4}, std::complex<double>{0, 1. / 4},
            std::complex<double>{0, 1. / 4}, std::complex<double>{1. / 4},
            std::complex<double>{1. / 4}, std::complex<double>{3. / 4}}));

TEST_P(StatePreparation, StatePreparationCircuitSimulation) {
  const auto& expectedAmplitudes = GetParam();
  qc::QuantumComputation qc;
  ASSERT_NO_THROW({ qc = qc::createStatePreparationCircuit(amplitudes); });
  auto dd = std::make_unique<dd::Package>(qc.getNqubits());
  dd::VectorDD e{};
  ASSERT_NO_THROW(
      { e = dd::simulate(qc, dd->makeZeroState(qc.getNqubits()), *dd); });
  auto result = e.getVector();
  for (size_t i = 0; i < expectedAmplitudes.size(); ++i) {
    ASSERT_NEAR(expectedAmplitudes[i].real(), result[i].real(), EPS);
    ASSERT_NEAR(expectedAmplitudes[i].imag(), result[i].imag(), EPS);
  }
}

TEST(StatePreparation, StatePreparationAmplitudesNotNormalized) {
  const auto amplitudes = std::vector<std::complex<double>>{
      std::complex<double>{1}, std::complex<double>{1}};

  ASSERT_THROW(std::ignore = qc::createStatePreparationCircuit(amplitudes),
               std::invalid_argument);
}

TEST(StatePreparation, StatePreparationsAmplitudesNotPowerOf2) {
  const auto amplitudes =
      std::vector<std::complex<double>>{std::complex<double>{1 / std::sqrt(3)},
                                        std::complex<double>{1 / std::sqrt(3)},
                                        std::complex<double>{1 / std::sqrt(3)}};

  ASSERT_THROW(std::ignore = qc::createStatePreparationCircuit(amplitudes),
               std::invalid_argument);
}
