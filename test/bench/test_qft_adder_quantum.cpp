/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Evaluation.hpp"
#include "bench/QFTAdderQuantum.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace {

using mqt::bench::Output;
using mqt::bench::QFTAdderQuantum;
using mqt::bench::QFTAdderQuantumOptions;

TEST(QFTAdderQuantum, StoresTheRegisterWidthAndOutput) {
  const QFTAdderQuantum benchmark{{.qubits = 7}};
  EXPECT_EQ(benchmark.options().qubits, 7);
  EXPECT_EQ(benchmark.output(), (Output{"result", 14}));
}

TEST(QFTAdderQuantum, ValidatesTheConfiguredInstance) {
  EXPECT_THROW(static_cast<void>(QFTAdderQuantum{{.qubits = 0}}),
               std::invalid_argument);
  EXPECT_NO_THROW(static_cast<void>(
      QFTAdderQuantum{{.qubits = QFTAdderQuantumOptions::MAX_QUBITS}}));
  EXPECT_THROW(static_cast<void>(QFTAdderQuantum{
                   {.qubits = QFTAdderQuantumOptions::MAX_QUBITS + 1}}),
               std::invalid_argument);
}

TEST(QFTAdderQuantum, GivesUniformWeightToCorrelatedSums) {
  const QFTAdderQuantum benchmark{{.qubits = 3}};
  for (const auto* outcome : {
           "000001",
           "001010",
           "010011",
           "011100",
           "100101",
           "101110",
           "110111",
           "111000",
       }) {
    EXPECT_DOUBLE_EQ(benchmark.probability(outcome), 1. / 8.);
  }

  EXPECT_DOUBLE_EQ(benchmark.probability("000000"), 0.);
  EXPECT_DOUBLE_EQ(benchmark.probability("111111"), 0.);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00001")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00000x")),
               std::invalid_argument);
}

TEST(QFTAdderQuantum, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const QFTAdderQuantum benchmark{{.qubits = 2}};
  const auto exact =
      benchmark.evaluate({{"0001", 1}, {"0110", 1}, {"1011", 1}, {"1100", 1}});
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(exact.successProbability);

  const auto biased = benchmark.evaluate({{"0001", 4}});
  EXPECT_DOUBLE_EQ(biased.totalVariationDistance, 0.75);
  EXPECT_DOUBLE_EQ(biased.squaredHellingerFidelity, 0.25);
  EXPECT_FALSE(biased.successProbability);
}

TEST(QFTAdderQuantum, KeepsTheLargestReferenceWeightRepresentable) {
  const QFTAdderQuantum benchmark{
      {.qubits = QFTAdderQuantumOptions::MAX_QUBITS}};
  const auto outcome = std::string(QFTAdderQuantumOptions::MAX_QUBITS, '1') +
                       std::string(QFTAdderQuantumOptions::MAX_QUBITS, '0');
  EXPECT_GT(benchmark.probability(outcome), 0.);
}

} // namespace
