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
#include "bench/Multiplexer.hpp"

#include <gtest/gtest.h>

#include <numbers>
#include <stdexcept>
#include <string>

namespace {

using mqt::bench::Multiplexer;
using mqt::bench::MultiplexerOptions;
using mqt::bench::Output;

TEST(Multiplexer, StoresTheTotalQubitCountAndOutput) {
  const Multiplexer benchmark{{.qubits = 7}};
  EXPECT_EQ(benchmark.options().qubits, 7);
  EXPECT_EQ(benchmark.output(), (Output{"result", 7}));
}

TEST(Multiplexer, ValidatesTheConfiguredInstance) {
  EXPECT_THROW(static_cast<void>(Multiplexer{{.qubits = 1}}),
               std::invalid_argument);
  EXPECT_NO_THROW(static_cast<void>(Multiplexer{{.qubits = 2}}));
  EXPECT_NO_THROW(static_cast<void>(
      Multiplexer{{.qubits = MultiplexerOptions::MAX_QUBITS}}));
  EXPECT_THROW(static_cast<void>(
                   Multiplexer{{.qubits = MultiplexerOptions::MAX_QUBITS + 1}}),
               std::invalid_argument);
}

TEST(Multiplexer, GivesTheUniformControlDistribution) {
  const Multiplexer benchmark{{.qubits = 3}};
  const auto high = (2. + std::numbers::sqrt2) / 16.;
  const auto low = (2. - std::numbers::sqrt2) / 16.;

  EXPECT_NEAR(benchmark.probability("000"), 0.25, 1e-15);
  EXPECT_NEAR(benchmark.probability("001"), 0., 1e-15);
  EXPECT_NEAR(benchmark.probability("010"), high, 1e-15);
  EXPECT_NEAR(benchmark.probability("011"), low, 1e-15);
  EXPECT_NEAR(benchmark.probability("100"), 0.125, 1e-15);
  EXPECT_NEAR(benchmark.probability("101"), 0.125, 1e-15);
  EXPECT_NEAR(benchmark.probability("110"), low, 1e-15);
  EXPECT_NEAR(benchmark.probability("111"), high, 1e-15);

  double total = 0.;
  for (const auto* outcome :
       {"000", "001", "010", "011", "100", "101", "110", "111"}) {
    total += benchmark.probability(outcome);
  }
  EXPECT_NEAR(total, 1., 1e-15);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00x")),
               std::invalid_argument);
}

TEST(Multiplexer, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const Multiplexer benchmark{{.qubits = 3}};
  const auto evaluation = benchmark.evaluate({{"000", 100}});
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.75);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 0.25);
  EXPECT_FALSE(evaluation.successProbability);
}

TEST(Multiplexer, KeepsTheLargestUniformControlWeightRepresentable) {
  const Multiplexer benchmark{{.qubits = MultiplexerOptions::MAX_QUBITS}};
  EXPECT_GT(
      benchmark.probability(std::string(MultiplexerOptions::MAX_QUBITS, '0')),
      0.);
}

} // namespace
