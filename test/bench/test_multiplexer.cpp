/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Error.hpp"
#include "bench/Evaluation.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/TestUtils.hpp"

#include "gtest/gtest.h"

#include <numbers>
#include <string>

namespace test = mqt::bench::test;

namespace {

using mqt::bench::Multiplexer;
using mqt::bench::MultiplexerOptions;
using mqt::bench::Output;

TEST(Multiplexer, StoresTheTotalQubitCountAndOutput) {
  const auto benchmark = test::value(Multiplexer::create({.qubits = 7}));
  EXPECT_EQ(benchmark.options().qubits, 7);
  EXPECT_EQ(benchmark.output(), (Output{"result", 7}));
}

TEST(Multiplexer, ValidatesTheConfiguredInstance) {
  EXPECT_EQ(test::errorKind(Multiplexer::create({.qubits = 1})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_NO_THROW(
      static_cast<void>(test::value(Multiplexer::create({.qubits = 2}))));
  EXPECT_NO_THROW(static_cast<void>(test::value(
      Multiplexer::create({.qubits = MultiplexerOptions::MAX_QUBITS}))));
  EXPECT_EQ(test::errorKind(Multiplexer::create(
                {.qubits = MultiplexerOptions::MAX_QUBITS + 1})),
            mqt::bench::Error::Kind::InvalidArgument);
}

TEST(Multiplexer, GivesTheUniformControlDistribution) {
  const auto benchmark = test::value(Multiplexer::create({.qubits = 3}));
  const auto high = (2. + std::numbers::sqrt2) / 16.;
  const auto low = (2. - std::numbers::sqrt2) / 16.;

  EXPECT_NEAR(test::value(benchmark.probability("000")), 0.25, 1e-15);
  EXPECT_NEAR(test::value(benchmark.probability("001")), 0., 1e-15);
  EXPECT_NEAR(test::value(benchmark.probability("010")), high, 1e-15);
  EXPECT_NEAR(test::value(benchmark.probability("011")), low, 1e-15);
  EXPECT_NEAR(test::value(benchmark.probability("100")), 0.125, 1e-15);
  EXPECT_NEAR(test::value(benchmark.probability("101")), 0.125, 1e-15);
  EXPECT_NEAR(test::value(benchmark.probability("110")), low, 1e-15);
  EXPECT_NEAR(test::value(benchmark.probability("111")), high, 1e-15);

  double total = 0.;
  for (const auto* outcome :
       {"000", "001", "010", "011", "100", "101", "110", "111"}) {
    total += test::value(benchmark.probability(outcome));
  }
  EXPECT_NEAR(total, 1., 1e-15);
  EXPECT_EQ(test::errorKind(benchmark.probability("00")),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(benchmark.probability("00x")),
            mqt::bench::Error::Kind::InvalidArgument);
}

TEST(Multiplexer, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const auto benchmark = test::value(Multiplexer::create({.qubits = 3}));
  const auto evaluation = test::value(benchmark.evaluate({{"000", 100}}));
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.75);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 0.25);
  EXPECT_FALSE(evaluation.successProbability);
}

TEST(Multiplexer, KeepsTheLargestUniformControlWeightRepresentable) {
  const auto benchmark = test::value(
      Multiplexer::create({.qubits = MultiplexerOptions::MAX_QUBITS}));
  EXPECT_GT(test::value(benchmark.probability(
                std::string(MultiplexerOptions::MAX_QUBITS, '0'))),
            0.);
}

} // namespace
