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
#include "bench/Teleportation.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string_view>

namespace {

using mqt::bench::Output;
using mqt::bench::Teleportation;

TEST(Teleportation, HasTheFixedOutput) {
  const Teleportation benchmark;
  EXPECT_EQ(benchmark.output(), (Output{"result", 3}));
}

TEST(Teleportation, HasTheExactUniformReference) {
  const Teleportation benchmark;
  for (const std::string_view outcome :
       {"000", "001", "010", "011", "100", "101", "110", "111"}) {
    EXPECT_DOUBLE_EQ(benchmark.probability(outcome), 1. / 8.);
  }
  EXPECT_THROW(static_cast<void>(benchmark.probability("00")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00x")),
               std::invalid_argument);
}

TEST(Teleportation, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const Teleportation benchmark;
  const auto exact = benchmark.evaluate({
      {"000", 1},
      {"001", 1},
      {"010", 1},
      {"011", 1},
      {"100", 1},
      {"101", 1},
      {"110", 1},
      {"111", 1},
  });
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(exact.successProbability);

  const auto biased = benchmark.evaluate({{"000", 8}});
  EXPECT_DOUBLE_EQ(biased.totalVariationDistance, 7. / 8.);
  EXPECT_DOUBLE_EQ(biased.squaredHellingerFidelity, 1. / 8.);
  EXPECT_FALSE(biased.successProbability);
}

} // namespace
