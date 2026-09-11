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

#include "gtest/gtest.h"

#include <stdexcept>

namespace {

using mqt::bench::Output;
using mqt::bench::Teleportation;

TEST(Teleportation, ChecksTheTeleportedState) {
  const Teleportation benchmark;
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
  EXPECT_DOUBLE_EQ(benchmark.probability("0"), 1.);
  EXPECT_DOUBLE_EQ(benchmark.probability("1"), 0.);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("x")),
               std::invalid_argument);

  const auto exact = benchmark.evaluate({{"0", 8}});
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_EQ(exact.successProbability, 1.);

  const auto noisy = benchmark.evaluate({{"0", 6}, {"1", 2}});
  EXPECT_DOUBLE_EQ(noisy.totalVariationDistance, 0.25);
  EXPECT_DOUBLE_EQ(noisy.squaredHellingerFidelity, 0.75);
  EXPECT_EQ(noisy.successProbability, 0.75);
}

} // namespace
