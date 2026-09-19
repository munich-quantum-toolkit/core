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
#include "bench/Teleportation.hpp"
#include "bench/TestUtils.hpp"

#include "gtest/gtest.h"

namespace test = mqt::bench::test;

namespace {

using mqt::bench::Output;
using mqt::bench::Teleportation;

TEST(Teleportation, ChecksTheTeleportedState) {
  const Teleportation benchmark;
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("0")), 1.);
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("1")), 0.);
  EXPECT_EQ(test::errorKind(benchmark.probability("00")),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(benchmark.probability("x")),
            mqt::bench::Error::Kind::InvalidArgument);

  const auto exact = test::value(benchmark.evaluate({{"0", 8}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_EQ(exact.successProbability, 1.);

  const auto noisy = test::value(benchmark.evaluate({{"0", 6}, {"1", 2}}));
  EXPECT_DOUBLE_EQ(noisy.totalVariationDistance, 0.25);
  EXPECT_DOUBLE_EQ(noisy.squaredHellingerFidelity, 0.75);
  EXPECT_EQ(noisy.successProbability, 0.75);
}

} // namespace
