/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Grover.hpp"
#include "bench/TestUtils.hpp"

#include "TestUtils.h"

#include "gtest/gtest.h"

namespace mqt::bench {

TEST(GenerateProgramTest, SamplesGroverAgainstReference) {
  test::expectSamplingMatchesReference(
      test::value(Grover::create({.markedBitstring = "01", .iterations = 1})));
}

} // namespace mqt::bench
