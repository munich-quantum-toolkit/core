/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestUtils.h"
#include "bench/Grover.hpp"

#include <gtest/gtest.h>

namespace mqt::bench {

TEST(GenerateProgramTest, SamplesGroverAgainstReference) {
  test::expectSamplingMatchesReference(
      Grover{{.markedBitstring = "01", .iterations = 1}});
}

} // namespace mqt::bench
