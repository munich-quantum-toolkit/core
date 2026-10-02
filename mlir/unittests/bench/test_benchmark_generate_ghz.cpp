/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/GHZ.hpp"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

namespace mqt::bench {

TEST(GenerateProgramTest, SamplesEveryGHZVariantAgainstReference) {
  for (const auto topology : {GHZTopology::Linear, GHZTopology::Star}) {
    for (const auto basis : {GHZBasis::Z, GHZBasis::X}) {
      SCOPED_TRACE(static_cast<int>(topology));
      SCOPED_TRACE(static_cast<int>(basis));
      test::expectSamplingMatchesReference(
          GHZ{{.qubits = 3, .topology = topology, .basis = basis}});
    }
  }
}

TEST(GenerateProgramTest, KeepsLargestGHZStructured) {
  for (const auto topology : {GHZTopology::Linear, GHZTopology::Star}) {
    auto program = generate(GHZ{{
        .qubits = GHZOptions::MAX_QUBITS,
        .topology = topology,
    }});
    ASSERT_TRUE(program);
    EXPECT_LT(test::countOperations(program->module()), 100U);
  }
}

} // namespace mqt::bench
