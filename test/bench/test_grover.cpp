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
#include "bench/Grover.hpp"
#include "bench/JSON.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace mqt::bench {

TEST(Grover, ResolvesTheDefaultIterationCountOnce) {
  const Grover grover{{.markedBitstring = "10"}};
  ASSERT_TRUE(grover.options().iterations.has_value());
  EXPECT_EQ(*grover.options().iterations, 1);
  EXPECT_EQ(grover.qubits(), 2);
  EXPECT_EQ(grover.output(), (Output{"result", 2}));
  EXPECT_DOUBLE_EQ(grover.probability("10"), 1.);
  EXPECT_NEAR(grover.probability("00"), 0., 1e-31);
}

TEST(Grover, ResolvesLargeDefaultIterationCountsWithoutProbabilityRounding) {
  const Grover grover{{.markedBitstring = std::string(62, '0')}};
  EXPECT_EQ(*grover.options().iterations, 1'686'629'713);
}

TEST(Grover, AcceptsAnExplicitZeroIterationCount) {
  const Grover grover{{.markedBitstring = "01", .iterations = 0}};
  EXPECT_EQ(*grover.options().iterations, 0);
  EXPECT_DOUBLE_EQ(grover.probability("01"), 0.25);
  EXPECT_DOUBLE_EQ(grover.probability("11"), 0.25);
}

TEST(Grover, RejectsUnsupportedOptions) {
  EXPECT_THROW(static_cast<void>(Grover{{.markedBitstring = "0"}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Grover{{.markedBitstring = "0x"}}),
               std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(Grover{{.markedBitstring = std::string(63, '0')}}),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(Grover{{
          .markedBitstring = "00",
          .iterations =
              static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1,
      }}),
      std::invalid_argument);
}

TEST(Grover, EvaluatesTheMarkedOutcomeAsSuccess) {
  const Grover grover{{.markedBitstring = "01", .iterations = 0}};
  const auto evaluation =
      grover.evaluate({{"00", 25}, {"01", 25}, {"10", 25}, {"11", 25}});
  EXPECT_NEAR(evaluation.totalVariationDistance, 0., 1e-15);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 1.);
  ASSERT_TRUE(evaluation.successProbability.has_value());
  EXPECT_DOUBLE_EQ(*evaluation.successProbability, 0.25);
}

TEST(Grover, RoundTripsJSON) {
  const auto defaults = groverFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"grover","parameters":{"marked_bitstring":"10"}})");
  ASSERT_TRUE(defaults.options().iterations);
  EXPECT_EQ(*defaults.options().iterations, 1);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"grover","parameters":{"iterations":1,"marked_bitstring":"10"},"schema_version":1})");

  const Grover configured{{.markedBitstring = "001", .iterations = 2}};
  const auto manifest = toManifestJSON(configured);
  EXPECT_EQ(toManifestJSON(groverFromManifestJSON(manifest)), manifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(manifest), "grover");
  EXPECT_NE(manifest.find("\"success_outcome\":\"001\""), std::string::npos);
}

TEST(Grover, DescribesJSONSchema) {
  EXPECT_NE(describeBenchmarkJSON("grover").find("\"maxLength\":62"),
            std::string::npos);
}

} // namespace mqt::bench
