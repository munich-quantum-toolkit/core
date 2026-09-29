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
#include "bench/JSON.hpp"

#include "gtest/gtest.h"

#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using mqt::bench::benchmarkIdFromInstanceSpecificationJSON;
using mqt::bench::countsFromJSON;
using mqt::bench::describeBenchmarkJSON;
using mqt::bench::Evaluation;
using mqt::bench::evaluationToJSON;
using mqt::bench::listBenchmarksJSON;

void expectInvalid(const std::function<void()>& operation,
                   const std::string_view diagnostic) {
  try {
    operation();
    FAIL() << "Expected invalid JSON input";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(diagnostic), std::string::npos)
        << error.what();
  }
}

TEST(BenchmarkJSON, RejectsDuplicateInstanceEnvelopeKey) {
  expectInvalid(
      [] {
        static_cast<void>(benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","benchmark":"qpe","parameters":{"qubits":2}})",
            "duplicate.json"));
      },
      "duplicate key 'benchmark'");
}

TEST(BenchmarkJSON, ListsBenchmarksAndRejectsUnknownSchemas) {
  EXPECT_EQ(
      listBenchmarksJSON(),
      R"({"benchmarks":[{"definition_version":1,"id":"bv"},{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"grover-weak-measurement"},{"definition_version":1,"id":"magic-state-distillation"},{"definition_version":1,"id":"modular-multiplier"},{"definition_version":1,"id":"multiplexer"},{"definition_version":1,"id":"qft"},{"definition_version":1,"id":"qft-adder"},{"definition_version":1,"id":"qpe"},{"definition_version":1,"id":"repeat-until-success"},{"definition_version":1,"id":"shor"},{"definition_version":1,"id":"teleportation"},{"definition_version":1,"id":"w-state"}],"schema_version":1})");
  EXPECT_THROW(static_cast<void>(describeBenchmarkJSON("unknown")),
               std::invalid_argument);
}

TEST(BenchmarkJSON, ParsesCountsAndValidatesEvaluations) {
  const auto counts =
      countsFromJSON(R"({"counts":{"11":50,"00":50},"schema_version":1})");
  EXPECT_EQ(counts.at("00"), 50);
  EXPECT_EQ(counts.at("11"), 50);

  expectInvalid(
      [] {
        static_cast<void>(
            countsFromJSON(R"({"schema_version":1,"counts":{"0":1,"0":2}})"));
      },
      "duplicate key '0'");
  expectInvalid(
      [] {
        static_cast<void>(
            countsFromJSON(R"({"schema_version":1,"counts":{"0x":1}})"));
      },
      "bitstrings");
  expectInvalid(
      [] {
        static_cast<void>(
            countsFromJSON(R"({"schema_version":1,"counts":{"00":0}})"));
      },
      "must be positive");
  const auto validCaseId = "sha256-" + std::string(64, '0');
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   "not-a-case", 1,
                   Evaluation{.totalVariationDistance = 0.,
                              .squaredHellingerFidelity = 1.,
                              .successProbability = std::nullopt})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   validCaseId, 1,
                   Evaluation{.totalVariationDistance =
                                  std::numeric_limits<double>::quiet_NaN(),
                              .squaredHellingerFidelity = 1.,
                              .successProbability = std::nullopt})),
               std::invalid_argument);
}

} // namespace
