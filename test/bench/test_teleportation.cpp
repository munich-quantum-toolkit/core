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
#include "bench/Teleportation.hpp"

#include "gtest/gtest.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using mqt::bench::benchmarkIdFromManifestJSON;
using mqt::bench::caseId;
using mqt::bench::describeBenchmarkJSON;
using mqt::bench::evaluateJSON;
using mqt::bench::Output;
using mqt::bench::Teleportation;
using mqt::bench::teleportationFromInstanceSpecificationJSON;
using mqt::bench::teleportationFromManifestJSON;
using mqt::bench::toInstanceSpecificationJSON;
using mqt::bench::toManifestJSON;

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

TEST(Teleportation, ParsesInstanceSpecificationsAndRoundTripsManifests) {
  const auto parsed = teleportationFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"teleportation","parameters":{}})");
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"teleportation","parameters":{},"schema_version":1})");

  const Teleportation benchmark;
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(teleportationFromManifestJSON(manifest)), manifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(manifest), "teleportation");
  EXPECT_NE(manifest.find("\"model\":\"teleportation\""), std::string::npos);
  EXPECT_NE(manifest.find("\"parameters\":{}"), std::string::npos);
}

TEST(Teleportation, UsesAStableJSONCaseIdAndDescribesSchema) {
  EXPECT_EQ(caseId(Teleportation{}), "sha256-de1348477e2604539b963a28bc19f5d3"
                                     "ed27ed86fc6608366bbc6eb9b55855f6");
  EXPECT_NE(
      describeBenchmarkJSON("teleportation")
          .find(
              R"("parameters":{"additionalProperties":false,"properties":{},"type":"object"})"),
      std::string::npos);
}

TEST(Teleportation, RejectsUnknownJSONParameters) {
  try {
    static_cast<void>(teleportationFromInstanceSpecificationJSON(
        R"({"schema_version":1,"benchmark":"teleportation","parameters":{"qubits":3}})"));
    FAIL() << "Expected invalid JSON input";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string_view(error.what()).find("unknown key 'qubits'"),
              std::string_view::npos)
        << error.what();
  }
}

TEST(Teleportation, EvaluatesCountsFromJSON) {
  const auto evaluation =
      evaluateJSON(toManifestJSON(Teleportation{}),
                   R"({"schema_version":1,"counts":{"0":8}})");
  EXPECT_NE(evaluation.find("\"success_probability\":1.0"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);
  EXPECT_NE(evaluation.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);
}

} // namespace
