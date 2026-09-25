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

#include "JSONTestUtils.hpp"
#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <string>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(Teleportation, ChecksTheTeleportedState) {
  const auto benchmark = Teleportation{};
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("0")), 1.);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("1")), 0.);
  EXPECT_EQ(::mqt::test::errorKind([&] { return benchmark.probability("00"); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] { return benchmark.probability("x"); }),
            ::mqt::ErrorCategory::InvalidArgument);

  const auto exact = ::mqt::test::value(benchmark.evaluate({{"0", 8}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_EQ(exact.successProbability, 1.);

  const auto noisy =
      ::mqt::test::value(benchmark.evaluate({{"0", 6}, {"1", 2}}));
  EXPECT_DOUBLE_EQ(noisy.totalVariationDistance, 0.25);
  EXPECT_DOUBLE_EQ(noisy.squaredHellingerFidelity, 0.75);
  EXPECT_EQ(noisy.successProbability, 0.75);
}

TEST(Teleportation, RoundTripsJSON) {
  const auto parsed =
      ::mqt::test::value(teleportationFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"teleportation","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"teleportation","parameters":{},"schema_version":1})");

  const auto benchmark = Teleportation{};
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(
                ::mqt::test::value(teleportationFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(manifest)),
            "teleportation");
  EXPECT_NE(manifest.find("\"model\":\"teleportation\""), std::string::npos);
  EXPECT_NE(manifest.find("\"parameters\":{}"), std::string::npos);
}

TEST(Teleportation, UsesSemanticCaseIds) {
  EXPECT_EQ(caseId(Teleportation{}), "sha256-de1348477e2604539b963a28bc19f5d3"
                                     "ed27ed86fc6608366bbc6eb9b55855f6");
}

TEST(Teleportation, DescribesJSONSchema) {
  EXPECT_NE(
      ::mqt::test::value(describeBenchmarkJSON("teleportation"))
          .find(
              R"("parameters":{"additionalProperties":false,"properties":{},"type":"object"})"),
      std::string::npos);
}

TEST(Teleportation, RejectsInvalidJSONParameters) {
  expectInvalidJSON(
      [] {
        static_cast<void>(teleportationFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"teleportation","parameters":{"qubits":3}})"));
      },
      "unknown key 'qubits'");
}

TEST(Teleportation, EvaluatesCountsFromJSON) {
  const auto evaluation = ::mqt::test::value(
      evaluateJSON(toManifestJSON(Teleportation{}),
                   R"({"schema_version":1,"counts":{"0":8}})"));
  EXPECT_NE(evaluation.find("\"success_probability\":1.0"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);
  EXPECT_NE(evaluation.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);
}

} // namespace mqt::bench
