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

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <string>
#include <system_error>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(Teleportation, ChecksTheTeleportedState) {
  const Teleportation benchmark;
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("0")), 1.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("1")), 0.);
  test::expectError(benchmark.probability("00"), std::errc::invalid_argument);
  test::expectError(benchmark.probability("x"), std::errc::invalid_argument);

  const auto exact = llvm::cantFail(benchmark.evaluate({{"0", 8}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_EQ(exact.successProbability, 1.);

  const auto noisy = llvm::cantFail(benchmark.evaluate({{"0", 6}, {"1", 2}}));
  EXPECT_DOUBLE_EQ(noisy.totalVariationDistance, 0.25);
  EXPECT_DOUBLE_EQ(noisy.squaredHellingerFidelity, 0.75);
  EXPECT_EQ(noisy.successProbability, 0.75);
}

TEST(Teleportation, RoundTripsJSON) {
  const auto parsed = llvm::cantFail(teleportationFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"teleportation","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"teleportation","parameters":{},"schema_version":1})");

  const Teleportation benchmark;
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(
      toManifestJSON(llvm::cantFail(teleportationFromManifestJSON(manifest))),
      manifest);
  EXPECT_EQ(llvm::cantFail(benchmarkIdFromManifestJSON(manifest)),
            "teleportation");
  EXPECT_NE(manifest.find("\"model\":\"teleportation\""), std::string::npos);
  EXPECT_NE(manifest.find("\"parameters\":{}"), std::string::npos);
}

TEST(Teleportation, UsesSemanticCaseIds) {
  EXPECT_EQ(caseId(Teleportation{}),
            "sha256-"
            "de1348477e2604539b963a28bc19f5d3ed27ed86fc6608366bbc6eb9b55855f6");
}

TEST(Teleportation, DescribesJSONSchema) {
  EXPECT_NE(
      llvm::cantFail(describeBenchmarkJSON("teleportation"))
          .find(
              R"("parameters":{"additionalProperties":false,"properties":{},"type":"object"})"),
      std::string::npos);
}

TEST(Teleportation, RejectsInvalidJSONParameters) {
  expectInvalidJSON(
      [] {
        return teleportationFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"teleportation","parameters":{"qubits":3}})");
      },
      "qubits");
}

TEST(Teleportation, EvaluatesCountsFromJSON) {
  const auto evaluation =
      llvm::cantFail(evaluateJSON(toManifestJSON(Teleportation{}),
                                  R"({"schema_version":1,"counts":{"0":8}})"));
  EXPECT_NE(evaluation.find("\"success_probability\":1"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":0"),
            std::string::npos);
  EXPECT_NE(evaluation.find("\"squared_hellinger_fidelity\":1"),
            std::string::npos);
}

} // namespace mqt::bench
