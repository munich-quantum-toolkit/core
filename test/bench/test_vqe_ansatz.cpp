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
#include "bench/VQEAnsatz.hpp"

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(VQEAnsatz, ValidatesTheConfiguredInstance) {
  EXPECT_THROW(static_cast<void>(VQEAnsatz{{.qubits = 0}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(VQEAnsatz{{.qubits = 1, .depth = 0}}),
               std::invalid_argument);
  EXPECT_NO_THROW(static_cast<void>(VQEAnsatz{{.qubits = 1}}));
  EXPECT_NO_THROW(
      static_cast<void>(VQEAnsatz{{.qubits = 1024, .depth = 1000}}));
}

TEST(VQEAnsatz, ValidatesParameterDimensionsWithoutOverflow) {
  const auto maximum = static_cast<size_t>(std::numeric_limits<int64_t>::max());
  const VQEAnsatz deepest{{.qubits = 1, .depth = maximum - 1}};
  EXPECT_EQ(deepest.numParameters(), maximum);
  EXPECT_THROW(static_cast<void>(VQEAnsatz{{.qubits = 1, .depth = maximum}}),
               std::invalid_argument);

  const VQEAnsatz widest{{.qubits = maximum / 2}};
  EXPECT_EQ(widest.numParameters(), maximum - 1);
  EXPECT_THROW(static_cast<void>(VQEAnsatz{{.qubits = (maximum / 2) + 1}}),
               std::invalid_argument);

  const VQEAnsatz largestProduct{{.qubits = 3, .depth = (maximum / 3) - 1}};
  EXPECT_EQ(largestProduct.numParameters(), maximum - (maximum % 3));
  EXPECT_THROW(
      static_cast<void>(VQEAnsatz{{.qubits = 3, .depth = maximum / 3}}),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(
                   VQEAnsatz{{.qubits = std::numeric_limits<size_t>::max()}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(VQEAnsatz{
                   {.qubits = 1, .depth = std::numeric_limits<size_t>::max()}}),
               std::invalid_argument);
}

TEST(VQEAnsatz, DescribesRotationsAndOutput) {
  const VQEAnsatz benchmark{{.qubits = 5, .depth = 3}};
  EXPECT_EQ(benchmark.options().qubits, 5);
  EXPECT_EQ(benchmark.options().depth, 3);
  EXPECT_EQ(benchmark.output().name, "result");
  EXPECT_EQ(benchmark.output().width, 5);
  EXPECT_EQ(benchmark.numParameters(), 20);
  EXPECT_EQ((VQEAnsatz{{.qubits = 1}}.numParameters()), 2);
}

TEST(VQEAnsatz, RoundTripsJSON) {
  const auto parsed = vqeAnsatzFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"vqe-ansatz","parameters":{"qubits":5,"depth":2}})");
  EXPECT_EQ(parsed.options().qubits, 5);
  EXPECT_EQ(parsed.options().depth, 2);
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"vqe-ansatz","parameters":{"depth":2,"qubits":5},"schema_version":1})");
  const auto manifest = toManifestJSON(parsed);
  EXPECT_EQ(toManifestJSON(vqeAnsatzFromManifestJSON(manifest)), manifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(manifest), "vqe-ansatz");
  EXPECT_NE(manifest.find(R"("kind":"none")"), std::string::npos);
  EXPECT_NE(caseId(parsed), caseId(VQEAnsatz{{.qubits = 5}}));
  EXPECT_NE(caseId(parsed), caseId(VQEAnsatz{{.qubits = 7, .depth = 2}}));
}

TEST(VQEAnsatz, ResolvesDefaultDepth) {
  const auto parsed = vqeAnsatzFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"vqe-ansatz","parameters":{"qubits":3}})");
  const VQEAnsatz defaults{{.qubits = 3}};
  EXPECT_EQ(toInstanceSpecificationJSON(parsed),
            toInstanceSpecificationJSON(defaults));
  EXPECT_EQ(caseId(parsed), caseId(defaults));
}

TEST(VQEAnsatz, DescribesJSONSchema) {
  const auto schema = describeBenchmarkJSON("vqe-ansatz");
  EXPECT_NE(schema.find(R"("minimum":1)"), std::string::npos);
  EXPECT_NE(schema.find(R"("depth":{"default":1,)"), std::string::npos);
  EXPECT_NE(schema.find(R"("required":["qubits"])"), std::string::npos);
}

TEST(VQEAnsatz, RejectsInvalidJSONParameters) {
  for (const auto* parameters : {
           R"({})",
           R"({"qubits":0})",
           R"({"qubits":1,"depth":0})",
           R"({"qubits":3,"depth":4611686018427387903})",
       }) {
    const auto specification =
        std::string(
            R"({"schema_version":1,"benchmark":"vqe-ansatz","parameters":)") +
        parameters + "}";
    EXPECT_THROW(static_cast<void>(
                     vqeAnsatzFromInstanceSpecificationJSON(specification)),
                 std::invalid_argument);
  }
  expectInvalidJSON(
      [] {
        static_cast<void>(vqeAnsatzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"vqe-ansatz","parameters":{"qubits":3,"thetas":[1]}})"));
      },
      "unknown key 'thetas'");
}

TEST(VQEAnsatz, DoesNotDefineAnEvaluation) {
  expectInvalidJSON(
      [] {
        static_cast<void>(
            evaluateJSON(toManifestJSON(VQEAnsatz{{.qubits = 1}}),
                         R"({"schema_version":1,"counts":{"0":1}})"));
      },
      "does not define a built-in evaluation");
}

} // namespace mqt::bench
