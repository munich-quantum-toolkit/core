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
#include "bench/QAOA.hpp"

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(QAOA, ValidatesTheConfiguredInstance) {
  EXPECT_THROW(static_cast<void>(QAOA{{.qubits = 2}}), std::invalid_argument);
  EXPECT_NO_THROW(static_cast<void>(QAOA{{.qubits = 3}}));
  const auto maximum = static_cast<size_t>(std::numeric_limits<int64_t>::max());
  EXPECT_NO_THROW(static_cast<void>(QAOA{{.qubits = maximum}}));
  if (maximum < std::numeric_limits<size_t>::max()) {
    EXPECT_THROW(static_cast<void>(QAOA{{.qubits = maximum + 1}}),
                 std::invalid_argument);
  }
  EXPECT_THROW(static_cast<void>(QAOA{{.qubits = 3, .depth = 0}}),
               std::invalid_argument);
  EXPECT_NO_THROW(
      static_cast<void>(QAOA{{.qubits = 3, .depth = QAOAOptions::MAX_DEPTH}}));
  EXPECT_THROW(static_cast<void>(
                   QAOA{{.qubits = 3, .depth = QAOAOptions::MAX_DEPTH + 1}}),
               std::invalid_argument);
}

TEST(QAOA, CountsEveryCycleEdge) {
  const QAOA triangle{{.qubits = 3}};
  EXPECT_EQ(triangle.cutSize("000"), 0);
  EXPECT_EQ(triangle.cutSize("001"), 2);
  EXPECT_EQ(triangle.cutSize("010"), 2);
  const QAOA square{{.qubits = 4}};
  EXPECT_EQ(square.cutSize("0000"), 0);
  EXPECT_EQ(square.cutSize("0001"), 2);
  EXPECT_EQ(square.cutSize("0011"), 2);
  EXPECT_EQ(square.cutSize("0101"), 4);
  EXPECT_EQ(square.cutSize("1010"), 4);
  EXPECT_THROW(static_cast<void>(square.cutSize("010")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(square.cutSize("010x")),
               std::invalid_argument);
}

TEST(QAOA, ScoresCutsBeyondMachineWordWidths) {
  constexpr size_t qubits = 513;
  const QAOA benchmark{{.qubits = qubits}};
  auto outcome = std::string(qubits, '0');
  for (size_t index = 1; index < qubits; index += 2) {
    outcome[index] = '1';
  }
  EXPECT_EQ(benchmark.cutSize(outcome), 512);
  const auto evaluation = benchmark.evaluate({{outcome, 1}});
  EXPECT_DOUBLE_EQ(evaluation.meanCut, 512.);
  EXPECT_DOUBLE_EQ(evaluation.approximationRatio, 1.);
}

TEST(QAOA, EvaluatesCutsAgainstTheCycleOptimum) {
  const auto triangle = QAOA{{.qubits = 3}}.evaluate({
      {"000", 1},
      {"010", 3},
  });
  EXPECT_DOUBLE_EQ(triangle.meanCut, 1.5);
  EXPECT_DOUBLE_EQ(triangle.approximationRatio, 0.75);
  const auto square = QAOA{{.qubits = 4}}.evaluate({
      {"0000", 1},
      {"0101", 3},
  });
  EXPECT_DOUBLE_EQ(square.meanCut, 3.);
  EXPECT_DOUBLE_EQ(square.approximationRatio, 0.75);
}

TEST(QAOA, AvoidsOverflowWhileWeightingCuts) {
  const auto evaluation = QAOA{{.qubits = 4}}.evaluate({
      {"0101", std::numeric_limits<size_t>::max()},
  });
  EXPECT_DOUBLE_EQ(evaluation.meanCut, 4.);
  EXPECT_DOUBLE_EQ(evaluation.approximationRatio, 1.);
}

TEST(QAOA, KeepsRoundedOptimalScoresWithinTheirBounds) {
  const QAOA benchmark{{.qubits = 3}};
  const auto maximum = std::numeric_limits<size_t>::max();
  const auto evaluation = benchmark.evaluate({
      {"001", maximum / 3},
      {"010", maximum / 5},
      {"100", maximum / 5},
  });
  EXPECT_DOUBLE_EQ(evaluation.meanCut, 2.);
  EXPECT_LE(evaluation.meanCut, 2.);
  EXPECT_DOUBLE_EQ(evaluation.approximationRatio, 1.);
  EXPECT_LE(evaluation.approximationRatio, 1.);
  static_cast<void>(llvm::cantFail(evaluationToJSON(
      caseId(benchmark), (maximum / 3) + (2 * (maximum / 5)), evaluation)));
}

TEST(QAOA, ValidatesEveryCountKey) {
  const QAOA benchmark{{.qubits = 4}};
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({{"0000", 0}})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({
                   {"0000", 1},
                   {"00x0", 0},
               })),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({
                   {"0000", std::numeric_limits<size_t>::max()},
                   {"0101", 1},
               })),
               std::overflow_error);
}

TEST(QAOA, RoundTripsJSON) {
  const auto parsed = llvm::cantFail(qaoaFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qaoa","parameters":{"qubits":5,"depth":2}})"));
  EXPECT_EQ(parsed.options().qubits, 5);
  EXPECT_EQ(parsed.options().depth, 2);
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"qaoa","parameters":{"depth":2,"qubits":5},"schema_version":1})");
  const auto manifest = toManifestJSON(parsed);
  EXPECT_EQ(toManifestJSON(llvm::cantFail(qaoaFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(llvm::cantFail(benchmarkIdFromManifestJSON(manifest)), "qaoa");
  EXPECT_NE(manifest.find(R"("kind":"verification")"), std::string::npos);
  EXPECT_NE(manifest.find(R"("model":"qaoa_max_cut_cycle")"),
            std::string::npos);
  EXPECT_NE(caseId(parsed), caseId(QAOA{{.qubits = 5}}));
  auto changed = parsed.options();
  changed.qubits = 7;
  EXPECT_NE(caseId(parsed), caseId(QAOA(changed)));
}

TEST(QAOA, ResolvesDefaultDepth) {
  const auto parsed = llvm::cantFail(qaoaFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qaoa","parameters":{"qubits":3}})"));
  const QAOA defaults{{.qubits = 3}};
  EXPECT_EQ(toInstanceSpecificationJSON(parsed),
            toInstanceSpecificationJSON(defaults));
  EXPECT_EQ(caseId(parsed), caseId(defaults));
}

TEST(QAOA, DescribesJSONSchema) {
  const auto schema = llvm::cantFail(describeBenchmarkJSON("qaoa"));
  EXPECT_NE(schema.find(R"("minimum":3)"), std::string::npos);
  EXPECT_NE(schema.find(R"("maximum":9223372036854775807)"), std::string::npos);
  EXPECT_NE(schema.find(R"("depth":{"default":1,"maximum":)" +
                        std::to_string(QAOAOptions::MAX_DEPTH) +
                        R"(,"minimum":1,"type":"integer"})"),
            std::string::npos);
  EXPECT_NE(schema.find(R"("required":["qubits"])"), std::string::npos);
}

TEST(QAOA, RejectsInvalidJSONParameters) {
  for (const auto* parameters : {
           R"({"qubits":2})",
           R"({"qubits":3,"depth":0})",
           R"({"qubits":3,"depth":-1})",
           R"({"qubits":3,"depth":1.5})",
           R"({"qubits":3,"depth":true})",
       }) {
    const auto specification =
        std::string(R"({"schema_version":1,"benchmark":"qaoa","parameters":)") +
        parameters + "}";
    test::expectInvalidJSON(
        [&] { return qaoaFromInstanceSpecificationJSON(specification); }, "");
  }
  expectInvalidJSON(
      [] {
        return qaoaFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qaoa","parameters":{"qubits":3,"gammas":[1]}})");
      },
      "gammas");
}

TEST(QAOA, ReportsTheParameterLocationOnce) {
  expectInvalidJSON(
      [] {
        return qaoaFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qaoa","parameters":{}})",
            "case.json");
      },
      "case.json:$/parameters.qubits");
  expectInvalidJSON(
      [] {
        return qaoaFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qaoa","parameters":{"qubits":3,"depth":"x"}})",
            "case.json");
      },
      "case.json:$/parameters.depth");
}

TEST(QAOA, EvaluatesCountsFromJSON) {
  const auto evaluation = llvm::cantFail(
      evaluateJSON(toManifestJSON(QAOA{{.qubits = 4}}),
                   R"({"schema_version":1,"counts":{"0000":1,"0101":3}})"));
  EXPECT_NE(evaluation.find(R"("mean_cut":3.0)"), std::string::npos);
  EXPECT_NE(evaluation.find(R"("approximation_ratio":0.75)"),
            std::string::npos);
  EXPECT_EQ(evaluation.find("total_variation_distance"), std::string::npos);
  EXPECT_EQ(evaluation.find("squared_hellinger_fidelity"), std::string::npos);
}

TEST(QAOA, RejectsInvalidEvaluationMetrics) {
  const auto id = caseId(QAOA{{.qubits = 3}});
  test::expectInvalidJSON(
      [&] { return evaluationToJSON(id, 0, QAOAEvaluation{}); }, "");
  test::expectInvalidJSON(
      [&] { return evaluationToJSON("invalid", 1, QAOAEvaluation{}); }, "");
  test::expectInvalidJSON(
      [&] { return evaluationToJSON(id, 1, QAOAEvaluation{.meanCut = -1.}); },
      "");
  test::expectInvalidJSON(
      [&] {
        return evaluationToJSON(id, 1,
                                QAOAEvaluation{.approximationRatio = 1.1});
      },
      "");
  test::expectInvalidJSON(
      [&] {
        return evaluationToJSON(
            id, 1,
            QAOAEvaluation{
                .meanCut = std::numeric_limits<double>::quiet_NaN(),
            });
      },
      "");
}

} // namespace mqt::bench
