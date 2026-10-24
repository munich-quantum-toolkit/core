/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/JSON.hpp"
#include "bench/QFTAdder.hpp"

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <string>
#include <system_error>

namespace mqt::bench {

TEST(QFTAdder, PreservesConfiguredOperandsAndOverflow) {
  for (const auto method :
       {QFTAdderMethod::Register, QFTAdderMethod::Constant}) {
    for (const auto overflow :
         {QFTAdderOverflow::Wrap, QFTAdderOverflow::Carry}) {
      const auto benchmark = llvm::cantFail(QFTAdder::create({
          .addend = "011",
          .accumulator = "110",
          .method = method,
          .overflow = overflow,
      }));
      const auto* const sum =
          overflow == QFTAdderOverflow::Carry ? "1001" : "001";
      const auto expected =
          (method == QFTAdderMethod::Register ? std::string{"011"}
                                              : std::string{}) +
          sum;
      EXPECT_EQ(benchmark.options().addend, "011");
      EXPECT_EQ(benchmark.options().accumulator, "110");
      EXPECT_EQ(benchmark.output().width, expected.size());
      EXPECT_EQ(benchmark.expectedResult(), expected);
      EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability(expected)), 1.);
      EXPECT_EQ(llvm::cantFail(benchmark.evaluate({{expected, 16}}))
                    .successProbability,
                1.);
    }
  }
}

TEST(QFTAdder, KeepsLeadingZerosAndRejectsUnsupportedInputs) {
  const auto zero = llvm::cantFail(QFTAdder::create({
      .addend = "000",
      .accumulator = "000",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }));
  EXPECT_EQ(zero.expectedResult(), "0000");
  for (const auto& options : {
           QFTAdderOptions{.addend = "", .accumulator = ""},
           QFTAdderOptions{.addend = "01", .accumulator = "1"},
           QFTAdderOptions{.addend = "x", .accumulator = "0"},
           QFTAdderOptions{.addend = "1", .accumulator = "+"},
           QFTAdderOptions{
               .addend = "+",
               .accumulator = "0",
               .method = QFTAdderMethod::Constant,
           },
       }) {
    test::expectError(QFTAdder::create(options), std::errc::invalid_argument);
  }
  test::expectError(zero.probability("000"), std::errc::invalid_argument);
  test::expectError(zero.probability("000x"), std::errc::invalid_argument);
}

TEST(QFTAdder, ScoresTheCorrelatedSuperposition) {
  const auto benchmark =
      llvm::cantFail(QFTAdder::create({.addend = "1+0", .accumulator = "001"}));
  EXPECT_FALSE(benchmark.expectedResult());
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("100101")), 0.5);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("110111")), 0.5);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("000001")), 0.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("100100")), 0.);
  const auto exact =
      llvm::cantFail(benchmark.evaluate({{"100101", 8}, {"110111", 8}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(exact.successProbability);
  const auto biased = llvm::cantFail(benchmark.evaluate({{"100101", 16}}));
  EXPECT_DOUBLE_EQ(biased.totalVariationDistance, 0.5);
  EXPECT_DOUBLE_EQ(biased.squaredHellingerFidelity, 0.5);
}

TEST(QFTAdder, BoundsTheSumWidthAndKeepsReferenceWeightsRepresentable) {
  const auto width = QFTAdderOptions::MAX_QUBITS;
  const auto accumulator = std::string(width - 1, '0') + "1";
  const auto maximum = llvm::cantFail(QFTAdder::create(
      {.addend = std::string(width, '+'), .accumulator = accumulator}));
  EXPECT_GT(llvm::cantFail(maximum.probability(std::string(width, '1') +
                                               std::string(width, '0'))),
            0.);
  test::expectError(QFTAdder::create({
                        .addend = std::string(width, '1'),
                        .accumulator = accumulator,
                        .overflow = QFTAdderOverflow::Carry,
                    }),
                    std::errc::invalid_argument);
  static_cast<void>(llvm::cantFail(QFTAdder::create({
      .addend = std::string(width - 1, '1'),
      .accumulator = accumulator.substr(1),
      .overflow = QFTAdderOverflow::Carry,
  })));
  test::expectError(QFTAdder::create({
                        .addend = std::string(width + 1, '0'),
                        .accumulator = std::string(width + 1, '0'),
                    }),
                    std::errc::invalid_argument);
}

TEST(QFTAdder, RoundTripsJSON) {
  const auto defaults = llvm::cantFail(qftAdderFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft-adder","parameters":{"addend":"+++","accumulator":"001"}})"));
  EXPECT_EQ(defaults.options().method, QFTAdderMethod::Register);
  EXPECT_EQ(defaults.options().overflow, QFTAdderOverflow::Wrap);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"qft-adder","parameters":{"accumulator":"001","addend":"+++","method":"register","overflow":"wrap"},"schema_version":1})");

  const auto benchmark =
      llvm::cantFail(QFTAdder::create({.addend = "+++", .accumulator = "001"}));
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(llvm::cantFail(qftAdderFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(llvm::cantFail(benchmarkIdFromManifestJSON(manifest)), "qft-adder");
  EXPECT_NE(manifest.find("\"model\":\"qft_adder\""), std::string::npos);
  EXPECT_NE(manifest.find("\"width\":6"), std::string::npos);

  const auto constant = llvm::cantFail(QFTAdder::create({
      .addend = "110",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }));
  const auto constantManifest = toManifestJSON(constant);
  EXPECT_EQ(toManifestJSON(
                llvm::cantFail(qftAdderFromManifestJSON(constantManifest))),
            constantManifest);
}

TEST(QFTAdder, UsesSemanticCaseIds) {
  EXPECT_EQ(caseId(llvm::cantFail(
                QFTAdder::create({.addend = "+++", .accumulator = "001"}))),
            caseId(llvm::cantFail(
                QFTAdder::create({.addend = "+++",
                                  .accumulator = "001",
                                  .method = QFTAdderMethod::Register,
                                  .overflow = QFTAdderOverflow::Wrap}))));
  EXPECT_NE(caseId(llvm::cantFail(
                QFTAdder::create({.addend = "+++", .accumulator = "001"}))),
            caseId(llvm::cantFail(
                QFTAdder::create({.addend = "++++", .accumulator = "0001"}))));
  EXPECT_NE(caseId(llvm::cantFail(
                QFTAdder::create({.addend = "110",
                                  .accumulator = "001",
                                  .method = QFTAdderMethod::Constant,
                                  .overflow = QFTAdderOverflow::Carry}))),
            caseId(llvm::cantFail(
                QFTAdder::create({.addend = "110", .accumulator = "001"}))));
}

TEST(QFTAdder, DescribesJSONSchema) {
  const auto schema = llvm::cantFail(describeBenchmarkJSON("qft-adder"));
  EXPECT_NE(schema.find("\"maxLength\":1024"), std::string::npos);
  EXPECT_NE(schema.find("\"minLength\":1"), std::string::npos);
}

TEST(QFTAdder, RejectsInvalidJSONParameters) {
  for (const auto* parameters : {
           R"({"addend":"","accumulator":""})",
           R"({"addend":"1","accumulator":"00"})",
           R"({"addend":"+","accumulator":"0","method":"constant"})",
           R"({"addend":"1","accumulator":"0","method":"unknown"})",
           R"({"addend":"1","accumulator":"0","overflow":"unknown"})",
           R"({"addend":"1","accumulator":"0","overflow":true})",
           R"({"addend":"1","accumulator":"0","qubits":1})",
           R"({"addend":"1"})",
       }) {
    const auto instance =
        std::string{
            R"({"schema_version":1,"benchmark":"qft-adder","parameters":)"} +
        parameters + "}";
    test::expectInvalidJSON(
        [&] { return qftAdderFromInstanceSpecificationJSON(instance); }, "");
  }
}

TEST(QFTAdder, EvaluatesCountsFromJSON) {
  const auto benchmark =
      llvm::cantFail(QFTAdder::create({.addend = "++", .accumulator = "01"}));
  const auto evaluation = llvm::cantFail(evaluateJSON(
      toManifestJSON(benchmark),
      R"({"schema_version":1,"counts":{"0001":1,"0110":1,"1011":1,"1100":1}})"));
  EXPECT_NE(evaluation.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":0"),
            std::string::npos);

  const auto constant = llvm::cantFail(QFTAdder::create({
      .addend = "110",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }));
  const auto constantEvaluation = llvm::cantFail(
      evaluateJSON(toManifestJSON(constant),
                   R"({"schema_version":1,"counts":{"0111":8,"0110":2}})"));
  EXPECT_NE(constantEvaluation.find("\"success_probability\":0.8"),
            std::string::npos);
}

} // namespace mqt::bench
