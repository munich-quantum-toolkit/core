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

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <string>

namespace mqt::bench {

TEST(QFTAdder, PreservesConfiguredOperandsAndOverflow) {
  for (const auto method :
       {QFTAdderMethod::Register, QFTAdderMethod::Constant}) {
    for (const auto overflow :
         {QFTAdderOverflow::Wrap, QFTAdderOverflow::Carry}) {
      const auto benchmark = ::mqt::test::value(QFTAdder::create({
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
      EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability(expected)), 1.);
      EXPECT_EQ(::mqt::test::value(benchmark.evaluate({{expected, 16}}))
                    .successProbability,
                1.);
    }
  }
}

TEST(QFTAdder, KeepsLeadingZerosAndRejectsUnsupportedInputs) {
  const auto zero = ::mqt::test::value(QFTAdder::create({
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
    EXPECT_EQ(::mqt::test::errorKind([&] { return QFTAdder::create(options); }),
              ::mqt::ErrorCategory::InvalidArgument);
  }
  EXPECT_EQ(::mqt::test::errorKind([&] { return zero.probability("000"); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] { return zero.probability("000x"); }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(QFTAdder, ScoresTheCorrelatedSuperposition) {
  const auto benchmark = ::mqt::test::value(
      QFTAdder::create({.addend = "1+0", .accumulator = "001"}));
  EXPECT_FALSE(benchmark.expectedResult());
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("100101")), 0.5);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("110111")), 0.5);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("000001")), 0.);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("100100")), 0.);
  const auto exact =
      ::mqt::test::value(benchmark.evaluate({{"100101", 8}, {"110111", 8}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(exact.successProbability);
  const auto biased = ::mqt::test::value(benchmark.evaluate({{"100101", 16}}));
  EXPECT_DOUBLE_EQ(biased.totalVariationDistance, 0.5);
  EXPECT_DOUBLE_EQ(biased.squaredHellingerFidelity, 0.5);
}

TEST(QFTAdder, BoundsTheSumWidthAndKeepsReferenceWeightsRepresentable) {
  const auto width = QFTAdderOptions::MAX_QUBITS;
  const auto accumulator = std::string(width - 1, '0') + "1";
  const auto maximum = ::mqt::test::value(QFTAdder::create(
      {.addend = std::string(width, '+'), .accumulator = accumulator}));
  EXPECT_GT(::mqt::test::value(maximum.probability(std::string(width, '1') +
                                                   std::string(width, '0'))),
            0.);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QFTAdder::create({.addend = std::string(width, '1'),
                                       .accumulator = accumulator,
                                       .overflow = QFTAdderOverflow::Carry});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_NO_THROW(static_cast<void>(::mqt::test::value(
      QFTAdder::create({.addend = std::string(width - 1, '1'),
                        .accumulator = accumulator.substr(1),
                        .overflow = QFTAdderOverflow::Carry}))));
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QFTAdder::create(
                  {.addend = std::string(width + 1, '0'),
                   .accumulator = std::string(width + 1, '0')});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(QFTAdder, RoundTripsJSON) {
  const auto defaults = ::mqt::test::value(qftAdderFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft-adder","parameters":{"addend":"+++","accumulator":"001"}})"));
  EXPECT_EQ(defaults.options().method, QFTAdderMethod::Register);
  EXPECT_EQ(defaults.options().overflow, QFTAdderOverflow::Wrap);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"qft-adder","parameters":{"accumulator":"001","addend":"+++","method":"register","overflow":"wrap"},"schema_version":1})");

  const auto benchmark = ::mqt::test::value(
      QFTAdder::create({.addend = "+++", .accumulator = "001"}));
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(
      toManifestJSON(::mqt::test::value(qftAdderFromManifestJSON(manifest))),
      manifest);
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(manifest)),
            "qft-adder");
  EXPECT_NE(manifest.find("\"model\":\"qft_adder\""), std::string::npos);
  EXPECT_NE(manifest.find("\"width\":6"), std::string::npos);

  const auto constant = ::mqt::test::value(QFTAdder::create({
      .addend = "110",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }));
  const auto constantManifest = toManifestJSON(constant);
  EXPECT_EQ(toManifestJSON(
                ::mqt::test::value(qftAdderFromManifestJSON(constantManifest))),
            constantManifest);
}

TEST(QFTAdder, UsesSemanticCaseIds) {
  EXPECT_EQ(caseId(::mqt::test::value(
                QFTAdder::create({.addend = "+++", .accumulator = "001"}))),
            caseId(::mqt::test::value(
                QFTAdder::create({.addend = "+++",
                                  .accumulator = "001",
                                  .method = QFTAdderMethod::Register,
                                  .overflow = QFTAdderOverflow::Wrap}))));
  EXPECT_NE(caseId(::mqt::test::value(
                QFTAdder::create({.addend = "+++", .accumulator = "001"}))),
            caseId(::mqt::test::value(
                QFTAdder::create({.addend = "++++", .accumulator = "0001"}))));
  EXPECT_NE(caseId(::mqt::test::value(
                QFTAdder::create({.addend = "110",
                                  .accumulator = "001",
                                  .method = QFTAdderMethod::Constant,
                                  .overflow = QFTAdderOverflow::Carry}))),
            caseId(::mqt::test::value(
                QFTAdder::create({.addend = "110", .accumulator = "001"}))));
}

TEST(QFTAdder, DescribesJSONSchema) {
  const auto schema = ::mqt::test::value(describeBenchmarkJSON("qft-adder"));
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
    EXPECT_EQ(::mqt::test::errorKind([&] {
                return qftAdderFromInstanceSpecificationJSON(instance);
              }),
              ::mqt::ErrorCategory::InvalidArgument);
  }
}

TEST(QFTAdder, EvaluatesCountsFromJSON) {
  const auto benchmark = ::mqt::test::value(
      QFTAdder::create({.addend = "++", .accumulator = "01"}));
  const auto evaluation = ::mqt::test::value(evaluateJSON(
      toManifestJSON(benchmark),
      R"({"schema_version":1,"counts":{"0001":1,"0110":1,"1011":1,"1100":1}})"));
  EXPECT_NE(evaluation.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);

  const auto constant = ::mqt::test::value(QFTAdder::create({
      .addend = "110",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }));
  const auto constantEvaluation = ::mqt::test::value(
      evaluateJSON(toManifestJSON(constant),
                   R"({"schema_version":1,"counts":{"0111":8,"0110":2}})"));
  EXPECT_NE(constantEvaluation.find("\"success_probability\":0.8"),
            std::string::npos);
}

} // namespace mqt::bench
