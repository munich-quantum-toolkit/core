/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"
#include "bench/Error.hpp"
#include "bench/Evaluation.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/JSON.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Teleportation.hpp"
#include "bench/TestUtils.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace test = mqt::bench::test;

namespace {

using mqt::bench::benchmarkIdFromInstanceSpecificationJSON;
using mqt::bench::benchmarkIdFromManifestJSON;
using mqt::bench::BV;
using mqt::bench::bvFromInstanceSpecificationJSON;
using mqt::bench::bvFromManifestJSON;
using mqt::bench::BVMethod;
using mqt::bench::caseId;
using mqt::bench::countsFromJSON;
using mqt::bench::describeBenchmarkJSON;
using mqt::bench::evaluateJSON;
using mqt::bench::Evaluation;
using mqt::bench::evaluationToJSON;
using mqt::bench::GHZ;
using mqt::bench::GHZBasis;
using mqt::bench::ghzFromInstanceSpecificationJSON;
using mqt::bench::ghzFromManifestJSON;
using mqt::bench::GHZTopology;
using mqt::bench::Grover;
using mqt::bench::groverFromInstanceSpecificationJSON;
using mqt::bench::groverFromManifestJSON;
using mqt::bench::listBenchmarksJSON;
using mqt::bench::ModularMultiplier;
using mqt::bench::modularMultiplierFromInstanceSpecificationJSON;
using mqt::bench::modularMultiplierFromManifestJSON;
using mqt::bench::Multiplexer;
using mqt::bench::multiplexerFromInstanceSpecificationJSON;
using mqt::bench::multiplexerFromManifestJSON;
using mqt::bench::Phase;
using mqt::bench::QFT;
using mqt::bench::QFTAdder;
using mqt::bench::qftAdderFromInstanceSpecificationJSON;
using mqt::bench::qftAdderFromManifestJSON;
using mqt::bench::QFTAdderMethod;
using mqt::bench::QFTAdderOverflow;
using mqt::bench::qftFromInstanceSpecificationJSON;
using mqt::bench::qftFromManifestJSON;
using mqt::bench::QFTMethod;
using mqt::bench::QPE;
using mqt::bench::qpeFromInstanceSpecificationJSON;
using mqt::bench::qpeFromManifestJSON;
using mqt::bench::QPEMethod;
using mqt::bench::RepeatUntilSuccess;
using mqt::bench::repeatUntilSuccessFromInstanceSpecificationJSON;
using mqt::bench::repeatUntilSuccessFromManifestJSON;
using mqt::bench::Teleportation;
using mqt::bench::teleportationFromInstanceSpecificationJSON;
using mqt::bench::teleportationFromManifestJSON;
using mqt::bench::toInstanceSpecificationJSON;
using mqt::bench::toManifestJSON;

template <class Action>
void expectInvalid(const Action& operation, const std::string_view diagnostic) {
  const auto result = operation();
  const auto* error = std::get_if<mqt::bench::Error>(&result);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->kind, mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_NE(error->message.find(diagnostic), std::string::npos)
      << error->message;
}

TEST(BenchmarkJSON, ReturnsNormalizedInstancesAndInputDiagnostics) {
  const auto benchmark = test::value(GHZ::create({.qubits = 3}));
  auto result = mqt::bench::parseInstanceSpecificationJSON(
      toInstanceSpecificationJSON(benchmark));
  ASSERT_TRUE(std::holds_alternative<mqt::bench::ParsedBenchmark>(result));
  const auto& parsed = std::get<mqt::bench::ParsedBenchmark>(result);
  EXPECT_TRUE(std::holds_alternative<GHZ>(parsed.instance));
  EXPECT_EQ(parsed.benchmarkId, "ghz");
  EXPECT_EQ(parsed.caseId, test::value(caseId(benchmark)));
  EXPECT_EQ(parsed.manifestJSON, test::value(toManifestJSON(benchmark)));

  for (
      const auto* invalid : {
          "{",
          R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":0}})",
      }) {
    result = mqt::bench::parseInstanceSpecificationJSON(invalid, "input.json");
    ASSERT_TRUE(std::holds_alternative<mqt::bench::Error>(result));
    EXPECT_NE(std::get<mqt::bench::Error>(result).message.find("input.json"),
              std::string::npos);
  }
  auto description = mqt::bench::describeBenchmarkJSON("unknown");
  ASSERT_TRUE(std::holds_alternative<mqt::bench::Error>(description));
  EXPECT_NE(std::get<mqt::bench::Error>(description).message.find("unknown"),
            std::string::npos);
  auto evaluation =
      mqt::bench::evaluateJSON(test::value(toManifestJSON(benchmark)), "{}",
                               "manifest.json", "counts.json");
  ASSERT_TRUE(std::holds_alternative<mqt::bench::Error>(evaluation));
  EXPECT_NE(std::get<mqt::bench::Error>(evaluation).message.find("counts.json"),
            std::string::npos);
}

TEST(BenchmarkJSON, ResolvesModularMultiplierInputs) {
  const auto benchmark = test::value(modularMultiplierFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{"multiplier":"011","modulus":"101","multiplicand":"111"}})"));
  EXPECT_EQ(benchmark.options().control, '1');
  EXPECT_EQ(benchmark.expectedResult(), "11110001");
  EXPECT_NE(test::value(toManifestJSON(benchmark))
                .find("\"success_outcome\":\"11110001\""),
            std::string::npos);
  EXPECT_EQ(test::value(modularMultiplierFromManifestJSON(
                            test::value(toManifestJSON(benchmark))))
                .expectedResult(),
            benchmark.expectedResult());
  EXPECT_EQ(test::value(caseId(benchmark)),
            test::value(caseId(
                test::value(ModularMultiplier::create({.multiplier = "011",
                                                       .modulus = "101",
                                                       .multiplicand = "111",
                                                       .control = '1'})))));
  EXPECT_NE(
      test::value(caseId(benchmark)),
      test::value(caseId(test::value(ModularMultiplier::create(
          {.multiplier = "011", .modulus = "101", .multiplicand = "110"})))));
  EXPECT_NE(test::value(caseId(benchmark)),
            test::value(caseId(
                test::value(ModularMultiplier::create({.multiplier = "011",
                                                       .modulus = "101",
                                                       .multiplicand = "111",
                                                       .control = '0'})))));
  for (const auto* control : {"\"\"", "\"11\"", "\"x\"", "true", "1"}) {
    const auto instance =
        std::string(
            R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{"multiplier":"011","modulus":"101","multiplicand":"111","control":)") +
        control + "}}";
    EXPECT_EQ(test::errorKind(
                  modularMultiplierFromInstanceSpecificationJSON(instance)),
              mqt::bench::Error::Kind::InvalidArgument);
  }
  EXPECT_EQ(
      test::errorKind(modularMultiplierFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{"multiplier":"011","modulus":"101"}})")),
      mqt::bench::Error::Kind::InvalidArgument);
}

TEST(BenchmarkJSON, ValidatesRepeatUntilSuccessWidth) {
  for (const auto* width : {"0", "1000001", "-1", "1.5", "true", "\"5\""}) {
    EXPECT_EQ(
        test::errorKind(repeatUntilSuccessFromInstanceSpecificationJSON(
            std::string(
                R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"data_qubits":)") +
            width + "}}")),
        mqt::bench::Error::Kind::InvalidArgument);
  }
  const auto benchmark =
      test::value(RepeatUntilSuccess::create({.dataQubits = 32}));
  EXPECT_EQ(test::value(repeatUntilSuccessFromManifestJSON(
                            test::value(toManifestJSON(benchmark))))
                .options()
                .dataQubits,
            32U);
}

TEST(BenchmarkJSON,
     ParsesInstanceSpecificationsAndSerializesResolvedParameters) {
  const auto bv = test::value(bvFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"bv","parameters":{"hidden_bitstring":"101"}})"));
  EXPECT_EQ(bv.options().method, BVMethod::Static);
  EXPECT_EQ(
      toInstanceSpecificationJSON(bv),
      R"({"benchmark":"bv","parameters":{"hidden_bitstring":"101","method":"static"},"schema_version":1})");

  const auto modularMultiplier =
      test::value(modularMultiplierFromInstanceSpecificationJSON(
          R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"011"},"schema_version":1})"));
  EXPECT_EQ(modularMultiplier.options().multiplier, "011");
  EXPECT_EQ(modularMultiplier.options().modulus, "101");
  EXPECT_EQ(
      toInstanceSpecificationJSON(modularMultiplier),
      R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"011"},"schema_version":1})");

  const auto ghz = test::value(ghzFromInstanceSpecificationJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})"));
  EXPECT_EQ(ghz.options().topology, GHZTopology::Linear);
  EXPECT_EQ(ghz.options().basis, GHZBasis::Z);
  EXPECT_EQ(
      toInstanceSpecificationJSON(ghz),
      R"({"benchmark":"ghz","parameters":{"basis":"z","qubits":3,"topology":"linear"},"schema_version":1})");

  const auto grover = test::value(groverFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"grover","parameters":{"marked_bitstring":"10"}})"));
  ASSERT_TRUE(grover.options().iterations);
  EXPECT_EQ(*grover.options().iterations, 1);
  EXPECT_EQ(
      toInstanceSpecificationJSON(grover),
      R"({"benchmark":"grover","parameters":{"iterations":1,"marked_bitstring":"10"},"schema_version":1})");

  const auto multiplexer = test::value(multiplexerFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7}})"));
  EXPECT_EQ(multiplexer.options().qubits, 7);
  EXPECT_EQ(
      toInstanceSpecificationJSON(multiplexer),
      R"({"benchmark":"multiplexer","parameters":{"qubits":7},"schema_version":1})");

  const auto qft = test::value(qftFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft","parameters":{"qubits":4,"period_exponent":2}})"));
  EXPECT_EQ(qft.options().method, QFTMethod::Standard);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qft),
      R"({"benchmark":"qft","parameters":{"method":"standard","period_exponent":2,"qubits":4},"schema_version":1})");

  const auto qftAdder = test::value(qftAdderFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft-adder","parameters":{"addend":"+++","accumulator":"001"}})"));
  EXPECT_EQ(qftAdder.options().method, QFTAdderMethod::Register);
  EXPECT_EQ(qftAdder.options().overflow, QFTAdderOverflow::Wrap);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qftAdder),
      R"({"benchmark":"qft-adder","parameters":{"accumulator":"001","addend":"+++","method":"register","overflow":"wrap"},"schema_version":1})");

  const auto qpe = test::value(qpeFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":4,"phase":{"numerator":10,"denominator":8},"method":"iterative"}})"));
  EXPECT_EQ(qpe.options().phase, test::value(Phase::create(1, 4)));
  EXPECT_EQ(qpe.options().method, QPEMethod::Iterative);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qpe),
      R"({"benchmark":"qpe","parameters":{"method":"iterative","phase":{"denominator":4,"numerator":1},"precision":4},"schema_version":1})");

  const auto repeatUntilSuccess =
      test::value(repeatUntilSuccessFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(repeatUntilSuccess),
      R"({"benchmark":"repeat-until-success","parameters":{"data_qubits":1},"schema_version":1})");

  const auto teleportation =
      test::value(teleportationFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"teleportation","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(teleportation),
      R"({"benchmark":"teleportation","parameters":{},"schema_version":1})");
}

TEST(BenchmarkJSON, RoundTripsSelfCheckingManifests) {
  const auto bv = test::value(
      BV::create({.hiddenBitstring = "101", .method = BVMethod::Dynamic}));
  const auto modularMultiplier = test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }));
  const auto ghz = test::value(GHZ::create(
      {.qubits = 4, .topology = GHZTopology::Star, .basis = GHZBasis::X}));
  const auto grover =
      test::value(Grover::create({.markedBitstring = "001", .iterations = 2}));
  const auto multiplexer = test::value(Multiplexer::create({.qubits = 7}));
  const auto qft = test::value(QFT::create(
      {.qubits = 4, .periodExponent = 2, .method = QFTMethod::Semiclassical}));
  const auto qftAdder =
      test::value(QFTAdder::create({.addend = "+++", .accumulator = "001"}));
  const auto qpe = test::value(QPE::create({
      .precision = 5,
      .phase = test::value(Phase::create(1, 3)),
      .method = QPEMethod::Iterative,
  }));
  const auto repeatUntilSuccess = test::value(RepeatUntilSuccess::create());
  const Teleportation teleportation;

  const auto bvManifest = test::value(toManifestJSON(bv));
  const auto modularMultiplierManifest =
      test::value(toManifestJSON(modularMultiplier));
  const auto ghzManifest = test::value(toManifestJSON(ghz));
  const auto groverManifest = test::value(toManifestJSON(grover));
  const auto multiplexerManifest = test::value(toManifestJSON(multiplexer));
  const auto qftManifest = test::value(toManifestJSON(qft));
  const auto qftAdderManifest = test::value(toManifestJSON(qftAdder));
  const auto qpeManifest = test::value(toManifestJSON(qpe));
  const auto repeatUntilSuccessManifest =
      test::value(toManifestJSON(repeatUntilSuccess));
  const auto teleportationManifest = test::value(toManifestJSON(teleportation));
  EXPECT_EQ(
      test::value(toManifestJSON(test::value(bvFromManifestJSON(bvManifest)))),
      bvManifest);
  EXPECT_EQ(test::value(toManifestJSON(test::value(
                modularMultiplierFromManifestJSON(modularMultiplierManifest)))),
            modularMultiplierManifest);
  EXPECT_EQ(test::value(
                toManifestJSON(test::value(ghzFromManifestJSON(ghzManifest)))),
            ghzManifest);
  EXPECT_EQ(test::value(toManifestJSON(
                test::value(groverFromManifestJSON(groverManifest)))),
            groverManifest);
  EXPECT_EQ(test::value(toManifestJSON(
                test::value(multiplexerFromManifestJSON(multiplexerManifest)))),
            multiplexerManifest);
  EXPECT_EQ(test::value(
                toManifestJSON(test::value(qftFromManifestJSON(qftManifest)))),
            qftManifest);
  EXPECT_EQ(test::value(toManifestJSON(
                test::value(qftAdderFromManifestJSON(qftAdderManifest)))),
            qftAdderManifest);
  EXPECT_EQ(test::value(
                toManifestJSON(test::value(qpeFromManifestJSON(qpeManifest)))),
            qpeManifest);
  EXPECT_EQ(
      test::value(toManifestJSON(test::value(
          repeatUntilSuccessFromManifestJSON(repeatUntilSuccessManifest)))),
      repeatUntilSuccessManifest);
  EXPECT_EQ(test::value(toManifestJSON(test::value(
                teleportationFromManifestJSON(teleportationManifest)))),
            teleportationManifest);
  EXPECT_EQ(test::value(benchmarkIdFromManifestJSON(bvManifest)), "bv");
  EXPECT_EQ(test::value(benchmarkIdFromManifestJSON(modularMultiplierManifest)),
            "modular-multiplier");
  EXPECT_EQ(test::value(benchmarkIdFromManifestJSON(ghzManifest)), "ghz");
  EXPECT_EQ(test::value(benchmarkIdFromManifestJSON(groverManifest)), "grover");
  EXPECT_EQ(test::value(benchmarkIdFromManifestJSON(multiplexerManifest)),
            "multiplexer");
  EXPECT_EQ(test::value(benchmarkIdFromManifestJSON(qftManifest)), "qft");
  EXPECT_EQ(test::value(benchmarkIdFromManifestJSON(qftAdderManifest)),
            "qft-adder");
  EXPECT_EQ(test::value(benchmarkIdFromManifestJSON(qpeManifest)), "qpe");
  EXPECT_EQ(
      test::value(benchmarkIdFromManifestJSON(repeatUntilSuccessManifest)),
      "repeat-until-success");
  EXPECT_EQ(test::value(benchmarkIdFromManifestJSON(teleportationManifest)),
            "teleportation");
  EXPECT_NE(
      ghzManifest.find("\"case_id\":\"" + test::value(caseId(ghz)) + "\""),
      std::string::npos);
  EXPECT_NE(groverManifest.find("\"success_outcome\":\"001\""),
            std::string::npos);
  EXPECT_NE(modularMultiplierManifest.find("\"model\":\"modular_multiplier\""),
            std::string::npos);
  EXPECT_NE(modularMultiplierManifest.find("\"width\":8"), std::string::npos);
  EXPECT_NE(multiplexerManifest.find("\"model\":\"multiplexer\""),
            std::string::npos);
  EXPECT_NE(qftAdderManifest.find("\"model\":\"qft_adder\""),
            std::string::npos);
  EXPECT_NE(qftAdderManifest.find("\"width\":6"), std::string::npos);
  EXPECT_EQ(qpeManifest.find("0.333"), std::string::npos);
  EXPECT_NE(
      repeatUntilSuccessManifest.find("\"model\":\"repeat_until_success\""),
      std::string::npos);
  EXPECT_NE(
      repeatUntilSuccessManifest.find("\"parameters\":{\"data_qubits\":1}"),
      std::string::npos);
  EXPECT_NE(teleportationManifest.find("\"model\":\"teleportation\""),
            std::string::npos);
  EXPECT_NE(teleportationManifest.find("\"parameters\":{}"), std::string::npos);
}

TEST(BenchmarkJSON, UsesStableSemanticCaseIds) {
  const auto linear = test::value(GHZ::create({.qubits = 3}));
  const auto same = test::value(GHZ::create({.qubits = 3}));
  const auto star =
      test::value(GHZ::create({.qubits = 3, .topology = GHZTopology::Star}));
  EXPECT_EQ(test::value(caseId(linear)), test::value(caseId(same)));
  EXPECT_NE(test::value(caseId(linear)), test::value(caseId(star)));
  EXPECT_NE(
      test::value(caseId(test::value(BV::create({.hiddenBitstring = "1"})))),
      test::value(caseId(test::value(
          BV::create({.hiddenBitstring = "1", .method = BVMethod::Dynamic})))));
  EXPECT_EQ(test::value(caseId(
                test::value(ModularMultiplier::create({.multiplier = "011",
                                                       .modulus = "101",
                                                       .multiplicand = "+++",
                                                       .control = '+'})))),
            test::value(caseId(
                test::value(ModularMultiplier::create({.multiplier = "011",
                                                       .modulus = "101",
                                                       .multiplicand = "+++",
                                                       .control = '+'})))));
  EXPECT_NE(test::value(caseId(
                test::value(ModularMultiplier::create({.multiplier = "011",
                                                       .modulus = "101",
                                                       .multiplicand = "+++",
                                                       .control = '+'})))),
            test::value(caseId(
                test::value(ModularMultiplier::create({.multiplier = "001",
                                                       .modulus = "101",
                                                       .multiplicand = "+++",
                                                       .control = '+'})))));
  EXPECT_NE(test::value(caseId(
                test::value(QFT::create({.qubits = 3, .periodExponent = 1})))),
            test::value(caseId(test::value(
                QFT::create({.qubits = 3,
                             .periodExponent = 1,
                             .method = QFTMethod::Semiclassical})))));
  EXPECT_EQ(test::value(caseId(test::value(
                QFTAdder::create({.addend = "+++", .accumulator = "001"})))),
            test::value(caseId(test::value(
                QFTAdder::create({.addend = "+++", .accumulator = "001"})))));
  EXPECT_NE(test::value(caseId(test::value(
                QFTAdder::create({.addend = "+++", .accumulator = "001"})))),
            test::value(caseId(test::value(
                QFTAdder::create({.addend = "++++", .accumulator = "0001"})))));
  EXPECT_EQ(
      test::value(caseId(test::value(Multiplexer::create({.qubits = 7})))),
      test::value(caseId(test::value(Multiplexer::create({.qubits = 7})))));
  EXPECT_NE(
      test::value(caseId(test::value(Multiplexer::create({.qubits = 7})))),
      test::value(caseId(test::value(Multiplexer::create({.qubits = 6})))));
  EXPECT_EQ(test::value(caseId(test::value(RepeatUntilSuccess::create()))),
            test::value(caseId(
                test::value(RepeatUntilSuccess::create({.dataQubits = 1})))));
  EXPECT_NE(test::value(caseId(test::value(RepeatUntilSuccess::create()))),
            test::value(caseId(
                test::value(RepeatUntilSuccess::create({.dataQubits = 5})))));
  EXPECT_EQ(test::value(caseId(Teleportation{})),
            "sha256-de1348477e2604539b963a28bc19f5d3"
            "ed27ed86fc6608366bbc6eb9b55855f6");
  EXPECT_EQ(test::value(caseId(linear)),
            "sha256-a222c0c57bcecb4f5e7ea72bab439683"
            "92861a52c5cb7c9c13aeaffffa059a65");
}

TEST(BenchmarkJSON,
     RejectsDuplicateUnknownAndMistypedInstanceSpecificationValues) {
  expectInvalid(
      [] {
        return benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","benchmark":"qpe","parameters":{"qubits":2}})",
            "duplicate.json");
      },
      "duplicate key 'benchmark'");
  expectInvalid(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"qubits":3}})");
      },
      "duplicate key 'qubits'");
  expectInvalid(
      [] {
        return qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4,"numerator":2}}})");
      },
      "duplicate key 'numerator'");
  expectInvalid(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2},"extra":true})");
      },
      "unknown key 'extra'");
  expectInvalid(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"extra":true}})");
      },
      "unknown key 'extra'");
  expectInvalid(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2.5}})");
      },
      "encoded as an integer");
  expectInvalid(
      [] {
        return qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":9007199254740993.0,"denominator":9007199254740994}}})");
      },
      "encoded as an integer");
  expectInvalid(
      [] {
        return qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":18446744073709551615,"phase":{"numerator":1,"denominator":4}}})");
      },
      "between 1 and 1000000");
  expectInvalid(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"basis":"x","qubits":1076}})");
      },
      "between 1 and 1075");
  expectInvalid(
      [] {
        return multiplexerFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":1}})");
      },
      "between 2 and 1024");
  expectInvalid(
      [] {
        return multiplexerFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7,"angles":[]}})");
      },
      "unknown key 'angles'");
  expectInvalid(
      [] {
        return modularMultiplierFromInstanceSpecificationJSON(
            R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"011","extra":true},"schema_version":1})");
      },
      "unknown key 'extra'");
  expectInvalid(
      [] {
        return modularMultiplierFromInstanceSpecificationJSON(
            R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"1001","multiplicand":"+++","multiplier":"011"},"schema_version":1})");
      },
      "equal widths");
  expectInvalid(
      [] {
        return modularMultiplierFromInstanceSpecificationJSON(
            R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"010","multiplicand":"+++","multiplier":"011"},"schema_version":1})");
      },
      "canonical");
  expectInvalid(
      [] {
        return modularMultiplierFromInstanceSpecificationJSON(
            R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"101"},"schema_version":1})");
      },
      "0 < a < N");
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
    EXPECT_EQ(test::errorKind(qftAdderFromInstanceSpecificationJSON(instance)),
              mqt::bench::Error::Kind::InvalidArgument);
  }
  expectInvalid(
      [] {
        return repeatUntilSuccessFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"attempts":1}})");
      },
      "unknown key 'attempts'");
  expectInvalid(
      [] {
        return teleportationFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"teleportation","parameters":{"qubits":3}})");
      },
      "unknown key 'qubits'");
  expectInvalid(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"new","parameters":{}})");
      },
      "unsupported benchmark 'new'");
  expectInvalid(
      [] {
        return qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":0}}})");
      },
      "denominator must not be zero");
}

TEST(BenchmarkJSON, RejectsAnInstanceSpecificationForAnotherConcreteType) {
  expectInvalid(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4}}})");
      },
      "must be 'ghz'");
}

TEST(BenchmarkJSON, RejectsAlteredOrUnresolvedManifestData) {
  const auto ghz = test::value(GHZ::create({.qubits = 3}));
  auto changedOutput = test::value(toManifestJSON(ghz));
  const auto width = changedOutput.find("\"width\":3");
  ASSERT_NE(width, std::string::npos);
  changedOutput.replace(width, std::string("\"width\":3").size(),
                        "\"width\":2");
  expectInvalid([&] { return ghzFromManifestJSON(changedOutput); },
                "does not match");

  auto changedNumericKind = test::value(toManifestJSON(ghz));
  const auto integerWidth = changedNumericKind.find(R"("width":3)");
  ASSERT_NE(integerWidth, std::string::npos);
  changedNumericKind.replace(integerWidth, std::string(R"("width":3)").size(),
                             R"("width":3.0)");
  expectInvalid([&] { return ghzFromManifestJSON(changedNumericKind); },
                "does not match");

  auto changedId = test::value(toManifestJSON(ghz));
  const auto digest = changedId.find("sha256-");
  ASSERT_NE(digest, std::string::npos);
  changedId[digest + 7U] = changedId[digest + 7U] == '0' ? '1' : '0';
  expectInvalid([&] { return ghzFromManifestJSON(changedId); }, "case ID");

  auto unresolved = test::value(toManifestJSON(ghz));
  const auto basis = unresolved.find(R"("basis":"z",)");
  ASSERT_NE(basis, std::string::npos);
  unresolved.erase(basis, std::string(R"("basis":"z",)").size());
  expectInvalid([&] { return ghzFromManifestJSON(unresolved); },
                "resolved benchmark instance");
}

TEST(BenchmarkJSON, ListsBenchmarksAndDescribesStandardSchemas) {
  EXPECT_EQ(
      listBenchmarksJSON(),
      R"({"benchmarks":[{"definition_version":1,"id":"bv"},{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"modular-multiplier"},{"definition_version":1,"id":"multiplexer"},{"definition_version":1,"id":"qft"},{"definition_version":1,"id":"qft-adder"},{"definition_version":1,"id":"qpe"},{"definition_version":1,"id":"repeat-until-success"},{"definition_version":1,"id":"teleportation"}],"schema_version":1})");
  const auto bv = test::value(describeBenchmarkJSON("bv"));
  const auto modularMultiplier =
      test::value(describeBenchmarkJSON("modular-multiplier"));
  const auto ghz = test::value(describeBenchmarkJSON("ghz"));
  const auto grover = test::value(describeBenchmarkJSON("grover"));
  const auto multiplexer = test::value(describeBenchmarkJSON("multiplexer"));
  const auto qft = test::value(describeBenchmarkJSON("qft"));
  const auto qftAdder = test::value(describeBenchmarkJSON("qft-adder"));
  const auto qpe = test::value(describeBenchmarkJSON("qpe"));
  const auto repeatUntilSuccess =
      test::value(describeBenchmarkJSON("repeat-until-success"));
  const auto teleportation =
      test::value(describeBenchmarkJSON("teleportation"));
  EXPECT_NE(ghz.find("https://json-schema.org/draft/2020-12/schema"),
            std::string::npos);
  EXPECT_NE(ghz.find("\"additionalProperties\":false"), std::string::npos);
  EXPECT_NE(ghz.find("\"maximum\":1000000"), std::string::npos);
  EXPECT_NE(ghz.find("\"maximum\":1075"), std::string::npos);
  EXPECT_NE(bv.find("\"dynamic\""), std::string::npos);
  EXPECT_NE(modularMultiplier.find("\"maxLength\":63"), std::string::npos);
  EXPECT_NE(modularMultiplier.find("\"pattern\":\"^1[01]+$\""),
            std::string::npos);
  EXPECT_NE(grover.find("\"maxLength\":62"), std::string::npos);
  EXPECT_NE(multiplexer.find("\"maximum\":1024"), std::string::npos);
  EXPECT_NE(multiplexer.find("\"minimum\":2"), std::string::npos);
  EXPECT_NE(qft.find("\"period_exponent\""), std::string::npos);
  EXPECT_NE(qftAdder.find("\"maxLength\":1024"), std::string::npos);
  EXPECT_NE(qftAdder.find("\"minLength\":1"), std::string::npos);
  EXPECT_NE(qpe.find("\"iterative\""), std::string::npos);
  EXPECT_NE(
      repeatUntilSuccess.find(
          R"("data_qubits":{"default":1,"maximum":1000000,"minimum":1,"type":"integer"})"),
      std::string::npos);
  EXPECT_NE(
      teleportation.find(
          R"("parameters":{"additionalProperties":false,"properties":{},"type":"object"})"),
      std::string::npos);
  EXPECT_EQ(test::errorKind(describeBenchmarkJSON("unknown")),
            mqt::bench::Error::Kind::InvalidArgument);
}

TEST(BenchmarkJSON, ParsesCountsAndSerializesEvaluations) {
  const auto counts = test::value(
      countsFromJSON(R"({"counts":{"11":50,"00":50},"schema_version":1})"));
  EXPECT_EQ(counts.at("00"), 50);
  EXPECT_EQ(counts.at("11"), 50);

  const auto ghz = test::value(GHZ::create({.qubits = 2}));
  const auto serialized = test::value(evaluationToJSON(
      test::value(caseId(ghz)), 100, test::value(ghz.evaluate(counts))));
  EXPECT_NE(serialized.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(serialized.find("\"total_variation_distance\":0.0"),
            std::string::npos);

  const auto bv = test::value(BV::create({.hiddenBitstring = "11"}));
  const auto generic = test::value(
      evaluateJSON(test::value(toManifestJSON(bv)),
                   R"({"schema_version":1,"counts":{"11":8,"00":2}})"));
  EXPECT_NE(generic.find("\"success_probability\":0.8"), std::string::npos);

  const auto multiplexer = test::value(Multiplexer::create({.qubits = 2}));
  const auto multiplexerEvaluation = test::value(
      evaluateJSON(test::value(toManifestJSON(multiplexer)),
                   R"({"schema_version":1,"counts":{"00":8,"01":2}})"));
  EXPECT_NE(multiplexerEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(multiplexerEvaluation.find("\"total_variation_distance\":"),
            std::string::npos);

  const auto modularMultiplier = test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }));
  const auto modularMultiplierEvaluation = test::value(evaluateJSON(
      test::value(toManifestJSON(modularMultiplier)),
      R"({"schema_version":1,"counts":{"00000000":1,"10000000":1,"00010000":1,"10010011":1,"00100000":1,"10100001":1,"00110000":1,"10110100":1,"01000000":1,"11000010":1,"01010000":1,"11010000":1,"01100000":1,"11100011":1,"01110000":1,"11110001":1}})"));
  EXPECT_NE(modularMultiplierEvaluation.find("\"success_probability\":1.0"),
            std::string::npos);
  EXPECT_NE(
      modularMultiplierEvaluation.find("\"total_variation_distance\":0.0"),
      std::string::npos);

  const auto qftAdder =
      test::value(QFTAdder::create({.addend = "++", .accumulator = "01"}));
  const auto qftAdderEvaluation = test::value(evaluateJSON(
      test::value(toManifestJSON(qftAdder)),
      R"({"schema_version":1,"counts":{"0001":1,"0110":1,"1011":1,"1100":1}})"));
  EXPECT_NE(qftAdderEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(qftAdderEvaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);

  const auto constant = test::value(QFTAdder::create({
      .addend = "110",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }));
  const auto constantEvaluation = test::value(
      evaluateJSON(test::value(toManifestJSON(constant)),
                   R"({"schema_version":1,"counts":{"0111":8,"0110":2}})"));
  EXPECT_NE(constantEvaluation.find("\"success_probability\":0.8"),
            std::string::npos);
  EXPECT_EQ(test::value(toManifestJSON(test::value(qftAdderFromManifestJSON(
                test::value(toManifestJSON(constant)))))),
            test::value(toManifestJSON(constant)));
  EXPECT_NE(test::value(caseId(constant)),
            test::value(caseId(test::value(
                QFTAdder::create({.addend = "110", .accumulator = "001"})))));

  const Teleportation teleportation;
  const auto teleportationEvaluation =
      test::value(evaluateJSON(test::value(toManifestJSON(teleportation)),
                               R"({"schema_version":1,"counts":{"0":8}})"));
  EXPECT_NE(teleportationEvaluation.find("\"success_probability\":1.0"),
            std::string::npos);
  EXPECT_NE(teleportationEvaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);
  EXPECT_NE(teleportationEvaluation.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);

  const auto repeatUntilSuccess = test::value(RepeatUntilSuccess::create());
  const auto repeatUntilSuccessEvaluation = test::value(
      evaluateJSON(test::value(toManifestJSON(repeatUntilSuccess)),
                   R"({"schema_version":1,"counts":{"0":993,"1":7}})"));
  EXPECT_NE(repeatUntilSuccessEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(repeatUntilSuccessEvaluation.find("\"total_variation_distance\":"),
            std::string::npos);

  expectInvalid(
      [] {
        return countsFromJSON(R"({"schema_version":1,"counts":{"0":1,"0":2}})");
      },
      "duplicate key '0'");
  expectInvalid(
      [] {
        return countsFromJSON(R"({"schema_version":1,"counts":{"0x":1}})");
      },
      "bitstrings");
  expectInvalid(
      [] {
        return countsFromJSON(R"({"schema_version":1,"counts":{"00":0}})");
      },
      "must be positive");
  EXPECT_EQ(test::errorKind(evaluationToJSON(
                "not-a-case", 1,
                Evaluation{.totalVariationDistance = 0.,
                           .squaredHellingerFidelity = 1.,
                           .successProbability = std::nullopt})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(evaluationToJSON(
                test::value(caseId(ghz)), 1,
                Evaluation{.totalVariationDistance =
                               std::numeric_limits<double>::quiet_NaN(),
                           .squaredHellingerFidelity = 1.,
                           .successProbability = std::nullopt})),
            mqt::bench::Error::Kind::InvalidArgument);
}

} // namespace
