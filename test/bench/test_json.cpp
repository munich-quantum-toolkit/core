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

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

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

TEST(BenchmarkJSON, ResolvesModularMultiplierInputs) {
  const auto benchmark = modularMultiplierFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{"multiplier":"011","modulus":"101","multiplicand":"111"}})");
  EXPECT_EQ(benchmark.options().control, '1');
  EXPECT_EQ(benchmark.expectedResult(), "11110001");
  EXPECT_NE(toManifestJSON(benchmark).find("\"success_outcome\":\"11110001\""),
            std::string::npos);
  EXPECT_EQ(modularMultiplierFromManifestJSON(toManifestJSON(benchmark))
                .expectedResult(),
            benchmark.expectedResult());
  EXPECT_EQ(caseId(benchmark), caseId(ModularMultiplier({.multiplier = "011",
                                                         .modulus = "101",
                                                         .multiplicand = "111",
                                                         .control = '1'})));
  EXPECT_NE(
      caseId(benchmark),
      caseId(ModularMultiplier(
          {.multiplier = "011", .modulus = "101", .multiplicand = "110"})));
  EXPECT_NE(caseId(benchmark), caseId(ModularMultiplier({.multiplier = "011",
                                                         .modulus = "101",
                                                         .multiplicand = "111",
                                                         .control = '0'})));
  for (const auto* control : {"\"\"", "\"11\"", "\"x\"", "true", "1"}) {
    const auto instance =
        std::string(
            R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{"multiplier":"011","modulus":"101","multiplicand":"111","control":)") +
        control + "}}";
    EXPECT_THROW(static_cast<void>(
                     modularMultiplierFromInstanceSpecificationJSON(instance)),
                 std::invalid_argument);
  }
  EXPECT_THROW(
      static_cast<void>(modularMultiplierFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{"multiplier":"011","modulus":"101"}})")),
      std::invalid_argument);
}

TEST(BenchmarkJSON,
     ParsesInstanceSpecificationsAndSerializesResolvedParameters) {
  const auto bv = bvFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"bv","parameters":{"hidden_bitstring":"101"}})");
  EXPECT_EQ(bv.options().method, BVMethod::Static);
  EXPECT_EQ(
      toInstanceSpecificationJSON(bv),
      R"({"benchmark":"bv","parameters":{"hidden_bitstring":"101","method":"static"},"schema_version":1})");

  const auto modularMultiplier = modularMultiplierFromInstanceSpecificationJSON(
      R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"011"},"schema_version":1})");
  EXPECT_EQ(modularMultiplier.options().multiplier, "011");
  EXPECT_EQ(modularMultiplier.options().modulus, "101");
  EXPECT_EQ(
      toInstanceSpecificationJSON(modularMultiplier),
      R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"011"},"schema_version":1})");

  const auto ghz = ghzFromInstanceSpecificationJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})");
  EXPECT_EQ(ghz.options().topology, GHZTopology::Linear);
  EXPECT_EQ(ghz.options().basis, GHZBasis::Z);
  EXPECT_EQ(
      toInstanceSpecificationJSON(ghz),
      R"({"benchmark":"ghz","parameters":{"basis":"z","qubits":3,"topology":"linear"},"schema_version":1})");

  const auto grover = groverFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"grover","parameters":{"marked_bitstring":"10"}})");
  ASSERT_TRUE(grover.options().iterations);
  EXPECT_EQ(*grover.options().iterations, 1);
  EXPECT_EQ(
      toInstanceSpecificationJSON(grover),
      R"({"benchmark":"grover","parameters":{"iterations":1,"marked_bitstring":"10"},"schema_version":1})");

  const auto multiplexer = multiplexerFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7}})");
  EXPECT_EQ(multiplexer.options().qubits, 7);
  EXPECT_EQ(
      toInstanceSpecificationJSON(multiplexer),
      R"({"benchmark":"multiplexer","parameters":{"qubits":7},"schema_version":1})");

  const auto qft = qftFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft","parameters":{"qubits":4,"period_exponent":2}})");
  EXPECT_EQ(qft.options().method, QFTMethod::Standard);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qft),
      R"({"benchmark":"qft","parameters":{"method":"standard","period_exponent":2,"qubits":4},"schema_version":1})");

  const auto qftAdder = qftAdderFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft-adder","parameters":{"addend":"+++","accumulator":"001"}})");
  EXPECT_EQ(qftAdder.options().method, QFTAdderMethod::Register);
  EXPECT_EQ(qftAdder.options().overflow, QFTAdderOverflow::Wrap);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qftAdder),
      R"({"benchmark":"qft-adder","parameters":{"accumulator":"001","addend":"+++","method":"register","overflow":"wrap"},"schema_version":1})");

  const auto qpe = qpeFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":4,"phase":{"numerator":10,"denominator":8},"method":"iterative"}})");
  EXPECT_EQ(qpe.options().phase, Phase(1, 4));
  EXPECT_EQ(qpe.options().method, QPEMethod::Iterative);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qpe),
      R"({"benchmark":"qpe","parameters":{"method":"iterative","phase":{"denominator":4,"numerator":1},"precision":4},"schema_version":1})");

  const auto repeatUntilSuccess = repeatUntilSuccessFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{}})");
  EXPECT_EQ(
      toInstanceSpecificationJSON(repeatUntilSuccess),
      R"({"benchmark":"repeat-until-success","parameters":{},"schema_version":1})");

  const auto teleportation = teleportationFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"teleportation","parameters":{}})");
  EXPECT_EQ(
      toInstanceSpecificationJSON(teleportation),
      R"({"benchmark":"teleportation","parameters":{},"schema_version":1})");
}

TEST(BenchmarkJSON, RoundTripsSelfCheckingManifests) {
  const BV bv{{.hiddenBitstring = "101", .method = BVMethod::Dynamic}};
  const ModularMultiplier modularMultiplier{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }};
  const GHZ ghz{
      {.qubits = 4, .topology = GHZTopology::Star, .basis = GHZBasis::X}};
  const Grover grover{{.markedBitstring = "001", .iterations = 2}};
  const Multiplexer multiplexer{{.qubits = 7}};
  const QFT qft{
      {.qubits = 4, .periodExponent = 2, .method = QFTMethod::Semiclassical}};
  const QFTAdder qftAdder{{.addend = "+++", .accumulator = "001"}};
  const QPE qpe{
      {.precision = 5, .phase = Phase(1, 3), .method = QPEMethod::Iterative}};
  const RepeatUntilSuccess repeatUntilSuccess;
  const Teleportation teleportation;

  const auto bvManifest = toManifestJSON(bv);
  const auto modularMultiplierManifest = toManifestJSON(modularMultiplier);
  const auto ghzManifest = toManifestJSON(ghz);
  const auto groverManifest = toManifestJSON(grover);
  const auto multiplexerManifest = toManifestJSON(multiplexer);
  const auto qftManifest = toManifestJSON(qft);
  const auto qftAdderManifest = toManifestJSON(qftAdder);
  const auto qpeManifest = toManifestJSON(qpe);
  const auto repeatUntilSuccessManifest = toManifestJSON(repeatUntilSuccess);
  const auto teleportationManifest = toManifestJSON(teleportation);
  EXPECT_EQ(toManifestJSON(bvFromManifestJSON(bvManifest)), bvManifest);
  EXPECT_EQ(toManifestJSON(
                modularMultiplierFromManifestJSON(modularMultiplierManifest)),
            modularMultiplierManifest);
  EXPECT_EQ(toManifestJSON(ghzFromManifestJSON(ghzManifest)), ghzManifest);
  EXPECT_EQ(toManifestJSON(groverFromManifestJSON(groverManifest)),
            groverManifest);
  EXPECT_EQ(toManifestJSON(multiplexerFromManifestJSON(multiplexerManifest)),
            multiplexerManifest);
  EXPECT_EQ(toManifestJSON(qftFromManifestJSON(qftManifest)), qftManifest);
  EXPECT_EQ(toManifestJSON(qftAdderFromManifestJSON(qftAdderManifest)),
            qftAdderManifest);
  EXPECT_EQ(toManifestJSON(qpeFromManifestJSON(qpeManifest)), qpeManifest);
  EXPECT_EQ(toManifestJSON(
                repeatUntilSuccessFromManifestJSON(repeatUntilSuccessManifest)),
            repeatUntilSuccessManifest);
  EXPECT_EQ(
      toManifestJSON(teleportationFromManifestJSON(teleportationManifest)),
      teleportationManifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(bvManifest), "bv");
  EXPECT_EQ(benchmarkIdFromManifestJSON(modularMultiplierManifest),
            "modular-multiplier");
  EXPECT_EQ(benchmarkIdFromManifestJSON(ghzManifest), "ghz");
  EXPECT_EQ(benchmarkIdFromManifestJSON(groverManifest), "grover");
  EXPECT_EQ(benchmarkIdFromManifestJSON(multiplexerManifest), "multiplexer");
  EXPECT_EQ(benchmarkIdFromManifestJSON(qftManifest), "qft");
  EXPECT_EQ(benchmarkIdFromManifestJSON(qftAdderManifest), "qft-adder");
  EXPECT_EQ(benchmarkIdFromManifestJSON(qpeManifest), "qpe");
  EXPECT_EQ(benchmarkIdFromManifestJSON(repeatUntilSuccessManifest),
            "repeat-until-success");
  EXPECT_EQ(benchmarkIdFromManifestJSON(teleportationManifest),
            "teleportation");
  EXPECT_NE(ghzManifest.find("\"case_id\":\"" + caseId(ghz) + "\""),
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
  EXPECT_NE(repeatUntilSuccessManifest.find("\"parameters\":{}"),
            std::string::npos);
  EXPECT_NE(teleportationManifest.find("\"model\":\"teleportation\""),
            std::string::npos);
  EXPECT_NE(teleportationManifest.find("\"parameters\":{}"), std::string::npos);
}

TEST(BenchmarkJSON, UsesStableSemanticCaseIds) {
  const GHZ linear{{.qubits = 3}};
  const GHZ same{{.qubits = 3}};
  const GHZ star{{.qubits = 3, .topology = GHZTopology::Star}};
  EXPECT_EQ(caseId(linear), caseId(same));
  EXPECT_NE(caseId(linear), caseId(star));
  EXPECT_NE(caseId(BV{{.hiddenBitstring = "1"}}),
            caseId(BV{{.hiddenBitstring = "1", .method = BVMethod::Dynamic}}));
  EXPECT_EQ(caseId(ModularMultiplier{{.multiplier = "011",
                                      .modulus = "101",
                                      .multiplicand = "+++",
                                      .control = '+'}}),
            caseId(ModularMultiplier{{.multiplier = "011",
                                      .modulus = "101",
                                      .multiplicand = "+++",
                                      .control = '+'}}));
  EXPECT_NE(caseId(ModularMultiplier{{.multiplier = "011",
                                      .modulus = "101",
                                      .multiplicand = "+++",
                                      .control = '+'}}),
            caseId(ModularMultiplier{{.multiplier = "001",
                                      .modulus = "101",
                                      .multiplicand = "+++",
                                      .control = '+'}}));
  EXPECT_NE(caseId(QFT{{.qubits = 3, .periodExponent = 1}}),
            caseId(QFT{{.qubits = 3,
                        .periodExponent = 1,
                        .method = QFTMethod::Semiclassical}}));
  EXPECT_EQ(caseId(QFTAdder{{.addend = "+++", .accumulator = "001"}}),
            caseId(QFTAdder{{.addend = "+++", .accumulator = "001"}}));
  EXPECT_NE(caseId(QFTAdder{{.addend = "+++", .accumulator = "001"}}),
            caseId(QFTAdder{{.addend = "++++", .accumulator = "0001"}}));
  EXPECT_EQ(caseId(Multiplexer{{.qubits = 7}}),
            caseId(Multiplexer{{.qubits = 7}}));
  EXPECT_NE(caseId(Multiplexer{{.qubits = 7}}),
            caseId(Multiplexer{{.qubits = 6}}));
  EXPECT_EQ(caseId(RepeatUntilSuccess{}),
            "sha256-3ff9fff1db965d838e4d0e3078cebe17"
            "f5ff18cdf6a63610e9fdb3159080a4fc");
  EXPECT_EQ(caseId(Teleportation{}), "sha256-de1348477e2604539b963a28bc19f5d3"
                                     "ed27ed86fc6608366bbc6eb9b55855f6");
  EXPECT_EQ(caseId(linear), "sha256-a222c0c57bcecb4f5e7ea72bab439683"
                            "92861a52c5cb7c9c13aeaffffa059a65");
}

TEST(BenchmarkJSON,
     RejectsDuplicateUnknownAndMistypedInstanceSpecificationValues) {
  expectInvalid(
      [] {
        static_cast<void>(benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","benchmark":"qpe","parameters":{"qubits":2}})",
            "duplicate.json"));
      },
      "duplicate key 'benchmark'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"qubits":3}})"));
      },
      "duplicate key 'qubits'");
  expectInvalid(
      [] {
        static_cast<void>(qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4,"numerator":2}}})"));
      },
      "duplicate key 'numerator'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2},"extra":true})"));
      },
      "unknown key 'extra'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"extra":true}})"));
      },
      "unknown key 'extra'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2.5}})"));
      },
      "encoded as an integer");
  expectInvalid(
      [] {
        static_cast<void>(qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":9007199254740993.0,"denominator":9007199254740994}}})"));
      },
      "encoded as an integer");
  expectInvalid(
      [] {
        static_cast<void>(qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":18446744073709551615,"phase":{"numerator":1,"denominator":4}}})"));
      },
      "between 1 and 1000000");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"basis":"x","qubits":1076}})"));
      },
      "between 1 and 1075");
  expectInvalid(
      [] {
        static_cast<void>(multiplexerFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":1}})"));
      },
      "between 2 and 1024");
  expectInvalid(
      [] {
        static_cast<void>(multiplexerFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7,"angles":[]}})"));
      },
      "unknown key 'angles'");
  expectInvalid(
      [] {
        static_cast<void>(modularMultiplierFromInstanceSpecificationJSON(
            R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"011","extra":true},"schema_version":1})"));
      },
      "unknown key 'extra'");
  expectInvalid(
      [] {
        static_cast<void>(modularMultiplierFromInstanceSpecificationJSON(
            R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"1001","multiplicand":"+++","multiplier":"011"},"schema_version":1})"));
      },
      "equal widths");
  expectInvalid(
      [] {
        static_cast<void>(modularMultiplierFromInstanceSpecificationJSON(
            R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"010","multiplicand":"+++","multiplier":"011"},"schema_version":1})"));
      },
      "canonical");
  expectInvalid(
      [] {
        static_cast<void>(modularMultiplierFromInstanceSpecificationJSON(
            R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"101"},"schema_version":1})"));
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
    EXPECT_THROW(
        static_cast<void>(qftAdderFromInstanceSpecificationJSON(instance)),
        std::invalid_argument);
  }
  expectInvalid(
      [] {
        static_cast<void>(repeatUntilSuccessFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"attempts":1}})"));
      },
      "unknown key 'attempts'");
  expectInvalid(
      [] {
        static_cast<void>(teleportationFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"teleportation","parameters":{"qubits":3}})"));
      },
      "unknown key 'qubits'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"new","parameters":{}})"));
      },
      "unsupported benchmark 'new'");
  expectInvalid(
      [] {
        static_cast<void>(qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":0}}})"));
      },
      "denominator must not be zero");
}

TEST(BenchmarkJSON, RejectsAnInstanceSpecificationForAnotherConcreteType) {
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4}}})"));
      },
      "must be 'ghz'");
}

TEST(BenchmarkJSON, RejectsAlteredOrUnresolvedManifestData) {
  const GHZ ghz{{.qubits = 3}};
  auto changedOutput = toManifestJSON(ghz);
  const auto width = changedOutput.find("\"width\":3");
  ASSERT_NE(width, std::string::npos);
  changedOutput.replace(width, std::string("\"width\":3").size(),
                        "\"width\":2");
  expectInvalid([&] { static_cast<void>(ghzFromManifestJSON(changedOutput)); },
                "does not match");

  auto changedNumericKind = toManifestJSON(ghz);
  const auto integerWidth = changedNumericKind.find(R"("width":3)");
  ASSERT_NE(integerWidth, std::string::npos);
  changedNumericKind.replace(integerWidth, std::string(R"("width":3)").size(),
                             R"("width":3.0)");
  expectInvalid(
      [&] { static_cast<void>(ghzFromManifestJSON(changedNumericKind)); },
      "does not match");

  auto changedId = toManifestJSON(ghz);
  const auto digest = changedId.find("sha256-");
  ASSERT_NE(digest, std::string::npos);
  changedId[digest + 7U] = changedId[digest + 7U] == '0' ? '1' : '0';
  expectInvalid([&] { static_cast<void>(ghzFromManifestJSON(changedId)); },
                "case ID");

  auto unresolved = toManifestJSON(ghz);
  const auto basis = unresolved.find(R"("basis":"z",)");
  ASSERT_NE(basis, std::string::npos);
  unresolved.erase(basis, std::string(R"("basis":"z",)").size());
  expectInvalid([&] { static_cast<void>(ghzFromManifestJSON(unresolved)); },
                "resolved benchmark instance");
}

TEST(BenchmarkJSON, ListsBenchmarksAndDescribesStandardSchemas) {
  EXPECT_EQ(
      listBenchmarksJSON(),
      R"({"benchmarks":[{"definition_version":1,"id":"bv"},{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"modular-multiplier"},{"definition_version":1,"id":"multiplexer"},{"definition_version":1,"id":"qft"},{"definition_version":1,"id":"qft-adder"},{"definition_version":1,"id":"qpe"},{"definition_version":1,"id":"repeat-until-success"},{"definition_version":1,"id":"teleportation"}],"schema_version":1})");
  const auto bv = describeBenchmarkJSON("bv");
  const auto modularMultiplier = describeBenchmarkJSON("modular-multiplier");
  const auto ghz = describeBenchmarkJSON("ghz");
  const auto grover = describeBenchmarkJSON("grover");
  const auto multiplexer = describeBenchmarkJSON("multiplexer");
  const auto qft = describeBenchmarkJSON("qft");
  const auto qftAdder = describeBenchmarkJSON("qft-adder");
  const auto qpe = describeBenchmarkJSON("qpe");
  const auto repeatUntilSuccess = describeBenchmarkJSON("repeat-until-success");
  const auto teleportation = describeBenchmarkJSON("teleportation");
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
          R"("parameters":{"additionalProperties":false,"properties":{},"type":"object"})"),
      std::string::npos);
  EXPECT_NE(
      teleportation.find(
          R"("parameters":{"additionalProperties":false,"properties":{},"type":"object"})"),
      std::string::npos);
  EXPECT_THROW(static_cast<void>(describeBenchmarkJSON("unknown")),
               std::invalid_argument);
}

TEST(BenchmarkJSON, ParsesCountsAndSerializesEvaluations) {
  const auto counts =
      countsFromJSON(R"({"counts":{"11":50,"00":50},"schema_version":1})");
  EXPECT_EQ(counts.at("00"), 50);
  EXPECT_EQ(counts.at("11"), 50);

  const GHZ ghz{{.qubits = 2}};
  const auto serialized =
      evaluationToJSON(caseId(ghz), 100, ghz.evaluate(counts));
  EXPECT_NE(serialized.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(serialized.find("\"total_variation_distance\":0.0"),
            std::string::npos);

  const BV bv{{.hiddenBitstring = "11"}};
  const auto generic = evaluateJSON(
      toManifestJSON(bv), R"({"schema_version":1,"counts":{"11":8,"00":2}})");
  EXPECT_NE(generic.find("\"success_probability\":0.8"), std::string::npos);

  const Multiplexer multiplexer{{.qubits = 2}};
  const auto multiplexerEvaluation =
      evaluateJSON(toManifestJSON(multiplexer),
                   R"({"schema_version":1,"counts":{"00":8,"01":2}})");
  EXPECT_NE(multiplexerEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(multiplexerEvaluation.find("\"total_variation_distance\":"),
            std::string::npos);

  const ModularMultiplier modularMultiplier{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }};
  const auto modularMultiplierEvaluation = evaluateJSON(
      toManifestJSON(modularMultiplier),
      R"({"schema_version":1,"counts":{"00000000":1,"10000000":1,"00010000":1,"10010011":1,"00100000":1,"10100001":1,"00110000":1,"10110100":1,"01000000":1,"11000010":1,"01010000":1,"11010000":1,"01100000":1,"11100011":1,"01110000":1,"11110001":1}})");
  EXPECT_NE(modularMultiplierEvaluation.find("\"success_probability\":1.0"),
            std::string::npos);
  EXPECT_NE(
      modularMultiplierEvaluation.find("\"total_variation_distance\":0.0"),
      std::string::npos);

  const QFTAdder qftAdder{{.addend = "++", .accumulator = "01"}};
  const auto qftAdderEvaluation = evaluateJSON(
      toManifestJSON(qftAdder),
      R"({"schema_version":1,"counts":{"0001":1,"0110":1,"1011":1,"1100":1}})");
  EXPECT_NE(qftAdderEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(qftAdderEvaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);

  const QFTAdder constant{{
      .addend = "110",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }};
  const auto constantEvaluation =
      evaluateJSON(toManifestJSON(constant),
                   R"({"schema_version":1,"counts":{"0111":8,"0110":2}})");
  EXPECT_NE(constantEvaluation.find("\"success_probability\":0.8"),
            std::string::npos);
  EXPECT_EQ(toManifestJSON(qftAdderFromManifestJSON(toManifestJSON(constant))),
            toManifestJSON(constant));
  EXPECT_NE(caseId(constant),
            caseId(QFTAdder{{.addend = "110", .accumulator = "001"}}));

  const Teleportation teleportation;
  const auto teleportationEvaluation =
      evaluateJSON(toManifestJSON(teleportation),
                   R"({"schema_version":1,"counts":{"0":8}})");
  EXPECT_NE(teleportationEvaluation.find("\"success_probability\":1.0"),
            std::string::npos);
  EXPECT_NE(teleportationEvaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);
  EXPECT_NE(teleportationEvaluation.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);

  const RepeatUntilSuccess repeatUntilSuccess;
  const auto repeatUntilSuccessEvaluation =
      evaluateJSON(toManifestJSON(repeatUntilSuccess),
                   R"({"schema_version":1,"counts":{"0":993,"1":7}})");
  EXPECT_NE(repeatUntilSuccessEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(repeatUntilSuccessEvaluation.find("\"total_variation_distance\":"),
            std::string::npos);

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
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   "not-a-case", 1,
                   Evaluation{.totalVariationDistance = 0.,
                              .squaredHellingerFidelity = 1.,
                              .successProbability = std::nullopt})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   caseId(ghz), 1,
                   Evaluation{.totalVariationDistance =
                                  std::numeric_limits<double>::quiet_NaN(),
                              .squaredHellingerFidelity = 1.,
                              .successProbability = std::nullopt})),
               std::invalid_argument);
}

} // namespace
