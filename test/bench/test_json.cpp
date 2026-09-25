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

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include "mlir/Support/LogicalResult.h"

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

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
  const auto error = ::mqt::test::diagnostic(operation);
  ASSERT_TRUE(error);
  EXPECT_EQ(error->category, ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_NE(error->message.find(diagnostic), std::string::npos)
      << error->message;
}

TEST(BenchmarkJSON, ReturnsNormalizedInstancesAndInputDiagnostics) {
  const auto benchmark = ::mqt::test::value(GHZ::create({.qubits = 3}));
  auto result = mqt::bench::parseInstanceSpecificationJSON(
      toInstanceSpecificationJSON(benchmark));
  ASSERT_TRUE(mlir::succeeded(result));
  const auto& parsed = *result;
  EXPECT_TRUE(std::holds_alternative<GHZ>(parsed.instance));
  EXPECT_EQ(parsed.benchmarkId, "ghz");
  EXPECT_EQ(parsed.caseId, caseId(benchmark));
  EXPECT_EQ(parsed.manifestJSON, toManifestJSON(benchmark));

  for (
      const auto* invalid : {
          "{",
          R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":0}})",
      }) {
    expectInvalid(
        [&] {
          return mqt::bench::parseInstanceSpecificationJSON(invalid,
                                                            "input.json");
        },
        "input.json");
  }
  expectInvalid([] { return mqt::bench::describeBenchmarkJSON("unknown"); },
                "unknown");
  expectInvalid(
      [&] {
        return mqt::bench::evaluateJSON(toManifestJSON(benchmark), "{}",
                                        "manifest.json", "counts.json");
      },
      "counts.json");
}

TEST(BenchmarkJSON, ResolvesModularMultiplierInputs) {
  const auto benchmark =
      ::mqt::test::value(modularMultiplierFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{"multiplier":"011","modulus":"101","multiplicand":"111"}})"));
  EXPECT_EQ(benchmark.options().control, '1');
  EXPECT_EQ(benchmark.expectedResult(), "11110001");
  EXPECT_NE(toManifestJSON(benchmark).find("\"success_outcome\":\"11110001\""),
            std::string::npos);
  EXPECT_EQ(::mqt::test::value(
                modularMultiplierFromManifestJSON(toManifestJSON(benchmark)))
                .expectedResult(),
            benchmark.expectedResult());
  EXPECT_EQ(caseId(benchmark),
            caseId(::mqt::test::value(
                ModularMultiplier::create({.multiplier = "011",
                                           .modulus = "101",
                                           .multiplicand = "111",
                                           .control = '1'}))));
  EXPECT_NE(
      caseId(benchmark),
      caseId(::mqt::test::value(ModularMultiplier::create(
          {.multiplier = "011", .modulus = "101", .multiplicand = "110"}))));
  EXPECT_NE(caseId(benchmark),
            caseId(::mqt::test::value(
                ModularMultiplier::create({.multiplier = "011",
                                           .modulus = "101",
                                           .multiplicand = "111",
                                           .control = '0'}))));
  for (const auto* control : {"\"\"", "\"11\"", "\"x\"", "true", "1"}) {
    const auto instance =
        std::string(
            R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{"multiplier":"011","modulus":"101","multiplicand":"111","control":)") +
        control + "}}";
    EXPECT_EQ(::mqt::test::errorKind([&] {
                return modularMultiplierFromInstanceSpecificationJSON(instance);
              }),
              ::mqt::ErrorCategory::InvalidArgument);
  }
  EXPECT_EQ(
      ::mqt::test::errorKind([&] {
        return modularMultiplierFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{"multiplier":"011","modulus":"101"}})");
      }),
      ::mqt::ErrorCategory::InvalidArgument);
}

TEST(BenchmarkJSON, ValidatesRepeatUntilSuccessWidth) {
  for (const auto* width : {"0", "1000001", "-1", "1.5", "true", "\"5\""}) {
    EXPECT_EQ(
        ::mqt::test::errorKind([&] {
          return repeatUntilSuccessFromInstanceSpecificationJSON(
              std::string(
                  R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"data_qubits":)") +
              width + "}}");
        }),
        ::mqt::ErrorCategory::InvalidArgument);
  }
  const auto benchmark =
      ::mqt::test::value(RepeatUntilSuccess::create({.dataQubits = 32}));
  EXPECT_EQ(::mqt::test::value(
                repeatUntilSuccessFromManifestJSON(toManifestJSON(benchmark)))
                .options()
                .dataQubits,
            32U);
}

TEST(BenchmarkJSON,
     ParsesInstanceSpecificationsAndSerializesResolvedParameters) {
  const auto bv = ::mqt::test::value(bvFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"bv","parameters":{"hidden_bitstring":"101"}})"));
  EXPECT_EQ(bv.options().method, BVMethod::Static);
  EXPECT_EQ(
      toInstanceSpecificationJSON(bv),
      R"({"benchmark":"bv","parameters":{"hidden_bitstring":"101","method":"static"},"schema_version":1})");

  const auto modularMultiplier =
      ::mqt::test::value(modularMultiplierFromInstanceSpecificationJSON(
          R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"011"},"schema_version":1})"));
  EXPECT_EQ(modularMultiplier.options().multiplier, "011");
  EXPECT_EQ(modularMultiplier.options().modulus, "101");
  EXPECT_EQ(
      toInstanceSpecificationJSON(modularMultiplier),
      R"({"benchmark":"modular-multiplier","parameters":{"control":"+","modulus":"101","multiplicand":"+++","multiplier":"011"},"schema_version":1})");

  const auto ghz = ::mqt::test::value(ghzFromInstanceSpecificationJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})"));
  EXPECT_EQ(ghz.options().topology, GHZTopology::Linear);
  EXPECT_EQ(ghz.options().basis, GHZBasis::Z);
  EXPECT_EQ(
      toInstanceSpecificationJSON(ghz),
      R"({"benchmark":"ghz","parameters":{"basis":"z","qubits":3,"topology":"linear"},"schema_version":1})");

  const auto grover = ::mqt::test::value(groverFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"grover","parameters":{"marked_bitstring":"10"}})"));
  ASSERT_TRUE(grover.options().iterations);
  EXPECT_EQ(*grover.options().iterations, 1);
  EXPECT_EQ(
      toInstanceSpecificationJSON(grover),
      R"({"benchmark":"grover","parameters":{"iterations":1,"marked_bitstring":"10"},"schema_version":1})");

  const auto multiplexer = ::mqt::test::value(multiplexerFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7}})"));
  EXPECT_EQ(multiplexer.options().qubits, 7);
  EXPECT_EQ(
      toInstanceSpecificationJSON(multiplexer),
      R"({"benchmark":"multiplexer","parameters":{"qubits":7},"schema_version":1})");

  const auto qft = ::mqt::test::value(qftFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft","parameters":{"qubits":4,"period_exponent":2}})"));
  EXPECT_EQ(qft.options().method, QFTMethod::Standard);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qft),
      R"({"benchmark":"qft","parameters":{"method":"standard","period_exponent":2,"qubits":4},"schema_version":1})");

  const auto qftAdder = ::mqt::test::value(qftAdderFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft-adder","parameters":{"addend":"+++","accumulator":"001"}})"));
  EXPECT_EQ(qftAdder.options().method, QFTAdderMethod::Register);
  EXPECT_EQ(qftAdder.options().overflow, QFTAdderOverflow::Wrap);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qftAdder),
      R"({"benchmark":"qft-adder","parameters":{"accumulator":"001","addend":"+++","method":"register","overflow":"wrap"},"schema_version":1})");

  const auto qpe = ::mqt::test::value(qpeFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":4,"phase":{"numerator":10,"denominator":8},"method":"iterative"}})"));
  EXPECT_EQ(qpe.options().phase, ::mqt::test::value(Phase::create(1, 4)));
  EXPECT_EQ(qpe.options().method, QPEMethod::Iterative);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qpe),
      R"({"benchmark":"qpe","parameters":{"method":"iterative","phase":{"denominator":4,"numerator":1},"precision":4},"schema_version":1})");

  const auto repeatUntilSuccess =
      ::mqt::test::value(repeatUntilSuccessFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(repeatUntilSuccess),
      R"({"benchmark":"repeat-until-success","parameters":{"data_qubits":1},"schema_version":1})");

  const auto teleportation =
      ::mqt::test::value(teleportationFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"teleportation","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(teleportation),
      R"({"benchmark":"teleportation","parameters":{},"schema_version":1})");
}

TEST(BenchmarkJSON, RoundTripsSelfCheckingManifests) {
  const auto bv = ::mqt::test::value(
      BV::create({.hiddenBitstring = "101", .method = BVMethod::Dynamic}));
  const auto modularMultiplier = ::mqt::test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }));
  const auto ghz = ::mqt::test::value(GHZ::create(
      {.qubits = 4, .topology = GHZTopology::Star, .basis = GHZBasis::X}));
  const auto grover = ::mqt::test::value(
      Grover::create({.markedBitstring = "001", .iterations = 2}));
  const auto multiplexer =
      ::mqt::test::value(Multiplexer::create({.qubits = 7}));
  const auto qft = ::mqt::test::value(QFT::create(
      {.qubits = 4, .periodExponent = 2, .method = QFTMethod::Semiclassical}));
  const auto qftAdder = ::mqt::test::value(
      QFTAdder::create({.addend = "+++", .accumulator = "001"}));
  const auto qpe = ::mqt::test::value(QPE::create({
      .precision = 5,
      .phase = ::mqt::test::value(Phase::create(1, 3)),
      .method = QPEMethod::Iterative,
  }));
  const auto repeatUntilSuccess =
      ::mqt::test::value(RepeatUntilSuccess::create());
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
  EXPECT_EQ(toManifestJSON(::mqt::test::value(bvFromManifestJSON(bvManifest))),
            bvManifest);
  EXPECT_EQ(toManifestJSON(::mqt::test::value(
                modularMultiplierFromManifestJSON(modularMultiplierManifest))),
            modularMultiplierManifest);
  EXPECT_EQ(
      toManifestJSON(::mqt::test::value(ghzFromManifestJSON(ghzManifest))),
      ghzManifest);
  EXPECT_EQ(toManifestJSON(
                ::mqt::test::value(groverFromManifestJSON(groverManifest))),
            groverManifest);
  EXPECT_EQ(toManifestJSON(::mqt::test::value(
                multiplexerFromManifestJSON(multiplexerManifest))),
            multiplexerManifest);
  EXPECT_EQ(
      toManifestJSON(::mqt::test::value(qftFromManifestJSON(qftManifest))),
      qftManifest);
  EXPECT_EQ(toManifestJSON(
                ::mqt::test::value(qftAdderFromManifestJSON(qftAdderManifest))),
            qftAdderManifest);
  EXPECT_EQ(
      toManifestJSON(::mqt::test::value(qpeFromManifestJSON(qpeManifest))),
      qpeManifest);
  EXPECT_EQ(
      toManifestJSON(::mqt::test::value(
          repeatUntilSuccessFromManifestJSON(repeatUntilSuccessManifest))),
      repeatUntilSuccessManifest);
  EXPECT_EQ(toManifestJSON(::mqt::test::value(
                teleportationFromManifestJSON(teleportationManifest))),
            teleportationManifest);
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(bvManifest)), "bv");
  EXPECT_EQ(::mqt::test::value(
                benchmarkIdFromManifestJSON(modularMultiplierManifest)),
            "modular-multiplier");
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(ghzManifest)),
            "ghz");
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(groverManifest)),
            "grover");
  EXPECT_EQ(
      ::mqt::test::value(benchmarkIdFromManifestJSON(multiplexerManifest)),
      "multiplexer");
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(qftManifest)),
            "qft");
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(qftAdderManifest)),
            "qft-adder");
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(qpeManifest)),
            "qpe");
  EXPECT_EQ(::mqt::test::value(
                benchmarkIdFromManifestJSON(repeatUntilSuccessManifest)),
            "repeat-until-success");
  EXPECT_EQ(
      ::mqt::test::value(benchmarkIdFromManifestJSON(teleportationManifest)),
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
  EXPECT_NE(
      repeatUntilSuccessManifest.find("\"parameters\":{\"data_qubits\":1}"),
      std::string::npos);
  EXPECT_NE(teleportationManifest.find("\"model\":\"teleportation\""),
            std::string::npos);
  EXPECT_NE(teleportationManifest.find("\"parameters\":{}"), std::string::npos);
}

TEST(BenchmarkJSON, UsesStableSemanticCaseIds) {
  const auto linear = ::mqt::test::value(GHZ::create({.qubits = 3}));
  const auto same = ::mqt::test::value(GHZ::create({.qubits = 3}));
  const auto star = ::mqt::test::value(
      GHZ::create({.qubits = 3, .topology = GHZTopology::Star}));
  EXPECT_EQ(caseId(linear), caseId(same));
  EXPECT_NE(caseId(linear), caseId(star));
  EXPECT_NE(caseId(::mqt::test::value(BV::create({.hiddenBitstring = "1"}))),
            caseId(::mqt::test::value(BV::create(
                {.hiddenBitstring = "1", .method = BVMethod::Dynamic}))));
  EXPECT_EQ(caseId(::mqt::test::value(
                ModularMultiplier::create({.multiplier = "011",
                                           .modulus = "101",
                                           .multiplicand = "+++",
                                           .control = '+'}))),
            caseId(::mqt::test::value(
                ModularMultiplier::create({.multiplier = "011",
                                           .modulus = "101",
                                           .multiplicand = "+++",
                                           .control = '+'}))));
  EXPECT_NE(caseId(::mqt::test::value(
                ModularMultiplier::create({.multiplier = "011",
                                           .modulus = "101",
                                           .multiplicand = "+++",
                                           .control = '+'}))),
            caseId(::mqt::test::value(
                ModularMultiplier::create({.multiplier = "001",
                                           .modulus = "101",
                                           .multiplicand = "+++",
                                           .control = '+'}))));
  EXPECT_NE(caseId(::mqt::test::value(
                QFT::create({.qubits = 3, .periodExponent = 1}))),
            caseId(::mqt::test::value(
                QFT::create({.qubits = 3,
                             .periodExponent = 1,
                             .method = QFTMethod::Semiclassical}))));
  EXPECT_EQ(caseId(::mqt::test::value(
                QFTAdder::create({.addend = "+++", .accumulator = "001"}))),
            caseId(::mqt::test::value(
                QFTAdder::create({.addend = "+++", .accumulator = "001"}))));
  EXPECT_NE(caseId(::mqt::test::value(
                QFTAdder::create({.addend = "+++", .accumulator = "001"}))),
            caseId(::mqt::test::value(
                QFTAdder::create({.addend = "++++", .accumulator = "0001"}))));
  EXPECT_EQ(caseId(::mqt::test::value(Multiplexer::create({.qubits = 7}))),
            caseId(::mqt::test::value(Multiplexer::create({.qubits = 7}))));
  EXPECT_NE(caseId(::mqt::test::value(Multiplexer::create({.qubits = 7}))),
            caseId(::mqt::test::value(Multiplexer::create({.qubits = 6}))));
  EXPECT_EQ(caseId(::mqt::test::value(RepeatUntilSuccess::create())),
            caseId(::mqt::test::value(
                RepeatUntilSuccess::create({.dataQubits = 1}))));
  EXPECT_NE(caseId(::mqt::test::value(RepeatUntilSuccess::create())),
            caseId(::mqt::test::value(
                RepeatUntilSuccess::create({.dataQubits = 5}))));
  EXPECT_EQ(caseId(Teleportation{}), "sha256-de1348477e2604539b963a28bc19f5d3"
                                     "ed27ed86fc6608366bbc6eb9b55855f6");
  EXPECT_EQ(caseId(linear), "sha256-a222c0c57bcecb4f5e7ea72bab439683"
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
    EXPECT_EQ(::mqt::test::errorKind([&] {
                return qftAdderFromInstanceSpecificationJSON(instance);
              }),
              ::mqt::ErrorCategory::InvalidArgument);
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

TEST(BenchmarkJSON, PreservesFieldDiagnosticsAcrossBenchmarkFamilies) {
  using Json = nlohmann::json;
  for (const auto& specification : std::vector<std::string>{
           toInstanceSpecificationJSON(
               ::mqt::test::value(BV::create({.hiddenBitstring = "01"}))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(GHZ::create({.qubits = 2}))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(Grover::create({.markedBitstring = "01"}))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(ModularMultiplier::create({
                   .multiplier = "011",
                   .modulus = "101",
                   .multiplicand = "+++",
               }))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(Multiplexer::create({.qubits = 2}))),
           toInstanceSpecificationJSON(::mqt::test::value(
               QFT::create({.qubits = 2, .periodExponent = 1}))),
           toInstanceSpecificationJSON(::mqt::test::value(
               QFTAdder::create({.addend = "01", .accumulator = "00"}))),
           toInstanceSpecificationJSON(::mqt::test::value(QPE::create({
               .precision = 2,
               .phase = ::mqt::test::value(Phase::create(1, 4)),
           }))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(RepeatUntilSuccess::create())),
           toInstanceSpecificationJSON(Teleportation{}),
       }) {
    SCOPED_TRACE(specification);
    const auto root = Json::parse(specification);
    const auto schema = Json::parse(::mqt::test::value(
        describeBenchmarkJSON(root["benchmark"].get<std::string>())));
    const auto check = [](const Json& invalid, const std::string& pointer) {
      expectInvalid(
          [&] {
            return mqt::bench::parseInstanceSpecificationJSON(invalid.dump(),
                                                              "instance.json");
          },
          "instance.json:$" + pointer);
    };
    for (const auto& [key, value] : root["parameters"].items()) {
      auto invalid = root;
      invalid["parameters"][key] = nullptr;
      check(invalid, "/parameters/" + key);
      if (value.is_string() &&
          (key == "method" || key == "basis" || key == "topology" ||
           key == "overflow" || key == "control")) {
        invalid["parameters"][key] = "unknown";
        check(invalid, "/parameters/" + key);
      }
    }
    auto invalid = root;
    invalid["parameters"]["unknown"] = true;
    check(invalid, "/parameters");
    for (const auto& name : schema["properties"]["parameters"].value(
             "required", std::vector<std::string>{})) {
      invalid = root;
      invalid["parameters"].erase(name);
      check(invalid, "/parameters/" + name);
    }
  }
}

TEST(BenchmarkJSON, RejectsMalformedEnvelopesAndPhaseFields) {
  using Json = nlohmann::json;
  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 2}));
  const auto manifest = Json::parse(toManifestJSON(ghz));
  for (const auto& item : manifest.items()) {
    const auto& key = item.key();
    SCOPED_TRACE(key);
    auto invalid = manifest;
    invalid.erase(key);
    expectInvalid(
        [&] {
          return benchmarkIdFromManifestJSON(invalid.dump(), "manifest.json");
        },
        "manifest.json:$/" + key);
    invalid[key] = nullptr;
    expectInvalid(
        [&] {
          return benchmarkIdFromManifestJSON(invalid.dump(), "manifest.json");
        },
        "manifest.json:$/" + key);
  }
  for (const auto* key : {"schema_version", "definition_version"}) {
    auto invalid = manifest;
    invalid[key] = 2;
    expectInvalid([&] { return benchmarkIdFromManifestJSON(invalid.dump()); },
                  key);
  }
  for (const auto* input : {
           "[]",
           R"({"values":[null,true,-1,1,1.5,"text",{}]} trailing)",
           "{\"num\":1e400}",
       }) {
    expectInvalid(
        [&] {
          return benchmarkIdFromInstanceSpecificationJSON(input, "syntax.json");
        },
        "syntax.json:");
  }
  const auto qpe =
      Json::parse(toInstanceSpecificationJSON(::mqt::test::value(QPE::create({
          .precision = 2,
          .phase = ::mqt::test::value(Phase::create(1, 4)),
      }))));
  for (const auto* key : {"numerator", "denominator"}) {
    auto invalid = qpe;
    invalid["parameters"]["phase"].erase(key);
    expectInvalid(
        [&] { return qpeFromInstanceSpecificationJSON(invalid.dump()); },
        std::string("$/parameters/phase/") + key);
    invalid["parameters"]["phase"][key] = nullptr;
    expectInvalid(
        [&] { return qpeFromInstanceSpecificationJSON(invalid.dump()); },
        std::string("$/parameters/phase/") + key);
  }
  auto invalid = qpe;
  invalid["parameters"]["phase"]["unknown"] = 1;
  expectInvalid(
      [&] { return qpeFromInstanceSpecificationJSON(invalid.dump()); },
      "unknown key");
}

TEST(BenchmarkJSON, RejectsAlteredOrUnresolvedManifestData) {
  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 3}));
  auto changedOutput = toManifestJSON(ghz);
  const auto width = changedOutput.find("\"width\":3");
  ASSERT_NE(width, std::string::npos);
  changedOutput.replace(width, std::string("\"width\":3").size(),
                        "\"width\":2");
  expectInvalid([&] { return ghzFromManifestJSON(changedOutput); },
                "does not match");

  auto changedNumericKind = toManifestJSON(ghz);
  const auto integerWidth = changedNumericKind.find(R"("width":3)");
  ASSERT_NE(integerWidth, std::string::npos);
  changedNumericKind.replace(integerWidth, std::string(R"("width":3)").size(),
                             R"("width":3.0)");
  expectInvalid([&] { return ghzFromManifestJSON(changedNumericKind); },
                "does not match");

  auto changedId = toManifestJSON(ghz);
  const auto digest = changedId.find("sha256-");
  ASSERT_NE(digest, std::string::npos);
  changedId[digest + 7U] = changedId[digest + 7U] == '0' ? '1' : '0';
  expectInvalid([&] { return ghzFromManifestJSON(changedId); }, "case ID");

  auto unresolved = toManifestJSON(ghz);
  const auto basis = unresolved.find(R"("basis":"z",)");
  ASSERT_NE(basis, std::string::npos);
  unresolved.erase(basis, std::string(R"("basis":"z",)").size());
  expectInvalid([&] { return ghzFromManifestJSON(unresolved); },
                "resolved benchmark instance");
}

TEST(BenchmarkJSON, ListsBenchmarksAndDescribesStandardSchemas) {
  EXPECT_EQ(
      listBenchmarksJSON(),
      R"({"benchmarks":[{"definition_version":1,"id":"bv"},{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"modular-multiplier"},{"definition_version":1,"id":"multiplexer"},{"definition_version":1,"id":"qft"},{"definition_version":1,"id":"qft-adder"},{"definition_version":1,"id":"qpe"},{"definition_version":1,"id":"repeat-until-success"},{"definition_version":1,"id":"shor"},{"definition_version":1,"id":"teleportation"},{"definition_version":1,"id":"w-state"}],"schema_version":1})");
  const auto bv = ::mqt::test::value(describeBenchmarkJSON("bv"));
  const auto modularMultiplier =
      ::mqt::test::value(describeBenchmarkJSON("modular-multiplier"));
  const auto ghz = ::mqt::test::value(describeBenchmarkJSON("ghz"));
  const auto grover = ::mqt::test::value(describeBenchmarkJSON("grover"));
  const auto multiplexer =
      ::mqt::test::value(describeBenchmarkJSON("multiplexer"));
  const auto qft = ::mqt::test::value(describeBenchmarkJSON("qft"));
  const auto qftAdder = ::mqt::test::value(describeBenchmarkJSON("qft-adder"));
  const auto qpe = ::mqt::test::value(describeBenchmarkJSON("qpe"));
  const auto repeatUntilSuccess =
      ::mqt::test::value(describeBenchmarkJSON("repeat-until-success"));
  const auto teleportation =
      ::mqt::test::value(describeBenchmarkJSON("teleportation"));
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
  EXPECT_EQ(
      ::mqt::test::errorKind([&] { return describeBenchmarkJSON("unknown"); }),
      ::mqt::ErrorCategory::InvalidArgument);
}

TEST(BenchmarkJSON, RejectsMalformedCountsAndPropagatesEvaluationErrors) {
  for (const auto* input : {
           "{",
           "[]",
           R"({"schema_version":2,"counts":{"0":1}})",
           R"({"schema_version":1,"counts":{"0":1},"extra":1})",
           R"({"schema_version":1})",
           R"({"schema_version":1,"counts":[]})",
           R"({"schema_version":1,"counts":{}})",
           R"({"schema_version":1,"counts":{"0":null}})",
           R"({"schema_version":1,"counts":{"0":18446744073709551615,"1":1}})",
       }) {
    SCOPED_TRACE(input);
    expectInvalid([&] { return countsFromJSON(input, "counts.json"); },
                  "counts.json:");
  }
  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 2}));
  const auto manifest = toManifestJSON(ghz);
  expectInvalid(
      [&] { return evaluateJSON("{}", "{}", "manifest.json", "counts.json"); },
      "manifest.json:");
  expectInvalid(
      [&] {
        return evaluateJSON(manifest, "{}", "manifest.json", "counts.json");
      },
      "counts.json:");
  expectInvalid([&] { return evaluationToJSON(caseId(ghz), 0, Evaluation{}); },
                "at least one shot");
}

TEST(BenchmarkJSON, ParsesCountsAndSerializesEvaluations) {
  const auto counts = ::mqt::test::value(
      countsFromJSON(R"({"counts":{"11":50,"00":50},"schema_version":1})"));
  EXPECT_EQ(counts.at("00"), 50);
  EXPECT_EQ(counts.at("11"), 50);

  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 2}));
  const auto serialized = ::mqt::test::value(evaluationToJSON(
      caseId(ghz), 100, ::mqt::test::value(ghz.evaluate(counts))));
  EXPECT_NE(serialized.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(serialized.find("\"total_variation_distance\":0.0"),
            std::string::npos);

  const auto bv = ::mqt::test::value(BV::create({.hiddenBitstring = "11"}));
  const auto generic = ::mqt::test::value(evaluateJSON(
      toManifestJSON(bv), R"({"schema_version":1,"counts":{"11":8,"00":2}})"));
  EXPECT_NE(generic.find("\"success_probability\":0.8"), std::string::npos);

  const auto multiplexer =
      ::mqt::test::value(Multiplexer::create({.qubits = 2}));
  const auto multiplexerEvaluation = ::mqt::test::value(
      evaluateJSON(toManifestJSON(multiplexer),
                   R"({"schema_version":1,"counts":{"00":8,"01":2}})"));
  EXPECT_NE(multiplexerEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(multiplexerEvaluation.find("\"total_variation_distance\":"),
            std::string::npos);

  const auto modularMultiplier = ::mqt::test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }));
  const auto modularMultiplierEvaluation = ::mqt::test::value(evaluateJSON(
      toManifestJSON(modularMultiplier),
      R"({"schema_version":1,"counts":{"00000000":1,"10000000":1,"00010000":1,"10010011":1,"00100000":1,"10100001":1,"00110000":1,"10110100":1,"01000000":1,"11000010":1,"01010000":1,"11010000":1,"01100000":1,"11100011":1,"01110000":1,"11110001":1}})"));
  EXPECT_NE(modularMultiplierEvaluation.find("\"success_probability\":1.0"),
            std::string::npos);
  EXPECT_NE(
      modularMultiplierEvaluation.find("\"total_variation_distance\":0.0"),
      std::string::npos);

  const auto qftAdder = ::mqt::test::value(
      QFTAdder::create({.addend = "++", .accumulator = "01"}));
  const auto qftAdderEvaluation = ::mqt::test::value(evaluateJSON(
      toManifestJSON(qftAdder),
      R"({"schema_version":1,"counts":{"0001":1,"0110":1,"1011":1,"1100":1}})"));
  EXPECT_NE(qftAdderEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(qftAdderEvaluation.find("\"total_variation_distance\":0.0"),
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
  EXPECT_EQ(toManifestJSON(::mqt::test::value(
                qftAdderFromManifestJSON(toManifestJSON(constant)))),
            toManifestJSON(constant));
  EXPECT_NE(caseId(constant), caseId(::mqt::test::value(QFTAdder::create(
                                  {.addend = "110", .accumulator = "001"}))));

  const Teleportation teleportation;
  const auto teleportationEvaluation = ::mqt::test::value(
      evaluateJSON(toManifestJSON(teleportation),
                   R"({"schema_version":1,"counts":{"0":8}})"));
  EXPECT_NE(teleportationEvaluation.find("\"success_probability\":1.0"),
            std::string::npos);
  EXPECT_NE(teleportationEvaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);
  EXPECT_NE(teleportationEvaluation.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);

  const auto repeatUntilSuccess =
      ::mqt::test::value(RepeatUntilSuccess::create());
  const auto repeatUntilSuccessEvaluation = ::mqt::test::value(
      evaluateJSON(toManifestJSON(repeatUntilSuccess),
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
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return evaluationToJSON(
                  "not-a-case", 1,
                  Evaluation{.totalVariationDistance = 0.,
                             .squaredHellingerFidelity = 1.,
                             .successProbability = std::nullopt});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return evaluationToJSON(
                  caseId(ghz), 1,
                  Evaluation{.totalVariationDistance =
                                 std::numeric_limits<double>::quiet_NaN(),
                             .squaredHellingerFidelity = 1.,
                             .successProbability = std::nullopt});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

} // namespace
