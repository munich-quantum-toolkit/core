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
#include "bench/ControlledMultiplicationModuloN.hpp"
#include "bench/Evaluation.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/JSON.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdderClassical.hpp"
#include "bench/QFTAdderQuantum.hpp"
#include "bench/QPE.hpp"
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
using mqt::bench::ControlledMultiplicationModuloN;
using mqt::bench::controlledMultiplicationModuloNFromInstanceSpecificationJSON;
using mqt::bench::controlledMultiplicationModuloNFromManifestJSON;
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
using mqt::bench::Multiplexer;
using mqt::bench::multiplexerFromInstanceSpecificationJSON;
using mqt::bench::multiplexerFromManifestJSON;
using mqt::bench::Phase;
using mqt::bench::QFT;
using mqt::bench::QFTAdderClassical;
using mqt::bench::qftAdderClassicalFromInstanceSpecificationJSON;
using mqt::bench::qftAdderClassicalFromManifestJSON;
using mqt::bench::QFTAdderQuantum;
using mqt::bench::qftAdderQuantumFromInstanceSpecificationJSON;
using mqt::bench::qftAdderQuantumFromManifestJSON;
using mqt::bench::qftFromInstanceSpecificationJSON;
using mqt::bench::qftFromManifestJSON;
using mqt::bench::QFTMethod;
using mqt::bench::QPE;
using mqt::bench::qpeFromInstanceSpecificationJSON;
using mqt::bench::qpeFromManifestJSON;
using mqt::bench::QPEMethod;
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

TEST(BenchmarkJSON,
     ParsesInstanceSpecificationsAndSerializesResolvedParameters) {
  const auto bv = bvFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"bv","parameters":{"hidden_bitstring":"101"}})");
  EXPECT_EQ(bv.options().method, BVMethod::Static);
  EXPECT_EQ(
      toInstanceSpecificationJSON(bv),
      R"({"benchmark":"bv","parameters":{"hidden_bitstring":"101","method":"static"},"schema_version":1})");

  const auto controlledMultiplicationModuloN =
      controlledMultiplicationModuloNFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"controlled-multiplication-modulo-n","parameters":{"multiplier":"011","modulus":"101"}})");
  EXPECT_EQ(controlledMultiplicationModuloN.options().multiplier, "011");
  EXPECT_EQ(controlledMultiplicationModuloN.options().modulus, "101");
  EXPECT_EQ(
      toInstanceSpecificationJSON(controlledMultiplicationModuloN),
      R"({"benchmark":"controlled-multiplication-modulo-n","parameters":{"modulus":"101","multiplier":"011"},"schema_version":1})");

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

  const auto qftAdderClassical = qftAdderClassicalFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft-adder-classical","parameters":{"addend":"001"}})");
  EXPECT_EQ(qftAdderClassical.options().addend, "001");
  EXPECT_EQ(qftAdderClassical.expectedResult(), "0010");
  EXPECT_EQ(
      toInstanceSpecificationJSON(qftAdderClassical),
      R"({"benchmark":"qft-adder-classical","parameters":{"addend":"001"},"schema_version":1})");

  const auto qftAdderQuantum = qftAdderQuantumFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft-adder-quantum","parameters":{"qubits":3}})");
  EXPECT_EQ(qftAdderQuantum.options().qubits, 3);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qftAdderQuantum),
      R"({"benchmark":"qft-adder-quantum","parameters":{"qubits":3},"schema_version":1})");

  const auto qpe = qpeFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":4,"phase":{"numerator":10,"denominator":8},"method":"iterative"}})");
  EXPECT_EQ(qpe.options().phase, Phase(1, 4));
  EXPECT_EQ(qpe.options().method, QPEMethod::Iterative);
  EXPECT_EQ(
      toInstanceSpecificationJSON(qpe),
      R"({"benchmark":"qpe","parameters":{"method":"iterative","phase":{"denominator":4,"numerator":1},"precision":4},"schema_version":1})");

  const auto teleportation = teleportationFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"teleportation","parameters":{}})");
  EXPECT_EQ(
      toInstanceSpecificationJSON(teleportation),
      R"({"benchmark":"teleportation","parameters":{},"schema_version":1})");
}

TEST(BenchmarkJSON, RoundTripsSelfCheckingManifests) {
  const BV bv{{.hiddenBitstring = "101", .method = BVMethod::Dynamic}};
  const ControlledMultiplicationModuloN controlledMultiplicationModuloN{
      {.multiplier = "011", .modulus = "101"}};
  const GHZ ghz{
      {.qubits = 4, .topology = GHZTopology::Star, .basis = GHZBasis::X}};
  const Grover grover{{.markedBitstring = "001", .iterations = 2}};
  const Multiplexer multiplexer{{.qubits = 7}};
  const QFT qft{
      {.qubits = 4, .periodExponent = 2, .method = QFTMethod::Semiclassical}};
  const QFTAdderClassical qftAdderClassical{{.addend = "110"}};
  const QFTAdderQuantum qftAdderQuantum{{.qubits = 3}};
  const QPE qpe{
      {.precision = 5, .phase = Phase(1, 3), .method = QPEMethod::Iterative}};
  const Teleportation teleportation;

  const auto bvManifest = toManifestJSON(bv);
  const auto controlledMultiplicationModuloNManifest =
      toManifestJSON(controlledMultiplicationModuloN);
  const auto ghzManifest = toManifestJSON(ghz);
  const auto groverManifest = toManifestJSON(grover);
  const auto multiplexerManifest = toManifestJSON(multiplexer);
  const auto qftManifest = toManifestJSON(qft);
  const auto qftAdderClassicalManifest = toManifestJSON(qftAdderClassical);
  const auto qftAdderQuantumManifest = toManifestJSON(qftAdderQuantum);
  const auto qpeManifest = toManifestJSON(qpe);
  const auto teleportationManifest = toManifestJSON(teleportation);
  EXPECT_EQ(toManifestJSON(bvFromManifestJSON(bvManifest)), bvManifest);
  EXPECT_EQ(toManifestJSON(controlledMultiplicationModuloNFromManifestJSON(
                controlledMultiplicationModuloNManifest)),
            controlledMultiplicationModuloNManifest);
  EXPECT_EQ(toManifestJSON(ghzFromManifestJSON(ghzManifest)), ghzManifest);
  EXPECT_EQ(toManifestJSON(groverFromManifestJSON(groverManifest)),
            groverManifest);
  EXPECT_EQ(toManifestJSON(multiplexerFromManifestJSON(multiplexerManifest)),
            multiplexerManifest);
  EXPECT_EQ(toManifestJSON(qftFromManifestJSON(qftManifest)), qftManifest);
  EXPECT_EQ(toManifestJSON(
                qftAdderClassicalFromManifestJSON(qftAdderClassicalManifest)),
            qftAdderClassicalManifest);
  EXPECT_EQ(
      toManifestJSON(qftAdderQuantumFromManifestJSON(qftAdderQuantumManifest)),
      qftAdderQuantumManifest);
  EXPECT_EQ(toManifestJSON(qpeFromManifestJSON(qpeManifest)), qpeManifest);
  EXPECT_EQ(
      toManifestJSON(teleportationFromManifestJSON(teleportationManifest)),
      teleportationManifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(bvManifest), "bv");
  EXPECT_EQ(
      benchmarkIdFromManifestJSON(controlledMultiplicationModuloNManifest),
      "controlled-multiplication-modulo-n");
  EXPECT_EQ(benchmarkIdFromManifestJSON(ghzManifest), "ghz");
  EXPECT_EQ(benchmarkIdFromManifestJSON(groverManifest), "grover");
  EXPECT_EQ(benchmarkIdFromManifestJSON(multiplexerManifest), "multiplexer");
  EXPECT_EQ(benchmarkIdFromManifestJSON(qftManifest), "qft");
  EXPECT_EQ(benchmarkIdFromManifestJSON(qftAdderClassicalManifest),
            "qft-adder-classical");
  EXPECT_EQ(benchmarkIdFromManifestJSON(qftAdderQuantumManifest),
            "qft-adder-quantum");
  EXPECT_EQ(benchmarkIdFromManifestJSON(qpeManifest), "qpe");
  EXPECT_EQ(benchmarkIdFromManifestJSON(teleportationManifest),
            "teleportation");
  EXPECT_NE(ghzManifest.find("\"case_id\":\"" + caseId(ghz) + "\""),
            std::string::npos);
  EXPECT_NE(groverManifest.find("\"success_outcome\":\"001\""),
            std::string::npos);
  EXPECT_NE(controlledMultiplicationModuloNManifest.find(
                "\"model\":\"controlled_multiplication_modulo_n\""),
            std::string::npos);
  EXPECT_NE(controlledMultiplicationModuloNManifest.find("\"width\":8"),
            std::string::npos);
  EXPECT_NE(multiplexerManifest.find("\"model\":\"multiplexer\""),
            std::string::npos);
  EXPECT_NE(qftAdderClassicalManifest.find("\"model\":\"qft_adder_classical\""),
            std::string::npos);
  EXPECT_NE(qftAdderClassicalManifest.find("\"success_outcome\":\"0111\""),
            std::string::npos);
  EXPECT_NE(qftAdderClassicalManifest.find("\"width\":4"), std::string::npos);
  EXPECT_NE(qftAdderQuantumManifest.find("\"model\":\"qft_adder_quantum\""),
            std::string::npos);
  EXPECT_NE(qftAdderQuantumManifest.find("\"width\":6"), std::string::npos);
  EXPECT_EQ(qpeManifest.find("0.333"), std::string::npos);
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
  EXPECT_EQ(caseId(ControlledMultiplicationModuloN{
                {.multiplier = "011", .modulus = "101"}}),
            caseId(ControlledMultiplicationModuloN{
                {.multiplier = "011", .modulus = "101"}}));
  EXPECT_NE(caseId(ControlledMultiplicationModuloN{
                {.multiplier = "011", .modulus = "101"}}),
            caseId(ControlledMultiplicationModuloN{
                {.multiplier = "001", .modulus = "101"}}));
  EXPECT_NE(caseId(QFT{{.qubits = 3, .periodExponent = 1}}),
            caseId(QFT{{.qubits = 3,
                        .periodExponent = 1,
                        .method = QFTMethod::Semiclassical}}));
  EXPECT_EQ(caseId(QFTAdderClassical{{.addend = "001"}}),
            caseId(QFTAdderClassical{{.addend = "001"}}));
  EXPECT_NE(caseId(QFTAdderClassical{{.addend = "001"}}),
            caseId(QFTAdderClassical{{.addend = "1"}}));
  EXPECT_EQ(caseId(QFTAdderQuantum{{.qubits = 3}}),
            caseId(QFTAdderQuantum{{.qubits = 3}}));
  EXPECT_NE(caseId(QFTAdderQuantum{{.qubits = 3}}),
            caseId(QFTAdderQuantum{{.qubits = 4}}));
  EXPECT_EQ(caseId(Multiplexer{{.qubits = 7}}),
            caseId(Multiplexer{{.qubits = 7}}));
  EXPECT_NE(caseId(Multiplexer{{.qubits = 7}}),
            caseId(Multiplexer{{.qubits = 6}}));
  EXPECT_EQ(caseId(Teleportation{}), "sha256-8abc3c4e4adb4f0fde27c0d3562acddb"
                                     "8c79442fbadf613098878d448b302251");
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
        static_cast<
            void>(controlledMultiplicationModuloNFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"controlled-multiplication-modulo-n","parameters":{"multiplier":"011","modulus":"101","control":true}})"));
      },
      "unknown key 'control'");
  expectInvalid(
      [] {
        static_cast<
            void>(controlledMultiplicationModuloNFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"controlled-multiplication-modulo-n","parameters":{"multiplier":"011","modulus":"1001"}})"));
      },
      "equal widths");
  expectInvalid(
      [] {
        static_cast<
            void>(controlledMultiplicationModuloNFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"controlled-multiplication-modulo-n","parameters":{"multiplier":"011","modulus":"010"}})"));
      },
      "canonical");
  expectInvalid(
      [] {
        static_cast<
            void>(controlledMultiplicationModuloNFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"controlled-multiplication-modulo-n","parameters":{"multiplier":"101","modulus":"101"}})"));
      },
      "0 < a < N");
  expectInvalid(
      [] {
        static_cast<void>(qftAdderClassicalFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qft-adder-classical","parameters":{"addend":""}})"));
      },
      "between 1 and 1023 bits");
  expectInvalid(
      [] {
        static_cast<void>(qftAdderClassicalFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qft-adder-classical","parameters":{"addend":"01x"}})"));
      },
      "only '0' and '1'");
  expectInvalid(
      [] {
        static_cast<void>(qftAdderClassicalFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qft-adder-classical","parameters":{"addend":"1","qubits":2}})"));
      },
      "unknown key 'qubits'");
  expectInvalid(
      [] {
        static_cast<void>(qftAdderQuantumFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qft-adder-quantum","parameters":{"qubits":0}})"));
      },
      "between 1 and 1024");
  expectInvalid(
      [] {
        static_cast<void>(qftAdderQuantumFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qft-adder-quantum","parameters":{"qubits":3,"addend":"1"}})"));
      },
      "unknown key 'addend'");
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
      R"({"benchmarks":[{"definition_version":1,"id":"bv"},{"definition_version":1,"id":"controlled-multiplication-modulo-n"},{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"multiplexer"},{"definition_version":1,"id":"qft"},{"definition_version":1,"id":"qft-adder-classical"},{"definition_version":1,"id":"qft-adder-quantum"},{"definition_version":1,"id":"qpe"},{"definition_version":1,"id":"teleportation"}],"schema_version":1})");
  const auto bv = describeBenchmarkJSON("bv");
  const auto controlledMultiplicationModuloN =
      describeBenchmarkJSON("controlled-multiplication-modulo-n");
  const auto ghz = describeBenchmarkJSON("ghz");
  const auto grover = describeBenchmarkJSON("grover");
  const auto multiplexer = describeBenchmarkJSON("multiplexer");
  const auto qft = describeBenchmarkJSON("qft");
  const auto qftAdderClassical = describeBenchmarkJSON("qft-adder-classical");
  const auto qftAdderQuantum = describeBenchmarkJSON("qft-adder-quantum");
  const auto qpe = describeBenchmarkJSON("qpe");
  const auto teleportation = describeBenchmarkJSON("teleportation");
  EXPECT_NE(ghz.find("https://json-schema.org/draft/2020-12/schema"),
            std::string::npos);
  EXPECT_NE(ghz.find("\"additionalProperties\":false"), std::string::npos);
  EXPECT_NE(ghz.find("\"maximum\":1000000"), std::string::npos);
  EXPECT_NE(ghz.find("\"maximum\":1075"), std::string::npos);
  EXPECT_NE(bv.find("\"dynamic\""), std::string::npos);
  EXPECT_NE(controlledMultiplicationModuloN.find("\"maxLength\":63"),
            std::string::npos);
  EXPECT_NE(controlledMultiplicationModuloN.find("\"pattern\":\"^1[01]+$\""),
            std::string::npos);
  EXPECT_NE(grover.find("\"maxLength\":62"), std::string::npos);
  EXPECT_NE(multiplexer.find("\"maximum\":1024"), std::string::npos);
  EXPECT_NE(multiplexer.find("\"minimum\":2"), std::string::npos);
  EXPECT_NE(qft.find("\"period_exponent\""), std::string::npos);
  EXPECT_NE(qftAdderClassical.find("\"maxLength\":1023"), std::string::npos);
  EXPECT_NE(qftAdderClassical.find("\"pattern\":\"^[01]+$\""),
            std::string::npos);
  EXPECT_NE(qftAdderQuantum.find("\"maximum\":1024"), std::string::npos);
  EXPECT_NE(qftAdderQuantum.find("\"minimum\":1"), std::string::npos);
  EXPECT_NE(qpe.find("\"iterative\""), std::string::npos);
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

  const ControlledMultiplicationModuloN controlledMultiplicationModuloN{
      {.multiplier = "011", .modulus = "101"}};
  const auto controlledMultiplicationModuloNEvaluation = evaluateJSON(
      toManifestJSON(controlledMultiplicationModuloN),
      R"({"schema_version":1,"counts":{"00000000":1,"10000000":1,"00010000":1,"10010011":1,"00100000":1,"10100001":1,"00110000":1,"10110100":1,"01000000":1,"11000010":1,"01010000":1,"11010000":1,"01100000":1,"11100011":1,"01110000":1,"11110001":1}})");
  EXPECT_NE(controlledMultiplicationModuloNEvaluation.find(
                "\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(controlledMultiplicationModuloNEvaluation.find(
                "\"total_variation_distance\":0.0"),
            std::string::npos);

  const QFTAdderQuantum qftAdderQuantum{{.qubits = 2}};
  const auto qftAdderQuantumEvaluation = evaluateJSON(
      toManifestJSON(qftAdderQuantum),
      R"({"schema_version":1,"counts":{"0001":1,"0110":1,"1011":1,"1100":1}})");
  EXPECT_NE(qftAdderQuantumEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(qftAdderQuantumEvaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);

  const QFTAdderClassical qftAdderClassical{{.addend = "110"}};
  const auto qftAdderClassicalEvaluation =
      evaluateJSON(toManifestJSON(qftAdderClassical),
                   R"({"schema_version":1,"counts":{"0111":8,"0110":2}})");
  EXPECT_NE(qftAdderClassicalEvaluation.find("\"success_probability\":0.8"),
            std::string::npos);
  EXPECT_NE(qftAdderClassicalEvaluation.find("\"total_variation_distance\":"),
            std::string::npos);

  const Teleportation teleportation;
  const auto teleportationEvaluation = evaluateJSON(
      toManifestJSON(teleportation),
      R"({"schema_version":1,"counts":{"000":1,"001":1,"010":1,"011":1,"100":1,"101":1,"110":1,"111":1}})");
  EXPECT_NE(teleportationEvaluation.find("\"success_probability\":null"),
            std::string::npos);
  EXPECT_NE(teleportationEvaluation.find("\"total_variation_distance\":0.0"),
            std::string::npos);
  EXPECT_NE(teleportationEvaluation.find("\"squared_hellinger_fidelity\":1.0"),
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
