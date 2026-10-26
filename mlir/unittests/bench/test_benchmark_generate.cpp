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
#include "bench/FanOut.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/GroverWeakMeasurement.hpp"
#include "bench/JSON.hpp"
#include "bench/MagicStateDistillation.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QAOA.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Shor.hpp"
#include "bench/Teleportation.hpp"
#include "bench/WState.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QIR/Execution/JIT/Session.h"
#include "mqt/Dialect/QIR/Execution/Runtime/Runtime.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, ResolvesInstanceMetadata) {
  const auto benchmark = llvm::cantFail(GHZ::create({.qubits = 3}));
  auto generated = generate(
      R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":3}})");
  ASSERT_TRUE(mlir::succeeded(generated));
  EXPECT_EQ(generated->benchmarkId, "ghz");
  EXPECT_EQ(generated->caseId, caseId(benchmark));
  EXPECT_EQ(generated->manifestJSON, toManifestJSON(benchmark));
  EXPECT_EQ(
      llvm::cantFail(ghzFromManifestJSON(generated->manifestJSON)).output(),
      benchmark.output());
}

TEST(GenerateProgramTest, RejectsInvalidInstanceSpecifications) {
  for (
      const auto* specification : {
          R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":0}})",
          R"({"schema_version":1,"benchmark":"unknown","parameters":{}})",
      }) {
    EXPECT_TRUE(mlir::failed(generate(specification)));
  }
}

template <class Benchmark>
static void expectQCAndJeff(const Benchmark& benchmark) {
  auto program = generate(benchmark);
  ASSERT_TRUE(mlir::succeeded(program));
  test::expectJeffRoundTrip(std::move(*program));
}

TEST(GenerateProgramTest, GeneratesEveryBenchmarkMethodAsQCAndJeff) {
  expectQCAndJeff(llvm::cantFail(BV::create({.hiddenBitstring = "101"})));
  expectQCAndJeff(llvm::cantFail(
      BV::create({.hiddenBitstring = "101", .method = BVMethod::Dynamic})));
  expectQCAndJeff(llvm::cantFail(FanOut::create({.controls = 3})));
  expectQCAndJeff(llvm::cantFail(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  })));
  expectQCAndJeff(llvm::cantFail(GHZ::create({.qubits = 3})));
  expectQCAndJeff(llvm::cantFail(Grover::create({.markedBitstring = "101"})));
  expectQCAndJeff(llvm::cantFail(
      GroverWeakMeasurement::create({.markedBitstring = "101"})));
  expectQCAndJeff(
      llvm::cantFail(MagicStateDistillation::create({.levels = 1})));
  expectQCAndJeff(llvm::cantFail(Multiplexer::create({.qubits = 3})));
  expectQCAndJeff(llvm::cantFail(QAOA::create({.qubits = 3})));
  expectQCAndJeff(
      llvm::cantFail(QFT::create({.qubits = 3, .periodExponent = 1})));
  expectQCAndJeff(llvm::cantFail(QFT::create(
      {.qubits = 3, .periodExponent = 1, .method = QFTMethod::Semiclassical})));
  expectQCAndJeff(llvm::cantFail(QFTAdder::create({
      .addend = "101",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  })));
  expectQCAndJeff(llvm::cantFail(
      QFTAdder::create({.addend = "+++", .accumulator = "001"})));
  expectQCAndJeff(llvm::cantFail(QPE::create(
      {.precision = 3, .phase = llvm::cantFail(Phase::create(3, 8))})));
  expectQCAndJeff(llvm::cantFail(QPE::create({
      .precision = 3,
      .phase = llvm::cantFail(Phase::create(3, 8)),
      .method = QPEMethod::Iterative,
  })));
  expectQCAndJeff(llvm::cantFail(RepeatUntilSuccess::create()));
  expectQCAndJeff(llvm::cantFail(Shor::create({.number = 15})));
  expectQCAndJeff(Teleportation{});
  expectQCAndJeff(llvm::cantFail(WState::create({.qubits = 3})));
}

template <class Benchmark>
static void expectReference(const Benchmark& benchmark, const Counts& counts) {
  EXPECT_LT(llvm::cantFail(benchmark.evaluate(counts)).totalVariationDistance,
            0.08);
}

static void expectReference(const Shor& benchmark, const Counts& counts) {
  const auto result = llvm::cantFail(benchmark.evaluate(counts));
  ASSERT_TRUE(result.factors);
  EXPECT_EQ(result.factors->first * result.factors->second,
            benchmark.options().number);
  for (const auto& [outcome, count] : counts) {
    EXPECT_TRUE(outcome == "00000000" || outcome == "01000000" ||
                outcome == "10000000" || outcome == "11000000");
  }
}

template <class Benchmark>
static void expectQIRSampling(const Benchmark& benchmark, QIRProgram qirProgram,
                              size_t shots) {
  const auto bitcode = std::move(qirProgram).intoBitcode();
  ASSERT_TRUE(mlir::succeeded(bitcode));
  ASSERT_FALSE(bitcode->empty());
  qir::JitSession session(
      llvm::StringRef(reinterpret_cast<const char*>(bitcode->data()),
                      bitcode->size()),
      "benchmark-bitcode", qir::Execution::Sampling, 17);
  session.runtime().disableOutput();
  std::vector<std::string> outcomes;
  ASSERT_EQ(session.sample(shots, outcomes), 0);
  ASSERT_EQ(outcomes.size(), shots);
  Counts qirCounts;
  for (const auto& outcome : outcomes) {
    // QIR records bit zero first; benchmark evaluators use big-endian strings.
    ++qirCounts[std::string(outcome.rbegin(), outcome.rend())];
  }
  expectReference(benchmark, qirCounts);
}

template <class Benchmark>
static void expectPortableExecution(const Benchmark& benchmark, size_t shots) {
  auto qc = generate(benchmark);
  ASSERT_TRUE(mlir::succeeded(qc));
  auto compiled =
      runDefaultPipeline(CompilerInput{qc->copy()}, ProgramFormat::Jeff);
  MQT_ASSERT_SUCCESS(compiled);
  auto& jeff = std::get<JeffProgram>(*compiled);
  const auto bytes = jeff.toBytes();
  ASSERT_FALSE(bytes.empty());
  auto restored = JeffProgram::fromBytes(bytes);
  MQT_ASSERT_SUCCESS(restored);
  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  auto counts = qco::sample(mlir::mqt::getEntryPoint(qco->module()), shots, 17);
  ASSERT_TRUE(succeeded(counts));
  expectReference(benchmark, *counts);

  compiled = runDefaultPipeline(CompilerInput{std::move(*qc)},
                                ProgramFormat::QIRAdaptive);
  MQT_ASSERT_SUCCESS(compiled);
  expectQIRSampling(benchmark, std::get<QIRProgram>(std::move(*compiled)),
                    shots);
}

TEST(GenerateProgramTest, ExecutesRuntimePhasesThroughJeffAndAdaptiveQIR) {
  for (auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    for (auto phase : {
             llvm::cantFail(Phase::create(1, 3)),
             llvm::cantFail(Phase::create(
                 uint64_t{1} << 63U, std::numeric_limits<uint64_t>::max())),
         }) {
      SCOPED_TRACE(static_cast<int>(method));
      SCOPED_TRACE(phase.numerator());
      expectPortableExecution(
          llvm::cantFail(
              QPE::create({.precision = 3, .phase = phase, .method = method})),
          2048);
    }
  }
  expectPortableExecution(llvm::cantFail(QFTAdder::create({
                              .addend = "101",
                              .accumulator = "011",
                              .method = QFTAdderMethod::Constant,
                              .overflow = QFTAdderOverflow::Carry,
                          })),
                          2048);
  expectPortableExecution(llvm::cantFail(ModularMultiplier::create({
                              .multiplier = "011",
                              .modulus = "101",
                              .multiplicand = "+++",
                              .control = '+',
                          })),
                          2048);
  expectPortableExecution(llvm::cantFail(Shor::create({.number = 15})), 64);
}

template <class Benchmark>
static void expectOpenQASMRoundTrip(const Benchmark& benchmark, size_t shots) {
  SCOPED_TRACE(toManifestJSON(benchmark));
  auto qc = generate(benchmark);
  ASSERT_TRUE(mlir::succeeded(qc));
  const bool hasLoop = test::countOps<scf::ForOp>(qc->module()) != 0;
  for (const bool optimize : {false, true}) {
    SCOPED_TRACE(optimize);
    FailureOr<OpenQASMProgram> qasm = failure();
    if (optimize) {
      auto compiled = runDefaultPipeline(CompilerInput{qc->copy()},
                                         ProgramFormat::OpenQASM3);
      MQT_ASSERT_SUCCESS(compiled);
      qasm = std::get<OpenQASMProgram>(std::move(*compiled));
    } else {
      qasm = qc->copy().intoOpenQASM3();
    }
    ASSERT_TRUE(mlir::succeeded(qasm));
    auto restored = QCProgram::fromOpenQASMString(qasm->source());
    MQT_ASSERT_SUCCESS(restored);
    if (hasLoop) {
      EXPECT_GT(test::countOps<scf::ForOp>(restored->module()), 0);
    }
    auto qco = std::move(*restored).intoQCO();
    ASSERT_TRUE(mlir::succeeded(qco));
    auto counts =
        qco::sample(mlir::mqt::getEntryPoint(qco->module()), shots, 17);
    ASSERT_TRUE(succeeded(counts));
    expectReference(benchmark, *counts);
  }
}

TEST(GenerateProgramTest,
     RoundTripsStructuredOpenQASMThroughBothCompilerPaths) {
  expectOpenQASMRoundTrip(
      llvm::cantFail(QPE::create({
          .precision = 8,
          .phase = llvm::cantFail(Phase::create(
              uint64_t{1} << 63U, std::numeric_limits<uint64_t>::max())),
      })),
      2048);
  expectOpenQASMRoundTrip(llvm::cantFail(QPE::create({
                              .precision = 8,
                              .phase = llvm::cantFail(Phase::create(3, 8)),
                              .method = QPEMethod::Iterative,
                          })),
                          2048);
  expectOpenQASMRoundTrip(llvm::cantFail(QFTAdder::create({
                              .addend = "101",
                              .accumulator = "011",
                              .method = QFTAdderMethod::Register,
                          })),
                          2048);
  expectOpenQASMRoundTrip(llvm::cantFail(Shor::create({.number = 15})), 64);
}

template <class Benchmark>
static void expectStaticTargetExecution(const Benchmark& benchmark) {
  auto qc = generate(benchmark);
  ASSERT_TRUE(mlir::succeeded(qc));
  auto target =
      CompilerTarget::create(16, CompilerTarget::Connectivity::allToAll(),
                             CompilerTarget::NativeOperations::unrestricted());
  MQT_ASSERT_SUCCESS(target);
  auto basePayload = PayloadSpecification::create(
      {.id = "qir", .version = "2.1", .profile = "base"});
  MQT_ASSERT_SUCCESS(basePayload);
  const TargetEnvironment baseTarget(*target, std::move(*basePayload));
  auto compiled = runDefaultPipeline(CompilerInput{qc->copy()}, baseTarget);
  MQT_ASSERT_SUCCESS(compiled);
  auto& qirProgram = std::get<QIRProgram>(*compiled);
  EXPECT_EQ(qirProgram.profile(), QIRProfile::Base);
  expectQIRSampling(benchmark, std::move(qirProgram), 2048);

  auto qasmPayload =
      PayloadSpecification::create({.id = "openqasm", .version = "3.1"});
  MQT_ASSERT_SUCCESS(qasmPayload);
  const TargetEnvironment qasmTarget(*target, std::move(*qasmPayload));
  compiled = runDefaultPipeline(CompilerInput{std::move(*qc)}, qasmTarget);
  MQT_ASSERT_SUCCESS(compiled);
  auto restored = QCProgram::fromOpenQASMString(
      std::get<OpenQASMProgram>(*compiled).source());
  MQT_ASSERT_SUCCESS(restored);
  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  auto sampled = qco::sample(mlir::mqt::getEntryPoint(qco->module()), 2048, 17);
  ASSERT_TRUE(succeeded(sampled));
  expectReference(benchmark, *sampled);
}

TEST(GenerateProgramTest, CompilesRuntimePhasesForStaticTargets) {
  expectStaticTargetExecution(llvm::cantFail(QPE::create(
      {.precision = 3, .phase = llvm::cantFail(Phase::create(1, 3))})));
  expectStaticTargetExecution(llvm::cantFail(QFTAdder::create({
      .addend = "101",
      .accumulator = "011",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  })));
  expectStaticTargetExecution(llvm::cantFail(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  })));
}

TEST(GenerateProgramTest, RoundTripsWStateThroughOpenQASM) {
  const auto benchmark = llvm::cantFail(WState::create({.qubits = 3}));
  auto qc = generate(benchmark);
  ASSERT_TRUE(mlir::succeeded(qc));
  auto qasm = std::move(*qc).intoOpenQASM3();
  ASSERT_TRUE(mlir::succeeded(qasm));
  auto restored = QCProgram::fromOpenQASMString(qasm->source());
  MQT_ASSERT_SUCCESS(restored);
  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  auto counts = qco::sample(mlir::mqt::getEntryPoint(qco->module()), 2048, 17);
  ASSERT_TRUE(succeeded(counts));
  expectReference(benchmark, *counts);
}

} // namespace mqt::bench
