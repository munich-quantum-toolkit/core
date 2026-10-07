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
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/MagicStateDistillation.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Shor.hpp"
#include "bench/Teleportation.hpp"
#include "bench/WState.hpp"
#include "bench/WeakMeasurementGrover.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QIR/Execution/JIT/Session.h"
#include "mqt/Dialect/QIR/Execution/Runtime/Runtime.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bench {

using namespace mlir;

template <class Benchmark>
static void expectQCAndJeff(const Benchmark& benchmark) {
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  test::expectJeffRoundTrip(std::move(*program));
}

TEST(GenerateProgramTest, GeneratesEveryBenchmarkMethodAsQCAndJeff) {
  expectQCAndJeff(BV{{.hiddenBitstring = "101"}});
  expectQCAndJeff(BV{{.hiddenBitstring = "101", .method = BVMethod::Dynamic}});
  expectQCAndJeff(ModularMultiplier{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }});
  expectQCAndJeff(GHZ{{.qubits = 3}});
  expectQCAndJeff(Grover{{.markedBitstring = "101"}});
  expectQCAndJeff(MagicStateDistillation{{.levels = 1}});
  expectQCAndJeff(Multiplexer{{.qubits = 3}});
  expectQCAndJeff(QFT{{.qubits = 3, .periodExponent = 1}});
  expectQCAndJeff(QFT{
      {.qubits = 3, .periodExponent = 1, .method = QFTMethod::Semiclassical}});
  expectQCAndJeff(QFTAdder{{
      .addend = "101",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }});
  expectQCAndJeff(QFTAdder{{.addend = "+++", .accumulator = "001"}});
  expectQCAndJeff(QPE{{.precision = 3, .phase = Phase(3, 8)}});
  expectQCAndJeff(QPE{
      {.precision = 3, .phase = Phase(3, 8), .method = QPEMethod::Iterative}});
  expectQCAndJeff(RepeatUntilSuccess{});
  expectQCAndJeff(Shor{{.number = 15}});
  expectQCAndJeff(Teleportation{});
  expectQCAndJeff(WState{{.qubits = 3}});
  expectQCAndJeff(WeakMeasurementGrover{{.markedBitstring = "101"}});
}

template <class Benchmark>
static void expectReference(const Benchmark& benchmark, const Counts& counts) {
  EXPECT_LT(benchmark.evaluate(counts).totalVariationDistance, 0.08);
}

static void expectReference(const Shor& benchmark, const Counts& counts) {
  const auto result = benchmark.evaluate(counts);
  ASSERT_TRUE(result.factors);
  EXPECT_EQ(result.factors->first * result.factors->second,
            benchmark.options().number);
  for (const auto& [outcome, count] : counts) {
    EXPECT_TRUE(outcome == "00000000" || outcome == "01000000" ||
                outcome == "10000000" || outcome == "11000000");
  }
}

template <class Benchmark>
static void expectQIRSampling(const Benchmark& benchmark,
                              QIRProgram& qirProgram, size_t shots) {
  const auto bitcode = qirProgram.toBitcode();
  ASSERT_TRUE(bitcode);
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
    /// QIR records bit zero first; benchmark evaluators use big-endian strings.
    ++qirCounts[std::string(outcome.rbegin(), outcome.rend())];
  }
  expectReference(benchmark, qirCounts);
}

template <class Benchmark>
static void expectPortableExecution(const Benchmark& benchmark, size_t shots) {
  auto qc = generate(benchmark);
  ASSERT_TRUE(qc);
  auto compiled =
      runDefaultPipeline(CompilerInput{qc->copy()}, ProgramFormat::Jeff);
  ASSERT_TRUE(compiled);
  auto& jeff = std::get<JeffProgram>(*compiled);
  const auto bytes = jeff.toBytes();
  ASSERT_FALSE(bytes.empty());
  auto restored = JeffProgram::fromBytes(bytes);
  ASSERT_TRUE(restored);
  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(qco);
  auto counts = qco::sample(mlir::mqt::getEntryPoint(qco->module()), shots, 17);
  ASSERT_TRUE(succeeded(counts));
  expectReference(benchmark, *counts);

  compiled = runDefaultPipeline(CompilerInput{std::move(*qc)},
                                ProgramFormat::QIRAdaptive);
  ASSERT_TRUE(compiled);
  expectQIRSampling(benchmark, std::get<QIRProgram>(*compiled), shots);
}

TEST(GenerateProgramTest, ExecutesRuntimePhasesThroughJeffAndAdaptiveQIR) {
  for (auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    for (auto phase : {
             Phase(1, 3),
             Phase(uint64_t{1} << 63U, std::numeric_limits<uint64_t>::max()),
         }) {
      SCOPED_TRACE(static_cast<int>(method));
      SCOPED_TRACE(phase.numerator());
      expectPortableExecution(
          QPE{{.precision = 3, .phase = phase, .method = method}}, 2048);
    }
  }
  expectPortableExecution(QFTAdder{{
                              .addend = "101",
                              .accumulator = "011",
                              .method = QFTAdderMethod::Constant,
                              .overflow = QFTAdderOverflow::Carry,
                          }},
                          2048);
  expectPortableExecution(ModularMultiplier{{
                              .multiplier = "011",
                              .modulus = "101",
                              .multiplicand = "+++",
                              .control = '+',
                          }},
                          2048);
  expectPortableExecution(Shor{{.number = 15}}, 64);
}

TEST(GenerateProgramTest, RoundTripsRuntimePhasesThroughOpenQASM) {
  for (auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    /// Direct import cannot prove nested QFT indices for larger registers.
    const size_t precision = method == QPEMethod::Standard ? 1U : 8U;
    const auto phase =
        method == QPEMethod::Standard
            ? Phase(uint64_t{1} << 63U, std::numeric_limits<uint64_t>::max())
            : Phase(3, 8);
    SCOPED_TRACE(static_cast<int>(method));
    SCOPED_TRACE(phase.numerator());
    const QPE benchmark{
        {.precision = precision, .phase = phase, .method = method}};
    auto qc = generate(benchmark);
    ASSERT_TRUE(qc);
    auto qasm = qc->toOpenQASM3();
    ASSERT_TRUE(qasm);
    auto restored = QCProgram::fromOpenQASMString(qasm->source());
    ASSERT_TRUE(restored);
    auto qco = std::move(*restored).intoQCO();
    ASSERT_TRUE(qco);
    auto counts =
        qco::sample(mlir::mqt::getEntryPoint(qco->module()), 2048, 17);
    ASSERT_TRUE(succeeded(counts));
    expectReference(benchmark, *counts);
  }
}

template <class Benchmark>
static void expectStaticTargetExecution(const Benchmark& benchmark) {
  auto qc = generate(benchmark);
  ASSERT_TRUE(qc);
  auto target =
      CompilerTarget::create(16, CompilerTarget::Connectivity::allToAll(),
                             CompilerTarget::NativeOperations::unrestricted());
  ASSERT_TRUE(static_cast<bool>(target));
  auto basePayload = PayloadSpecification::create(
      {.id = "qir", .version = "2.1", .profile = "base"});
  ASSERT_TRUE(static_cast<bool>(basePayload));
  const TargetEnvironment baseTarget(*target, std::move(*basePayload));
  auto compiled = runDefaultPipeline(CompilerInput{qc->copy()}, baseTarget);
  ASSERT_TRUE(compiled);
  auto& qirProgram = std::get<QIRProgram>(*compiled);
  EXPECT_EQ(qirProgram.profile(), QIRProfile::Base);
  expectQIRSampling(benchmark, qirProgram, 2048);

  auto qasmPayload =
      PayloadSpecification::create({.id = "openqasm", .version = "3.1"});
  ASSERT_TRUE(static_cast<bool>(qasmPayload));
  const TargetEnvironment qasmTarget(*target, std::move(*qasmPayload));
  compiled = runDefaultPipeline(CompilerInput{std::move(*qc)}, qasmTarget);
  ASSERT_TRUE(compiled);
  auto restored = QCProgram::fromOpenQASMString(
      std::get<OpenQASMProgram>(*compiled).source());
  ASSERT_TRUE(restored);
  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(qco);
  auto sampled = qco::sample(mlir::mqt::getEntryPoint(qco->module()), 2048, 17);
  ASSERT_TRUE(succeeded(sampled));
  expectReference(benchmark, *sampled);
}

TEST(GenerateProgramTest, ExportsConstantAdderRuntimePhasesToOpenQASM) {
  auto program = generate(QFTAdder{{
      .addend = "101",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }});
  ASSERT_TRUE(program);
  EXPECT_TRUE(program->toOpenQASM3());
}

TEST(GenerateProgramTest, CompilesRuntimePhasesForStaticTargets) {
  expectStaticTargetExecution(QPE{{.precision = 3, .phase = Phase(1, 3)}});
  expectStaticTargetExecution(QFTAdder{{
      .addend = "101",
      .accumulator = "011",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }});
  expectStaticTargetExecution(ModularMultiplier{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }});
}

TEST(GenerateProgramTest, RoundTripsWStateThroughOpenQASM) {
  const WState benchmark{{.qubits = 3}};
  auto qc = generate(benchmark);
  ASSERT_TRUE(qc);
  auto qasm = qc->toOpenQASM3();
  ASSERT_TRUE(qasm);
  auto restored = QCProgram::fromOpenQASMString(qasm->source());
  ASSERT_TRUE(restored);
  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(qco);
  auto counts = qco::sample(mlir::mqt::getEntryPoint(qco->module()), 2048, 17);
  ASSERT_TRUE(succeeded(counts));
  expectReference(benchmark, *counts);
}

} // namespace mqt::bench
