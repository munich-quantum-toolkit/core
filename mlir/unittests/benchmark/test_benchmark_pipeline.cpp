/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/Generate.h"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Compiler/Programs.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace mqt::benchmark {

namespace {
/// The size the benchmarks are generated at. A program that needs more
/// qubits than this is generated at its own minimum.
constexpr uint64_t PIPELINE_SIZE = 7;
} // namespace

/**
 * @brief Runs one benchmark through the default pipeline to @p format.
 *
 * @details The formats the benchmarks reach are QCO, QC, `jeff`, and QIR for
 * the Adaptive Profile. OpenQASM 3 and the QIR Base Profile are left out: the
 * OpenQASM emitter needs constant qubit indices, and the Base Profile allows
 * neither control flow nor a classical register that is read at runtime, so
 * every program that indexes a register in a loop is outside both of them.
 */
static bool reaches(const Benchmark& benchmark,
                    const mlir::ProgramFormat format) {
  auto program = generateProgram(
      benchmark, std::max(PIPELINE_SIZE, benchmark.minimumSize));
  if (!program) {
    return false;
  }
  return mlir::runDefaultPipeline(std::move(*program), format).has_value();
}

namespace {

class PipelineBenchmarkTest : public testing::TestWithParam<Benchmark> {};

INSTANTIATE_TEST_SUITE_P(Benchmarks, PipelineBenchmarkTest,
                         testing::ValuesIn(benchmarks()),
                         [](const testing::TestParamInfo<Benchmark>& info) {
                           auto name = info.param.name.str();
                           std::ranges::replace(name, '-', '_');
                           return name;
                         });

TEST_P(PipelineBenchmarkTest, ReachesQCO) {
  EXPECT_TRUE(reaches(GetParam(), mlir::ProgramFormat::QCO));
}

TEST_P(PipelineBenchmarkTest, ReachesOptimizedQCO) {
  EXPECT_TRUE(reaches(GetParam(), mlir::ProgramFormat::QCOOptimized));
}

TEST_P(PipelineBenchmarkTest, ReachesQC) {
  EXPECT_TRUE(reaches(GetParam(), mlir::ProgramFormat::QC));
}

TEST_P(PipelineBenchmarkTest, ReachesJeff) {
  EXPECT_TRUE(reaches(GetParam(), mlir::ProgramFormat::Jeff));
}

TEST_P(PipelineBenchmarkTest, ReachesQIRAdaptive) {
  EXPECT_TRUE(reaches(GetParam(), mlir::ProgramFormat::QIRAdaptive));
}

} // namespace

} // namespace mqt::benchmark
