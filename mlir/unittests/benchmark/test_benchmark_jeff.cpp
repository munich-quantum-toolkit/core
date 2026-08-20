/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "BenchmarkTestUtils.h"
#include "mlir/Benchmark/Compile.h"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Compiler/Programs.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>

namespace mqt::benchmark {

namespace {

/// The size the benchmarks are generated at.
constexpr uint64_t JEFF_SIZE = 7;

class JeffBenchmarkTest : public testing::TestWithParam<Benchmark> {};

INSTANTIATE_TEST_SUITE_P(Benchmarks, JeffBenchmarkTest,
                         testing::ValuesIn(benchmarks()),
                         [](const testing::TestParamInfo<Benchmark>& info) {
                           return testName(info.param.name);
                         });

/**
 * @brief The benchmarks exist to exercise structured control flow.
 *
 * @details A program that loses its loops and conditionals on the way to `jeff`
 * no longer tests what it was written for, so every program must keep at least
 * one structured operation.
 */
TEST_P(JeffBenchmarkTest, KeepsStructuredControlFlow) {
  const auto& benchmark = GetParam();
  auto program = buildQCProgram(benchmark, JEFF_SIZE);
  ASSERT_TRUE(program.has_value());
  const auto compiled =
      mlir::runDefaultPipeline(std::move(*program), mlir::ProgramFormat::Jeff);
  ASSERT_TRUE(compiled.has_value());

  const auto assembly = std::get<mlir::JeffProgram>(*compiled).str();
  EXPECT_TRUE(assembly.find("jeff.for") != std::string::npos ||
              assembly.find("jeff.while") != std::string::npos ||
              assembly.find("jeff.switch") != std::string::npos)
      << assembly;
}

TEST_P(JeffBenchmarkTest, RejectsSizesBelowTheMinimum) {
  const auto& benchmark = GetParam();
  if (benchmark.minimumSize == 0) {
    GTEST_SKIP() << "the benchmark accepts every size";
  }
  EXPECT_FALSE(buildQCProgram(benchmark, benchmark.minimumSize - 1));
}

TEST_P(JeffBenchmarkTest, RejectsSizesAboveTheMaximum) {
  const auto& benchmark = GetParam();
  if (benchmark.maximumSize == 0) {
    GTEST_SKIP() << "the benchmark has no upper size limit";
  }
  EXPECT_FALSE(buildQCProgram(benchmark, benchmark.maximumSize + 1));
}

} // namespace

} // namespace mqt::benchmark
