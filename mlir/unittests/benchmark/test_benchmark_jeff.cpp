/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Generate.h"
#include "programs/Programs.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace mqt::jeff::benchmarks {
namespace {

/// The size the benchmarks are generated at.
constexpr uint64_t SIZE = 7;

/// Replaces the characters that a test name cannot contain.
std::string testName(llvm::StringRef name) {
  auto sanitized = name.str();
  for (auto& character : sanitized) {
    if (character == '-') {
      character = '_';
    }
  }
  return sanitized;
}

class JeffBenchmarkTest : public testing::TestWithParam<Benchmark> {};

INSTANTIATE_TEST_SUITE_P(Benchmarks, JeffBenchmarkTest,
                         testing::ValuesIn(benchmarks()),
                         [](const testing::TestParamInfo<Benchmark>& info) {
                           return testName(info.param.name);
                         });

TEST_P(JeffBenchmarkTest, LowersToJeff) {
  const auto& benchmark = GetParam();
  const auto program = buildJeffProgram(benchmark, SIZE);
  ASSERT_TRUE(program.has_value());
  EXPECT_FALSE(program->toBytes().empty());
}

/**
 * @brief The benchmarks exist to exercise structured control flow.
 *
 * @details A program that loses its loops and conditionals on the way to `jeff`
 * no longer tests what it was written for, so every program must keep at least
 * one structured operation.
 */
TEST_P(JeffBenchmarkTest, KeepsStructuredControlFlow) {
  const auto& benchmark = GetParam();
  const auto program = buildJeffProgram(benchmark, SIZE);
  ASSERT_TRUE(program.has_value());

  const auto assembly = program->str();
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
  EXPECT_FALSE(buildJeffProgram(benchmark, benchmark.minimumSize - 1));
}

} // namespace
} // namespace mqt::jeff::benchmarks
