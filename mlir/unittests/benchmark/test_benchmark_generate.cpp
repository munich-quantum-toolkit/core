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
#include "mlir/Benchmark/Generate.h"
#include "mlir/Benchmark/Programs.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace mqt::benchmark {

namespace {

class GenerateBenchmarkTest : public testing::TestWithParam<Benchmark> {};

INSTANTIATE_TEST_SUITE_P(Benchmarks, GenerateBenchmarkTest,
                         testing::ValuesIn(benchmarks()),
                         [](const testing::TestParamInfo<Benchmark>& info) {
                           return testName(info.param.name);
                         });

TEST_P(GenerateBenchmarkTest, AcceptsTheMinimumSize) {
  const auto& benchmark = GetParam();
  EXPECT_TRUE(generateProgram(benchmark, benchmark.minimumSize));
}

TEST_P(GenerateBenchmarkTest, RejectsSizesBelowTheMinimum) {
  const auto& benchmark = GetParam();
  if (benchmark.minimumSize == 0) {
    GTEST_SKIP() << "the benchmark accepts every size";
  }
  EXPECT_FALSE(generateProgram(benchmark, benchmark.minimumSize - 1));
}

TEST_P(GenerateBenchmarkTest, RejectsSizesAboveTheMaximum) {
  const auto& benchmark = GetParam();
  if (benchmark.maximumSize == 0) {
    GTEST_SKIP() << "the benchmark has no upper size limit";
  }
  EXPECT_FALSE(generateProgram(benchmark, benchmark.maximumSize + 1));
}

/**
 * @brief The programs size their registers with signed dimensions.
 *
 * @details The first size that a signed dimension cannot hold must be
 * rejected, whichever limit the benchmark reaches first.
 */
TEST_P(GenerateBenchmarkTest, RejectsSizesBeyondTheSignedRange) {
  constexpr auto FIRST_UNSIGNED =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1;
  EXPECT_FALSE(generateProgram(GetParam(), FIRST_UNSIGNED));
}

} // namespace

} // namespace mqt::benchmark
