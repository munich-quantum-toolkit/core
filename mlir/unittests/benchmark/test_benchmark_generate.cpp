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
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <gtest/gtest.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace mqt::benchmark {

using namespace mlir;

/// A program that ignores its size and measures one qubit into one bit.
static SmallVector<Value> oneBit(qc::QCProgramBuilder& b, uint64_t /*n*/) {
  auto c = b.allocClassicalBitRegister(1, "c");
  auto q = b.allocQubit();
  b.reset(q);
  b.measure(q, c, 0);
  return {c};
}

namespace {

/// A benchmark that accepts every size from two upwards.
constexpr Benchmark UNBOUNDED{
    .name = "unbounded", .build = &oneBit, .minimumSize = 2};

/// A benchmark that accepts the sizes from two to four.
constexpr Benchmark BOUNDED{
    .name = "bounded", .build = &oneBit, .minimumSize = 2, .maximumSize = 4};

TEST(GenerateProgramTest, AcceptsTheMinimumSize) {
  EXPECT_TRUE(generateProgram(UNBOUNDED, UNBOUNDED.minimumSize));
}

TEST(GenerateProgramTest, RejectsSizesBelowTheMinimum) {
  EXPECT_FALSE(generateProgram(UNBOUNDED, UNBOUNDED.minimumSize - 1));
}

TEST(GenerateProgramTest, AcceptsTheMaximumSize) {
  EXPECT_TRUE(generateProgram(BOUNDED, BOUNDED.maximumSize));
}

TEST(GenerateProgramTest, RejectsSizesAboveTheMaximum) {
  EXPECT_FALSE(generateProgram(BOUNDED, BOUNDED.maximumSize + 1));
}

TEST(GenerateProgramTest, AcceptsAnySizeWithoutAMaximum) {
  EXPECT_TRUE(generateProgram(UNBOUNDED, BOUNDED.maximumSize + 1));
}

TEST(GenerateProgramTest, RejectsSizesBeyondTheSignedRange) {
  constexpr auto firstUnsigned =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1;
  EXPECT_FALSE(generateProgram(UNBOUNDED, firstUnsigned));
}

class BenchmarkTest : public testing::TestWithParam<Benchmark> {};

INSTANTIATE_TEST_SUITE_P(Benchmarks, BenchmarkTest,
                         testing::ValuesIn(benchmarks()),
                         [](const testing::TestParamInfo<Benchmark>& info) {
                           auto name = info.param.name.str();
                           std::ranges::replace(name, '-', '_');
                           return name;
                         });

TEST_P(BenchmarkTest, BuildsAtItsMinimumSize) {
  const auto& benchmark = GetParam();
  EXPECT_TRUE(generateProgram(benchmark, benchmark.minimumSize));
}

} // namespace

} // namespace mqt::benchmark
