/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/GHZ.hpp"
#include "benchmarks/Grover.hpp"
#include "benchmarks/QPE.hpp"
#include "mlir/Benchmark/Generate.h"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
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

template <class Op> [[nodiscard]] size_t countOps(ModuleOp moduleOp) {
  size_t count = 0;
  moduleOp.walk([&](Op) { ++count; });
  return count;
}

[[nodiscard]] DenseElementsAttr angleTable(ModuleOp moduleOp) {
  DenseElementsAttr result;
  moduleOp.walk([&](arith::ConstantOp op) {
    if (const auto table = dyn_cast<DenseElementsAttr>(op.getValue())) {
      EXPECT_FALSE(result);
      result = table;
    }
  });
  return result;
}

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

TEST(GenerateProgramTest, EmitsConfiguredGHZWithoutEagerRegisterLoads) {
  const benchmarks::GHZ benchmark({.qubits = 64,
                                   .topology = benchmarks::GHZTopology::Star,
                                   .basis = benchmarks::GHZBasis::X});
  auto program = generateProgram(benchmark);
  ASSERT_TRUE(program);

  EXPECT_LT(countOps<memref::LoadOp>(program->module()), 10U);
}

TEST(GenerateProgramTest, EmitsDirectGroverOracleWithBigEndianMarkedState) {
  const benchmarks::Grover benchmark(
      {.markedBitstring = "01", .iterations = 2});
  auto program = generateProgram(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(countOps<qc::AllocOp>(moduleOp), 0U);
  EXPECT_EQ(countOps<qc::CtrlOp>(moduleOp), 2U);

  scf::ForOp iterations;
  moduleOp.walk([&](qc::CtrlOp op) {
    if (!iterations) {
      iterations = op->getParentOfType<scf::ForOp>();
    }
  });
  ASSERT_TRUE(iterations);

  size_t markedStateFlips = 0;
  iterations.walk([&](qc::XOp op) {
    if (op->getParentOp() != iterations.getOperation()) {
      return;
    }
    ++markedStateFlips;
    auto load = op.getQubit(0).getDefiningOp<memref::LoadOp>();
    ASSERT_TRUE(load);
    auto index =
        load.getIndices().front().getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(index);
    EXPECT_EQ(index.value(), 1);
  });
  EXPECT_EQ(markedStateFlips, 2U);
}

TEST(GenerateProgramTest, KeepsStandardQPEPowerAndResultOrderAligned) {
  const benchmarks::QPE benchmark(
      {.precision = 2, .phase = benchmarks::Phase(1, 4)});
  EXPECT_DOUBLE_EQ(benchmark.probability("01"), 1.);

  auto program = generateProgram(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  auto table = angleTable(moduleOp);
  ASSERT_TRUE(table);
  const auto angles = llvm::to_vector(table.getValues<double>());
  ASSERT_EQ(angles.size(), 2U);
  EXPECT_NEAR(angles[0], std::numbers::pi / 2., 1e-15);
  EXPECT_NEAR(angles[1], std::numbers::pi, 1e-15);

  tensor::ExtractOp extract;
  moduleOp.walk([&](tensor::ExtractOp op) { extract = op; });
  ASSERT_TRUE(extract);
  auto powerLoop = extract->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(powerLoop);
  EXPECT_EQ(extract.getIndices().front(), powerLoop.getInductionVar());

  qc::CtrlOp controlledPower;
  powerLoop.walk([&](qc::CtrlOp op) { controlledPower = op; });
  ASSERT_TRUE(controlledPower);
  auto controlLoad =
      controlledPower.getControl(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(controlLoad);
  auto controlIndex =
      controlLoad.getIndices().front().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(controlIndex);
  EXPECT_EQ(controlIndex.getRhs(), powerLoop.getInductionVar());

  qc::MeasureOp measure;
  moduleOp.walk([&](qc::MeasureOp op) { measure = op; });
  ASSERT_TRUE(measure);
  auto measurementLoop = measure->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(measurementLoop);
  auto measuredLoad = measure.getQubit().getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(measuredLoad);
  EXPECT_EQ(measuredLoad.getIndices().front(),
            measurementLoop.getInductionVar());

  ASSERT_FALSE(measure.getResult().use_empty());
  auto* user = *measure.getResult().getUsers().begin();
  auto store = dyn_cast<cbit::StoreOp>(user);
  ASSERT_TRUE(store);
  auto resultIndex = store.getIndex().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(resultIndex);
  EXPECT_EQ(resultIndex.getRhs(), measurementLoop.getInductionVar());
}

TEST(GenerateProgramTest, KeepsLargeQPEFiniteAndStructured) {
  constexpr size_t precision = 1025;
  for (const auto method :
       {benchmarks::QPEMethod::Standard, benchmarks::QPEMethod::Iterative}) {
    SCOPED_TRACE(static_cast<int>(method));
    const benchmarks::QPE benchmark(
        {.precision = precision,
         .phase = benchmarks::Phase(std::numeric_limits<uint64_t>::max() - 1,
                                    std::numeric_limits<uint64_t>::max()),
         .method = method});
    auto program = generateProgram(benchmark);
    ASSERT_TRUE(program);
    auto moduleOp = program->module();

    auto table = angleTable(moduleOp);
    ASSERT_TRUE(table);
    EXPECT_EQ(table.getNumElements(), precision);
    for (const auto angle : table.getValues<double>()) {
      EXPECT_TRUE(std::isfinite(angle));
    }

    size_t operations = 0;
    moduleOp.walk([&](Operation*) { ++operations; });
    EXPECT_LT(operations, 150U);
    EXPECT_EQ(countOps<tensor::ExtractOp>(moduleOp), 1U);
  }
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
