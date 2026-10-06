/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QPE.hpp"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace mqt::bench {

using namespace mlir;

static void evaluateResidueLoop(scf::ForOp loop,
                                DenseMap<Value, Attribute>& arguments,
                                std::vector<uint64_t>& residues,
                                std::vector<double>& angles) {
  auto lower = dyn_cast_or_null<IntegerAttr>(
      test::evaluateArithmetic(loop.getLowerBound(), arguments));
  auto upper = dyn_cast_or_null<IntegerAttr>(
      test::evaluateArithmetic(loop.getUpperBound(), arguments));
  auto step = dyn_cast_or_null<IntegerAttr>(
      test::evaluateArithmetic(loop.getStep(), arguments));
  ASSERT_TRUE(lower);
  ASSERT_TRUE(upper);
  ASSERT_TRUE(step);
  ASSERT_GT(step.getInt(), 0);
  ASSERT_EQ(loop.getInitArgs().size(), 1U);
  auto carried =
      test::evaluateArithmetic(loop.getInitArgs().front(), arguments);
  ASSERT_TRUE(carried);
  for (auto index = lower.getInt(); index < upper.getInt();
       index += step.getInt()) {
    arguments[loop.getInductionVar()] =
        IntegerAttr::get(loop.getInductionVar().getType(), index);
    arguments[loop.getRegionIterArg(0)] = carried;
    for (auto& operation : *loop.getBody()) {
      if (auto control = dyn_cast<qc::CtrlOp>(operation)) {
        qc::POp phase;
        control.walk([&](qc::POp op) { phase = op; });
        ASSERT_TRUE(phase);
        auto angle = dyn_cast_or_null<FloatAttr>(
            test::evaluateArithmetic(phase.getTheta(), arguments));
        ASSERT_TRUE(angle);
        residues.push_back(
            cast<IntegerAttr>(carried).getValue().getZExtValue());
        angles.push_back(angle.getValueAsDouble());
      }
    }
    carried = test::evaluateArithmetic(
        loop.getBody()->getTerminator()->getOperand(0), arguments);
    ASSERT_TRUE(carried);
  }
  arguments[loop.getResult(0)] = carried;
}

TEST(GenerateProgramTest, KeepsQPEPhaseStateScalarAtMaximumPrecision) {
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    SCOPED_TRACE(static_cast<int>(method));
    const QPE benchmark({
        .precision = QPEOptions::MAX_PRECISION,
        .phase = Phase(std::numeric_limits<uint64_t>::max() - 1,
                       std::numeric_limits<uint64_t>::max()),
        .method = method,
    });
    auto program = generate(benchmark);
    ASSERT_TRUE(program);
    auto moduleOp = program->module();
    moduleOp.walk([&](arith::ConstantOp op) {
      EXPECT_FALSE(isa<DenseFPElementsAttr>(op.getValue()));
    });
    EXPECT_LT(test::countOperations(moduleOp), 150U);
    test::expectJeffRoundTrip(program->copy());
  }
}

TEST(GenerateProgramTest, KeepsStandardQPEPowerAndResultOrderAligned) {
  const QPE benchmark({.precision = 2, .phase = Phase(1, 4)});
  EXPECT_DOUBLE_EQ(benchmark.probability("01"), 1.);

  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  scf::ForOp powerLoop;
  moduleOp.walk([&](arith::UIToFPOp op) {
    EXPECT_FALSE(powerLoop);
    powerLoop = op->getParentOfType<scf::ForOp>();
  });
  ASSERT_TRUE(powerLoop);

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

  EXPECT_EQ(test::countOps<qc::SWAPOp>(moduleOp), 0U);
}

TEST(GenerateProgramTest, KeepsLargeQPEFiniteAndStructured) {
  constexpr size_t precision = 1025;
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    SCOPED_TRACE(static_cast<int>(method));
    const QPE benchmark({
        .precision = precision,
        .phase = Phase(std::numeric_limits<uint64_t>::max() - 1,
                       std::numeric_limits<uint64_t>::max()),
        .method = method,
    });
    auto program = generate(benchmark);
    ASSERT_TRUE(program);
    auto moduleOp = program->module();

    moduleOp.walk([&](arith::ConstantOp op) {
      if (auto angle = dyn_cast<FloatAttr>(op.getValue())) {
        EXPECT_TRUE(std::isfinite(angle.getValueAsDouble()));
      }
    });
    EXPECT_LT(test::countOperations(moduleOp), 150U);
  }
}

TEST(GenerateProgramTest, ComputesExactQPEResiduesAtRuntime) {
  constexpr auto maximum = std::numeric_limits<uint64_t>::max();
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    for (const auto phase : {
             Phase(uint64_t{1} << 63U, maximum),
             Phase(maximum - 2, maximum - 1),
             Phase((uint64_t{1} << 63U) - 1, uint64_t{1} << 63U),
             Phase(5, 12),
             Phase(1, 3),
         }) {
      for (const size_t precision : {4U, 65U, 1025U}) {
        SCOPED_TRACE(static_cast<int>(method));
        SCOPED_TRACE(phase.numerator());
        SCOPED_TRACE(phase.denominator());
        SCOPED_TRACE(precision);
        auto program = generate(
            QPE{{.precision = precision, .phase = phase, .method = method}});
        ASSERT_TRUE(program);
        DenseMap<Value, Attribute> arguments;
        std::vector<uint64_t> residues;
        std::vector<double> angles;
        for (auto& operation :
             mlir::mqt::getEntryPoint(program->module()).getBody().front()) {
          if (auto loop = dyn_cast<scf::ForOp>(operation);
              loop && loop.getNumResults() == 1 &&
              loop.getResult(0).getType().isInteger(64)) {
            evaluateResidueLoop(loop, arguments, residues, angles);
          }
        }
        ASSERT_EQ(residues.size(), precision);
        ASSERT_EQ(angles.size(), precision);
        const llvm::APInt denominator(128, phase.denominator());
        auto remainder = phase.numerator();
        for (size_t power = 0; power < precision; ++power) {
          const auto index =
              method == QPEMethod::Standard ? power : precision - 1 - power;
          EXPECT_EQ(residues[index], remainder);
          EXPECT_TRUE(std::isfinite(angles[index]));
          /// Allow f64 conversion/division/multiplication rounding against the
          /// long-double reference while keeping phase reduction exact.
          EXPECT_NEAR(angles[index],
                      static_cast<double>(
                          2.L * std::numbers::pi_v<long double> *
                          static_cast<long double>(remainder) /
                          static_cast<long double>(phase.denominator())),
                      3e-15);
          remainder = llvm::APInt(128, remainder)
                          .shl(1)
                          .urem(denominator)
                          .getZExtValue();
        }
      }
    }
  }
}

TEST(GenerateProgramTest, SamplesQPEAgainstReference) {
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    for (const auto phase :
         {Phase(0, 1), Phase(3, 8), Phase(1, 3), Phase(5, 12), Phase(7, 16)}) {
      SCOPED_TRACE(static_cast<int>(method));
      for (const size_t precision : {1U, 3U}) {
        SCOPED_TRACE(precision);
        test::expectSamplingMatchesReference(
            QPE{{.precision = precision, .phase = phase, .method = method}});
      }
    }
  }
}

} // namespace mqt::bench
