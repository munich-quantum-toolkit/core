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
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <utility>
#include <variant>
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

TEST(GenerateProgramTest, BoundsQPEPayloadAtMaximumPrecision) {
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    SCOPED_TRACE(static_cast<int>(method));
    const QPE benchmark({
        .precision = QPEOptions::MAX_PRECISION,
        .phase = Phase(std::numeric_limits<uint64_t>::max() - 1,
                       std::numeric_limits<uint64_t>::max()),
        .method = method,
    });
    auto program = generate(benchmark);
    ASSERT_TRUE(mlir::succeeded(program));
    EXPECT_LT(program->str().size(), 4096U);
    auto compiled = runDefaultPipeline(CompilerInput{std::move(*program)},
                                       ProgramFormat::Jeff);
    ASSERT_TRUE(mlir::succeeded(compiled));
    const auto bytes = std::get<JeffProgram>(*compiled).toBytes();
    EXPECT_LT(bytes.size(), 16'384U);
    auto restored = JeffProgram::fromBytes(bytes);
    ASSERT_TRUE(mlir::succeeded(restored));
    EXPECT_EQ(restored->toBytes(), bytes);
  }
}

TEST(GenerateProgramTest, ComputesExactQPEResiduesAtRuntime) {
  constexpr size_t precision = 1026;
  constexpr auto maximum = std::numeric_limits<uint64_t>::max();
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    for (const auto phase : {
             Phase(uint64_t{1} << 63U, maximum),
             Phase(maximum - 2, maximum - 1),
             Phase((uint64_t{1} << 63U) - 1, uint64_t{1} << 63U),
             Phase(5, 12),
         }) {
      SCOPED_TRACE(static_cast<int>(method));
      SCOPED_TRACE(phase.numerator());
      SCOPED_TRACE(phase.denominator());
      auto program = generate(
          QPE{{.precision = precision, .phase = phase, .method = method}});
      ASSERT_TRUE(mlir::succeeded(program));
      DenseMap<Value, Attribute> arguments;
      std::vector<uint64_t> residues;
      std::vector<double> angles;
      for (auto& operation :
           mlir::mqt::getEntryPoint(program->module()).getBody().front()) {
        if (auto loop = dyn_cast<scf::ForOp>(operation);
            loop && loop.getNumResults() >= 1 &&
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
        // Allow f64 conversion/division/multiplication rounding against the
        // long-double reference while keeping phase reduction exact.
        EXPECT_NEAR(
            angles[index],
            static_cast<double>(2.L * std::numbers::pi_v<long double> *
                                static_cast<long double>(remainder) /
                                static_cast<long double>(phase.denominator())),
            3e-15);
        remainder =
            llvm::APInt(128, remainder).shl(1).urem(denominator).getZExtValue();
      }
    }
  }
}

TEST(GenerateProgramTest, SamplesQPEAgainstReference) {
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    for (const auto phase : {Phase(3, 8), Phase(1, 3)}) {
      SCOPED_TRACE(static_cast<int>(method));
      test::expectSamplingMatchesReference(
          QPE{{.precision = 8, .phase = phase, .method = method}});
    }
  }
}

} // namespace mqt::bench
