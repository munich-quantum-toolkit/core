/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Evaluation.hpp"
#include "bench/JSON.hpp"
#include "bench/QFTAdder.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <numbers>
#include <numeric>
#include <string>
#include <utility>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, BoundsLargestQuantumQFTAdderPayload) {
  auto program = generate(QFTAdder{{
      .addend = std::string(QFTAdderOptions::MAX_QUBITS, '+'),
      .accumulator = std::string(QFTAdderOptions::MAX_QUBITS - 1, '0') + "1",
  }});
  ASSERT_TRUE(program);
  EXPECT_LT(program->str().size(), 8192U);
}

TEST(GenerateProgramTest, ComputesWideClassicalQFTAdderPhasesAccurately) {
  for (const auto overflow :
       {QFTAdderOverflow::Wrap, QFTAdderOverflow::Carry}) {
    const auto carry = overflow == QFTAdderOverflow::Carry;
    const auto width = QFTAdderOptions::MAX_QUBITS - (carry ? 1U : 0U);
    for (const bool allOnes : {false, true}) {
      SCOPED_TRACE(static_cast<int>(overflow));
      SCOPED_TRACE(allOnes);
      auto addend = std::string(width, allOnes ? '1' : '0');
      if (!allOnes) {
        for (const auto bit :
             std::array<size_t, 5>{0, 63, 64, 511, width - 2}) {
          addend[width - 1 - bit] = '1';
        }
      }
      auto program = generate(QFTAdder{{
          .addend = addend,
          .accumulator = std::string(width, '0'),
          .method = QFTAdderMethod::Constant,
          .overflow = overflow,
      }});
      ASSERT_TRUE(program);
      EXPECT_LT(program->str().size(), 8192U);
      qc::POp phase;
      program->module().walk([&](qc::POp op) {
        if (!op->getParentOfType<qc::CtrlOp>()) {
          phase = op;
        }
      });
      ASSERT_TRUE(phase);
      auto loop = phase->getParentOfType<scf::ForOp>();
      ASSERT_TRUE(loop);
      auto targetLoad = phase.getQubit(0).getDefiningOp<memref::LoadOp>();
      ASSERT_TRUE(targetLoad);
      DenseMap<Value, Attribute> arguments;
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
      size_t phases = 0;
      for (auto index = lower.getInt(); index < upper.getInt();
           index += step.getInt()) {
        arguments[loop.getInductionVar()] =
            IntegerAttr::get(loop.getInductionVar().getType(), index);
        arguments[loop.getRegionIterArg(0)] = carried;
        auto target = dyn_cast_or_null<IntegerAttr>(test::evaluateArithmetic(
            targetLoad.getIndices().front(), arguments));
        auto angle = dyn_cast_or_null<FloatAttr>(
            test::evaluateArithmetic(phase.getTheta(), arguments));
        ASSERT_TRUE(target);
        ASSERT_TRUE(angle);
        ASSERT_GE(target.getInt(), 0);
        ASSERT_LT(target.getInt(), QFTAdderOptions::MAX_QUBITS);
        const auto targetIndex = static_cast<size_t>(target.getInt());
        long double expected = 0.L;
        for (size_t bit = 0; bit < width && bit <= targetIndex; ++bit) {
          if (addend[width - 1 - bit] == '1') {
            expected += std::numbers::pi_v<long double> *
                        std::ldexp(1.L, -static_cast<int>(targetIndex - bit));
          }
        }
        const auto actual = angle.getValueAsDouble();
        EXPECT_NEAR(actual, static_cast<double>(expected), 3e-15)
            << targetIndex;
        carried = test::evaluateArithmetic(
            loop.getBody()->getTerminator()->getOperand(0), arguments);
        ASSERT_TRUE(carried);
        ++phases;
      }
      EXPECT_EQ(phases, QFTAdderOptions::MAX_QUBITS);
    }
  }
}

TEST(GenerateProgramTest, SamplesEverySmallQFTAdderOperandPair) {
  for (const auto method :
       {QFTAdderMethod::Register, QFTAdderMethod::Constant}) {
    for (const auto overflow :
         {QFTAdderOverflow::Wrap, QFTAdderOverflow::Carry}) {
      for (size_t width = 1; width <= 3; ++width) {
        const auto sumWidth =
            width + (overflow == QFTAdderOverflow::Carry ? 1U : 0U);
        for (size_t addend = 0; addend < (size_t{1} << width); ++addend) {
          for (size_t accumulator = 0; accumulator < (size_t{1} << width);
               ++accumulator) {
            const auto addendBits = dd::intToBinaryString(addend, width);
            const QFTAdder benchmark{{
                .addend = addendBits,
                .accumulator = dd::intToBinaryString(accumulator, width),
                .method = method,
                .overflow = overflow,
            }};
            SCOPED_TRACE(toInstanceSpecificationJSON(benchmark));
            const auto total = (addend + accumulator) % (size_t{1} << sumWidth);
            const auto expected =
                (method == QFTAdderMethod::Register ? addendBits : "") +
                dd::intToBinaryString(total, sumWidth);
            EXPECT_EQ(benchmark.expectedResult(), expected);
            auto program = test::generateQCO(benchmark);
            ASSERT_TRUE(program);
            auto counts = qco::sample(
                mlir::mqt::getEntryPoint(program->module()), 32, 17);
            ASSERT_TRUE(succeeded(counts));
            EXPECT_EQ(*counts, (Counts{{expected, 32}}));
          }
        }
      }
    }
  }
}

TEST(GenerateProgramTest, PreservesQFTAdderRelativePhases) {
  for (const auto overflow :
       {QFTAdderOverflow::Wrap, QFTAdderOverflow::Carry}) {
    for (size_t width = 1; width <= 3; ++width) {
      const auto sumWidth =
          width + (overflow == QFTAdderOverflow::Carry ? 1U : 0U);
      for (size_t accumulator = 0; accumulator < (size_t{1} << width);
           ++accumulator) {
        const QFTAdder benchmark{{
            .addend = std::string(width, '+'),
            .accumulator = dd::intToBinaryString(accumulator, width),
            .overflow = overflow,
        }};
        SCOPED_TRACE(toInstanceSpecificationJSON(benchmark));
        auto program = test::generateQCO(benchmark);
        ASSERT_TRUE(program);
        dd::Package package(0);
        auto state = qco::simulateStatevector(
            mlir::mqt::getEntryPoint(program->module()), package);
        ASSERT_TRUE(succeeded(state));
        const auto actual = state->getVector();
        package.decRef(*state);
        dd::CVec expected(size_t{1} << (width + sumWidth));
        for (size_t addend = 0; addend < (size_t{1} << width); ++addend) {
          const auto total = (addend + accumulator) % (size_t{1} << sumWidth);
          expected[(total << width) | addend] =
              1. / std::sqrt(static_cast<double>(size_t{1} << width));
        }
        ASSERT_EQ(actual.size(), expected.size());
        const auto overlap =
            std::inner_product(expected.begin(), expected.end(), actual.begin(),
                               std::complex<double>{}, std::plus<>(),
                               [](const auto& lhs, const auto& rhs) {
                                 return std::conj(lhs) * rhs;
                               });
        const auto phase = std::polar(1., std::arg(overlap));
        for (size_t i = 0; i < actual.size(); ++i) {
          EXPECT_NEAR(std::abs(actual[i] - phase * expected[i]), 0., 1e-12)
              << i;
        }
      }
    }
  }
}

TEST(GenerateProgramTest, SamplesPartlySuperposedQFTAdder) {
  test::expectSamplingMatchesReference(
      QFTAdder{{.addend = "1+0", .accumulator = "001"}});
}

} // namespace mqt::bench
