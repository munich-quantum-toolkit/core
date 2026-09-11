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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numbers>
#include <numeric>
#include <string>
#include <utility>

namespace mqt::bench {

using namespace mlir;

static void expectConstantIndex(Value value, int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(constant);
  EXPECT_EQ(constant.value(), expected);
}

static void expectConstantFloat(Value value, double expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(constant);
  auto attribute = dyn_cast<FloatAttr>(constant.getValue());
  ASSERT_TRUE(attribute);
  EXPECT_DOUBLE_EQ(attribute.getValueAsDouble(), expected);
}

TEST(GenerateProgramTest, EmitsQuantumQFTAdderCircuit) {
  constexpr int64_t qubits = 3;
  auto program = generate(QFTAdder{{.addend = "+++", .accumulator = "001"}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  /// Unlike the QFT phases, the addition phase connects the two registers.
  qc::CtrlOp addition;
  moduleOp.walk([&](qc::CtrlOp op) {
    auto control = op.getControl(0).getDefiningOp<memref::LoadOp>();
    auto target = op.getTarget(0).getDefiningOp<memref::LoadOp>();
    if (control && target && control.getMemref() != target.getMemref()) {
      EXPECT_FALSE(addition);
      addition = op;
    }
  });
  ASSERT_TRUE(addition);

  auto sourceLoad = addition.getControl(0).getDefiningOp<memref::LoadOp>();
  auto targetLoad = addition.getTarget(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(sourceLoad);
  ASSERT_TRUE(targetLoad);

  auto inner = addition->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(inner);
  auto outer = inner->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(outer);

  auto target = targetLoad.getIndices().front();
  auto targetIndex = target.getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(targetIndex);
  expectConstantIndex(targetIndex.getLhs(), qubits - 1);
  EXPECT_EQ(targetIndex.getRhs(), outer.getInductionVar());

  auto sourceIndex =
      sourceLoad.getIndices().front().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(sourceIndex);
  EXPECT_EQ(sourceIndex.getLhs(), target);
  EXPECT_EQ(sourceIndex.getRhs(), inner.getInductionVar());

  auto upper = inner.getUpperBound().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(upper);
  expectConstantIndex(upper.getLhs(), qubits);
  EXPECT_EQ(upper.getRhs(), outer.getInductionVar());
  expectConstantIndex(inner.getLowerBound(), 0);
  expectConstantIndex(inner.getStep(), 1);

  ASSERT_EQ(inner.getInitArgs().size(), 1U);
  expectConstantFloat(inner.getInitArgs().front(), std::numbers::pi);
  qc::POp phase;
  addition.walk([&](qc::POp op) { phase = op; });
  ASSERT_TRUE(phase);
  EXPECT_EQ(phase.getTheta(), inner.getRegionIterArg(0));

  auto yield = dyn_cast<scf::YieldOp>(inner.getBody()->getTerminator());
  ASSERT_TRUE(yield);
  ASSERT_EQ(yield.getNumOperands(), 1U);
  auto nextAngle = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(nextAngle);
  EXPECT_EQ(nextAngle.getLhs(), inner.getRegionIterArg(0));
  expectConstantFloat(nextAngle.getRhs(), 0.5);
}

TEST(GenerateProgramTest, KeepsLargestQuantumQFTAdderFiniteAndStructured) {
  auto program = generate(QFTAdder{{
      .addend = std::string(QFTAdderOptions::MAX_QUBITS, '+'),
      .accumulator = std::string(QFTAdderOptions::MAX_QUBITS - 1, '0') + "1",
  }});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_LT(test::countOperations(moduleOp), 200U);
  moduleOp.walk([&](arith::ConstantOp op) {
    if (auto value = dyn_cast<FloatAttr>(op.getValue())) {
      EXPECT_TRUE(std::isfinite(value.getValueAsDouble()));
    }
  });
}

TEST(GenerateProgramTest, UsesConfiguredClassicalQFTAdderPhases) {
  auto program = generate(QFTAdder{{
      .addend = "101",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  auto table = test::angleTable(moduleOp);
  ASSERT_TRUE(table);
  const auto angles = llvm::to_vector(table.getValues<double>());
  ASSERT_EQ(angles.size(), 4U);
  EXPECT_DOUBLE_EQ(angles[0], std::numbers::pi);
  EXPECT_DOUBLE_EQ(angles[1], std::numbers::pi / 2.);
  EXPECT_DOUBLE_EQ(angles[2], 5. * std::numbers::pi / 4.);
  EXPECT_DOUBLE_EQ(angles[3], 5. * std::numbers::pi / 8.);

  tensor::ExtractOp extract;
  moduleOp.walk([&](tensor::ExtractOp op) {
    EXPECT_FALSE(extract);
    extract = op;
  });
  ASSERT_TRUE(extract);
  auto loop = extract->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(loop);
  expectConstantIndex(loop.getLowerBound(), 0);
  expectConstantIndex(loop.getUpperBound(), 4);
  expectConstantIndex(loop.getStep(), 1);
  EXPECT_EQ(extract.getIndices().front(), loop.getInductionVar());

  qc::POp phase;
  moduleOp.walk([&](qc::POp op) {
    if (!op->getParentOfType<qc::CtrlOp>()) {
      EXPECT_FALSE(phase);
      phase = op;
    }
  });
  ASSERT_TRUE(phase);
  EXPECT_EQ(phase->getParentOfType<scf::ForOp>(), loop);
  EXPECT_EQ(phase.getTheta(), extract.getResult());
  auto target = phase.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(target);
  EXPECT_EQ(target.getIndices().front(), loop.getInductionVar());
}

TEST(GenerateProgramTest, KeepsLargestClassicalQFTAdderFiniteAndStructured) {
  auto addend = std::string((QFTAdderOptions::MAX_QUBITS - 1U), '1');
  auto program = generate(QFTAdder{{
      .addend = std::move(addend),
      .accumulator = std::string(QFTAdderOptions::MAX_QUBITS - 2, '0') + "1",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  auto table = test::angleTable(moduleOp);
  ASSERT_TRUE(table);
  EXPECT_EQ(table.getNumElements(), (QFTAdderOptions::MAX_QUBITS - 1U) + 1U);
  for (const auto angle : table.getValues<double>()) {
    EXPECT_TRUE(std::isfinite(angle));
  }

  EXPECT_LT(test::countOperations(moduleOp), 100U);
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
