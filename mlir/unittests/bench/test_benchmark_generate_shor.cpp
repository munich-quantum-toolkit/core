/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Shor.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/bench/Generate.h"

#include "ModularArithmetic.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/APInt.h"

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bench {
using namespace mlir;

/// Tiny independent modular-permutation reference followed by a discrete
/// Fourier transform.
static std::vector<double> shorReference(uint64_t number, uint64_t base) {
  const auto size = size_t{1} << (2U * std::bit_width(number));
  std::vector<size_t> residues(size);
  residues[0] = 1;
  for (size_t x = 1; x < size; ++x) {
    residues[x] = (residues[x - 1] * base) % number;
  }
  std::vector<double> probabilities(size);
  for (size_t y = 0; y < size; ++y) {
    std::vector<std::complex<double>> amplitudes(number);
    for (size_t x = 0; x < size; ++x) {
      const auto angle = 2. * std::numbers::pi *
                         static_cast<double>((x * y) % size) /
                         static_cast<double>(size);
      amplitudes[residues[x]] +=
          std::polar(1. / static_cast<double>(size), angle);
    }
    for (auto amplitude : amplitudes) {
      probabilities[y] += std::norm(amplitude);
    }
  }
  return probabilities;
}

TEST(GenerateProgramTest, SamplesShorAndRecoversFactors) {
  for (uint64_t number : {15ULL, 21ULL, 35ULL}) {
    SCOPED_TRACE(number);
    const Shor benchmark({.number = number});
    auto program = test::generateQCO(benchmark);
    ASSERT_TRUE(program);
    EXPECT_GE(test::countOps<func::CallOp>(program->module()), 3U);
    auto counts =
        qco::sample(mlir::mqt::getEntryPoint(program->module()), 64, 17);
    ASSERT_TRUE(succeeded(counts));
    auto evaluation = benchmark.evaluate(*counts);
    ASSERT_TRUE(evaluation.factors);
    EXPECT_EQ(evaluation.factors->first * evaluation.factors->second, number);
    const auto reference = shorReference(number, 2);
    double distance = 0.;
    for (size_t phase = 0; phase < reference.size(); ++phase) {
      auto it =
          counts->find(dd::intToBinaryString(phase, benchmark.output().width));
      const auto observed =
          it == counts->end() ? 0. : static_cast<double>(it->second) / 64.;
      distance += std::abs(reference[phase] - observed) / 2.;
    }
    EXPECT_LT(distance, 0.4);
  }
}

static std::optional<QCOProgram>
inPlaceMultiplier(uint64_t number, uint64_t multiplier,
                  bool composeInverse = false) {
  const auto bits = static_cast<int64_t>(std::bit_width(number));
  const auto width = static_cast<unsigned>(bits + 1);
  uint64_t inverse = 1;
  while ((inverse * multiplier) % number != 1) {
    ++inverse;
  }
  auto context = createCompilerContext();
  auto moduleOp = qc::QCProgramBuilder::build(
      context.get(), [&](qc::QCProgramBuilder& builder) {
        SmallVector<double> angles;
        for (auto value : {multiplier, inverse, multiplier}) {
          detail::appendModularPhaseAngles(angles, llvm::APInt(width, value),
                                           llvm::APInt(width, number));
        }
        auto type = RankedTensorType::get({static_cast<int64_t>(angles.size())},
                                          builder.getF64Type());
        auto helper = detail::createInPlaceMultiplier(builder, bits, type);
        auto table = arith::ConstantOp::create(
            builder, DenseElementsAttr::get(type, ArrayRef<double>(angles)));
        auto control = builder.allocQubit();
        auto value = builder.allocQubitRegisterStorage(bits);
        auto accumulator = builder.allocQubitRegisterStorage(bits + 1);
        auto work = builder.allocQubit();
        builder.call(helper, {
                                 control,
                                 value,
                                 accumulator,
                                 work,
                                 table,
                                 builder.indexConstant(0),
                             });
        if (composeInverse) {
          builder.call(helper,
                       {
                           control,
                           value,
                           accumulator,
                           work,
                           table,
                           builder.indexConstant((bits + 1) * (bits + 1)),
                       });
        }
        return SmallVector<Value>{};
      });
  auto qcProgram = QCProgram::fromModule(context, std::move(moduleOp));
  if (!qcProgram) {
    return std::nullopt;
  }
  auto compiled = runDefaultPipeline(CompilerInput{std::move(*qcProgram)},
                                     ProgramFormat::QCO);
  if (!compiled) {
    return std::nullopt;
  }
  return std::get<QCOProgram>(std::move(*compiled));
}

TEST(GenerateProgramTest, VerifiesSmallInPlaceMultiplierBasisStates) {
  for (uint64_t number = 2; number < 16; ++number) {
    const auto bits = std::bit_width(number);
    const auto qubits = 2U * bits + 3U;
    for (uint64_t multiplier = 1; multiplier < number; ++multiplier) {
      if (std::gcd(multiplier, number) != 1) {
        continue;
      }
      auto program = inPlaceMultiplier(number, multiplier);
      ASSERT_TRUE(program);
      dd::Package package(qubits);
      auto functionality = qco::buildFunctionality(
          mlir::mqt::getEntryPoint(program->module()), package);
      ASSERT_TRUE(succeeded(functionality));
      for (uint64_t value = 0; value < number; ++value) {
        for (size_t control = 0; control < 2; ++control) {
          SCOPED_TRACE(testing::PrintToString(
              std::vector<uint64_t>{number, multiplier, value, control}));
          dd::CVec input(size_t{1} << qubits);
          input[(value << 1U) | control] = 1.;
          auto state = package.applyOperation(
              *functionality, dd::makeStateFromVector(input, package));
          auto output = state.getVector();
          package.decRef(state);
          const auto product =
              control != 0 ? multiplier * value % number : value;
          const auto expected = (product << 1U) | control;
          EXPECT_NEAR(std::norm(output[expected]), 1., 1e-11);
        }
      }
      package.decRef(*functionality);
    }
  }
}

TEST(GenerateProgramTest, PreservesMultiplierCoherenceAndUncomputesWorkspace) {
  for (const bool composeInverse : {false, true}) {
    for (uint64_t number : {3ULL, 5ULL, 7ULL, 15ULL}) {
      auto program = inPlaceMultiplier(number, 2, composeInverse);
      ASSERT_TRUE(program);
      const auto qubits = 2U * std::bit_width(number) + 3U;
      dd::CVec input(size_t{1} << qubits);
      dd::CVec expected(input.size());
      for (size_t control = 0; control < 2; ++control) {
        for (uint64_t value = 0; value < number; ++value) {
          const auto index = (value << 1U) | control;
          const auto amplitude =
              std::polar(1. / std::sqrt(2. * static_cast<double>(number)),
                         0.137 * static_cast<double>(index));
          input[index] = amplitude;
          const auto product =
              control != 0 && !composeInverse ? 2 * value % number : value;
          expected[(product << 1U) | control] = amplitude;
        }
      }
      dd::Package package(qubits);
      auto functionality = qco::buildFunctionality(
          mlir::mqt::getEntryPoint(program->module()), package);
      ASSERT_TRUE(succeeded(functionality));
      auto state = package.applyOperation(
          *functionality, dd::makeStateFromVector(input, package));
      auto output = state.getVector();
      package.decRef(state);
      package.decRef(*functionality);
      ASSERT_EQ(output.size(), expected.size());
      const auto overlap = std::inner_product(
          expected.begin(), expected.end(), output.begin(),
          std::complex<double>{}, std::plus<>(),
          [](auto lhs, auto rhs) { return std::conj(lhs) * rhs; });
      const auto phase = std::polar(1., std::arg(overlap));
      for (size_t index = 0; index < output.size(); ++index) {
        EXPECT_NEAR(std::abs(output[index] - phase * expected[index]), 0.,
                    1e-11);
      }
    }
  }
}

TEST(GenerateProgramTest, KeepsLargestShorStructuredAndCompilable) {
  const Shor benchmark({.number = ShorOptions::MAX_NUMBER});
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  auto table = test::angleTable(program->module());
  ASSERT_TRUE(table);
  EXPECT_EQ(table.getNumElements(), 4U * 31U * 32U * 32U);
  EXPECT_LT(test::countOperations(program->module()), 600U);
  EXPECT_EQ(test::countOps<qc::AllocOp>(program->module()), 2U);
  auto qcoProgram = std::move(*program).intoQCO();
  ASSERT_TRUE(qcoProgram);
  EXPECT_TRUE(succeeded(qco::verifyLinearity(qcoProgram->module())));
  auto qir = runDefaultPipeline(CompilerInput{std::move(*qcoProgram)},
                                ProgramFormat::QIRAdaptive);
  ASSERT_TRUE(qir);
}

TEST(GenerateProgramTest, ShorCutoffCoversTheExactCircuit) {
  auto exact = generate(Shor({.number = 15}));
  auto full = generate(Shor({.number = 15, .qftCutoff = 64}));
  ASSERT_TRUE(exact);
  ASSERT_TRUE(full);
  EXPECT_EQ(exact->str(), full->str());
  const Shor approximate({.number = 15, .qftCutoff = 3});
  auto program = test::generateQCO(approximate);
  ASSERT_TRUE(program);
  auto counts =
      qco::sample(mlir::mqt::getEntryPoint(program->module()), 64, 17);
  ASSERT_TRUE(succeeded(counts));
  const auto evaluation = approximate.evaluate(*counts);
  ASSERT_TRUE(evaluation.factors);
  EXPECT_EQ(evaluation.factors->first * evaluation.factors->second, 15U);
}

} // namespace mqt::bench
