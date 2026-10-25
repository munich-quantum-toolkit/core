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
#include "dd/Edge.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/bench/Generate.h"

#include "ShorMultiplier.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bench {
using namespace mlir;

TEST(GenerateProgramTest, SamplesShorAndRecoversFactors) {
  // Keep the circuit small enough for unoptimized coverage builds.
  constexpr uint64_t number = 15;
  const auto benchmark = llvm::cantFail(Shor::create({.number = number}));
  auto program = test::generateQCO(benchmark);
  ASSERT_TRUE(mlir::succeeded(program));
  auto counts =
      qco::sample(mlir::mqt::getEntryPoint(program->module()), 64, 17);
  ASSERT_TRUE(succeeded(counts));
  auto evaluation = llvm::cantFail(benchmark.evaluate(*counts));
  ASSERT_TRUE(evaluation.factors);
  EXPECT_EQ(evaluation.factors->first * evaluation.factors->second, number);
  // Base 2 has order 4 modulo 15: four equally likely phases, with no tails.
  double distance = 1.;
  for (const auto* phase : {"00000000", "01000000", "10000000", "11000000"}) {
    if (auto it = counts->find(phase); it != counts->end()) {
      distance -= std::min(0.25, static_cast<double>(it->second) / 64.);
    }
  }
  EXPECT_LT(distance, 0.4);
  auto shared =
      qco::sample(mlir::mqt::getEntryPoint(program->module()), 256, 17,
                  qco::DDArgumentBindings{}, nullptr, nullptr, {}, 1);
  ASSERT_TRUE(succeeded(shared));
  EXPECT_TRUE(llvm::cantFail(benchmark.evaluate(*shared)).factors);
}

static mlir::FailureOr<QCOProgram> inPlaceMultiplier(uint64_t number,
                                                     uint64_t multiplier) {
  const auto bits = static_cast<int64_t>(std::bit_width(number));
  uint64_t inverse = 1;
  while ((inverse * multiplier) % number != 1) {
    ++inverse;
  }
  auto context = createCompilerContext();
  auto moduleOp = qc::QCProgramBuilder::build(
      context.get(), [&](qc::QCProgramBuilder& builder) {
        auto helper = detail::createInPlaceMultiplier(builder, bits);
        auto multiplierValue =
            builder.intConstant(static_cast<int64_t>(multiplier));
        auto inverseValue = builder.intConstant(static_cast<int64_t>(inverse));
        auto modulus = builder.intConstant(static_cast<int64_t>(number));
        auto control = builder.allocQubit();
        auto value = builder.allocQubitRegisterStorage(bits);
        auto accumulator = builder.allocQubitRegisterStorage(bits + 1);
        auto work = builder.allocQubit();
        builder.call(helper, {
                                 control,
                                 value,
                                 accumulator,
                                 work,
                                 multiplierValue,
                                 inverseValue,
                                 modulus,
                             });
        return SmallVector<Value>{};
      });
  auto qcProgram = QCProgram::fromModule(context, std::move(moduleOp));
  if (mlir::failed(qcProgram)) {
    return mlir::failure();
  }
  auto compiled = runDefaultPipeline(CompilerInput{std::move(*qcProgram)},
                                     ProgramFormat::QCO);
  if (mlir::failed(compiled)) {
    return mlir::failure();
  }
  return std::get<QCOProgram>(std::move(*compiled));
}

TEST(GenerateProgramTest,
     PreservesSmallInPlaceMultiplierCoherenceAndWorkspace) {
  for (uint64_t number = 2; number < 16; ++number) {
    const auto bits = std::bit_width(number);
    const auto qubits = 2U * bits + 3U;
    for (uint64_t multiplier = 1; multiplier < number; ++multiplier) {
      if (std::gcd(multiplier, number) != 1) {
        continue;
      }
      auto program = inPlaceMultiplier(number, multiplier);
      ASSERT_TRUE(mlir::succeeded(program));
      dd::Package package(qubits);
      auto functionality = qco::buildFunctionality(
          mlir::mqt::getEntryPoint(program->module()), package);
      ASSERT_TRUE(succeeded(functionality));
      const auto phase = dd::getValueByIndex(*functionality, qubits, 0, 0);
      EXPECT_NEAR(std::norm(phase), 1., 1e-11);
      dd::CVec input(size_t{1} << qubits);
      dd::CVec expectedOutput(input.size());
      // A common unit-modulus amplitude for every valid basis column proves
      // the permutation, clean workspace, and coherence of any superposition.
      for (uint64_t value = 0; value < number; ++value) {
        for (size_t control = 0; control < 2; ++control) {
          SCOPED_TRACE(testing::PrintToString(
              std::vector<uint64_t>{number, multiplier, value, control}));
          const auto product =
              control != 0 ? multiplier * value % number : value;
          const auto expected = (product << 1U) | control;
          const auto amplitude = dd::getValueByIndex(
              *functionality, qubits, expected, (value << 1U) | control);
          EXPECT_NEAR(std::abs(amplitude - phase), 0., 1e-11);
          const auto index = (value << 1U) | control;
          input[index] =
              std::polar(1. / std::sqrt(2. * static_cast<double>(number)),
                         0.137 * static_cast<double>(index));
          expectedOutput[expected] = phase * input[index];
        }
      }
      auto state = package.applyOperation(
          *functionality, dd::makeStateFromVector(input, package));
      const auto output = dd::getVector(state);
      for (size_t index = 0; index < output.size(); ++index) {
        EXPECT_NEAR(std::abs(output[index] - expectedOutput[index]), 0., 1e-11);
      }
      package.decRef(state);
      package.decRef(*functionality);
    }
  }
}

TEST(GenerateProgramTest, KeepsLargestShorStructuredAndCompilable) {
  const auto benchmark =
      llvm::cantFail(Shor::create({.number = ShorOptions::MAX_NUMBER}));
  auto program = generate(benchmark);
  ASSERT_TRUE(mlir::succeeded(program));
  EXPECT_LT(program->str().size(), 32'768U);
  auto qir = runDefaultPipeline(CompilerInput{std::move(*program)},
                                ProgramFormat::QIRAdaptive);
  ASSERT_TRUE(mlir::succeeded(qir));
}

} // namespace mqt::bench
