/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestUtils.h"
#include "bench/ControlledMultiplicationModuloN.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>

namespace mqt::bench {

using namespace mlir;

static void expectCoherentControlledMultiplication(const size_t bits,
                                                   const size_t multiplier,
                                                   const size_t modulus) {
  const ControlledMultiplicationModuloN benchmark{{
      .multiplier = dd::intToBinaryString(multiplier, bits),
      .modulus = dd::intToBinaryString(modulus, bits),
  }};
  auto program = test::generateQCO(benchmark);
  ASSERT_TRUE(program);

  dd::Package package(0);
  auto state = qco::simulateStatevector(
      mlir::mqt::getEntryPoint(program->module()), package);
  ASSERT_TRUE(succeeded(state));
  const auto actual = state->getVector();
  package.decRef(*state);

  dd::CVec expected(size_t{1} << ((2U * bits) + 3U));
  const auto amplitude =
      1. / std::sqrt(static_cast<double>(size_t{1} << (bits + 1U)));
  for (size_t control = 0; control < 2U; ++control) {
    for (size_t multiplicand = 0; multiplicand < (size_t{1} << bits);
         ++multiplicand) {
      const auto product =
          control == 0 ? 0 : (multiplier * multiplicand) % modulus;
      const auto index =
          (product << (bits + 1U)) | (multiplicand << 1U) | control;
      expected[index] = amplitude;
    }
  }

  ASSERT_EQ(actual.size(), expected.size());
  const auto overlap = std::inner_product(
      expected.begin(), expected.end(), actual.begin(), std::complex<double>{},
      std::plus<>(),
      [](const auto& lhs, const auto& rhs) { return std::conj(lhs) * rhs; });
  const auto phase = std::polar(1., std::arg(overlap));
  for (size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(std::abs(actual[index] - phase * expected[index]), 0., 1e-12)
        << index;
  }
}

TEST(GenerateProgramTest, PreservesControlledMultiplicationCoherence) {
  expectCoherentControlledMultiplication(/*bits=*/2, /*multiplier=*/2,
                                         /*modulus=*/3);
  expectCoherentControlledMultiplication(/*bits=*/3, /*multiplier=*/3,
                                         /*modulus=*/5);
  expectCoherentControlledMultiplication(/*bits=*/4, /*multiplier=*/6,
                                         /*modulus=*/12);
  expectCoherentControlledMultiplication(/*bits=*/5, /*multiplier=*/7,
                                         /*modulus=*/29);
}

TEST(GenerateProgramTest,
     KeepsLargestControlledMultiplicationFiniteAndStructured) {
  constexpr size_t bits = ControlledMultiplicationModuloNOptions::MAX_BITS;
  const auto multiplier = std::string(bits - 1U, '0') + "1";
  const auto modulus = "1" + std::string(bits - 1U, '0');
  auto program = generate(ControlledMultiplicationModuloN{
      {.multiplier = multiplier, .modulus = modulus}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  const auto table = test::angleTable(moduleOp);
  ASSERT_TRUE(table);
  EXPECT_EQ(table.getNumElements(), (bits + 1U) * (bits + 1U));
  for (const auto angle : table.getValues<double>()) {
    EXPECT_TRUE(std::isfinite(angle));
  }
  EXPECT_EQ(test::countOps<tensor::ExtractOp>(moduleOp), 5U);
  EXPECT_LT(test::countOperations(moduleOp), 200U);
}

TEST(GenerateProgramTest,
     SamplesControlledMultiplicationModuloNAgainstReference) {
  test::expectSamplingMatchesReference(
      ControlledMultiplicationModuloN{{.multiplier = "011", .modulus = "101"}});
  test::expectSamplingMatchesReference(
      ControlledMultiplicationModuloN{{.multiplier = "010", .modulus = "100"}});
}

} // namespace mqt::bench
