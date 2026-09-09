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
#include "bench/ModularMultiplier.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <utility>

namespace mqt::bench {

using namespace mlir;

static void expectCoherentModularMultiplier(const size_t bits,
                                            const size_t multiplier,
                                            const size_t modulus,
                                            std::string pattern = {},
                                            const char controlInput = '+') {
  if (pattern.empty()) {
    pattern = std::string(bits, '+');
  }
  const ModularMultiplier benchmark{{
      .multiplier = dd::intToBinaryString(multiplier, bits),
      .modulus = dd::intToBinaryString(modulus, bits),
      .multiplicand = pattern,
      .control = controlInput,
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
  const auto superposed =
      std::ranges::count(pattern, '+') + (controlInput == '+' ? 1 : 0);
  const auto amplitude =
      std::sqrt(std::ldexp(1., -static_cast<int>(superposed)));
  for (size_t control = 0; control < 2U; ++control) {
    for (size_t multiplicand = 0; multiplicand < (size_t{1} << bits);
         ++multiplicand) {
      if (controlInput != '+' && !std::cmp_equal(control, controlInput - '0')) {
        continue;
      }
      const auto input = dd::intToBinaryString(multiplicand, bits);
      bool matches = true;
      for (size_t bit = 0; bit < bits; ++bit) {
        matches =
            matches && (pattern[bit] == '+' || pattern[bit] == input[bit]);
      }
      if (!matches) {
        continue;
      }
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

TEST(GenerateProgramTest, PreservesModularMultiplierCoherence) {
  expectCoherentModularMultiplier(/*bits=*/2, /*multiplier=*/2,
                                  /*modulus=*/3);
  expectCoherentModularMultiplier(/*bits=*/3, /*multiplier=*/3,
                                  /*modulus=*/5);
  expectCoherentModularMultiplier(/*bits=*/4, /*multiplier=*/6,
                                  /*modulus=*/12);
  expectCoherentModularMultiplier(/*bits=*/5, /*multiplier=*/7,
                                  /*modulus=*/29);
  expectCoherentModularMultiplier(3, 3, 5, "1+0", '+');
  expectCoherentModularMultiplier(3, 2, 4, "0++", '1');
  expectCoherentModularMultiplier(3, 3, 5, "1+1", '0');
}

TEST(GenerateProgramTest, VerifiesEverySmallModularMultiplierBasisInput) {
  for (size_t bits = 2; bits <= 3; ++bits) {
    const auto limit = size_t{1} << bits;
    for (size_t modulus = limit / 2; modulus < limit; ++modulus) {
      for (size_t multiplier = 1; multiplier < modulus; ++multiplier) {
        for (size_t input = 0; input < limit; ++input) {
          for (size_t control = 0; control < 2; ++control) {
            const ModularMultiplier benchmark({
                .multiplier = dd::intToBinaryString(multiplier, bits),
                .modulus = dd::intToBinaryString(modulus, bits),
                .multiplicand = dd::intToBinaryString(input, bits),
                .control = control == 0 ? '0' : '1',
            });
            const auto expected =
                std::to_string(control) + dd::intToBinaryString(input, bits) +
                dd::intToBinaryString(control * multiplier * input % modulus,
                                      bits + 1);
            SCOPED_TRACE(expected);
            ASSERT_EQ(benchmark.expectedResult(), expected);
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

TEST(GenerateProgramTest, KeepsLargestModularMultiplierFiniteAndStructured) {
  constexpr size_t bits = ModularMultiplierOptions::MAX_BITS;
  const auto multiplier = std::string(bits - 1U, '0') + "1";
  const auto modulus = "1" + std::string(bits - 1U, '0');
  auto program = generate(ModularMultiplier{{
      .multiplier = multiplier,
      .modulus = modulus,
      .multiplicand = std::string(bits, '+'),
      .control = '+',
  }});
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

TEST(GenerateProgramTest, SamplesModularMultiplierAgainstReference) {
  test::expectSamplingMatchesReference(ModularMultiplier{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }});
  test::expectSamplingMatchesReference(ModularMultiplier{{
      .multiplier = "010",
      .modulus = "100",
      .multiplicand = "+++",
      .control = '+',
  }});
}

} // namespace mqt::bench
