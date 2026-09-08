/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <numbers>
#include <random>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;

static Matrix4x4 reconstruct(const TwoQubitNativeDecomposition& result) {
  const double s = std::numbers::sqrt2 / 2.;
  const auto gate =
      Matrix4x4::fromElements(1., 0., 0., 0., 0., s, Complex(0., s), 0., 0.,
                              Complex(0., s), s, 0., 0., 0., 0., 1.);
  auto matrix = Matrix4x4::identity();
  for (size_t i = 0; i <= result.numBasisUses; ++i) {
    matrix = Matrix4x4::kron(result.singleQubitFactors[2 * i + 1],
                             result.singleQubitFactors[2 * i]) *
             matrix;
    if (i < result.numBasisUses)
      matrix = gate * matrix;
  }
  return std::polar(1., result.globalPhase) * matrix;
}

TEST(SqrtISwap, ChamberGridWithLocalFactorsAndPhase) {
  constexpr int steps = 12;
  const auto left =
      Matrix4x4::kron(RXOp::unitaryMatrix(.71), RYOp::unitaryMatrix(-1.3));
  const auto right =
      Matrix4x4::kron(RZOp::unitaryMatrix(2.11), RXOp::unitaryMatrix(.37));
  for (int a = 0; a <= steps; ++a) {
    for (int b = 0; b <= a; ++b) {
      for (int c = -b; c <= b; ++c) {
        const auto step = std::numbers::pi / (4. * steps);
        const auto target = std::polar(1., .42) * left *
                            TwoQubitWeylDecomposition::getCanonicalMatrix(
                                a * step, b * step, c * step) *
                            right;
        const auto result = decomposeSqrtISwap(target);
        SCOPED_TRACE(::testing::Message() << a << "," << b << "," << c);
        const int expected = a == 0                                         ? 0
                             : (a == steps / 2 && b == steps / 2 && c == 0) ? 1
                             : a >= b + std::abs(c)                         ? 2
                                                                            : 3;
        EXPECT_EQ(result.numBasisUses, expected);
        EXPECT_TRUE(reconstruct(result).isApprox(target, 1e-8));
      }
    }
  }
}

TEST(SqrtISwap, NearChamberBoundaries) {
  for (double epsilon : {1e-4, 1e-8, 1e-10}) {
    const auto p = std::numbers::pi / 4.;
    for (const auto& coordinates :
         {std::array{epsilon, 0., 0.}, std::array{p - epsilon, 0., 0.},
          std::array{p, epsilon, epsilon}, std::array{p - epsilon, .2, -.1},
          std::array{.3, .2, .1 - epsilon}, std::array{.3, .2, .1 + epsilon},
          std::array{p, p, p - epsilon}}) {
      const auto target = TwoQubitWeylDecomposition::getCanonicalMatrix(
          coordinates[0], coordinates[1], coordinates[2]);
      const auto result = decomposeSqrtISwap(target);
      SCOPED_TRACE(::testing::Message()
                   << coordinates[0] << "," << coordinates[1] << ","
                   << coordinates[2] << " eps=" << epsilon);
      EXPECT_TRUE(reconstruct(result).isApprox(target, 1e-9));
    }
  }
}

TEST(SqrtISwap, RandomInteractionsAndLocalFactors) {
  std::mt19937 generator(42);
  std::uniform_real_distribution<double> sample(0., 1.);
  for (int i = 0; i < 1000; ++i) {
    std::array coordinates{sample(generator), sample(generator),
                           sample(generator)};
    std::ranges::sort(coordinates, std::greater<>());
    for (auto& coefficient : coordinates)
      coefficient *= std::numbers::pi / 4.;
    if (i % 2 == 0)
      coordinates[2] = -coordinates[2];
    const auto left =
        Matrix4x4::kron(RXOp::unitaryMatrix(6. * sample(generator)),
                        RYOp::unitaryMatrix(6. * sample(generator)));
    const auto right =
        Matrix4x4::kron(RZOp::unitaryMatrix(6. * sample(generator)),
                        RXOp::unitaryMatrix(6. * sample(generator)));
    const auto target = std::polar(1., 6. * sample(generator)) * left *
                        TwoQubitWeylDecomposition::getCanonicalMatrix(
                            coordinates[0], coordinates[1], coordinates[2]) *
                        right;
    const auto result = decomposeSqrtISwap(target);
    SCOPED_TRACE(i);
    EXPECT_EQ(result.numBasisUses,
              coordinates[0] >= coordinates[1] + std::abs(coordinates[2]) ? 2
                                                                          : 3);
    EXPECT_TRUE(reconstruct(result).isApprox(target, 1e-9));
  }
}
