/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"

#include <gtest/gtest.h>

#include <complex>

TEST(DDBackportTest, WidePhasedHadamardsKeepNonzeroRoot) {
  constexpr dd::Qubit width = 83;
  dd::Package package(width);
  const std::complex<dd::fp> weight{0.5, -0.5};
  const dd::GateMatrix gate{weight, weight, weight, -weight};
  auto matrix = dd::Package::makeIdent();
  package.incRef(matrix);
  for (dd::Qubit q = 0; q < width; ++q) {
    const auto next = package.multiply(package.makeGateDD(gate, q), matrix);
    package.incRef(next);
    package.decRef(matrix);
    matrix = next;
    package.garbageCollect();
  }
  EXPECT_FALSE(matrix.isZeroTerminal());
  const auto root = static_cast<dd::ComplexValue>(matrix.w);
  EXPECT_DOUBLE_EQ(root.r, -0x1p-42);
  EXPECT_DOUBLE_EQ(root.i, -0x1p-42);
  package.decRef(matrix);
}
