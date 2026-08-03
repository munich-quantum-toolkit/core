/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/Layout.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

using namespace mlir;

TEST(LayoutTest, ConstructFromPermutation) {
  constexpr std::array<size_t, 3> mapping{2, 0, 1};
  const auto layout = qco::Layout::fromMapping(mapping);

  EXPECT_EQ(layout.nqubits(), mapping.size());
  EXPECT_EQ(layout.getProgramToHardware(), ArrayRef<size_t>(mapping));
  EXPECT_EQ(layout.getProgramIndex(0), 1);
  EXPECT_EQ(layout.getProgramIndex(1), 2);
  EXPECT_EQ(layout.getProgramIndex(2), 0);
}

TEST(LayoutDeathTest, RejectDuplicateHardwareIndex) {
  constexpr std::array<size_t, 3> mapping{0, 0, 2};
  EXPECT_DEATH((void)qco::Layout::fromMapping(mapping),
               "mapping must be a permutation");
}

TEST(LayoutDeathTest, RejectOutOfRangeHardwareIndex) {
  constexpr std::array<size_t, 3> mapping{0, 1, 3};
  EXPECT_DEATH((void)qco::Layout::fromMapping(mapping),
               "mapping must be a permutation");
}
