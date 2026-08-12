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
#include <llvm/ADT/DenseSet.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>

using namespace mlir;
using namespace mlir::qco;

TEST(LayoutTest, DefaultConstructedIsEmpty) {
  const Layout layout;
  EXPECT_EQ(layout.nProgramQubits(), 0UL);
  EXPECT_EQ(layout.nHardwareQubits(), 0UL);
}

TEST(LayoutTest, ConstructFromPermutation) {
  constexpr std::array<size_t, 3> mapping{2, 0, 1};
  const auto layout = Layout::fromMapping(mapping);

  EXPECT_EQ(layout.nHardwareQubits(), mapping.size());
  EXPECT_EQ(ArrayRef<size_t>(layout.getProgramToHardware()),
            ArrayRef<size_t>(mapping));
  EXPECT_EQ(layout.getProgramIndex(0), 1);
  EXPECT_EQ(layout.getProgramIndex(1), 2);
  EXPECT_EQ(layout.getProgramIndex(2), 0);
}

TEST(LayoutDeathTest, RejectDuplicateHardwareIndex) {
  constexpr std::array<size_t, 3> mapping{0, 0, 2};
  EXPECT_DEATH((void)Layout::fromMapping(mapping),
               "mapping must be a permutation");
}

TEST(LayoutDeathTest, RejectOutOfRangeHardwareIndex) {
  constexpr std::array<size_t, 3> mapping{0, 1, 3};
  EXPECT_DEATH((void)Layout::fromMapping(mapping),
               "mapping must be a permutation");
}

TEST(LayoutTest, RandomPlacesEveryProgramOnDistinctHardware) {
  constexpr size_t nProg = 3;
  constexpr size_t nHw = 5;
  const auto layout = Layout::random(nProg, nHw, /*seed=*/42);

  EXPECT_EQ(layout.nProgramQubits(), nProg);
  EXPECT_EQ(layout.nHardwareQubits(), nHw);

  llvm::DenseSet<size_t> mappedHwIndices;
  for (size_t prog = 0; prog < nProg; ++prog) {
    const auto hw = layout.getHardwareIndex(prog);
    EXPECT_LT(hw, nHw);
    EXPECT_TRUE(layout.hasProgramAt(hw));
    EXPECT_EQ(layout.getProgramIndex(hw), prog);
    mappedHwIndices.insert(hw);
  }
  EXPECT_EQ(mappedHwIndices.size(), nProg);
}

TEST(LayoutTest, RandomLeavesExtraHardwareUnmapped) {
  constexpr size_t nProg = 2;
  constexpr size_t nHw = 5;
  const auto layout = Layout::random(nProg, nHw, /*seed=*/0);

  const auto mappedCount =
      std::ranges::count_if(std::views::iota(size_t{0}, nHw),
                            [&](size_t hw) { return layout.hasProgramAt(hw); });
  EXPECT_EQ(mappedCount, nProg);
}

TEST(LayoutTest, RandomAcceptsZeroPrograms) {
  const auto layout =
      Layout::random(/*nProgramQubits=*/0, /*nHardwareQubits=*/4, /*seed=*/0);
  EXPECT_EQ(layout.nProgramQubits(), 0UL);
  EXPECT_EQ(layout.nHardwareQubits(), 4UL);
  for (size_t hw = 0; hw < 4; ++hw) {
    EXPECT_FALSE(layout.hasProgramAt(hw));
  }
}

TEST(LayoutTest, HasProgramAtDistinguishesMappedAndUnmapped) {
  constexpr size_t nHw = 4;
  const auto layout = Layout::random(/*nProgramQubits=*/2, nHw, /*seed=*/0);

  const auto mappedCount =
      std::ranges::count_if(std::views::iota(size_t{0}, nHw),
                            [&](size_t hw) { return layout.hasProgramAt(hw); });
  EXPECT_EQ(mappedCount, 2);
  EXPECT_EQ(nHw - static_cast<size_t>(mappedCount), 2UL);
}

TEST(LayoutTest, SwapBetweenMappedExchangesPrograms) {
  auto layout =
      Layout::random(/*nProgramQubits=*/2, /*nHardwareQubits=*/4, /*seed=*/0);
  const auto hwA = layout.getHardwareIndex(0);
  const auto hwB = layout.getHardwareIndex(1);
  ASSERT_NE(hwA, hwB);

  layout.swap(hwA, hwB);

  EXPECT_EQ(layout.getProgramIndex(hwA), 1UL);
  EXPECT_EQ(layout.getProgramIndex(hwB), 0UL);
  EXPECT_EQ(layout.getHardwareIndex(0), hwB);
  EXPECT_EQ(layout.getHardwareIndex(1), hwA);
}

TEST(LayoutTest, SwapSameHardwareIsNoOp) {
  auto layout =
      Layout::random(/*nProgramQubits=*/2, /*nHardwareQubits=*/4, /*seed=*/0);
  const auto before = layout;

  layout.swap(0, 0);

  EXPECT_EQ(layout, before);
}

TEST(LayoutTest, EqualityReflectsMapping) {
  const auto a =
      Layout::random(/*nProgramQubits=*/3, /*nHardwareQubits=*/5, /*seed=*/1);
  const auto b =
      Layout::random(/*nProgramQubits=*/3, /*nHardwareQubits=*/5, /*seed=*/1);
  const auto c =
      Layout::random(/*nProgramQubits=*/3, /*nHardwareQubits=*/5, /*seed=*/2);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(LayoutTest, SwapCommutesWithItself) {
  auto layout =
      Layout::random(/*nProgramQubits=*/2, /*nHardwareQubits=*/4, /*seed=*/0);
  const auto before = layout;
  const auto hwA = layout.getHardwareIndex(0);
  const auto hwB = layout.getHardwareIndex(1);

  layout.swap(hwA, hwB);
  layout.swap(hwA, hwB);

  EXPECT_EQ(layout, before);
}
