/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Utils/Layout.h"

#include "gtest/gtest.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using namespace mlir;
using namespace mlir::qco;

namespace {

template <typename T> class LayoutTest : public ::testing::Test {};
using IndexTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(LayoutTest, IndexTypes);

TYPED_TEST(LayoutTest, DefaultConstructedIsEmpty) {
  const Layout<TypeParam> layout;
  EXPECT_EQ(layout.nProgramQubits(), 0UL);
  EXPECT_EQ(layout.nHardwareQubits(), 0UL);
}

TYPED_TEST(LayoutTest, ConstructFromPermutation) {
  constexpr std::array<TypeParam, 3> mapping{2, 0, 1};
  const auto layout = Layout<TypeParam>::fromMapping(mapping);
  static_assert(
      std::is_same_v<decltype(layout.getHardwareIndex(0)), TypeParam>);
  static_assert(std::is_same_v<decltype(layout.getProgramIndex(0)), TypeParam>);

  EXPECT_EQ(layout.nHardwareQubits(), mapping.size());
  EXPECT_EQ(ArrayRef<TypeParam>(layout.getProgramToHardware()),
            ArrayRef<TypeParam>(mapping));
  EXPECT_EQ(layout.getProgramIndex(0), 1);
  EXPECT_EQ(layout.getProgramIndex(1), 2);
  EXPECT_EQ(layout.getProgramIndex(2), 0);
}

TYPED_TEST(LayoutTest, RejectDuplicateHardwareIndex) {
  constexpr std::array<TypeParam, 3> mapping{0, 0, 2};
  EXPECT_DEATH(Layout<TypeParam>::fromMapping(mapping),
               "mapping must be a permutation");
}

TYPED_TEST(LayoutTest, RejectOutOfRangeHardwareIndex) {
  constexpr std::array<TypeParam, 3> mapping{0, 1, 3};
  EXPECT_DEATH(Layout<TypeParam>::fromMapping(mapping),
               "mapping must be a permutation");
}

TYPED_TEST(LayoutTest, RandomPlacesEveryProgramOnDistinctHardware) {
  constexpr size_t nProg = 3;
  constexpr size_t nHw = 5;
  const auto layout = Layout<TypeParam>::random(nProg, nHw, /*seed=*/42);

  EXPECT_EQ(layout.nProgramQubits(), nProg);
  EXPECT_EQ(layout.nHardwareQubits(), nHw);
  llvm::DenseSet<TypeParam> mappedHwIndices;
  for (size_t prog = 0; prog < nProg; ++prog) {
    const auto hw = layout.getHardwareIndex(prog);
    EXPECT_LT(hw, nHw);
    EXPECT_TRUE(layout.hasProgramAt(hw));
    EXPECT_EQ(layout.getProgramIndex(hw), prog);
    mappedHwIndices.insert(hw);
  }
  EXPECT_EQ(mappedHwIndices.size(), nProg);
}

TYPED_TEST(LayoutTest, RandomLeavesExtraHardwareUnmapped) {
  constexpr size_t nProg = 2;
  constexpr size_t nHw = 5;
  const auto layout = Layout<TypeParam>::random(nProg, nHw, /*seed=*/0);

  const auto mappedCount = llvm::count_if(
      llvm::seq(nHw), [&](size_t hw) { return layout.hasProgramAt(hw); });
  EXPECT_EQ(mappedCount, nProg);
}

TYPED_TEST(LayoutTest, RandomAcceptsZeroPrograms) {
  const auto layout = Layout<TypeParam>::random(
      /*nProgramQubits=*/0, /*nHardwareQubits=*/4, /*seed=*/0);
  EXPECT_EQ(layout.nProgramQubits(), 0UL);
  EXPECT_EQ(layout.nHardwareQubits(), 4UL);
  for (size_t hw = 0; hw < 4; ++hw) {
    EXPECT_FALSE(layout.hasProgramAt(hw));
  }
}

TYPED_TEST(LayoutTest, HasProgramAtDistinguishesMappedAndUnmapped) {
  constexpr size_t nHw = 4;
  const auto layout =
      Layout<TypeParam>::random(/*nProgramQubits=*/2, nHw, /*seed=*/0);

  const auto mappedCount = llvm::count_if(
      llvm::seq(nHw), [&](size_t hw) { return layout.hasProgramAt(hw); });
  EXPECT_EQ(mappedCount, 2);
  EXPECT_EQ(nHw - static_cast<size_t>(mappedCount), 2UL);
}

TYPED_TEST(LayoutTest, SwapBetweenMappedExchangesPrograms) {
  auto layout = Layout<TypeParam>::random(/*nProgramQubits=*/2,
                                          /*nHardwareQubits=*/4, /*seed=*/0);
  const auto hwA = layout.getHardwareIndex(0);
  const auto hwB = layout.getHardwareIndex(1);
  ASSERT_NE(hwA, hwB);

  layout.swap(hwA, hwB);

  EXPECT_EQ(layout.getProgramIndex(hwA), 1UL);
  EXPECT_EQ(layout.getProgramIndex(hwB), 0UL);
  EXPECT_EQ(layout.getHardwareIndex(0), hwB);
  EXPECT_EQ(layout.getHardwareIndex(1), hwA);
}

TYPED_TEST(LayoutTest, SwapSameHardwareIsNoOp) {
  auto layout = Layout<TypeParam>::random(/*nProgramQubits=*/2,
                                          /*nHardwareQubits=*/4, /*seed=*/0);
  const auto before = layout;

  layout.swap(0, 0);

  EXPECT_EQ(layout, before);
}

TEST(LayoutBoundaryTest, LargestByteLayoutPreservesEveryIndex) {
  for (const auto& layout : {
           Layout<uint8_t>::identity(255),
           Layout<uint8_t>::random(255, 255, 42),
       }) {
    ASSERT_EQ(layout.nHardwareQubits(), 255);
    for (uint8_t prog = 0; prog < 255; ++prog) {
      const auto hw = layout.getHardwareIndex(prog);
      EXPECT_LT(hw, 255);
      EXPECT_EQ(layout.getProgramIndex(hw), prog);
    }
  }
}

TEST(LayoutBoundaryTest, RejectCountsBeyondIndexCapacity) {
  const std::array<uint8_t, 256> mapping{};
  EXPECT_DEATH(Layout<uint8_t>::identity(256), "qubit index capacity");
  EXPECT_DEATH(Layout<uint8_t>::random(1, 256, 0), "qubit index capacity");
  EXPECT_DEATH(Layout<uint8_t>::fromMapping(mapping), "qubit index capacity");
  EXPECT_DEATH(Layout<uint8_t>::random(2, 1, 0),
               "cannot map more program qubits");
}

TEST(LayoutBoundaryTest, WiderIndexSupportsMoreThan65535Sites) {
  const auto layout = Layout<uint32_t>::identity(65536);
  EXPECT_EQ(layout.nHardwareQubits(), 65536);
  EXPECT_EQ(layout.getHardwareIndex(65535), 65535);
}

#ifndef NDEBUG
TEST(LayoutBoundaryTest, RejectWideInputBeforeNarrowing) {
  auto layout = Layout<uint8_t>::identity(1);
  EXPECT_DEATH(layout.getHardwareIndex(256UL), "program index out of bounds");
  EXPECT_DEATH(layout.getHardwareIndices(256UL), "program index out of bounds");
  EXPECT_DEATH(layout.swap(0, 256UL), "hardware index out of bounds");
}
#endif

TYPED_TEST(LayoutTest, EqualityReflectsMapping) {
  const auto a = Layout<TypeParam>::random(/*nProgramQubits=*/3,
                                           /*nHardwareQubits=*/5, /*seed=*/1);
  const auto b = Layout<TypeParam>::random(/*nProgramQubits=*/3,
                                           /*nHardwareQubits=*/5, /*seed=*/1);
  const auto c = Layout<TypeParam>::random(/*nProgramQubits=*/3,
                                           /*nHardwareQubits=*/5, /*seed=*/2);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TYPED_TEST(LayoutTest, SwapCommutesWithItself) {
  auto layout = Layout<TypeParam>::random(/*nProgramQubits=*/2,
                                          /*nHardwareQubits=*/4, /*seed=*/0);
  const auto before = layout;
  const auto hwA = layout.getHardwareIndex(0);
  const auto hwB = layout.getHardwareIndex(1);

  layout.swap(hwA, hwB);
  layout.swap(hwA, hwB);

  EXPECT_EQ(layout, before);
}

} // namespace
