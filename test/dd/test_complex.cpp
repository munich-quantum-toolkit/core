/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/ComplexNumbers.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Export.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

using namespace dd;

namespace {

class CNTest : public testing::Test {
protected:
  const fp savedTolerance = RealNumber::eps;
  void TearDown() override { ComplexNumbers::setTolerance(savedTolerance); }

  MemoryManager mm{MemoryManager::create<RealNumber>()};
  RealNumberUniqueTable ut{mm};
  ComplexNumbers cn{ut};
};

} // namespace

TEST_F(CNTest, ComplexNumberCreation) {
  EXPECT_TRUE(cn.lookup(Complex::zero()).exactlyZero());
  EXPECT_TRUE(cn.lookup(Complex::one()).exactlyOne());
  EXPECT_TRUE(cn.lookup(1e-16, 0.).exactlyZero());
  EXPECT_EQ(RealNumber::val(cn.lookup(1e-16, 1.).r), 0.);
  EXPECT_EQ(RealNumber::val(cn.lookup(1e-16, 1.).i), 1.);
  EXPECT_EQ(RealNumber::val(cn.lookup(1e-16, -1.).r), 0.);
  EXPECT_EQ(RealNumber::val(cn.lookup(1e-16, -1.).i), -1.);
  EXPECT_EQ(RealNumber::val(cn.lookup(-1., -1.).r), -1.);
  EXPECT_EQ(RealNumber::val(cn.lookup(-1., -1.).i), -1.);
  auto c = cn.lookup(0., -1.);
  std::cout << c << "\n";
  EXPECT_EQ(RealNumber::val(cn.lookup(c).r), 0.);
  EXPECT_EQ(RealNumber::val(cn.lookup(c).i), -1.);
  c = cn.lookup(0., 1.);
  EXPECT_EQ(RealNumber::val(cn.lookup(c).r), 0.);
  EXPECT_EQ(RealNumber::val(cn.lookup(c).i), 1.);
  c = cn.lookup(0., -0.5);
  std::cout << c << "\n";
  EXPECT_EQ(RealNumber::val(cn.lookup(c).r), 0.);
  EXPECT_EQ(RealNumber::val(cn.lookup(c).i), -0.5);
  c = cn.lookup(-1., -1.);
  EXPECT_EQ(RealNumber::val(cn.lookup(c).r), -1.);
  EXPECT_EQ(RealNumber::val(cn.lookup(c).i), -1.);
  std::cout << c << "\n";

  auto const e = cn.lookup(1., -1.);
  std::cout << e << "\n";
  std::cout << ComplexValue{1., 1.} << "\n";
  std::cout << ComplexValue{1., -1.} << "\n";
  std::cout << ComplexValue{1., -0.5} << "\n";
  ut.print();
  std::cout << ut.getStats();
}

TEST_F(CNTest, NearZeroLookup) {
  auto const d = cn.lookup(RealNumber::eps / 10., RealNumber::eps / 10.);
  EXPECT_TRUE(d.exactlyZero());
}

TEST_F(CNTest, NearestEntriesAcrossCellBoundaries) {
  constexpr fp border = 0.25;
  const auto tolerance = RealNumber::eps;
  auto* lower = ut.lookup(border - (0.75 * tolerance));
  auto* upper = ut.lookup(border + (0.75 * tolerance));
  ASSERT_NE(lower, upper);
  EXPECT_EQ(ut.lookup(border), lower);
  EXPECT_EQ(ut.lookup(border + (tolerance / 4)), upper);
  EXPECT_EQ(ut.lookup(border - (tolerance / 4)), lower);
  EXPECT_EQ(ut.lookup(-border), RealNumber::getNegativePointer(lower));

  /// Rounded endpoints must not skip a cell containing an existing match.
  auto* half = ut.lookup(0.5);
  EXPECT_EQ(ut.lookup(std::nextafter(0.5, 0.)), half);
  EXPECT_EQ(ut.lookup(std::nextafter(0.5, 1.)), half);
  EXPECT_EQ(ut.lookup(-0.), &constants::zero);
  EXPECT_EQ(ut.lookup(1. + (tolerance / 2)), &constants::one);
  EXPECT_EQ(ut.lookup(SQRT2_2 - (tolerance / 2)), &constants::sqrt2over2);
}

TEST_F(CNTest, ReusesLargeFiniteValues) {
  for (const fp value : {2., 1e15, std::numeric_limits<fp>::max()}) {
    const auto* entry = ut.lookup(value);
    EXPECT_EQ(entry->value, value);
    EXPECT_EQ(ut.lookup(value), entry);
    EXPECT_EQ(ut.lookup(-value), RealNumber::getNegativePointer(entry));
  }
}

TEST_F(CNTest, GrowthPreservesEntriesAndCollectionFlags) {
  const auto initialBuckets = ut.getTable().size();
  auto* half = ut.lookup(0.5);
  std::vector<RealNumber*> retained;
  for (size_t i = 0; i <= initialBuckets; ++i) {
    auto* entry = ut.lookup(2. + (static_cast<fp>(i) / 1024.));
    if (i % 8192 == 0) {
      RealNumber::mark(entry);
      retained.push_back(entry);
    }
  }
  ASSERT_GT(ut.getTable().size(), initialBuckets);
  EXPECT_TRUE(RealNumber::isImmortal(half));
  const auto before = ut.getStats().numEntries;
  for (auto* entry : retained) {
    EXPECT_TRUE(RealNumber::isMarked(entry));
    EXPECT_EQ(ut.lookup(entry->value), entry);
  }
  EXPECT_EQ(ut.garbageCollect(true), before - retained.size() - 1);
  EXPECT_EQ(ut.getStats().numEntries, retained.size() + 1);
  EXPECT_EQ(ut.lookup(0.5), half);
  for (auto* entry : retained) {
    EXPECT_EQ(ut.lookup(entry->value), entry);
    RealNumber::unmark(entry);
  }
  EXPECT_EQ(ut.garbageCollect(true), retained.size());
  EXPECT_EQ(ut.getStats().numEntries, 1U);
  ut.clear();
  mm.reset();
  EXPECT_EQ(ut.getStats().numEntries, 0U);
  EXPECT_EQ(ut.lookup(0.25)->value, 0.25);
}

TEST_F(CNTest, CollectsSingleEntryWithoutDynamicImmortals) {
  ComplexNumbers::setTolerance(1.);
  auto manager = MemoryManager::create<RealNumber>();
  RealNumberUniqueTable table(manager);
  EXPECT_EQ(table.getStats().numEntries, 0U);
  EXPECT_EQ(table.lookup(4.)->value, 4.);
  EXPECT_EQ(table.garbageCollect(true), 1U);
  EXPECT_EQ(table.getStats().numEntries, 0U);
}

TEST_F(CNTest, ToleranceChangesPreserveNearestLookup) {
  auto* first = ut.lookup(0.25);
  auto* second = ut.lookup(0.25 + (4 * RealNumber::eps));
  ASSERT_NE(first, second);
  RealNumber::mark(first);
  ComplexNumbers::setTolerance(RealNumber::eps * 16);
  EXPECT_EQ(ut.lookup(second->value), second);
  EXPECT_EQ(ut.lookup(first->value), first);
  EXPECT_TRUE(RealNumber::isMarked(first));

  /// Compare against every retained entry, independently of index layout.
  for (const fp tolerance : {
           0.,
           std::numeric_limits<fp>::denorm_min(),
           1e-300,
           1e-15,
           savedTolerance,
           1e-4,
           1e100,
           std::numeric_limits<fp>::max() / 4,
       }) {
    ComplexNumbers::setTolerance(tolerance);
    for (const fp value : {
             std::numeric_limits<fp>::denorm_min(),
             1e-299,
             0.25,
             std::nextafter(0.25, 0.),
             std::nextafter(0.25, 1.),
             0.5,
             0.75,
             1.,
             2.,
             1e100,
             std::numeric_limits<fp>::max(),
         }) {
      RealNumber* expected = nullptr;
      if (value <= tolerance) {
        expected = &constants::zero;
      } else if (std::abs(value - 1.) <= tolerance) {
        expected = &constants::one;
      } else if (std::abs(value - SQRT2_2) <= tolerance) {
        expected = &constants::sqrt2over2;
      } else {
        auto distance = tolerance;
        for (auto* head : ut.getTable()) {
          for (auto* entry = head; entry != nullptr; entry = entry->next()) {
            const auto difference = std::abs(entry->value - value);
            if (difference <= distance &&
                (expected == nullptr || difference < distance ||
                 entry->value < expected->value)) {
              expected = entry;
              distance = difference;
            }
          }
        }
      }
      const auto* actual = ut.lookup(value);
      if (expected != nullptr) {
        EXPECT_EQ(actual, expected);
      } else {
        EXPECT_EQ(actual->value, value);
      }
    }
  }
}

TEST(DDComplexTest, LowestFractions) {
  EXPECT_THAT(ComplexValue::getLowestFraction(0.0), ::testing::Pair(0, 1));
  EXPECT_THAT(ComplexValue::getLowestFraction(0.2), ::testing::Pair(1, 5));
  EXPECT_THAT(ComplexValue::getLowestFraction(0.25), ::testing::Pair(1, 4));
  EXPECT_THAT(ComplexValue::getLowestFraction(0.5), ::testing::Pair(1, 2));
  EXPECT_THAT(ComplexValue::getLowestFraction(0.75), ::testing::Pair(3, 4));
  EXPECT_THAT(ComplexValue::getLowestFraction(1.5), ::testing::Pair(3, 2));
  EXPECT_THAT(ComplexValue::getLowestFraction(2.0), ::testing::Pair(2, 1));
  EXPECT_THAT(ComplexValue::getLowestFraction(2047.0 / 2048.0, 1024U),
              ::testing::Pair(1, 1));
}

TEST_F(CNTest, NumberPrintingToString) {
  auto const imag = cn.lookup(0., 1.);
  auto const imagStr = imag.toString(false);
  EXPECT_STREQ(imagStr.c_str(), "1i");
  auto const imagStrFormatted = imag.toString(true);
  EXPECT_STREQ(imagStrFormatted.c_str(), "+i");

  auto const superposition = cn.lookup(SQRT2_2, SQRT2_2);
  auto const superpositionStr = superposition.toString(false, 3);
  EXPECT_STREQ(superpositionStr.c_str(), "0.707+0.707i");
  auto const superpositionStrFormatted = superposition.toString(true, 3);
  EXPECT_STREQ(superpositionStrFormatted.c_str(), "1/√2(1+i)");
  auto const negSuperposition = cn.lookup(SQRT2_2, -SQRT2_2);
  auto const negSuperpositionStrFormatted = negSuperposition.toString(true, 3);
  EXPECT_STREQ(negSuperpositionStrFormatted.c_str(), "1/√2(1-i)");
}

TEST(DDComplexTest, NumberPrintingFormattedFractions) {
  std::stringstream ss{};

  ComplexValue::printFormatted(ss, 0.0, false);
  EXPECT_STREQ(ss.str().c_str(), "0");
  ss.str("");
  ComplexValue::printFormatted(ss, -0.0, false);
  EXPECT_STREQ(ss.str().c_str(), "-0");
  ss.str("");

  ComplexValue::printFormatted(ss, 0.0, true);
  EXPECT_STREQ(ss.str().c_str(), "+0i");
  ss.str("");
  ComplexValue::printFormatted(ss, -0.0, true);
  EXPECT_STREQ(ss.str().c_str(), "-0i");
  ss.str("");

  ComplexValue::printFormatted(ss, 0.25, false);
  EXPECT_STREQ(ss.str().c_str(), "1/4");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.25, true);
  EXPECT_STREQ(ss.str().c_str(), "+i/4");
  ss.str("");

  ComplexValue::printFormatted(ss, 0.5, false);
  EXPECT_STREQ(ss.str().c_str(), "1/2");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.5, true);
  EXPECT_STREQ(ss.str().c_str(), "+i/2");
  ss.str("");

  ComplexValue::printFormatted(ss, 0.75, false);
  EXPECT_STREQ(ss.str().c_str(), "3/4");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.75, true);
  EXPECT_STREQ(ss.str().c_str(), "+3i/4");
  ss.str("");

  ComplexValue::printFormatted(ss, 1, false);
  EXPECT_STREQ(ss.str().c_str(), "1");
  ss.str("");
  ComplexValue::printFormatted(ss, 1, true);
  EXPECT_STREQ(ss.str().c_str(), "+i");
  ss.str("");

  ComplexValue::printFormatted(ss, 1.5, false);
  EXPECT_STREQ(ss.str().c_str(), "3/2");
  ss.str("");
  ComplexValue::printFormatted(ss, 1.5, true);
  EXPECT_STREQ(ss.str().c_str(), "+3i/2");
  ss.str("");

  ComplexValue::printFormatted(ss, 2, false);
  EXPECT_STREQ(ss.str().c_str(), "2");
  ss.str("");
  ComplexValue::printFormatted(ss, 2, true);
  EXPECT_STREQ(ss.str().c_str(), "+2i");
  ss.str("");
}

TEST(DDComplexTest, NumberPrintingFormattedFractionsSqrt) {
  std::stringstream ss{};

  ComplexValue::printFormatted(ss, 0.25 * SQRT2_2, false);
  EXPECT_STREQ(ss.str().c_str(), "1/(4√2)");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.25 * SQRT2_2, true);
  EXPECT_STREQ(ss.str().c_str(), "+i/(4√2)");
  ss.str("");

  ComplexValue::printFormatted(ss, 0.5 * SQRT2_2, false);
  EXPECT_STREQ(ss.str().c_str(), "1/(2√2)");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.5 * SQRT2_2, true);
  EXPECT_STREQ(ss.str().c_str(), "+i/(2√2)");
  ss.str("");

  ComplexValue::printFormatted(ss, 0.75 * SQRT2_2, false);
  EXPECT_STREQ(ss.str().c_str(), "3/(4√2)");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.75 * SQRT2_2, true);
  EXPECT_STREQ(ss.str().c_str(), "+3i/(4√2)");
  ss.str("");

  ComplexValue::printFormatted(ss, SQRT2_2, false);
  EXPECT_STREQ(ss.str().c_str(), "1/√2");
  ss.str("");
  ComplexValue::printFormatted(ss, SQRT2_2, true);
  EXPECT_STREQ(ss.str().c_str(), "+i/√2");
  ss.str("");

  ComplexValue::printFormatted(ss, 1.5 * SQRT2_2, false);
  EXPECT_STREQ(ss.str().c_str(), "3/(2√2)");
  ss.str("");
  ComplexValue::printFormatted(ss, 1.5 * SQRT2_2, true);
  EXPECT_STREQ(ss.str().c_str(), "+3i/(2√2)");
  ss.str("");

  ComplexValue::printFormatted(ss, 2 * SQRT2_2, false);
  EXPECT_STREQ(ss.str().c_str(), "2/√2");
  ss.str("");
  ComplexValue::printFormatted(ss, 2 * SQRT2_2, true);
  EXPECT_STREQ(ss.str().c_str(), "+2i/√2");
  ss.str("");
}

TEST(DDComplexTest, NumberPrintingFormattedFractionsPi) {
  std::stringstream ss{};

  ComplexValue::printFormatted(ss, 0.25 * PI, false);
  EXPECT_STREQ(ss.str().c_str(), "π/4");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.25 * PI, true);
  EXPECT_STREQ(ss.str().c_str(), "+πi/4");
  ss.str("");

  ComplexValue::printFormatted(ss, 0.5 * PI, false);
  EXPECT_STREQ(ss.str().c_str(), "π/2");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.5 * PI, true);
  EXPECT_STREQ(ss.str().c_str(), "+πi/2");
  ss.str("");

  ComplexValue::printFormatted(ss, 0.75 * PI, false);
  EXPECT_STREQ(ss.str().c_str(), "3π/4");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.75 * PI, true);
  EXPECT_STREQ(ss.str().c_str(), "+3πi/4");
  ss.str("");

  ComplexValue::printFormatted(ss, PI, false);
  EXPECT_STREQ(ss.str().c_str(), "π");
  ss.str("");
  ComplexValue::printFormatted(ss, PI, true);
  EXPECT_STREQ(ss.str().c_str(), "+πi");
  ss.str("");

  ComplexValue::printFormatted(ss, 1.5 * PI, false);
  EXPECT_STREQ(ss.str().c_str(), "3π/2");
  ss.str("");
  ComplexValue::printFormatted(ss, 1.5 * PI, true);
  EXPECT_STREQ(ss.str().c_str(), "+3πi/2");
  ss.str("");

  ComplexValue::printFormatted(ss, 2 * PI, false);
  EXPECT_STREQ(ss.str().c_str(), "2π");
  ss.str("");
  ComplexValue::printFormatted(ss, 2 * PI, true);
  EXPECT_STREQ(ss.str().c_str(), "+2πi");
  ss.str("");
}

TEST(DDComplexTest, NumberPrintingFormattedFloating) {
  std::stringstream ss{};
  ComplexValue::printFormatted(ss, 0.1234, false);
  EXPECT_STREQ(ss.str().c_str(), "0.1234");
  ss.str("");
  ComplexValue::printFormatted(ss, 0.1234, true);
  EXPECT_STREQ(ss.str().c_str(), "+0.1234i");
  ss.str("");
}

TEST_F(CNTest, ComplexTableAllocation) {
  auto mem = MemoryManager::create<RealNumber>();
  const auto allocs = mem.getStats().numAllocated;
  std::cout << allocs << "\n";
  std::vector<RealNumber*> nums{allocs};
  // get all the numbers that are pre-allocated
  for (auto i = 0U; i < allocs; ++i) {
    nums[i] = mem.get<RealNumber>();
  }

  // trigger new allocation
  const auto* num = mem.get<RealNumber>();
  ASSERT_NE(num, nullptr);
  EXPECT_EQ(mem.getStats().numAllocated,
            (1. + MemoryManager::GROWTH_FACTOR) * static_cast<fp>(allocs));

  // clearing the complex table should reduce the allocated size to the original
  // size
  mem.reset();
  EXPECT_EQ(mem.getStats().numAllocated, allocs);

  EXPECT_EQ(mem.getStats().numAvailableForReuse, 0U);
  // obtain entry
  auto* entry = mem.get<RealNumber>();
  // immediately return entry
  mem.returnEntry(*entry);
  EXPECT_EQ(mem.getStats().numAvailableForReuse, 1U);
  // obtain the same entry again, but this time from the available stack
  auto* entry2 = mem.get<RealNumber>();
  EXPECT_EQ(entry, entry2);
}

TEST_F(CNTest, DoubleHitInFindOrInsert) {
  // insert a number somewhere in a bucket
  constexpr fp num1 = 0.5;
  const auto* tnum1 = ut.lookup(num1);
  EXPECT_EQ(tnum1->value, num1);

  // insert a second number that is farther away than the tolerance, but closer
  // than twice the tolerance
  const fp num2 = num1 + (2.1 * RealNumber::eps);
  const auto* tnum2 = ut.lookup(num2);
  EXPECT_EQ(tnum2->value, num2);

  // insert a third number that is close to both previously inserted numbers,
  // but closer to the second
  const fp num3 = num1 + (2.2 * RealNumber::eps);
  const auto* tnum3 = ut.lookup(num3);
  EXPECT_EQ(tnum3->value, num2);
}

TEST_F(CNTest, DoubleHitAcrossBuckets) {
  std::cout << std::setprecision(std::numeric_limits<fp>::max_digits10);

  // insert a number at a lower bucket border
  const fp num1 = 8191.5 / (static_cast<fp>(ut.getTable().size()) - 1);
  const auto* tnum1 = ut.lookup(num1);
  EXPECT_EQ(tnum1->value, num1);

  // insert a second number that is farther away than the tolerance towards the
  // lower bucket, but closer than twice the tolerance
  const fp num2 = num1 - (1.5 * RealNumber::eps);
  const auto* tnum2 = ut.lookup(num2);
  EXPECT_EQ(tnum2->value, num2);

  // insert a third number that is close to both previously inserted numbers,
  // but closer to the second
  const fp num3 = num1 - (0.9 * RealNumber::eps);
  const auto* tnum3 = ut.lookup(num3);
  EXPECT_EQ(tnum3->value, num2);

  // insert a third number that is close to both previously inserted numbers,
  // but closer to the first
  const fp num4 = num1 - (0.6 * RealNumber::eps);
  const auto* tnum4 = ut.lookup(num4);
  EXPECT_EQ(tnum4->value, num1);
}

TEST_F(CNTest, exactlyZeroComparison) {
  const auto notZero = cn.lookup(0, 2 * RealNumber::eps);
  const auto zero = cn.lookup(0, 0);
  EXPECT_TRUE(!notZero.exactlyZero());
  EXPECT_TRUE(zero.exactlyZero());
}

TEST_F(CNTest, exactlyOneComparison) {
  const auto notOne = cn.lookup(1 + (2 * RealNumber::eps), 0);
  const auto one = cn.lookup(1, 0);
  EXPECT_TRUE(!notOne.exactlyOne());
  EXPECT_TRUE(one.exactlyOne());
}

TEST_F(CNTest, ExportConditionalFormat1) {
  EXPECT_STREQ(conditionalFormat(cn.lookup(1, 0)).c_str(), "1");
}

TEST_F(CNTest, ExportConditionalFormat2) {
  EXPECT_STREQ(conditionalFormat(cn.lookup(0, 1)).c_str(), "i");
}

TEST_F(CNTest, ExportConditionalFormat3) {
  EXPECT_STREQ(conditionalFormat(cn.lookup(-1, 0)).c_str(), "-1");
}

TEST_F(CNTest, ExportConditionalFormat4) {
  EXPECT_STREQ(conditionalFormat(cn.lookup(0, -1)).c_str(), "-i");
}

TEST_F(CNTest, ExportConditionalFormat5) {
  const auto num = cn.lookup(-SQRT2_2, -SQRT2_2);
  EXPECT_STREQ(conditionalFormat(num).c_str(), "ℯ(-iπ 3/4)");
  EXPECT_STREQ(conditionalFormat(num, false).c_str(), "-1/√2(1+i)");
}

TEST_F(CNTest, ExportConditionalFormat6) {
  EXPECT_STREQ(conditionalFormat(cn.lookup(-1, -1)).c_str(), "2/√2 ℯ(-iπ 3/4)");
}

TEST_F(CNTest, ExportConditionalFormat7) {
  EXPECT_STREQ(conditionalFormat(cn.lookup(-SQRT2_2, 0)).c_str(), "-1/√2");
}

TEST(DDComplexTest, HashesSignedQuantizedWeights) {
  const std::hash<ComplexValue> hash;
  for (const bool imaginary : {false, true}) {
    std::unordered_set<size_t> hashes;
    for (const fp value : {-0.125, -0.25, -0.5, -0.75}) {
      const ComplexValue weight =
          imaginary ? ComplexValue{0., value} : ComplexValue{value, 0.};
      hashes.insert(hash(weight));
      auto nearby = weight;
      (imaginary ? nearby.i : nearby.r) += RealNumber::eps / 4.;
      EXPECT_EQ(hash(weight), hash(nearby));
    }
    EXPECT_GT(hashes.size(), 1U);
  }
  for (const fp zero : {0., -0., RealNumber::eps / 4., -RealNumber::eps / 4.}) {
    EXPECT_EQ(hash({zero, zero}), hash({0., 0.}));
    EXPECT_EQ(hash({zero, 0.}), hash({0., 0.}));
    EXPECT_EQ(hash({0., zero}), hash({0., 0.}));
    EXPECT_EQ(hash({zero, 0.5}), hash({0., 0.5}));
    EXPECT_EQ(hash({0.5, zero}), hash({0.5, 0.}));
  }
}

TEST(DDComplexTest, PreservesFlagsWhenRelinkingNumbers) {
  RealNumber entry{};
  RealNumber neighbor{};
  RealNumber::immortalize(&entry);
  RealNumber::mark(&entry);
  entry.setNext(&neighbor);
  EXPECT_EQ(entry.next(), &neighbor);
  EXPECT_TRUE(RealNumber::isImmortal(&entry));
  EXPECT_TRUE(RealNumber::isMarked(&entry));
  entry.setNext(nullptr);
  EXPECT_EQ(entry.next(), nullptr);
  EXPECT_TRUE(RealNumber::isImmortal(&entry));
  EXPECT_TRUE(RealNumber::isMarked(&entry));
  RealNumber::unmark(&entry);
  EXPECT_FALSE(RealNumber::isMarked(&entry));
  EXPECT_TRUE(RealNumber::isImmortal(&entry));
}

TEST_F(CNTest, ClearsFlagsWhenReusingNumbers) {
  auto* entry = mm.get<RealNumber>();
  RealNumber::mark(entry);
  RealNumber::immortalize(entry);
  mm.returnEntry(*entry);
  auto* reused = ut.lookup(0.123);
  ASSERT_EQ(reused, entry);
  EXPECT_FALSE(RealNumber::isMarked(reused));
  EXPECT_FALSE(RealNumber::isImmortal(reused));

  ut.clear();
  mm.reset();
  reused = ut.lookup(0.321);
  EXPECT_FALSE(RealNumber::isMarked(reused));
  EXPECT_FALSE(RealNumber::isImmortal(reused));
}

TEST(DDComplexTest, ScalarComplexDivisorsPreserveRange) {
  for (const fp scale : {1e-200, 1e200}) {
    for (const ComplexValue divisor : {ComplexValue{scale, 0.}, {0., scale}}) {
      const auto quotient = (divisor * 2.) / divisor;
      EXPECT_EQ(quotient.r, 2.);
      EXPECT_EQ(quotient.i, 0.);
    }
  }
  const ComplexValue mixed{1e300, 1e-300};
  EXPECT_EQ(mixed / ComplexValue{1.}, mixed);
}

TEST_F(CNTest, MatrixNormalizationPreservesSubnormalComponents) {
  ComplexNumbers::setTolerance(0.);
  Package package(1);
  const auto tiny = std::numeric_limits<fp>::denorm_min();
  const GateMatrix matrix{
      std::complex<fp>{.5, .5},
      {tiny, tiny},
      {},
      {},
  };
  std::array<mEdge, NEDGE> edges{};
  for (size_t i = 0; i < NEDGE; ++i) {
    edges[i] = mEdge::terminal(package.cn.lookup(ComplexValue{matrix[i]}));
  }
  for (const auto& result :
       {package.makeGateDD(matrix, 0), package.makeDDNode(0, edges)}) {
    EXPECT_EQ(result.getValueByIndex(1, 0, 0), matrix[0]);
    EXPECT_EQ(result.getValueByIndex(1, 0, 1), matrix[1]);
  }
}

TEST(DDComplexTest, ComplexTextRejectsUnrepresentableValues) {
  ComplexValue value;
  EXPECT_THROW(value.fromString("1e-400", ""), std::out_of_range);
  EXPECT_THROW(value.fromString("", "1e400i"), std::out_of_range);
  EXPECT_THROW(value.fromString("invalid", ""), std::invalid_argument);
}
