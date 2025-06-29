/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Export.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"

#include <array>
#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

using namespace dd;

class CNTest : public testing::Test {
protected:
  MemoryManager mm{MemoryManager::create<RealNumber>()};
  RealNumberUniqueTable ut{mm};
  ComplexNumbers cn{ut};
};

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

  auto e = cn.lookup(1., -1.);
  std::cout << e << "\n";
  std::cout << ComplexValue{1., 1.} << "\n";
  std::cout << ComplexValue{1., -1.} << "\n";
  std::cout << ComplexValue{1., -0.5} << "\n";
  ut.print();
  std::cout << ut.getStats();
}

TEST_F(CNTest, NearZeroLookup) {
  auto d = cn.lookup(RealNumber::eps / 10., RealNumber::eps / 10.);
  EXPECT_TRUE(d.exactlyZero());
}

TEST_F(CNTest, SortedBuckets) {
  constexpr fp num = 0.25;

  const std::array<fp, 7> numbers = {
      num + (2. * RealNumber::eps), num - (2. * RealNumber::eps),
      num + (4. * RealNumber::eps), num,
      num - (4. * RealNumber::eps), num + (6. * RealNumber::eps),
      num + (8. * RealNumber::eps)};

  const auto theBucket =
      static_cast<std::size_t>(RealNumberUniqueTable::hash(num));

  for (auto const& number : numbers) {
    ASSERT_EQ(theBucket, ut.hash(number));
    const auto* entry = ut.lookup(number);
    ASSERT_NE(entry, nullptr);
  }

  RealNumber* p = ut.getTable().at(theBucket);
  ASSERT_NE(p, nullptr);

  constexpr fp last = std::numeric_limits<fp>::min();
  std::size_t counter = 0;
  while (p != nullptr) {
    ASSERT_LT(last, p->value);
    p = p->next();
    ++counter;
  }
  std::cout << ut.getStats();
  EXPECT_EQ(counter, numbers.size());
}

TEST_F(CNTest, GarbageCollectSomeInBucket) {
  EXPECT_EQ(ut.garbageCollect(), 0);

  constexpr auto num = 0.25;
  const auto [r, i] = cn.lookup(num, 0.0);
  ASSERT_NE(r, nullptr);
  ASSERT_NE(i, nullptr);

  const fp num2 = num + (2. * RealNumber::eps);
  auto lookup2 = cn.lookup(num2, 0.0);
  ASSERT_NE(lookup2.r, nullptr);
  ASSERT_NE(lookup2.i, nullptr);
  lookup2.mark();

  // num2 should be placed in same bucket as num
  auto key = RealNumberUniqueTable::hash(num);
  auto key2 = RealNumberUniqueTable::hash(num2);
  ASSERT_EQ(key, key2);

  const auto& table = ut.getTable();
  const auto* p = table[static_cast<std::size_t>(key)];
  EXPECT_NEAR(p->value, num, RealNumber::eps);

  ASSERT_NE(p->next(), nullptr);
  EXPECT_NEAR((p->next())->value, num2, RealNumber::eps);

  ut.garbageCollect(true); // num should be collected
  lookup2.unmark();
  const auto* q = table[static_cast<std::size_t>(key)];
  ASSERT_NE(q, nullptr);
  EXPECT_NEAR(q->value, num2, RealNumber::eps);
  EXPECT_EQ(q->next(), nullptr);
}

TEST_F(CNTest, LookupInNeighbouringBuckets) {
  std::clog << "Current rounding mode: " << std::numeric_limits<fp>::round_style
            << "\n";
  const auto mask = ut.getTable().size() - 1;
  const auto fpMask = static_cast<fp>(mask);
  const std::size_t nbucket = mask + 1U;
  auto preHash = [fpMask](const fp val) { return val * fpMask; };

  // lower border of a bucket
  const fp numBucketBorder = (0.25 * fpMask - 0.5) / (fpMask);
  const auto hashBucketBorder = RealNumberUniqueTable::hash(numBucketBorder);
  std::cout.flush();
  std::clog << "numBucketBorder          = "
            << std::setprecision(std::numeric_limits<fp>::max_digits10)
            << numBucketBorder << "\n";
  std::clog << "preHash(numBucketBorder) = " << preHash(numBucketBorder)
            << "\n";
  std::clog << "hashBucketBorder         = " << hashBucketBorder << "\n"
            << std::flush;
  EXPECT_EQ(hashBucketBorder, nbucket / 4);

  // insert a number slightly away from the border
  const fp numAbove = numBucketBorder + (2 * RealNumber::eps);
  const auto lookupAbove = cn.lookup(numAbove, 0.0);
  ASSERT_NE(lookupAbove.r, nullptr);
  ASSERT_NE(lookupAbove.i, nullptr);
  const auto key = RealNumberUniqueTable::hash(numAbove);
  EXPECT_EQ(key, nbucket / 4);

  // insert a number barely in the bucket below
  const fp numBarelyBelow = numBucketBorder - (RealNumber::eps / 10);
  const auto lookupBarelyBelow = cn.lookup(numBarelyBelow, 0.0);
  ASSERT_NE(lookupBarelyBelow.r, nullptr);
  ASSERT_NE(lookupBarelyBelow.i, nullptr);
  const auto hashBarelyBelow = RealNumberUniqueTable::hash(numBarelyBelow);
  std::clog << "numBarelyBelow          = "
            << std::setprecision(std::numeric_limits<fp>::max_digits10)
            << numBarelyBelow << "\n";
  std::clog << "preHash(numBarelyBelow) = " << preHash(numBarelyBelow) << "\n";
  std::clog << "hashBarelyBelow         = " << hashBarelyBelow << "\n";
  EXPECT_EQ(hashBarelyBelow, (nbucket / 4) - 1);

  // insert another number in the bucket below a bit farther away from the
  // border
  const fp numBelow = numBucketBorder - (2 * RealNumber::eps);
  const auto lookupBelow = cn.lookup(numBelow, 0.0);
  ASSERT_NE(lookupBelow.r, nullptr);
  ASSERT_NE(lookupBelow.i, nullptr);
  const auto hashBelow = RealNumberUniqueTable::hash(numBelow);
  std::clog << "numBelow          = "
            << std::setprecision(std::numeric_limits<fp>::max_digits10)
            << numBelow << "\n";
  std::clog << "preHash(numBelow) = " << preHash(numBelow) << "\n";
  std::clog << "hashBelow         = " << hashBelow << "\n";
  EXPECT_EQ(hashBelow, (nbucket / 4) - 1);

  // insert border number that is too far away from the number in the bucket,
  // but is close enough to a number in the bucket below
  const fp num4 = numBucketBorder;
  const auto c = cn.lookup(num4, 0.0);
  const auto key4 = RealNumberUniqueTable::hash(num4 - RealNumber::eps);
  EXPECT_EQ(hashBarelyBelow, key4);
  EXPECT_NEAR(c.r->value, numBarelyBelow, RealNumber::eps);

  // insert a number in the higher bucket
  const fp numNextBorder = numBucketBorder +
                           (1.0 / static_cast<double>(nbucket - 1)) +
                           RealNumber::eps;
  const auto lookupNextBorder = cn.lookup(numNextBorder, 0.0);
  ASSERT_NE(lookupNextBorder.r, nullptr);
  ASSERT_NE(lookupNextBorder.i, nullptr);
  const auto hashNextBorder = RealNumberUniqueTable::hash(numNextBorder);
  std::clog << "numNextBorder          = "
            << std::setprecision(std::numeric_limits<fp>::max_digits10)
            << numNextBorder << "\n";
  std::clog << "preHash(numNextBorder) = " << preHash(numNextBorder) << "\n";
  std::clog << "hashNextBorder         = " << hashNextBorder << "\n";
  EXPECT_EQ(hashNextBorder, (nbucket / 4) + 1);

  // search for a number in the lower bucket that is ultimately close enough to
  // a number in the upper bucket
  const fp num6 = numNextBorder - (RealNumber::eps / 10);
  const auto d = cn.lookup(num6, 0.0);
  const auto key6 = RealNumberUniqueTable::hash(num6 + RealNumber::eps);
  EXPECT_EQ(hashNextBorder, key6);
  EXPECT_NEAR(d.r->value, numNextBorder, RealNumber::eps);
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
  auto imag = cn.lookup(0., 1.);
  auto imagStr = imag.toString(false);
  EXPECT_STREQ(imagStr.c_str(), "1i");
  auto imagStrFormatted = imag.toString(true);
  EXPECT_STREQ(imagStrFormatted.c_str(), "+i");

  auto superposition = cn.lookup(SQRT2_2, SQRT2_2);
  auto superpositionStr = superposition.toString(false, 3);
  EXPECT_STREQ(superpositionStr.c_str(), "0.707+0.707i");
  auto superpositionStrFormatted = superposition.toString(true, 3);
  EXPECT_STREQ(superpositionStrFormatted.c_str(), "1/√2(1+i)");
  auto negSuperposition = cn.lookup(SQRT2_2, -SQRT2_2);
  auto negSuperpositionStrFormatted = negSuperposition.toString(true, 3);
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
