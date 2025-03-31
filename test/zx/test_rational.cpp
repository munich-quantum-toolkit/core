/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "zx/Rational.hpp"
#include "zx/ZXDefinitions.hpp"

#include <gtest/gtest.h>

namespace zx {
class RationalTest : public ::testing::Test {};

TEST_F(RationalTest, normalize) {
  const PiRational r(-33, 16);
  EXPECT_EQ(r, PiRational(-1, 16));
}

TEST_F(RationalTest, fromDouble) {
  const PiRational r(-PI / 8);
  EXPECT_EQ(r, PiRational(-1, 8));
}

TEST_F(RationalTest, fromDouble2) {
  const PiRational r(-3 * PI / 4);
  EXPECT_EQ(r, PiRational(-3, 4));
}

TEST_F(RationalTest, fromDouble3) {
  const PiRational r(-7 * PI / 8);
  EXPECT_EQ(r, PiRational(-7, 8));
}

TEST_F(RationalTest, fromDouble4) {
  const PiRational r(-1 * PI / 32);
  EXPECT_EQ(r, PiRational(-1, 32));
}

TEST_F(RationalTest, fromDouble5) {
  const PiRational r((5000 * PI) + (PI / 4));
  EXPECT_EQ(r, PiRational(1, 4));
}

TEST_F(RationalTest, fromDouble6) {
  const PiRational r((-5000 * PI) + (5 * PI / 4));
  EXPECT_EQ(r, PiRational(-3, 4));
}

TEST_F(RationalTest, add) {
  const PiRational r0(1, 8);
  const PiRational r1(7, 8);
  const auto r = r0 + r1;

  EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, add2) {
  const PiRational r0(9, 8);
  const PiRational r1(7, 8);
  const auto r = r0 + r1;

  EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub) {
  const PiRational r0(9, 8);
  const PiRational r1(-7, 8);
  const auto r = r0 - r1;

  EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub2) {
  const PiRational r0(-1, 2);
  const PiRational r1(1, 2);
  const auto r = r0 - r1;

  EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, mul) {
  const PiRational r0(1, 8);
  const PiRational r1(1, 2);
  const auto r = r0 * r1;

  EXPECT_EQ(r, PiRational(1, 16));
}

TEST_F(RationalTest, mul2) {
  const PiRational r0(1, 8);
  const PiRational r1(0, 1);
  const auto r = r0 * r1;

  EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, div) {
  const PiRational r0(1, 2);
  const PiRational r1(1, 2);
  const auto r = r0 / r1;

  EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, approximateDivPi) {
  const PiRational r(1'000'000'000'000'000 - 1, 1'000'000'000'000'000);
  EXPECT_TRUE(r.isCloseDivPi(1, 1e-9));
}

TEST_F(RationalTest, approximate) {
  const PiRational r(1, 1);
  EXPECT_TRUE(r.isClose(3.14159, 1e-5));
}
} // namespace zx
