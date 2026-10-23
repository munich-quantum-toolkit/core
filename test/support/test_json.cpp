/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "support/JSON.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <bit>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {
TEST(JSON, StrictSyntaxAndUnicode) {
  for (const auto* text : {
           "01",
           "1.",
           "+1",
           "1e",
           "NaN",
           "Infinity",
           "1e309",
           "[1,]",
           "{\"a\":1,}",
           "true false",
           R"("\uD800")",
           R"("\uDC00")",
           R"("\uD800\u0041")",
           "\"\xFF\"",
           R"({"a":1,"\u0061":2})",
       }) {
    auto result = mqt::json::parse(text, "input.json");
    ASSERT_FALSE(result) << text;
    EXPECT_NE(llvm::toString(result.takeError()).find("input.json"),
              std::string::npos);
  }
  auto parsed =
      mqt::json::parse(R"({"key":"a\u0000b","emoji":"\uD83D\uDE00"})");
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  EXPECT_EQ(parsed->getAsObject()->getString("key"),
            llvm::StringRef("a\0b", 3));
  EXPECT_EQ(parsed->getAsObject()->getString("emoji"), "😀");
}

TEST(JSON, BoundsNesting) {
  auto accepted =
      mqt::json::parse(std::string(256, '[') + "0" + std::string(256, ']'));
  ASSERT_TRUE(static_cast<bool>(accepted));
  auto rejected =
      mqt::json::parse(std::string(257, '[') + "0" + std::string(257, ']'));
  ASSERT_FALSE(rejected);
  EXPECT_NE(llvm::toString(rejected.takeError()).find("nesting"),
            std::string::npos);
}

TEST(JSON, NumbersAndCanonicalOrdering) {
  auto parsed = mqt::json::parse(
      R"({"z":18446744073709551615,"a":-9223372036854775808})");
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  EXPECT_EQ(parsed->getAsObject()->get("z")->getAsUINT64(),
            std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(mqt::json::serialize(*parsed),
            R"({"a":-9223372036854775808,"z":18446744073709551615})");
  for (const auto value : {
           0.,
           -0.,
           1.25,
           -1.25,
           0.035329071259631153,
           std::numeric_limits<double>::denorm_min(),
           std::numeric_limits<double>::min(),
           std::numeric_limits<double>::max(),
           0x1p63,
           -0x1p63,
       }) {
    const auto encoded = mqt::json::serialize(value);
    auto decoded = mqt::json::parse(encoded);
    ASSERT_TRUE(static_cast<bool>(decoded)) << encoded;
    ASSERT_TRUE(decoded->getAsNumber());
    EXPECT_EQ(std::bit_cast<uint64_t>(*decoded->getAsNumber()),
              std::bit_cast<uint64_t>(value))
        << encoded;
  }
  EXPECT_EQ(mqt::json::serialize(std::numeric_limits<double>::infinity()),
            "null");
  auto underflow = mqt::json::parse("-1e-999");
  ASSERT_TRUE(static_cast<bool>(underflow));
  EXPECT_TRUE(std::signbit(*underflow->getAsNumber()));
  auto fraction = mqt::json::parse("1.0");
  ASSERT_TRUE(static_cast<bool>(fraction));
  EXPECT_FALSE(fraction->getAsUINT64());
}

TEST(JSON, IndependentOfNumericLocale) {
  const std::string original = std::setlocale(LC_NUMERIC, nullptr);
  struct Restore {
    std::string locale;
    ~Restore() { std::setlocale(LC_NUMERIC, locale.c_str()); }
  };
  const Restore restore{original};
  if (std::setlocale(LC_NUMERIC, "de_DE.UTF-8") == nullptr &&
      std::setlocale(LC_NUMERIC, "de_DE.utf8") == nullptr) {
    GTEST_SKIP() << "comma-decimal locale is not installed";
  }
  auto parsed = mqt::json::parse("1.25");
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  EXPECT_EQ(parsed->getAsNumber(), 1.25);
  EXPECT_EQ(mqt::json::serialize(*parsed), "1.25");
}
} // namespace
