/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/sc/Generator.hpp"

#include <gtest/gtest.h>
#include <sstream>
// clang-tidy wants to include the forward header, but we need the full
// NOLINTNEXTLINE(misc-include-cleaner)
#include <nlohmann/json.hpp>

namespace sc {
namespace {
// clang-tidy wants to include the forward header, but we have the full
// NOLINTNEXTLINE(misc-include-cleaner)
auto testPopulation(const nlohmann::json& json) -> void {
  for (const auto& [key, value] : json.items()) {
    if (value.is_array()) {
      // Array field should have at least one default entry
      EXPECT_GT(value.size(), 0) << "Array field '" << key
                                 << "' should have at least one default entry";
      for (const auto& item : value) {
        // Each entry in the array should not be null
        EXPECT_FALSE(item.is_null())
            << "Array field '" << key << "' should not have null entries";
        testPopulation(item);
      }
    } else if (value.is_object()) {
      testPopulation(value);
    }
  }
}
} // namespace

TEST(ScGeneratorTest, WriteJSONSchema) {
  std::ostringstream os;
  EXPECT_NO_THROW(writeJSONSchema(os));
  // clang-tidy wants to include the forward header, but we have the full
  // NOLINTNEXTLINE(misc-include-cleaner)
  nlohmann::json json;
  EXPECT_NO_THROW(json = nlohmann::json::parse(os.str()));
  EXPECT_TRUE(json.is_object());
  EXPECT_GT(json.size(), 0);
  testPopulation(json);
}

} // namespace sc
