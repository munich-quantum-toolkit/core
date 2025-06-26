/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/device/Generator.hpp"

#include <gtest/gtest.h>
#include <sstream>
// clang-tidy wants to include the forward header, but we need the full
// NOLINTNEXTLINE(misc-include-cleaner)
#include <nlohmann/json.hpp>

namespace na {

TEST(GeneratorTest, WriteJSONSchema) {
  std::ostringstream os;
  EXPECT_NO_THROW(writeJSONSchema(os));
  // clang-tidy wants to include the forward header, but we have the full
  // NOLINTNEXTLINE(misc-include-cleaner)
  nlohmann::json json;
  EXPECT_NO_THROW(json = nlohmann::json::parse(os.str()));
  EXPECT_TRUE(json.is_object());
  EXPECT_GT(json.size(), 0);
}

} // namespace na
