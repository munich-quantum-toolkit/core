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

#include <cstddef>
#include <fstream>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace na {

TEST(GeneratorTest, WriteJSONSchema) {
  std::ostringstream os;
  EXPECT_NO_THROW(writeJSONSchema(os));
  nlohmann::json json;
  EXPECT_NO_THROW(json = nlohmann::json::parse(os.str()));
  EXPECT_TRUE(json.is_object());
  EXPECT_GT(json.size(), 0);
}

} // namespace na
