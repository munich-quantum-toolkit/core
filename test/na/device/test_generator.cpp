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
#include "na/device/device.pb.h"

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
// clang-tidy wants to include the forward header, but we need the full
// NOLINTNEXTLINE(misc-include-cleaner)
#include <nlohmann/json.hpp>
#include <stdexcept>

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

TEST(GeneratorTest, TimeUnitNanosecond) {
  std::istringstream is(R"({
  "timeUnit": {
    "value": 5,
    "unit": "ns"
  }
})");
  Device device;
  ASSERT_NO_THROW(device = readJSON(is));
  EXPECT_EQ(device.time_unit().value(), 5);
  EXPECT_EQ(device.time_unit().unit(), "ns");
}

TEST(GeneratorTest, TimeUnitInvalid) {
  std::istringstream is(R"({
  "timeUnit": {
    "value": 1,
    "unit": "ts"
  }
})");
  Device device;
  ASSERT_NO_THROW(device = readJSON(is));
  EXPECT_THROW(writeHeader(device, std::cout), std::runtime_error);
}

TEST(GeneratorTest, LengthUnitNanometer) {
  std::istringstream is(R"({
  "lengthUnit": {
    "value": 5,
    "unit": "nm"
  }
})");
  Device device;
  ASSERT_NO_THROW(device = readJSON(is));
  EXPECT_EQ(device.time_unit().value(), 5);
  EXPECT_EQ(device.time_unit().unit(), "nm");
}

TEST(GeneratorTest, LengthUnitInvalid) {
  std::istringstream is(R"({
  "lengthUnit": {
    "value": 1,
    "unit": "tm"
  }
})");
  Device device;
  ASSERT_NO_THROW(device = readJSON(is));
  EXPECT_THROW(writeHeader(device, std::cout), std::runtime_error);
}

} // namespace na
