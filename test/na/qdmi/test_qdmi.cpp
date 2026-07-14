/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/qdmi/Device.hpp"
#include "qdmi/DeviceManager.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <utility>

namespace na {
namespace {
// NOLINTNEXTLINE(misc-include-cleaner)
auto canonicallyOrderLatticeVectors(nlohmann::json& device) -> void {
  for (auto& lattice : device["traps"]) {
    const auto& v1 = lattice["latticeVector1"];
    const auto& v2 = lattice["latticeVector2"];
    if (v1["x"] > v2["x"] || (v1["x"] == v2["x"] && v1["y"] > v2["y"])) {
      std::swap(lattice["latticeVector1"], lattice["latticeVector2"]);
    }
  }
}

auto getDevice() -> qdmi::Device {
  ::qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides = {{
      .id = "mqt.na.default",
      .library = NA_DEVICE_LIBRARY,
      .prefix = "MQT_NA",
  }};
  auto device = ::qdmi::DeviceManager(options).open("mqt.na.default");
  auto converted = qdmi::Device::tryCreateFromDevice(device);
  if (!converted) {
    throw std::runtime_error("Built-in NA device is missing required metadata");
  }
  return *std::move(converted);
}
} // namespace
// ignore the linter warning regarding nlohmann::json and the compile time
// definitions
// NOLINTBEGIN(misc-include-cleaner)
TEST(TestNAQDMI, TrapsJSONRoundTrip) {
  nlohmann::json expectedDevice;
  // Open the file
  std::ifstream file(NA_DEVICE_JSON);
  ASSERT_TRUE(file.is_open()) << "Failed to open json file: " NA_DEVICE_JSON;
  // Parse the JSON file
  try {
    expectedDevice = nlohmann::json::parse(file);
  } catch (const nlohmann::json::parse_error& e) {
    GTEST_FAIL() << "JSON parsing error: " << e.what();
  }
  nlohmann::json actualDevice = getDevice();
  canonicallyOrderLatticeVectors(expectedDevice);
  canonicallyOrderLatticeVectors(actualDevice);
  EXPECT_EQ(expectedDevice["traps"], actualDevice["traps"]);
}
TEST(TestNAQDMI, FullJSONRoundTrip) {
  nlohmann::json jsonDevice;
  // Open the file
  std::ifstream file(NA_DEVICE_JSON);
  ASSERT_TRUE(file.is_open()) << "Failed to open json file: " NA_DEVICE_JSON;
  // Parse the JSON file
  try {
    jsonDevice = nlohmann::json::parse(file);
  } catch (const nlohmann::json::parse_error& e) {
    GTEST_FAIL() << "JSON parsing error: " << e.what();
  }
  nlohmann::json qdmiDevice = getDevice();
  canonicallyOrderLatticeVectors(jsonDevice);
  canonicallyOrderLatticeVectors(qdmiDevice);
  EXPECT_EQ(jsonDevice, qdmiDevice);
}
// NOLINTEND(misc-include-cleaner)
} // namespace na
