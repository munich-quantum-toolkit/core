/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/fomac/Device.hpp"

#include "gtest/gtest.h"
#include <fstream>
#include <nlohmann/json.hpp>
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
} // namespace
TEST(TestNAFoMaC, TrapsJSONRoundTrip) {
  // NOLINTNEXTLINE(misc-include-cleaner)
  nlohmann::json fomacDevice;
  // Open the file
  // NOLINTNEXTLINE(misc-include-cleaner)
  std::ifstream file(NA_DEVICE_JSON);
  ASSERT_TRUE(file.is_open()) << "Failed to open json file: " NA_DEVICE_JSON;
  // Parse the JSON file
  try {
    // NOLINTNEXTLINE(misc-include-cleaner)
    fomacDevice = nlohmann::json::parse(file);
  } catch (const nlohmann::json::parse_error& e) {
    GTEST_FAIL() << "JSON parsing error: " << e.what();
  }
  nlohmann::json qdmiDevice = FoMaC::getDevices().front();
  canonicallyOrderLatticeVectors(fomacDevice);
  canonicallyOrderLatticeVectors(qdmiDevice);
  EXPECT_EQ(fomacDevice["traps"], qdmiDevice["traps"]);
}
TEST(TestNAFoMaC, FullJSONRoundTrip) {
  // NOLINTNEXTLINE(misc-include-cleaner)
  nlohmann::json jsonDevice;
  // Open the file
  // NOLINTNEXTLINE(misc-include-cleaner)
  std::ifstream file(NA_DEVICE_JSON);
  ASSERT_TRUE(file.is_open()) << "Failed to open json file: " NA_DEVICE_JSON;
  // Parse the JSON file
  try {
    // NOLINTNEXTLINE(misc-include-cleaner)
    jsonDevice = nlohmann::json::parse(file);
  } catch (const nlohmann::json::parse_error& e) {
    GTEST_FAIL() << "JSON parsing error: " << e.what();
  }
  nlohmann::json fomacDevice = FoMaC::getDevices().front();
  canonicallyOrderLatticeVectors(jsonDevice);
  canonicallyOrderLatticeVectors(fomacDevice);
  EXPECT_EQ(jsonDevice, fomacDevice);
}
} // namespace na
