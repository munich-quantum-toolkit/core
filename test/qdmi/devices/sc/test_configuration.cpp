/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/devices/sc/Configuration.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp> // NOLINT(misc-include-cleaner)
#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sc {
namespace {
using Json = nlohmann::json;

[[nodiscard]] Json bundledJson() {
  std::ifstream input(SC_DEVICE_JSON);
  if (!input) {
    throw std::runtime_error("failed to open bundled SC configuration");
  }
  return Json::parse(input);
}

void expectInvalid(const std::function<void(Json&)>& mutate,
                   const std::string& diagnostic) {
  auto value = bundledJson();
  mutate(value);
  try {
    static_cast<void>(readJSON(value.dump(), "test-source"));
    FAIL() << "Expected invalid SC configuration";
  } catch (const std::invalid_argument& error) {
    EXPECT_THAT(error.what(), testing::HasSubstr("test-source:$"));
    EXPECT_THAT(error.what(), testing::HasSubstr(diagnostic));
  }
}
} // namespace

TEST(ScConfigurationTest, ParsesBundledDeviceStrictly) {
  const auto device = readJSON(SC_DEVICE_JSON);
  EXPECT_EQ(device.schemaVersion, 1);
  EXPECT_EQ(device.name, "MQT SC Default QDMI Device");
  EXPECT_EQ(device.numQubits, 100);
  EXPECT_EQ(device.operations.size(), 3);
  EXPECT_EQ(device.operations[0].name, "r");
  EXPECT_EQ(device.operations[1].name, "cz");
  EXPECT_EQ(device.operations[2].name, "measure");
  ASSERT_EQ(device.couplings.size(), 180);
  EXPECT_EQ(device.couplings.front(), (std::pair<uint64_t, uint64_t>{0, 1}));
  EXPECT_EQ(device.couplings.back(), (std::pair<uint64_t, uint64_t>{89, 99}));
}

TEST(ScConfigurationTest, RejectsTopLevelSchemaErrors) {
  expectInvalid([](auto& root) { root["unknown"] = true; }, "unknown");
  expectInvalid([](auto& root) { root.erase("qubitProperties"); },
                "qubitProperties");
  expectInvalid([](auto& root) { root["schema-version"] = 2; },
                "schema-version");
  expectInvalid([](auto& root) { root["name"] = ""; }, "name");
  expectInvalid([](auto& root) { root["numQubits"] = 0; }, "numQubits");
  expectInvalid([](auto& root) { root["numQubits"] = "100"; }, "numQubits");
  expectInvalid(
      [](auto& root) {
        root["numQubits"] = std::numeric_limits<uint64_t>::max();
      },
      "numQubits");
}

TEST(ScConfigurationTest, RejectsInvalidUnitsAndCalibration) {
  expectInvalid([](auto& root) { root["durationUnit"]["unit"] = "fortnight"; },
                "durationUnit");
  expectInvalid([](auto& root) { root["durationUnit"]["scaleFactor"] = 0.; },
                "durationUnit");
  expectInvalid(
      [](auto& root) {
        root["qubitProperties"]["overrides"].push_back(
            {{"qubit", 100}, {"t1", 1}});
      },
      "qubitProperties/overrides");
  expectInvalid(
      [](auto& root) {
        root["qubitProperties"]["overrides"].push_back(
            {{"qubit", 8}, {"t1", 0}});
      },
      "qubitProperties/overrides");
}

TEST(ScConfigurationTest, RejectsInvalidOrderedTopology) {
  expectInvalid(
      [](auto& root) { root["couplings"].push_back(root["couplings"][0]); },
      "couplings");
  expectInvalid([](auto& root) { root["couplings"][0] = {0, 0}; }, "couplings");
  expectInvalid([](auto& root) { root["couplings"][0] = {0, 100}; },
                "couplings");

  auto root = bundledJson();
  root["couplings"] = {{1, 0}, {0, 1}};
  root["operations"][1]["siteOverrides"] = Json::array();
  const auto parsed = readJSON(root.dump(), "ordered");
  EXPECT_EQ(parsed.couplings[0], (std::pair<uint64_t, uint64_t>{1, 0}));
  EXPECT_EQ(parsed.couplings[1], (std::pair<uint64_t, uint64_t>{0, 1}));
}

TEST(ScConfigurationTest, ValidatesOperationSupportAndCalibration) {
  expectInvalid(
      [](auto& root) {
        root["operations"].push_back(root["operations"].front());
      },
      "operations");
  expectInvalid(
      [](auto& root) {
        root["operations"][0]["numQubits"] = 3;
        root["operations"][0].erase("sites");
      },
      "sites");
  expectInvalid(
      [](auto& root) {
        root["operations"][1]["sites"] = {{0, 2}};
        root["operations"][1]["siteOverrides"] = Json::array();
      },
      "device connectivity");
  expectInvalid([](auto& root) { root["operations"][0]["fidelity"] = 1.1; },
                "fidelity");
  expectInvalid(
      [](auto& root) {
        root["operations"][1]["siteOverrides"][0]["sites"] = {1, 0};
      },
      "siteOverrides");
  expectInvalid(
      [](auto& root) {
        root["operations"][1]["siteOverrides"][0].erase("duration");
        root["operations"][1]["siteOverrides"][0].erase("fidelity");
      },
      "siteOverrides");
}

TEST(ScConfigurationTest, AcceptsZeroDurationAndRejectsOverflowingJsonNumber) {
  auto root = bundledJson();
  root["operations"][0]["duration"] = 0;
  root["operations"][1]["siteOverrides"][0]["duration"] = 0;
  EXPECT_NO_THROW(static_cast<void>(readJSON(root.dump(), "zero-duration")));
  EXPECT_THROW(
      static_cast<void>(readJSON(
          R"({"schema-version":1e400,"name":"x","numQubits":1,"durationUnit":{"unit":"ns","scaleFactor":1},"qubitProperties":{"defaults":{},"overrides":[]},"couplings":[],"operations":[]})",
          "overflow")),
      std::invalid_argument);
}

TEST(ScConfigurationTest, AcceptsExplicitHigherAritySites) {
  auto root = bundledJson();
  root["numQubits"] = 3;
  root["couplings"] = Json::array();
  root["qubitProperties"]["overrides"] = Json::array();
  root["operations"] = Json::array(
      {{{"name", "three"},
        {"numParameters", 1},
        {"numQubits", 3},
        {"sites", {{2, 0, 1}}},
        {"duration", 0},
        {"siteOverrides", {{{"sites", {2, 0, 1}}, {"fidelity", 0.75}}}}}});
  const auto parsed = readJSON(root.dump(), "higher-arity");
  ASSERT_EQ(parsed.operations.size(), 1);
  ASSERT_TRUE(parsed.operations[0].sites.has_value());
  EXPECT_EQ(parsed.operations[0].sites->front(),
            (std::vector<uint64_t>{2, 0, 1}));
}

} // namespace sc
