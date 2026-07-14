/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/DeviceManager.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
class TemporaryDirectory {
public:
  TemporaryDirectory() {
    path_ =
        std::filesystem::temp_directory_path() / "mqt-core-qdmi-manager-test";
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

  void write(const std::filesystem::path& relative,
             const std::string& contents) const {
    std::ofstream output(path_ / relative);
    output << contents;
  }

private:
  std::filesystem::path path_;
};

TEST(DeviceRegistry, ParsesInlineConfigurationWithoutLoadingLibraries) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.baseDirectory = "/configuration";
  options.inlineOverrides = nlohmann::json::parse(R"({
    "schema-version": 1,
    "qdmi": {
      "devices": [{
        "id": "example.device",
        "library": "libexample.so",
        "prefix": "EXAMPLE",
        "session": {"auth-file": "secret.json", "custom1": "value"}
      }]
    }
  })");

  const qdmi::DeviceRegistry registry(options);
  ASSERT_EQ(registry.definitions().size(), 1);
  const auto& definition = registry.definitions().front();
  EXPECT_EQ(definition.id, "example.device");
  EXPECT_EQ(definition.library,
            std::filesystem::path("/configuration/libexample.so"));
  ASSERT_TRUE(definition.session.authFile.has_value());
  EXPECT_EQ(*definition.session.authFile,
            std::filesystem::path("/configuration/secret.json"));
  EXPECT_EQ(definition.session.custom1, "value");
}

TEST(DeviceRegistry, RejectsDuplicateIdsAndUnknownKeys) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.inlineOverrides = nlohmann::json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [
      {"id": "duplicate", "library": "one", "prefix": "ONE"},
      {"id": "duplicate", "library": "two", "prefix": "TWO"}
    ]}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);

  options.inlineOverrides = nlohmann::json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [
      {"id": "invalid", "library": "one", "prefix": "ONE", "typo": true}
    ]}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);
}

TEST(DeviceRegistry, DisabledOverrideMasksInheritedDefinition) {
  qdmi::DeviceDefinition definition{
      .id = "masked", .library = "unused", .prefix = "UNUSED"};
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(definition);
  options.inlineOverrides = nlohmann::json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [{"id": "masked", "enabled": false}]}
  })");

  // Runtime overrides have the highest precedence and deliberately re-enable
  // complete definitions.
  EXPECT_EQ(qdmi::DeviceRegistry(options).definitions().size(), 1);

  options.runtimeOverrides.clear();
  EXPECT_TRUE(qdmi::DeviceRegistry(options).definitions().empty());
}

TEST(DeviceRegistry, MergesExplicitConfigurationOverBuiltInFragments) {
  const TemporaryDirectory directory;
  directory.write("builtin.qdmi.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "merged", "library": "device.so", "prefix": "DEVICE",
      "session": {"base-url": "https://default.invalid", "custom1": "one"}
    }]}
  })");
  directory.write("override.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "merged", "session": {"custom1": "two"}
    }]}
  })");
  qdmi::ConfigOptions options;
  options.configRoot = directory.path();
  options.explicitFile = directory.path() / "override.json";

  const qdmi::DeviceRegistry registry(options);
  const auto& definitions = registry.definitions();
  ASSERT_EQ(definitions.size(), 1);
  EXPECT_EQ(definitions.front().library, directory.path() / "device.so");
  EXPECT_EQ(definitions.front().session.baseUrl, "https://default.invalid");
  EXPECT_EQ(definitions.front().session.custom1, "two");
}

TEST(DeviceRegistry, ReadsProjectConfigurationFromPyprojectToml) {
  const TemporaryDirectory directory;
  directory.write("pyproject.toml", R"(
    [tool.mqt-core.qdmi]
    devices = [
      { id = "toml", library = "device.so", prefix = "TOML" }
    ]
  )");
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.baseDirectory = directory.path();

  const qdmi::DeviceRegistry registry(options);
  const auto& definitions = registry.definitions();
  ASSERT_EQ(definitions.size(), 1);
  EXPECT_EQ(definitions.front().id, "toml");
  EXPECT_EQ(definitions.front().library, directory.path() / "device.so");
}

TEST(DeviceManager, LazilyOpensAndKeepsDeviceAlive) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "mqt.sc.test", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"});

  auto device = qdmi::DeviceManager(options).open("mqt.sc.test");
  EXPECT_EQ(device.getName(), "MQT SC Default QDMI Device");
  EXPECT_FALSE(device.getSites().empty());
}

TEST(DeviceManager, OpenAllIsolatesFailures) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "good", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"});
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "bad", .library = "does-not-exist", .prefix = "MISSING"});

  qdmi::DeviceManager manager(options);
  auto result = manager.openAll();
  EXPECT_TRUE(result.devices.contains("good"));
  EXPECT_TRUE(result.errors.contains("bad"));
}
} // namespace
