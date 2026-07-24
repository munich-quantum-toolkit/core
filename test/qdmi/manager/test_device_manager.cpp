/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Device.hpp"
#include "qdmi/DeviceManager.hpp"
#include "qdmi/DeviceRegistry.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp> // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json; // NOLINT(misc-include-cleaner)

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    path_ = std::filesystem::temp_directory_path() /
            ("mqt-core-qdmi-manager-test-" +
             std::to_string(std::random_device{}()));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

  void write(const std::filesystem::path& relative,
             const std::string& contents) const {
    std::filesystem::create_directories((path_ / relative).parent_path());
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
  options.inlineOverrides = Json::parse(R"({
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
  options.inlineOverrides = Json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [
      {"id": "duplicate", "library": "one", "prefix": "ONE"},
      {"id": "duplicate", "library": "two", "prefix": "TWO"}
    ]}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);

  options.inlineOverrides = Json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [
      {"id": "invalid", "library": "one", "prefix": "ONE", "typo": true}
    ]}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);
}

TEST(DeviceRegistry, DisabledOverrideMasksInheritedDefinition) {
  const qdmi::DeviceDefinition definition{
      .id = "masked", .library = "unused", .prefix = "UNUSED"};
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(definition);
  options.inlineOverrides = Json::parse(R"({
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

TEST(DeviceRegistry, DiscoversWindowsRuntimeLayout) {
  const TemporaryDirectory directory;
  directory.write("bin/runtime.qdmi.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "runtime", "library": "device.dll", "prefix": "RUNTIME"
    }]}
  })");
  directory.write("override.json", R"({
    "schema-version": 1,
    "qdmi": {}
  })");
  qdmi::ConfigOptions options;
  options.configRoot = directory.path();
  options.explicitFile = directory.path() / "override.json";

  const qdmi::DeviceRegistry registry(options);
  ASSERT_EQ(registry.definitions().size(), 1);
  EXPECT_EQ(registry.definitions().front().library,
            directory.path() / "bin" / "device.dll");
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

TEST(DeviceRegistry, DedicatedProjectFileWinsOverPyproject) {
  const TemporaryDirectory directory;
  directory.write("pyproject.toml", R"(
    [tool.mqt-core.qdmi]
    devices = [{ id = "toml", library = "toml.so", prefix = "TOML" }]
  )");
  directory.write("mqt-core.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [
      {"id": "json", "library": "json.so", "prefix": "JSON"}
    ]}
  })");
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.baseDirectory = directory.path();

  const qdmi::DeviceRegistry registry(options);
  ASSERT_EQ(registry.definitions().size(), 1);
  EXPECT_EQ(registry.definitions().front().id, "json");
}

TEST(DeviceRegistry, ReportsInvalidDocumentsAndSchemaPaths) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.inlineOverrides = Json::object();
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);

  options.inlineOverrides = Json::parse(R"({
    "schema-version": 2,
    "qdmi": {}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);

  options.inlineOverrides = Json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": {}}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);
}

TEST(DeviceRegistry, ValidatesDefinitionAndSessionTypes) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.inlineOverrides = Json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [{"id": 4}]}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);

  options.inlineOverrides = Json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "invalid", "library": "device", "prefix": "P",
      "enabled": "yes"
    }]}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);

  options.inlineOverrides = Json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "invalid", "library": "device", "prefix": "P",
      "session": {"token": 42}
    }]}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);
}

TEST(DeviceRegistry, RejectsIncompleteDefinitionsAndUnknownKeys) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.inlineOverrides = Json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [{"id": "missing", "prefix": "P"}]}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);

  options.inlineOverrides = Json::parse(R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "unknown", "library": "device", "prefix": "P", "unexpected": true
    }]}
  })");
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);
}

TEST(DeviceRegistry, ReportsInvalidExplicitJsonAndToml) {
  const TemporaryDirectory directory;
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.explicitFile = directory.path() / "missing.json";
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::runtime_error);

  directory.write("invalid.json", "{");
  options.explicitFile = directory.path() / "invalid.json";
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);

  directory.write("pyproject.toml", "[tool.mqt-core.qdmi\n");
  options.explicitFile.reset();
  options.baseDirectory = directory.path();
  EXPECT_THROW(static_cast<void>(qdmi::DeviceRegistry(options)),
               std::invalid_argument);
}

TEST(DeviceRegistry, RegistersReplacesAndUnregistersDefinitions) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.baseDirectory = std::filesystem::temp_directory_path();
  qdmi::DeviceRegistry registry(options);
  registry.registerDevice({.id = "example", .library = "one", .prefix = "ONE"});
  EXPECT_THROW(registry.registerDevice(
                   {.id = "example", .library = "two", .prefix = "TWO"}),
               std::invalid_argument);
  registry.registerDevice({.id = "example", .library = "two", .prefix = "TWO"},
                          true);
  EXPECT_EQ(registry.definitions().front().prefix, "TWO");
  registry.registerDevice(
      {.id = "example", .library = "two", .prefix = "TWO", .enabled = false},
      true);
  EXPECT_TRUE(registry.definitions().empty());
  registry.registerDevice({.id = "example", .library = "two", .prefix = "TWO"});
  EXPECT_TRUE(registry.unregisterDevice("example"));
  EXPECT_FALSE(registry.unregisterDevice("example"));
  EXPECT_THROW(registry.registerDevice({}), std::invalid_argument);
}

TEST(DeviceManager, LazilyOpensAndKeepsDeviceAlive) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "mqt.sc.test", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"});

  const auto device = qdmi::DeviceManager(options).open("mqt.sc.test");
  EXPECT_EQ(device.getName(), "MQT SC Default QDMI Device");
  EXPECT_FALSE(device.getSites().empty());
}

TEST(DeviceManager, OpensDefinitionsIndividually) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "good", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"});
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "bad", .library = "does-not-exist", .prefix = "MISSING"});

  qdmi::DeviceManager manager(options);
  EXPECT_EQ(manager.open("good").getName(), "MQT SC Default QDMI Device");
  EXPECT_THROW(static_cast<void>(manager.open("bad")), std::runtime_error);
  EXPECT_THROW(static_cast<void>(manager.open("missing")), std::out_of_range);
}

TEST(DeviceManager, OpensAllDefinitionsAndIsolatesFailures) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides = {
      {.id = "good", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"},
      {.id = "bad", .library = "does-not-exist", .prefix = "MISSING"},
  };
  const qdmi::DeviceManager manager(options);
  const auto result = manager.openAll();
  ASSERT_EQ(result.devices.size(), 1);
  EXPECT_EQ(result.devices.at("good").getName(), "MQT SC Default QDMI Device");
  ASSERT_EQ(result.errors.size(), 1);
  EXPECT_FALSE(result.errors.at("bad").empty());
}

TEST(DeviceManager, DefinitionsAreStableSnapshots) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "snapshot", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"});
  qdmi::DeviceManager manager(options);
  const auto snapshot = manager.definitions();
  EXPECT_TRUE(manager.unregisterDevice("snapshot"));
  ASSERT_EQ(snapshot.size(), 1);
  EXPECT_EQ(snapshot.front().id, "snapshot");
  EXPECT_TRUE(manager.definitions().empty());
}

TEST(DeviceManager, ConcurrentOpenCallsShareTheLibrarySafely) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "concurrent", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"});
  qdmi::DeviceManager manager(options);
  std::vector<std::future<std::string>> names;
  names.reserve(4);
  for (size_t i = 0; i < 4; ++i) {
    names.emplace_back(std::async(std::launch::async, [&manager] {
      return manager.open("concurrent").getName();
    }));
  }
  for (auto& name : names) {
    EXPECT_EQ(name.get(), "MQT SC Default QDMI Device");
  }
}

TEST(DeviceManager, SharesLibrariesButKeepsSessionsIndependent) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides = {
      {.id = "first", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"},
      {.id = "second", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"},
  };
  qdmi::DeviceManager manager(options);
  const auto first = manager.open("first");
  const auto second = manager.open("second");
  EXPECT_EQ(first.getName(), second.getName());
  EXPECT_EQ(first.getSites().size(), second.getSites().size());
}

TEST(DeviceManager, OpenDevicesOutliveRegistrationsAndManager) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "persistent", .library = SC_DEVICE_LIBRARY, .prefix = "MQT_SC"});
  const qdmi::Device device = [&options] {
    qdmi::DeviceManager manager(options);
    auto opened = manager.open("persistent");
    EXPECT_TRUE(manager.unregisterDevice("persistent"));
    return opened;
  }();
  EXPECT_EQ(device.getName(), "MQT SC Default QDMI Device");
}

TEST(DeviceManager, RejectsIncompleteSymbolSet) {
  qdmi::ConfigOptions options;
  options.isolated = true;
  options.runtimeOverrides.emplace_back(qdmi::DeviceDefinition{
      .id = "wrong-prefix", .library = SC_DEVICE_LIBRARY, .prefix = "MISSING"});
  qdmi::DeviceManager manager(options);
  EXPECT_THROW(static_cast<void>(manager.open("wrong-prefix")),
               std::runtime_error);
}
} // namespace
