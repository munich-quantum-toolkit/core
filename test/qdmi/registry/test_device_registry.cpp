/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "DeviceRegistry.hpp"
#include "qdmi/TestUtils.hpp"
#include "qdmi/driver/Driver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    path_ = std::filesystem::temp_directory_path() /
            ("mqt-core-qdmi-registry-test-" +
             std::to_string(std::random_device{}()));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

  [[nodiscard]] std::filesystem::path
  write(const std::filesystem::path& relative,
        const std::string& contents) const {
    const auto path = path_ / relative;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << contents;
    return path;
  }

private:
  std::filesystem::path path_;
};

using mqt::test::ScopedEnvironmentVariable;

class ScopedCurrentPath {
public:
  explicit ScopedCurrentPath(const std::filesystem::path& path)
      : previous_(std::filesystem::current_path()) {
    std::filesystem::current_path(path);
  }
  ~ScopedCurrentPath() { std::filesystem::current_path(previous_); }

  ScopedCurrentPath(const ScopedCurrentPath&) = delete;
  ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;
  ScopedCurrentPath(ScopedCurrentPath&&) = delete;
  ScopedCurrentPath& operator=(ScopedCurrentPath&&) = delete;

private:
  std::filesystem::path previous_;
};

[[nodiscard]] auto findDefinition(const qdmi::detail::DeviceRegistry& registry,
                                  const std::string_view id)
    -> const qdmi::DeviceDefinition* {
  const auto& definitions = registry.definitions();
  const auto found =
      std::ranges::find(definitions, id, &qdmi::DeviceDefinition::id);
  return found == definitions.end() ? nullptr : &*found;
}

[[nodiscard]] auto emptyConfig(const TemporaryDirectory& directory)
    -> ScopedEnvironmentVariable {
  const auto path =
      directory.write("empty.json", R"({"schema-version": 1, "qdmi": {}})");
  return {"MQT_CORE_QDMI_CONFIG_FILE", path.string()};
}

TEST(DeviceRegistry, ParsesEnvironmentConfigurationWithoutLoadingLibraries) {
  const TemporaryDirectory directory;
  const ScopedCurrentPath currentPath(directory.path());
  const auto configFile = emptyConfig(directory);
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "example.device", "library": "libexample.so", "prefix": "EXAMPLE",
      "session": {"auth-file": "secret.json", "custom1": "value"}
    }]}
  })");

  const qdmi::detail::DeviceRegistry registry;
  const auto* definition = findDefinition(registry, "example.device");
  ASSERT_NE(definition, nullptr);
  EXPECT_EQ(std::filesystem::weakly_canonical(definition->library),
            std::filesystem::weakly_canonical(directory.path()) /
                "libexample.so");
  ASSERT_TRUE(definition->session.authFile.has_value());
  EXPECT_EQ(std::filesystem::weakly_canonical(*definition->session.authFile),
            std::filesystem::weakly_canonical(directory.path()) /
                "secret.json");
  EXPECT_EQ(definition->session.custom1, "value");
}

TEST(DeviceRegistry, RejectsDuplicateIdsAndUnsupportedKeys) {
  const TemporaryDirectory directory;
  const auto configFile = emptyConfig(directory);
  {
    const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", R"({
      "schema-version": 1,
      "qdmi": {"devices": [
        {"id": "duplicate", "library": "one", "prefix": "ONE"},
        {"id": "duplicate", "library": "two", "prefix": "TWO"}
      ]}
    })");
    EXPECT_THROW(static_cast<void>(qdmi::detail::DeviceRegistry()),
                 std::invalid_argument);
  }
  {
    const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", R"({
      "schema-version": 1,
      "qdmi": {"device-config": {"model": "unused"}}
    })");
    EXPECT_THROW(static_cast<void>(qdmi::detail::DeviceRegistry()),
                 std::invalid_argument);
  }
}

TEST(DeviceRegistry, MergesEnvironmentJsonOverExplicitFile) {
  const TemporaryDirectory directory;
  const auto path = directory.write("environment.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "environment", "library": "file.so", "prefix": "FILE",
      "session": {"custom1": "from-file", "custom2": "preserved"}
    }]}
  })");
  const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE",
                                             path.string());
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "environment", "session": {"custom1": "from-json"}
    }]}
  })");

  const qdmi::detail::DeviceRegistry registry;
  const auto* definition = findDefinition(registry, "environment");
  ASSERT_NE(definition, nullptr);
  EXPECT_EQ(definition->library, directory.path() / "file.so");
  EXPECT_EQ(definition->prefix, "FILE");
  EXPECT_EQ(definition->session.custom1, "from-json");
  EXPECT_EQ(definition->session.custom2, "preserved");
}

TEST(DeviceRegistry, DeviceConfigurationSourceReplacesAtomically) {
  const TemporaryDirectory directory;
  const ScopedCurrentPath currentPath(directory.path());
  const auto path = directory.write("environment.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "environment", "library": "file.so", "prefix": "FILE",
      "session": {"device-config": {"inline": {
        "schema-version": 1, "name": "inline"
      }}}
    }]}
  })");
  const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE",
                                             path.string());
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "environment",
      "session": {"device-config": {"file": "overrides/device.json"}}
    }]}
  })");

  const qdmi::detail::DeviceRegistry registry;
  const auto* definition = findDefinition(registry, "environment");
  ASSERT_NE(definition, nullptr);
  ASSERT_TRUE(definition->session.deviceConfiguration);
  const auto* file = std::get_if<qdmi::FileDeviceConfiguration>(
      &*definition->session.deviceConfiguration);
  ASSERT_NE(file, nullptr);
  EXPECT_EQ(std::filesystem::weakly_canonical(file->path),
            std::filesystem::weakly_canonical(directory.path()) /
                "overrides/device.json");
}

TEST(DeviceRegistry, SerializesInlineDeviceConfigurationCompactly) {
  const TemporaryDirectory directory;
  const auto configFile = emptyConfig(directory);
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "inline", "library": "file.so", "prefix": "FILE",
      "session": {"device-config": {"inline": {
        "schema-version": 1, "name": "inline"
      }}}
    }]}
  })");

  const qdmi::detail::DeviceRegistry registry;
  const auto* definition = findDefinition(registry, "inline");
  ASSERT_NE(definition, nullptr);
  ASSERT_TRUE(definition->session.deviceConfiguration);
  const auto* inlineConfig = std::get_if<qdmi::InlineDeviceConfiguration>(
      &*definition->session.deviceConfiguration);
  ASSERT_NE(inlineConfig, nullptr);
  EXPECT_EQ(inlineConfig->json, R"({"name":"inline","schema-version":1})");
}

TEST(DeviceRegistry, RejectsInvalidDeviceConfigurationSources) {
  const TemporaryDirectory directory;
  const auto configFile = emptyConfig(directory);
  const std::vector<std::string_view> invalidSources{
      R"({})",
      R"({"unknown": true})",
      R"({"inline": []})",
      R"({"file": ""})",
      R"({"file": 1})",
      R"({"inline": {"schema-version": 1}, "file": "device.json"})",
  };
  for (const auto source : invalidSources) {
    SCOPED_TRACE(source);
    const auto json =
        R"({"schema-version":1,"qdmi":{"devices":[{"id":"invalid",)"
        R"("library":"file.so","prefix":"FILE","session":{"device-config":)" +
        std::string(source) + "}}]}}";
    const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON",
                                               json);
    EXPECT_THROW(static_cast<void>(qdmi::detail::DeviceRegistry()),
                 std::invalid_argument);
  }
}

TEST(DeviceRegistry, DisabledEnvironmentEntryMasksExplicitDefinition) {
  const TemporaryDirectory directory;
  const auto path = directory.write("complete.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [
      {"id": "masked", "library": "device.so", "prefix": "DEVICE"}
    ]}
  })");
  const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE",
                                             path.string());
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{"id": "masked", "enabled": false}]}
  })");

  const qdmi::detail::DeviceRegistry registry;
  EXPECT_EQ(findDefinition(registry, "masked"), nullptr);
  ASSERT_EQ(registry.disabledIds().size(), 1);
  EXPECT_EQ(registry.disabledIds().front(), "masked");
}

TEST(DeviceRegistry, HigherPrecedenceDefinitionMustExplicitlyReenableDevice) {
  const TemporaryDirectory directory;
  const auto path = directory.write("disabled.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{"id": "masked", "enabled": false}]}
  })");
  const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE",
                                             path.string());

  {
    const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", R"({
          "schema-version": 1,
          "qdmi": {"devices": [{
            "id": "masked", "library": "device.so", "prefix": "DEVICE"
          }]}
        })");
    const qdmi::detail::DeviceRegistry registry;
    EXPECT_EQ(findDefinition(registry, "masked"), nullptr);
    ASSERT_EQ(registry.disabledIds().size(), 1);
    EXPECT_EQ(registry.disabledIds().front(), "masked");
  }

  {
    const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", R"({
          "schema-version": 1,
          "qdmi": {"devices": [{
            "id": "masked", "library": "device.so", "prefix": "DEVICE",
            "enabled": true
          }]}
        })");
    const qdmi::detail::DeviceRegistry registry;
    const auto* definition = findDefinition(registry, "masked");
    ASSERT_NE(definition, nullptr);
    EXPECT_EQ(definition->library,
              std::filesystem::current_path() / "device.so");
    EXPECT_EQ(definition->prefix, "DEVICE");
    EXPECT_TRUE(registry.disabledIds().empty());
  }
}

TEST(DeviceRegistry, ResolvesRelativeConfigurationPathsBeforeCwdChanges) {
  const TemporaryDirectory directory;
  directory.write("config/device.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "relative", "library": "libdevice.so", "prefix": "RELATIVE",
      "session": {"auth-file": "auth.json"}
    }]}
  })");

  std::filesystem::path library;
  std::filesystem::path authFile;
  {
    const ScopedCurrentPath currentPath(directory.path());
    const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE",
                                               "config/device.json");
    const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", "");
    const qdmi::detail::DeviceRegistry registry;
    const auto* definition = findDefinition(registry, "relative");
    ASSERT_NE(definition, nullptr);
    library = definition->library;
    ASSERT_TRUE(definition->session.authFile.has_value());
    authFile = *definition->session.authFile;
  }

  EXPECT_TRUE(library.is_absolute());
  EXPECT_TRUE(authFile.is_absolute());
  EXPECT_EQ(std::filesystem::weakly_canonical(library),
            std::filesystem::weakly_canonical(directory.path()) / "config" /
                "libdevice.so");
  EXPECT_EQ(std::filesystem::weakly_canonical(authFile),
            std::filesystem::weakly_canonical(directory.path()) / "config" /
                "auth.json");
}

TEST(DeviceRegistry, DiscoversGeneratedBuildTreeManifests) {
  const TemporaryDirectory directory;
  const auto configFile = emptyConfig(directory);
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", "");

  const qdmi::detail::DeviceRegistry registry;
  ASSERT_EQ(registry.definitions().size(), 5);
  EXPECT_EQ(registry.definitions().at(0).id, "mqt.ddsim.default");
  EXPECT_EQ(registry.definitions().at(1).id, "mqt.na.default");
  EXPECT_EQ(registry.definitions().at(2).id, "mqt.sc.default");
  EXPECT_EQ(registry.definitions().at(3).id, "mqt.sc.iqm.emerald");
  EXPECT_EQ(registry.definitions().at(4).id, "mqt.sc.iqm.garnet");
  for (const auto& definition : registry.definitions()) {
    EXPECT_TRUE(std::filesystem::is_regular_file(definition.library));
  }

  const auto assertPackagedModel = [&](const std::string_view id,
                                       const std::string_view filename) {
    const auto* definition = findDefinition(registry, id);
    ASSERT_NE(definition, nullptr);
    ASSERT_TRUE(definition->session.deviceConfiguration);
    const auto* file = std::get_if<qdmi::FileDeviceConfiguration>(
        &*definition->session.deviceConfiguration);
    ASSERT_NE(file, nullptr);
    EXPECT_EQ(file->path.filename(), filename);
    EXPECT_EQ(file->path.parent_path(), definition->library.parent_path());
    EXPECT_TRUE(std::filesystem::is_regular_file(file->path));
  };
  assertPackagedModel("mqt.sc.iqm.garnet", "iqm-garnet.json");
  assertPackagedModel("mqt.sc.iqm.emerald", "iqm-emerald.json");
}

TEST(DeviceRegistry, ReadsProjectConfigurationFromNearestQdmiJson) {
  const TemporaryDirectory directory;
  directory.write("qdmi.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [
      {"id": "json", "library": "device.so", "prefix": "JSON"}
    ]}
  })");
  const ScopedCurrentPath currentPath(directory.path());
  const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE", "");
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", "");

  const qdmi::detail::DeviceRegistry registry;
  const auto* definition = findDefinition(registry, "json");
  ASSERT_NE(definition, nullptr);
  EXPECT_EQ(std::filesystem::weakly_canonical(definition->library),
            std::filesystem::weakly_canonical(directory.path()) / "device.so");
}

TEST(DeviceRegistry, MergesProjectConfigurationOverUserConfiguration) {
  const TemporaryDirectory directory;
  directory.write("user/mqt-core/qdmi.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{
      "id": "layered", "library": "user.so", "prefix": "USER",
      "session": {"custom1": "user-default"}
    }]}
  })");
  directory.write("project/qdmi.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [{"id": "layered", "prefix": "PROJECT"}]}
  })");
  const ScopedCurrentPath currentPath(directory.path() / "project");
  const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE", "");
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", "");
#ifdef _WIN32
  const ScopedEnvironmentVariable programData("PROGRAMDATA",
                                              directory.path().string());
  const ScopedEnvironmentVariable userConfig(
      "APPDATA", (directory.path() / "user").string());
#else
  const ScopedEnvironmentVariable userConfig(
      "XDG_CONFIG_HOME", (directory.path() / "user").string());
#endif

  const qdmi::detail::DeviceRegistry registry;
  const auto* definition = findDefinition(registry, "layered");
  ASSERT_NE(definition, nullptr);
  EXPECT_EQ(definition->library,
            directory.path() / "user" / "mqt-core" / "user.so");
  EXPECT_EQ(definition->prefix, "PROJECT");
  EXPECT_EQ(definition->session.custom1, "user-default");
}

TEST(DeviceRegistry, ReportsInvalidDocumentsAndDefinitionTypes) {
  const TemporaryDirectory directory;
  const auto configFile = emptyConfig(directory);
  for (
      const auto* document : {
          R"({})",
          R"({"schema-version": 2, "qdmi": {}})",
          R"({"schema-version": 1, "qdmi": {"devices": {}}})",
          R"({"schema-version": 1, "qdmi": {"devices": [{"id": 4}]}})",
          R"({"schema-version": 1, "qdmi": {"devices": [{"id": "invalid", "library": "device", "prefix": "P", "enabled": "yes"}]}})",
          R"({"schema-version": 1, "qdmi": {"devices": [{"id": "invalid", "library": "device", "prefix": "P", "session": {"token": 42}}]}})",
          R"({"schema-version": 1, "qdmi": {"devices": [{"id": "missing", "prefix": "P"}]}})",
          R"({"schema-version": 1, "qdmi": {"devices": [{"id": "unknown", "library": "device", "prefix": "P", "unexpected": true}]}})",
      }) {
    const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON",
                                               document);
    EXPECT_THROW(static_cast<void>(qdmi::detail::DeviceRegistry()),
                 std::invalid_argument);
  }
}

TEST(DeviceRegistry, ReportsInvalidExplicitJson) {
  const TemporaryDirectory directory;
  {
    const ScopedEnvironmentVariable configFile(
        "MQT_CORE_QDMI_CONFIG_FILE",
        (directory.path() / "missing.json").string());
    EXPECT_THROW(static_cast<void>(qdmi::detail::DeviceRegistry()),
                 std::runtime_error);
  }
  {
    const auto invalid = directory.write("invalid.json", "{");
    const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE",
                                               invalid.string());
    EXPECT_THROW(static_cast<void>(qdmi::detail::DeviceRegistry()),
                 std::invalid_argument);
  }
}

} // namespace
