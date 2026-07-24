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
#include "qdmi/driver/Driver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

class ScopedEnvironmentVariable {
public:
  ScopedEnvironmentVariable(std::string name, const std::string& value)
      : name_(std::move(name)) {
    if (const auto* previous = std::getenv(name_.c_str());
        previous != nullptr) {
      previous_ = previous;
    }
    set(value);
  }

  ~ScopedEnvironmentVariable() {
    if (previous_) {
      static_cast<void>(setWithoutChecking(*previous_));
    } else {
#ifdef _WIN32
      static_cast<void>(_putenv_s(name_.c_str(), ""));
#else
      // NOLINTNEXTLINE(misc-include-cleaner)
      static_cast<void>(unsetenv(name_.c_str()));
#endif
    }
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable&
  operator=(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable(ScopedEnvironmentVariable&&) = delete;
  ScopedEnvironmentVariable& operator=(ScopedEnvironmentVariable&&) = delete;

private:
  void set(const std::string& value) const {
    if (!setWithoutChecking(value)) {
      throw std::runtime_error("Failed to set environment variable " + name_);
    }
  }

  [[nodiscard]] bool setWithoutChecking(const std::string& value) const {
#ifdef _WIN32
    return _putenv_s(name_.c_str(), value.c_str()) == 0;
#else
    // NOLINTNEXTLINE(misc-include-cleaner)
    return setenv(name_.c_str(), value.c_str(), 1) == 0;
#endif
  }

  std::string name_;
  std::optional<std::string> previous_;
};

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
  ASSERT_EQ(registry.definitions().size(), 3);
  EXPECT_EQ(registry.definitions().at(0).id, "mqt.ddsim.default");
  EXPECT_EQ(registry.definitions().at(1).id, "mqt.na.default");
  EXPECT_EQ(registry.definitions().at(2).id, "mqt.sc.default");
  for (const auto& definition : registry.definitions()) {
    EXPECT_TRUE(std::filesystem::is_regular_file(definition.library));
  }
}

TEST(DeviceRegistry, ReadsProjectConfigurationFromPyprojectToml) {
  const TemporaryDirectory directory;
  directory.write("pyproject.toml", R"(
    [tool.qdmi]
    devices = [{ id = "toml", library = "device.so", prefix = "TOML" }]
  )");
  const ScopedCurrentPath currentPath(directory.path());
  const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE", "");
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", "");
#ifdef _WIN32
  const ScopedEnvironmentVariable userConfig("APPDATA",
                                             directory.path().string());
#else
  const ScopedEnvironmentVariable userConfig("XDG_CONFIG_HOME",
                                             directory.path().string());
#endif

  const qdmi::detail::DeviceRegistry registry;
  const auto* definition = findDefinition(registry, "toml");
  ASSERT_NE(definition, nullptr);
  EXPECT_EQ(std::filesystem::weakly_canonical(definition->library),
            std::filesystem::weakly_canonical(directory.path()) / "device.so");
}

TEST(DeviceRegistry, DedicatedProjectFileWinsOverPyproject) {
  const TemporaryDirectory directory;
  directory.write("pyproject.toml", R"(
    [tool.qdmi]
    devices = [{ id = "toml", library = "toml.so", prefix = "TOML" }]
  )");
  directory.write("qdmi.json", R"({
    "schema-version": 1,
    "qdmi": {"devices": [
      {"id": "json", "library": "json.so", "prefix": "JSON"}
    ]}
  })");
  const ScopedCurrentPath currentPath(directory.path());
  const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE", "");
  const ScopedEnvironmentVariable configJson("MQT_CORE_QDMI_CONFIG_JSON", "");

  const qdmi::detail::DeviceRegistry registry;
  EXPECT_NE(findDefinition(registry, "json"), nullptr);
  EXPECT_EQ(findDefinition(registry, "toml"), nullptr);
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

TEST(DeviceRegistry, ReportsInvalidExplicitJsonAndToml) {
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
  {
    directory.write("pyproject.toml", "[tool.qdmi\n");
    const ScopedCurrentPath currentPath(directory.path());
    const ScopedEnvironmentVariable configFile("MQT_CORE_QDMI_CONFIG_FILE", "");
    EXPECT_THROW(static_cast<void>(qdmi::detail::DeviceRegistry()),
                 std::invalid_argument);
  }
}

} // namespace
