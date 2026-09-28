/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Client.hpp"

#include "TestUtils.hpp"

/// Required for JSON construction inside GoogleTest macro expansions.
#include "nlohmann/json.hpp" /// NOLINT(misc-include-cleaner)
#include "nlohmann/json_fwd.hpp"
#include "qdmi/constants.h"

#include <filesystem>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdmi {
namespace {
using mqt::test::ScopedEnvironmentVariable;

TEST(ClientRuntimeTest, ValidatesDriversAndRetainsSessions) {
  const ScopedEnvironmentVariable config{"QDMI_CONF",
                                         MQT_CORE_QDMI_EXAMPLE_CONFIG};
  const ScopedEnvironmentVariable driver{"MQT_CORE_QDMI_DRIVER",
                                         "missing-driver"};
  EXPECT_THAT([] { return Session{}; },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("Cannot load QDMI driver")));
  EXPECT_THAT(
      [] {
        return Session{
            SessionConfig{.driverPath = MQT_CORE_QDMI_INCOMPLETE_DRIVER}};
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("missing symbol QDMI_session_alloc")));
  EXPECT_THAT(
      [] {
        return Session{
            SessionConfig{.driverPath = MQT_CORE_QDMI_INCOMPATIBLE_DRIVER}};
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("incompatible ABI")));

  /// The upstream example requires a token. A failed initialization must allow
  /// retry.
  EXPECT_THAT(
      [] {
        return Session{SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER}};
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Permission denied")));
  const SessionConfig firstConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "first-token",
  };
  Session first(firstConfig);
  Session second(SessionConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "second-token",
  });
  EXPECT_EQ(first.getDeviceIds(), std::vector<std::string>{"test.example"});
  EXPECT_THAT(first.getDevice("test.example").getName(),
              testing::HasSubstr("token=first-token"));
  EXPECT_THAT(second.getDevice("test.example").getName(),
              testing::HasSubstr("token=second-token"));

  const auto retained = Session::openDevice("test.example", firstConfig);
  EXPECT_THAT(retained.getName(), testing::HasSubstr("token=first-token"));
  const auto job = Session::openDevice("test.example", firstConfig)
                       .submitJob("payload", QDMI_PROGRAM_FORMAT_CUSTOM1);
  EXPECT_EQ(job.getId(), "session-job");
}

TEST(BuiltinDriverExtensionTest, DiscoversThenOpensIndependentSessions) {
  const mqt::test::TemporaryDirectory directory;
  const auto definition = [](const std::string& id) {
    return nlohmann::json::object({
        {"id", id},
        {"library", MQT_CORE_QDMI_SESSION_DEVICE},
        {"prefix", "TEST_SESSION"},
        {"session", {{"custom4", "idle"}}},
    });
  };
  const auto manifest = [&](const std::filesystem::path& name,
                            const std::string& id) {
    return directory.write(name,
                           nlohmann::json{
                               {"schema-version", 1},
                               {"qdmi", {{"devices", {definition(id)}}}},
                           }
                               .dump());
  };
  const auto configured = manifest("device.qdmi.json", "test.session");
  const auto conflicting = manifest("duplicate.qdmi.json", "test.session");
  const auto unicode = manifest(
      std::filesystem::path{u8"device-ünicode.qdmi.json"}, "test.unicode");
  const auto late = manifest("late.qdmi.json", "test.late");
  const auto malformed = directory.write("malformed.qdmi.json", "{");
  EXPECT_THROW(builtin_driver::addManifest(directory.path() / "missing"),
               std::runtime_error);
  EXPECT_THROW(builtin_driver::addManifest(malformed), std::invalid_argument);

  const ScopedEnvironmentVariable driver{"MQT_CORE_QDMI_DRIVER",
                                         MQT_CORE_QDMI_TEST_DRIVER};
  builtin_driver::addManifest(configured);
  builtin_driver::addManifest(configured);
  EXPECT_THROW(builtin_driver::addManifest(conflicting), std::invalid_argument);
  builtin_driver::addManifest(unicode);
  {
    const ScopedEnvironmentVariable invalidConfig{"MQT_CORE_QDMI_CONFIG_JSON",
                                                  "{"};
    EXPECT_THROW(static_cast<void>(builtin_driver::openDevice("test.session")),
                 std::invalid_argument);
  }
  /// Failed driver initialization must leave manifest registration available.
  builtin_driver::addManifest(late);
  const auto first = [&] {
    const ScopedEnvironmentVariable overrides{
        "MQT_CORE_QDMI_CONFIG_JSON",
        R"({"schema-version":1,"qdmi":{"devices":[{"id":"test.session","session":{"custom4":"busy"}}]}})"};
    return builtin_driver::openDevice("test.session");
  }();
  const auto second =
      builtin_driver::openDevice("test.session", R"({"custom4":"offline"})");
  EXPECT_EQ(first.getId(), "test.session");
  EXPECT_EQ(second.getId(), "test.session");
  EXPECT_NE(first, second);
  EXPECT_EQ(first.getStatus(), QDMI_DEVICE_STATUS_BUSY);
  EXPECT_EQ(second.getStatus(), QDMI_DEVICE_STATUS_OFFLINE);
  EXPECT_EQ(builtin_driver::openDevice("test.unicode").getId(), "test.unicode");
  EXPECT_THROW(
      builtin_driver::addManifest(manifest("new.qdmi.json", "test.new")),
      std::runtime_error);

  const ScopedEnvironmentVariable config{"QDMI_CONF",
                                         MQT_CORE_QDMI_EXAMPLE_CONFIG};
  Session replacement(SessionConfig{.token = "replacement"});
  EXPECT_EQ(replacement.getDeviceIds(),
            std::vector<std::string>{"test.example"});
  EXPECT_EQ(first.getStatus(), QDMI_DEVICE_STATUS_BUSY);
  EXPECT_THAT(
      [] {
        return builtin_driver::openDevice(
            "unused", {},
            std::optional<std::filesystem::path>{MQT_CORE_QDMI_TEST_DRIVER});
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("does not support targeted sessions")));
}

} // namespace
} // namespace qdmi
