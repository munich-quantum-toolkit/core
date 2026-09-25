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

#include "support/TestSupport.hpp"

/// Required for JSON construction inside GoogleTest macro expansions.
#include "nlohmann/json.hpp" /// NOLINT(misc-include-cleaner)
#include "nlohmann/json_fwd.hpp"
#include "qdmi/constants.h"

#include <filesystem>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <optional>
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
  EXPECT_THAT(::mqt::test::errorMessage([] { return Session::create(); }),
              testing::HasSubstr("Cannot load QDMI driver"));
  EXPECT_THAT(::mqt::test::errorMessage([] {
                return Session::create(SessionConfig{
                    .driverPath = MQT_CORE_QDMI_INCOMPLETE_DRIVER});
              }),
              testing::HasSubstr("missing symbol QDMI_session_alloc"));
  EXPECT_THAT(::mqt::test::errorMessage([] {
                return Session::create(SessionConfig{
                    .driverPath = MQT_CORE_QDMI_INCOMPATIBLE_DRIVER});
              }),
              testing::HasSubstr("incompatible ABI"));

  /// The upstream example requires a token. A failed initialization must allow
  /// retry.
  EXPECT_THAT(::mqt::test::errorMessage([] {
                return Session::create(
                    SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER});
              }),
              testing::HasSubstr("Permission denied"));
  const SessionConfig firstConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "first-token",
  };
  auto first = ::mqt::test::value(Session::create(firstConfig));
  auto second = ::mqt::test::value(Session::create(SessionConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "second-token",
  }));
  EXPECT_EQ(::mqt::test::value(first.getDeviceIds()),
            std::vector<std::string>{"test.example"});
  EXPECT_THAT(
      ::mqt::test::value(
          ::mqt::test::value(first.getDevice("test.example")).getName()),
      testing::HasSubstr("token=first-token"));
  EXPECT_THAT(
      ::mqt::test::value(
          ::mqt::test::value(second.getDevice("test.example")).getName()),
      testing::HasSubstr("token=second-token"));

  const auto retained =
      ::mqt::test::value(Session::openDevice("test.example", firstConfig));
  EXPECT_THAT(::mqt::test::value(retained.getName()),
              testing::HasSubstr("token=first-token"));
  const auto job = ::mqt::test::value(
      ::mqt::test::value(Session::openDevice("test.example", firstConfig))
          .submitJob("payload", QDMI_PROGRAM_FORMAT_CUSTOM1));
  EXPECT_EQ(::mqt::test::value(job.getId()), "session-job");
  {
    const ScopedEnvironmentVariable optionalToken{
        "MQT_CORE_QDMI_TEST_DEVICE_FAILURE", "token-unsupported"};
    EXPECT_EQ(
        ::mqt::test::value(
            ::mqt::test::value(Session::openDevice("test.example", firstConfig))
                .getId()),
        "test.example");
  }
  {
    const ScopedEnvironmentVariable deniedToken{
        "MQT_CORE_QDMI_TEST_DEVICE_FAILURE", "token-denied"};
    EXPECT_THAT(
        ::mqt::test::errorMessage([&] { return Session::create(firstConfig); }),
        testing::HasSubstr("Permission denied"));
  }
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
  EXPECT_EQ(::mqt::test::errorStatus([&] {
              return builtin_driver::addManifest(directory.path() / "missing");
            }),
            QDMI_ERROR_LIBNOTFOUND);
  EXPECT_EQ(::mqt::test::errorStatus(
                [&] { return builtin_driver::addManifest(malformed); }),
            QDMI_ERROR_INVALIDARGUMENT);

  const ScopedEnvironmentVariable driver{"MQT_CORE_QDMI_DRIVER",
                                         MQT_CORE_QDMI_TEST_DRIVER};
  ::mqt::test::value(builtin_driver::addManifest(configured));
  ::mqt::test::value(builtin_driver::addManifest(configured));
  EXPECT_EQ(::mqt::test::errorStatus(
                [&] { return builtin_driver::addManifest(conflicting); }),
            QDMI_ERROR_INVALIDARGUMENT);
  ::mqt::test::value(builtin_driver::addManifest(unicode));
  {
    const ScopedEnvironmentVariable invalidConfig{"MQT_CORE_QDMI_CONFIG_JSON",
                                                  "{"};
    EXPECT_EQ(::mqt::test::errorStatus(
                  [&] { return builtin_driver::openDevice("test.session"); }),
              QDMI_ERROR_INVALIDARGUMENT);
  }
  /// Failed driver initialization must leave manifest registration available.
  ::mqt::test::value(builtin_driver::addManifest(late));
  const auto first = [&] {
    const ScopedEnvironmentVariable overrides{
        "MQT_CORE_QDMI_CONFIG_JSON",
        R"({"schema-version":1,"qdmi":{"devices":[{"id":"test.session","session":{"custom4":"busy"}}]}})"};
    return ::mqt::test::value(builtin_driver::openDevice("test.session"));
  }();
  const auto second = ::mqt::test::value(
      builtin_driver::openDevice("test.session", R"({"custom4":"offline"})"));
  EXPECT_EQ(::mqt::test::value(first.getId()), "test.session");
  EXPECT_EQ(::mqt::test::value(second.getId()), "test.session");
  EXPECT_NE(first, second);
  EXPECT_EQ(::mqt::test::value(first.getStatus()), QDMI_DEVICE_STATUS_BUSY);
  EXPECT_EQ(::mqt::test::value(second.getStatus()), QDMI_DEVICE_STATUS_OFFLINE);
  EXPECT_EQ(::mqt::test::value(
                ::mqt::test::value(builtin_driver::openDevice("test.unicode"))
                    .getId()),
            "test.unicode");
  EXPECT_EQ(::mqt::test::errorStatus([&] {
              return builtin_driver::addManifest(
                  manifest("new.qdmi.json", "test.new"));
            }),
            QDMI_ERROR_BADSTATE);

  const ScopedEnvironmentVariable config{"QDMI_CONF",
                                         MQT_CORE_QDMI_EXAMPLE_CONFIG};
  auto replacement = ::mqt::test::value(
      Session::create(SessionConfig{.token = "replacement"}));
  EXPECT_EQ(::mqt::test::value(replacement.getDeviceIds()),
            std::vector<std::string>{"test.example"});
  EXPECT_EQ(::mqt::test::value(first.getStatus()), QDMI_DEVICE_STATUS_BUSY);
  EXPECT_THAT(
      ::mqt::test::errorMessage([] {
        return builtin_driver::openDevice(
            "unused", {},
            std::optional<std::filesystem::path>{MQT_CORE_QDMI_TEST_DRIVER});
      }),
      testing::HasSubstr("does not support targeted sessions"));
}

} // namespace
} // namespace qdmi
