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
#include "qdmi/common/Common.hpp"

#include "TestUtils.hpp"

#include "qdmi/constants.h"

#include <filesystem>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <new>
#include <optional>
#include <stdexcept>
/// POSIX declares setenv and unsetenv in <stdlib.h>.
/// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdlib.h>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace qdmi {
namespace {

void setDriverEnvironment(const std::optional<std::string>& value) {
#ifdef _WIN32
  ASSERT_EQ(_putenv_s("MQT_CORE_QDMI_DRIVER", value.value_or("").c_str()), 0);
#else
  if (value) {
    ASSERT_EQ(setenv("MQT_CORE_QDMI_DRIVER", value->c_str(), 1), 0);
  } else {
    ASSERT_EQ(unsetenv("MQT_CORE_QDMI_DRIVER"), 0);
  }
#endif
}

TEST(ClientRuntimeTest, ValidatesDriversAndRetainsSessions) {
  const auto missing =
      std::filesystem::path(MQT_CORE_QDMI_TEST_DRIVER).parent_path() /
      "missing-driver";
  setDriverEnvironment(missing.string());
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

  const SessionConfig firstConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "first-token",
  };
  setDriverEnvironment(MQT_CORE_QDMI_TEST_DRIVER);
  {
    const mqt::test::ScopedEnvironmentVariable failAllocation{
        "MQT_CORE_QDMI_FAKE_FAIL_ALLOCATION", "1"};
    EXPECT_THROW(static_cast<void>(Session{firstConfig}), std::bad_alloc);
  }
  EXPECT_THAT(
      [] {
        return Session{
            SessionConfig{.driverPath = MQT_CORE_QDMI_INCOMPLETE_DRIVER}};
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("missing symbol QDMI_session_alloc")));

  Session first(firstConfig);
  {
    const mqt::test::ScopedEnvironmentVariable nullAllocation{
        "MQT_CORE_QDMI_FAKE_FAIL_ALLOCATION", "success-null"};
    EXPECT_THAT([&] { return Session{firstConfig}; },
                testing::ThrowsMessage<std::runtime_error>(
                    testing::HasSubstr("returned a null session")));
  }
  Session second(SessionConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "second-token",
  });
  const auto alternate =
      std::filesystem::temp_directory_path() / "mqt-alternate-qdmi-driver" /
      std::filesystem::path(MQT_CORE_QDMI_TEST_DRIVER).filename();
  std::filesystem::create_directories(alternate.parent_path());
  std::filesystem::copy_file(MQT_CORE_QDMI_TEST_DRIVER, alternate,
                             std::filesystem::copy_options::overwrite_existing);
  {
    Session replacement(
        SessionConfig{.driverPath = alternate, .token = "replacement"});
    EXPECT_EQ(replacement.getDevice("test.fake.client").getName(),
              "replacement");
    EXPECT_EQ(first.getDeviceIds(),
              std::vector<std::string>{"test.fake.client"});
    EXPECT_EQ(first.getDevice("test.fake.client").getName(), "first-token");
  }
  /// Windows keeps a loaded driver DLL open until the process exits.
  std::error_code cleanupError;
  std::filesystem::remove(alternate, cleanupError);
  const auto firstDevices = first.getDevices();
  const auto secondDevices = second.getDevices();
  ASSERT_EQ(firstDevices.size(), 1U);
  ASSERT_EQ(secondDevices.size(), 1U);
  EXPECT_EQ(firstDevices.front().getId(), "test.fake.client");
  EXPECT_EQ(secondDevices.front().getId(), "test.fake.client");
  EXPECT_EQ(firstDevices.front().getName(), "first-token");
  EXPECT_EQ(secondDevices.front().getName(), "second-token");

  Session oddSize(SessionConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "odd-size",
  });
  EXPECT_THROW(static_cast<void>(oddSize.getDevices()), std::runtime_error);
  Session oddDeviceSize(SessionConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "odd-device-size",
  });
  EXPECT_THROW(static_cast<void>(oddDeviceSize.getDevices().front().getSites()),
               std::runtime_error);
  Session oddOperationSize(SessionConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "odd-operation-size",
  });
  EXPECT_THROW(static_cast<void>(oddOperationSize.getDevices()
                                     .front()
                                     .getOperations()
                                     .front()
                                     .getSites()),
               std::runtime_error);

  EXPECT_THAT(
      [] {
        return Session{
            SessionConfig{.driverPath = MQT_CORE_QDMI_INCOMPLETE_DRIVER}};
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("missing symbol QDMI_session_alloc")));

  const auto site = [] {
    const auto device = Session::openDevice(
        "test.fake.client", SessionConfig{
                                .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
                                .token = "retained-token",
                            });
    auto sites = device.getSites();
    return sites.front();
  }();
  EXPECT_EQ(site.getIndex(), 0U);

  setDriverEnvironment(std::nullopt);
}

TEST(ClientTargetedSelectionTest,
     FailedInitializationLeavesOtherDriversUsable) {
  EXPECT_THAT(
      [] {
        return builtin_driver::openDevice(
            "unused", {},
            std::optional<std::filesystem::path>{
                MQT_CORE_QDMI_TARGETED_INIT_FAILURE_DRIVER});
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Permission denied")));
  Session session(SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER});
  EXPECT_EQ(session.getDeviceIds(),
            std::vector<std::string>{"test.fake.client"});
}

void setEnvironment(const char* const name, const std::string_view value) {
#ifdef _WIN32
  ASSERT_EQ(_putenv_s(name, std::string(value).c_str()), 0);
#else
  if (value.empty()) {
    ASSERT_EQ(unsetenv(name), 0);
  } else {
    ASSERT_EQ(setenv(name, std::string(value).c_str(), 1), 0);
  }
#endif
}

void setConfigurationJson(const std::string_view value) {
  setEnvironment("MQT_CORE_QDMI_CONFIG_JSON", value);
}

TEST(BuiltinDriverExtensionTest, StagesThenOpensStrictFreshSessions) {
  EXPECT_THAT([] { builtin_driver::addManifest("missing-device-manifest"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("Library not found")));
  EXPECT_THAT(
      [] { builtin_driver::addManifest(MQT_CORE_QDMI_MALFORMED_MANIFEST); },
      testing::ThrowsMessage<std::invalid_argument>(
          testing::HasSubstr("Invalid argument")));
  EXPECT_THAT(
      [] {
        builtin_driver::addManifest(MQT_CORE_QDMI_MISSING_LIBRARY_MANIFEST);
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Library not found")));
  EXPECT_THROW(
      builtin_driver::addManifest(MQT_CORE_QDMI_MISSING_PREFIX_MANIFEST),
      std::invalid_argument);
  EXPECT_THROW(builtin_driver::addManifest(MQT_CORE_QDMI_NUL_MANIFEST),
               std::invalid_argument);
  EXPECT_THROW(
      builtin_driver::addManifest(MQT_CORE_QDMI_CONFLICTING_CONFIG_MANIFEST),
      std::invalid_argument);

  setEnvironment("MQT_CORE_QDMI_DRIVER", MQT_CORE_QDMI_FAKE_CLIENT);
  builtin_driver::addManifest(MQT_CORE_QDMI_DEVICE_MANIFEST);
  setEnvironment("MQT_CORE_QDMI_DRIVER", {});
  builtin_driver::addManifest(MQT_CORE_QDMI_DEVICE_MANIFEST);
  EXPECT_THROW(builtin_driver::addManifest(MQT_CORE_QDMI_CONFLICTING_MANIFEST),
               std::invalid_argument);

  constexpr std::string_view unicodeFilename = "device-\xC3\xBC"
                                               "nicode.qdmi.json";
  const auto unicodeManifestSource =
      detail::pathFromString(MQT_CORE_QDMI_UNICODE_MANIFEST_SOURCE);
  const auto unicodeManifest = unicodeManifestSource.parent_path() /
                               detail::pathFromString(unicodeFilename);
  EXPECT_EQ(detail::pathToString(unicodeManifest.filename()), unicodeFilename);
  ASSERT_TRUE(std::filesystem::copy_file(
      unicodeManifestSource, unicodeManifest,
      std::filesystem::copy_options::overwrite_existing));
  builtin_driver::addManifest(unicodeManifest);

  setConfigurationJson("{");
  EXPECT_THROW(static_cast<void>(builtin_driver::openDevice("test.session")),
               std::invalid_argument);
  setConfigurationJson({});
  builtin_driver::addManifest(MQT_CORE_QDMI_LATE_MANIFEST);
  builtin_driver::addManifest(MQT_CORE_QDMI_INCOMPATIBLE_MANIFEST);

  setConfigurationJson(
      R"({"schema-version":1,"qdmi":{"devices":[{"id":"test.session","session":{"custom4":"busy","custom5":"with-child"}}]}})");
  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_FAILURE", "alloc-error-handle");
  EXPECT_THAT([] { return builtin_driver::openDevice("test.session"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("Permission denied")));

  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_FAILURE", "alloc-null");
  EXPECT_THAT([] { return builtin_driver::openDevice("test.session"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("A fatal error")));
  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_FAILURE", {});
  EXPECT_THAT(
      [] {
        return builtin_driver::openDevice(
            "unused", {},
            std::optional<std::filesystem::path>{MQT_CORE_QDMI_FAKE_CLIENT});
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("does not support targeted sessions")));

  setEnvironment("MQT_CORE_QDMI_DRIVER", MQT_CORE_QDMI_FAKE_CLIENT);
  const auto first = builtin_driver::openDevice("test.session");
  setConfigurationJson({});
  setEnvironment("MQT_CORE_QDMI_DRIVER", {});
  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_FAILURE", {});
  EXPECT_EQ(first.getId(), "test.session");
  EXPECT_THAT(first.getName(), testing::HasSubstr("active=2"));
  EXPECT_EQ(first.getStatus(), QDMI_DEVICE_STATUS_BUSY);
  EXPECT_EQ(first.getChildDevices().size(), 1U);

  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_FAILURE", "children-null");
  EXPECT_THROW(static_cast<void>(builtin_driver::openDevice("test.session")),
               std::runtime_error);
  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_FAILURE", {});

  const auto second =
      builtin_driver::openDevice("test.session", R"({"custom4":"offline"})");
  EXPECT_NE(first, second);
  EXPECT_EQ(second.getStatus(), QDMI_DEVICE_STATUS_OFFLINE);

  EXPECT_THROW(
      static_cast<void>(builtin_driver::openDevice("test.session", "{")),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(builtin_driver::openDevice(
                   "test.session",
                   R"({"device-config":{"inline":{}},"custom1":"raw"})")),
               std::invalid_argument);
  const auto nullId = std::string("test.session") + '\0' + "alias";
  EXPECT_THROW(static_cast<void>(builtin_driver::openDevice(nullId)),
               std::invalid_argument);

  EXPECT_THAT([] { return builtin_driver::openDevice("test.child-error"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("Permission denied")));
  EXPECT_THAT([] { return builtin_driver::openDevice("test.incompatible"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("A fatal error")));
  EXPECT_EQ(builtin_driver::openDevice("test.unicode").getId(), "test.unicode");
  EXPECT_TRUE(std::filesystem::remove(unicodeManifest));

  builtin_driver::addManifest(unicodeManifest);
  builtin_driver::addManifest(MQT_CORE_QDMI_DEVICE_MANIFEST);
  EXPECT_THAT(
      [] { builtin_driver::addManifest(MQT_CORE_QDMI_POST_FREEZE_MANIFEST); },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Bad state")));
}

} // namespace
} // namespace qdmi
