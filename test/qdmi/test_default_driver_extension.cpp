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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <qdmi/constants.h>
/// POSIX declares setenv and unsetenv in this compatibility header.
/// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdlib.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace qdmi {
namespace {

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

TEST(DefaultDriverExtensionTest, StagesThenOpensStrictFreshSessions) {
  EXPECT_THAT([] { default_driver::addManifest("missing-package-manifest"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("Library not found")));
  EXPECT_THAT(
      [] { default_driver::addManifest(MQT_CORE_QDMI_MALFORMED_MANIFEST); },
      testing::ThrowsMessage<std::invalid_argument>(
          testing::HasSubstr("Invalid argument")));
  EXPECT_THAT(
      [] {
        default_driver::addManifest(MQT_CORE_QDMI_MISSING_LIBRARY_MANIFEST);
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Library not found")));
  EXPECT_THROW(
      default_driver::addManifest(MQT_CORE_QDMI_MISSING_PREFIX_MANIFEST),
      std::invalid_argument);
  EXPECT_THROW(default_driver::addManifest(MQT_CORE_QDMI_NUL_MANIFEST),
               std::invalid_argument);
  EXPECT_THROW(
      default_driver::addManifest(MQT_CORE_QDMI_CONFLICTING_CONFIG_MANIFEST),
      std::invalid_argument);

  setEnvironment("MQT_CORE_QDMI_DRIVER", MQT_CORE_QDMI_FAKE_CLIENT);
  default_driver::addManifest(MQT_CORE_QDMI_PACKAGE_MANIFEST);
  setEnvironment("MQT_CORE_QDMI_DRIVER", {});
  default_driver::addManifest(MQT_CORE_QDMI_PACKAGE_MANIFEST);
  EXPECT_THROW(default_driver::addManifest(MQT_CORE_QDMI_CONFLICTING_MANIFEST),
               std::invalid_argument);

  constexpr std::string_view unicodeFilename = "package-\xC3\xBC"
                                               "nicode.qdmi.json";
  const auto unicodeManifestSource =
      detail::pathFromUtf8(MQT_CORE_QDMI_UNICODE_MANIFEST_SOURCE);
  const auto unicodeManifest = unicodeManifestSource.parent_path() /
                               detail::pathFromUtf8(unicodeFilename);
  EXPECT_EQ(detail::pathToUtf8(unicodeManifest.filename()), unicodeFilename);
  ASSERT_TRUE(std::filesystem::copy_file(
      unicodeManifestSource, unicodeManifest,
      std::filesystem::copy_options::overwrite_existing));
  default_driver::addManifest(unicodeManifest);

  setConfigurationJson("{");
  EXPECT_THROW(static_cast<void>(default_driver::openDevice("package.session")),
               std::invalid_argument);
  setConfigurationJson({});
  default_driver::addManifest(MQT_CORE_QDMI_LATE_MANIFEST);
  default_driver::addManifest(MQT_CORE_QDMI_INCOMPATIBLE_MANIFEST);

  setConfigurationJson(
      R"({"schema-version":1,"qdmi":{"devices":[{"id":"package.session","session":{"custom4":"busy","custom5":"with-child"}}]}})");
  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_WARNING", "alloc-error-handle");
  EXPECT_THAT([] { return default_driver::openDevice("package.session"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("Permission denied")));

  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_WARNING", "alloc-null");
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  EXPECT_THAT([] { return default_driver::openDevice("package.session"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("A fatal error")));
  const auto warningOutput = testing::internal::GetCapturedStdout() +
                             testing::internal::GetCapturedStderr();
  EXPECT_THAT(warningOutput,
              testing::Not(testing::HasSubstr("general warning")));
  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_WARNING", {});
  EXPECT_THAT(
      [] {
        return default_driver::openDevice(
            "unused", {},
            std::optional<std::filesystem::path>{MQT_CORE_QDMI_FAKE_CLIENT});
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("does not support targeted sessions")));

  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_WARNING", "all");
  setEnvironment("MQT_CORE_QDMI_DRIVER", MQT_CORE_QDMI_FAKE_CLIENT);
  const auto first = default_driver::openDevice("package.session");
  setConfigurationJson({});
  setEnvironment("MQT_CORE_QDMI_DRIVER", {});
  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_WARNING", {});
  EXPECT_EQ(first.getId(), "package.session");
  EXPECT_THAT(first.getName(), testing::HasSubstr("active=2"));
  EXPECT_EQ(first.getStatus(), QDMI_DEVICE_STATUS_BUSY);
  EXPECT_EQ(first.getChildDevices().size(), 1U);

  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_WARNING", "children-null");
  EXPECT_THROW(static_cast<void>(default_driver::openDevice("package.session")),
               std::invalid_argument);
  setEnvironment("MQT_CORE_QDMI_TEST_DEVICE_WARNING", {});

  const auto second =
      default_driver::openDevice("package.session", R"({"custom4":"offline"})");
  EXPECT_NE(first, second);
  EXPECT_EQ(second.getStatus(), QDMI_DEVICE_STATUS_OFFLINE);

  EXPECT_THROW(
      static_cast<void>(default_driver::openDevice("package.session", "{")),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(default_driver::openDevice(
                   "package.session",
                   R"({"device-config":{"inline":{}},"custom1":"raw"})")),
               std::invalid_argument);
  const std::string nullId{"package.session\0alias", 21};
  EXPECT_THROW(static_cast<void>(default_driver::openDevice(nullId)),
               std::invalid_argument);

  EXPECT_THAT([] { return default_driver::openDevice("package.child-error"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("Permission denied")));
  EXPECT_THAT([] { return default_driver::openDevice("package.incompatible"); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("A fatal error")));
  EXPECT_EQ(default_driver::openDevice("package.unicode").getId(),
            "package.unicode");
  EXPECT_TRUE(std::filesystem::remove(unicodeManifest));

  default_driver::addManifest(unicodeManifest);
  default_driver::addManifest(MQT_CORE_QDMI_PACKAGE_MANIFEST);
  EXPECT_THAT(
      [] { default_driver::addManifest(MQT_CORE_QDMI_POST_FREEZE_MANIFEST); },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Bad state")));
}

} // namespace
} // namespace qdmi
