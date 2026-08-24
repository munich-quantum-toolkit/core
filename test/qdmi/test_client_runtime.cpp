/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestUtils.hpp"
#include "qdmi/Client.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
/// POSIX declares setenv and unsetenv in this compatibility header.
/// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdlib.h>

#include <filesystem>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>

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

TEST(ClientRuntimeTest, ValidatesThenFreezesOneDriverAndRetainsSessions) {
  const auto missing =
      std::filesystem::path(MQT_CORE_QDMI_TEST_DRIVER).parent_path() /
      "missing-client-driver";
  setDriverEnvironment(missing.string());
  EXPECT_THAT([] { return Session{}; },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("Cannot load QDMI Client driver")));

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

  const SessionConfig firstConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER,
                                  .token = "first-token"};
  setDriverEnvironment(MQT_CORE_QDMI_TEST_DRIVER);
  const mqt::test::ScopedEnvironmentVariable failAllocation{
      "MQT_CORE_QDMI_FAKE_FAIL_ALLOCATION", "1"};
  EXPECT_THROW(static_cast<void>(Session{firstConfig}), std::bad_alloc);
  EXPECT_THAT(
      [] {
        return Session{
            SessionConfig{.driverPath = MQT_CORE_QDMI_INCOMPLETE_DRIVER}};
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("missing symbol QDMI_session_alloc")));

  Session first(firstConfig);
  Session second(SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER,
                               .token = "second-token"});
  const auto firstDevices = first.getDevices();
  const auto secondDevices = second.getDevices();
  ASSERT_EQ(firstDevices.size(), 1U);
  ASSERT_EQ(secondDevices.size(), 1U);
  EXPECT_EQ(firstDevices.front().getId(), "test.fake.client");
  EXPECT_EQ(secondDevices.front().getId(), "test.fake.client");
  EXPECT_EQ(firstDevices.front().getName(), "first-token");
  EXPECT_EQ(secondDevices.front().getName(), "second-token");

  Session oddSize(SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER,
                                .token = "odd-size"});
  EXPECT_THROW(static_cast<void>(oddSize.getDevices()), std::invalid_argument);
  Session oddDeviceSize(SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER,
                                      .token = "odd-device-size"});
  EXPECT_THROW(static_cast<void>(oddDeviceSize.getDevices().front().getSites()),
               std::invalid_argument);
  Session oddOperationSize(SessionConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER, .token = "odd-operation-size"});
  EXPECT_THROW(static_cast<void>(oddOperationSize.getDevices()
                                     .front()
                                     .getOperations()
                                     .front()
                                     .getSites()),
               std::invalid_argument);

  Session malformedFormat(SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER,
                                        .token = "malformed-format"});
  EXPECT_THROW(
      static_cast<void>(
          malformedFormat.getDevices().front().getSupportedProgramFormats()),
      std::runtime_error);
  Session malformedFeature(SessionConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER, .token = "malformed-feature"});
  EXPECT_THROW(static_cast<void>(
                   malformedFeature.getDevices().front().tryGetProgramFeatures(
                       OPENQASM3)),
               std::runtime_error);

  EXPECT_THAT(
      [] {
        return Session{
            SessionConfig{.driverPath = MQT_CORE_QDMI_INCOMPLETE_DRIVER}};
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("already selected")));

  const auto site = [] {
    auto device = Session::openDevice(
        "test.fake.client",
        SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER,
                      .token = "retained-token"});
    auto sites = device.getSites();
    return sites.front();
  }();
  EXPECT_EQ(site.getIndex(), 0U);

  setDriverEnvironment(std::nullopt);
}

} // namespace
} // namespace qdmi
