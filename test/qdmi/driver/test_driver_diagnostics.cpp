/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Driver.hpp"
#include "support/TestSupport.hpp"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "qdmi/client.h"

#include <cstdlib>
#include <vector>

namespace {

TEST(DriverDiagnosticDeathTest,
     InvalidConfigurationDoesNotEscapeSessionAllocation) {
  const auto probe = [] {
    const mqt::test::ScopedEnvironmentVariable config(
        "MQT_CORE_QDMI_CONFIG_JSON", "invalid-json");
    if (QDMI_session_alloc(nullptr) != QDMI_ERROR_INVALIDARGUMENT) {
      std::_Exit(2);
    }
    QDMI_Session_impl_d sentinel(std::vector<QDMI_Device>{});
    auto* session = &sentinel;
    const auto status = QDMI_session_alloc(&session);
    std::_Exit(status == QDMI_ERROR_INVALIDARGUMENT && session == nullptr ? 0
                                                                          : 3);
  };
  EXPECT_EXIT(probe(), testing::ExitedWithCode(0), "");
}

TEST(DriverDiagnosticTest, ReportsSkippedConfiguredDevice) {
  const mqt::test::ScopedEnvironmentVariable config(
      "MQT_CORE_QDMI_CONFIG_FILE", MQT_CORE_QDMI_DIAGNOSTIC_CONFIG_FILE);

  testing::internal::CaptureStderr();
  QDMI_Session session = nullptr;
  const auto status = QDMI_session_alloc(&session);
  const auto diagnostic = testing::internal::GetCapturedStderr();

  ASSERT_EQ(status, QDMI_SUCCESS);
  EXPECT_THAT(
      diagnostic,
      testing::AllOf(testing::HasSubstr("[mqt-core] [warning]"),
                     testing::HasSubstr("Skipping configured QDMI device "
                                        "'broken.diagnostic'"),
                     testing::HasSubstr("missing-diagnostic-device-library")));
  QDMI_session_free(session);
}

} // namespace
