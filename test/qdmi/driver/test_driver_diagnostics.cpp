/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <qdmi/client.h>

#include <cstdlib>

namespace {

TEST(DriverDiagnosticTest, ReportsSkippedConfiguredDevice) {
#ifdef _WIN32
  ASSERT_EQ(_putenv_s("MQT_CORE_QDMI_CONFIG_FILE",
                      MQT_CORE_QDMI_DIAGNOSTIC_CONFIG_FILE),
            0);
#else
  // POSIX exposes setenv through <cstdlib>, but include-cleaner does not
  // associate the global declaration with that C++ header.
  // NOLINTNEXTLINE(misc-include-cleaner)
  ASSERT_EQ(setenv("MQT_CORE_QDMI_CONFIG_FILE",
                   MQT_CORE_QDMI_DIAGNOSTIC_CONFIG_FILE, 1),
            0);
#endif

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
