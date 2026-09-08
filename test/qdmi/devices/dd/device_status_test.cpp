/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/* DDSIM QDMI device status transitions. */
#include "helpers/controlled_job.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <gtest/gtest.h>

namespace {
QDMI_Device_Status queryStatus(MQT_DDSIM_QDMI_Device_Session session) {
  QDMI_Device_Status st = QDMI_DEVICE_STATUS_OFFLINE;
  const auto rc = MQT_DDSIM_QDMI_device_session_query_device_property(
      session, QDMI_DEVICE_PROPERTY_STATUS, sizeof(QDMI_Device_Status), &st,
      nullptr);
  EXPECT_EQ(rc, QDMI_SUCCESS);
  return st;
}
} // namespace

TEST(DeviceStatus, TransitionsBusyThenIdleAfterJob) {
  const qdmi_test::SessionGuard s{};

  EXPECT_EQ(queryStatus(s.session), QDMI_DEVICE_STATUS_IDLE);

  const qdmi_test::JobGuard j{s.session};
  qdmi_test::ControlledJob running{j.job};
  EXPECT_EQ(queryStatus(s.session), QDMI_DEVICE_STATUS_BUSY);
  running.finish(j.job);

  /// After completion, the status should be IDLE.
  EXPECT_EQ(queryStatus(s.session), QDMI_DEVICE_STATUS_IDLE);
}
