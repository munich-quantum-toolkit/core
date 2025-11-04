/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * DDSIM QDMI Device - Device status transitions (OFFLINE/BUSY/IDLE)
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"

#include <atomic>
#include <gtest/gtest.h>
#include <thread>

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
  qdmi_test::SessionGuard s{};

  // Initial status can be OFFLINE depending on implementation; do not assert
  // it. Submit a job to force BUSY then completion to IDLE.
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 4096), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_SUCCESS);

  // Poll while running to observe BUSY at least once.
  auto sawBusy = false;
  std::atomic<bool> done{false};
  std::thread poller([&]() {
    while (!done.load()) {
      if (const auto st = queryStatus(s.session);
          st == QDMI_DEVICE_STATUS_BUSY) {
        sawBusy = true;
      }
    }
  });

  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_SUCCESS);
  done.store(true);
  poller.join();

  EXPECT_TRUE(sawBusy);

  // After completion, the status should be IDLE.
  EXPECT_EQ(queryStatus(s.session), QDMI_DEVICE_STATUS_IDLE);
}

TEST(DeviceStatus, MultipleConcurrentJobsKeepBusyUntilLastFinishes) {
  const qdmi_test::SessionGuard s{};

  const qdmi_test::JobGuard j1{s.session};
  const qdmi_test::JobGuard j2{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j1.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setProgram(j2.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Heavy_Sampling5),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j1.job, 1024), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j2.job, 16384), QDMI_SUCCESS);

  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j1.job), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j2.job), QDMI_SUCCESS);

  // Wait for first to finish
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(j1.job, 0), QDMI_SUCCESS);
  // Status should still be BUSY while the second runs
  EXPECT_EQ(queryStatus(s.session), QDMI_DEVICE_STATUS_BUSY);

  // Wait for second, then status should go IDLE
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(j2.job, 0), QDMI_SUCCESS);
  EXPECT_EQ(queryStatus(s.session), QDMI_DEVICE_STATUS_IDLE);
}
