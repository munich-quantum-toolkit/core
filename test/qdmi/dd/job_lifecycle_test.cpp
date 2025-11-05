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
 * DDSIM QDMI Device - Job lifecycle (submit/cancel/check/wait/free)
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/device.h"

#include <atomic>
#include <cstddef>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <qdmi/constants.h>
#include <thread>

using testing::AnyOf;

TEST(JobLifecycle, SubmitAndWaitSampling) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 256), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, SubmitAndWaitStatevector) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_State),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, WaitInvalidBeforeSubmitAndIdempotentAfterDone) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  // wait before submit is invalid
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0),
            QDMI_ERROR_INVALIDARGUMENT);
  // now run a quick job
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 64), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
  // waiting again succeeds
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, WaitTimeoutPath) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  // More shots to increase runtime slightly
  ASSERT_EQ(qdmi_test::setShots(j.job, 4096), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_SUCCESS);
  // 1-second timeout may or may not be enough → accept either
  const auto rc = MQT_DDSIM_QDMI_device_job_wait(j.job, 1);
  EXPECT_THAT(rc, AnyOf(QDMI_SUCCESS, QDMI_ERROR_TIMEOUT));
  // Ensure completion for cleanup
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, CancelFromCreatedAndFromRunningAndFromDone) {
  // From CREATED
  {
    const qdmi_test::SessionGuard s{};
    const qdmi_test::JobGuard j{s.session};
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_cancel(j.job), QDMI_SUCCESS);
  }
  // From RUNNING → we submit and then cancel (which effectively waits)
  {
    const qdmi_test::SessionGuard s{};
    const qdmi_test::JobGuard j{s.session};
    ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                    qdmi_test::QASM3_Bell_Sampling),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(j.job, 4096), QDMI_SUCCESS);
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_SUCCESS);
    EXPECT_THAT(MQT_DDSIM_QDMI_device_job_cancel(j.job),
                AnyOf(QDMI_SUCCESS, QDMI_ERROR_INVALIDARGUMENT));
  }
  // From DONE/FAILED → INVALIDARGUMENT
  {
    const qdmi_test::SessionGuard s{};
    const qdmi_test::JobGuard j{s.session};
    ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                    qdmi_test::QASM3_Bell_Sampling),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(j.job, 1), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_cancel(j.job),
              QDMI_ERROR_INVALIDARGUMENT);
  }
}

TEST(JobLifecycle, FreeWhileRunningWaitsForCompletion) {
  const qdmi_test::SessionGuard s{};
  MQT_DDSIM_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_create_device_job(s.session, &job),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setProgram(job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(job, 4096), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(job), QDMI_SUCCESS);
  // Freeing should wait internally without crashing; we cannot directly assert
  // timing but should not deadlock
  MQT_DDSIM_QDMI_device_job_free(job);
}
