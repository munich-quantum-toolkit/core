/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <thread>

using testing::AnyOf;

TEST(JobLifecycle, SubmitAndWaitSampling) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 256), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, SubmitAndWaitStatevector) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, WaitInvalidBeforeSubmitAndIdempotentAfterDone) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  // wait before submit is invalid
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_ERROR_BADSTATE);
  // now run a quick job
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
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
                                  qdmi_test::QASM3_BELL_SAMPLING),
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
                                    qdmi_test::QASM3_BELL_SAMPLING),
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
                                    qdmi_test::QASM3_BELL_SAMPLING),
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
                                  qdmi_test::QASM3_HEAVY_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(job, 4096), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(job), QDMI_SUCCESS);
  // Freeing should wait internally without crashing; we cannot directly assert
  // timing but should not deadlock
  MQT_DDSIM_QDMI_device_job_free(job);
}

TEST(JobLifecycle, KeepsSuccessfulResultsWhenAnotherProgramFails) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setPrograms(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                   {qdmi_test::QASM3_ZERO, "invalid OpenQASM",
                                    qdmi_test::QASM3_ONE}),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 32), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
  QDMI_Job_Status status{};
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(j.job, &status), QDMI_SUCCESS);
  EXPECT_EQ(status, QDMI_JOB_STATUS_FAILED);
  EXPECT_EQ(qdmi_test::getProgramStatus(j.job, 0), QDMI_JOB_STATUS_DONE);
  EXPECT_EQ(qdmi_test::getProgramStatus(j.job, 1), QDMI_JOB_STATUS_FAILED);
  EXPECT_EQ(qdmi_test::getProgramStatus(j.job, 2), QDMI_JOB_STATUS_DONE);
  for (const auto index : {size_t{0}, size_t{2}}) {
    std::array<char, 2> key{};
    size_t count = 0;
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, index, QDMI_JOB_RESULT_HIST_KEYS, key.size(),
                  key.data(), nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, index, QDMI_JOB_RESULT_HIST_VALUES, sizeof(count),
                  &count, nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(std::string(key.data()), index == 0 ? "0" : "1");
    EXPECT_EQ(count, 32);
  }
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, 1, QDMI_JOB_RESULT_SHOTS, 0, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, 3, QDMI_JOB_RESULT_SHOTS, 0, nullptr, nullptr),
            QDMI_ERROR_OUTOFRANGE);
}

TEST(JobLifecycle, CancelsAllProgramsBeforeSubmission) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(
      qdmi_test::setPrograms(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                             {qdmi_test::QASM3_ZERO, qdmi_test::QASM3_ONE}),
      QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_cancel(j.job), QDMI_SUCCESS);
  EXPECT_EQ(qdmi_test::getProgramStatus(j.job, 0), QDMI_JOB_STATUS_CANCELED);
  EXPECT_EQ(qdmi_test::getProgramStatus(j.job, 1), QDMI_JOB_STATUS_CANCELED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_ERROR_BADSTATE);
  EXPECT_EQ(qdmi_test::setPrograms(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                   {qdmi_test::QASM3_ONE}),
            QDMI_ERROR_BADSTATE);
}

TEST(JobLifecycle, CancellationReturnsOnlyAfterAllProgramsStop) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(
      qdmi_test::setPrograms(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                             {qdmi_test::QASM3_HEAVY_SAMPLING,
                              qdmi_test::QASM3_ZERO, qdmi_test::QASM3_ONE}),
      QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 8192), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_SUCCESS);
  while (qdmi_test::getProgramStatus(j.job, 0) == QDMI_JOB_STATUS_CREATED) {
    std::this_thread::yield();
  }
  const auto result = MQT_DDSIM_QDMI_device_job_cancel(j.job);
  QDMI_Job_Status status{};
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(j.job, &status), QDMI_SUCCESS);
  if (result == QDMI_SUCCESS) {
    EXPECT_EQ(status, QDMI_JOB_STATUS_CANCELED);
  } else {
    EXPECT_EQ(result, QDMI_ERROR_INVALIDARGUMENT);
    EXPECT_EQ(status, QDMI_JOB_STATUS_DONE);
  }
  for (size_t i = 0; i < 3; ++i) {
    const auto programStatus = qdmi_test::getProgramStatus(j.job, i);
    EXPECT_TRUE(programStatus == QDMI_JOB_STATUS_CANCELED ||
                programStatus == QDMI_JOB_STATUS_DONE);
    if (programStatus == QDMI_JOB_STATUS_DONE) {
      EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                    j.job, i, QDMI_JOB_RESULT_SHOTS, 0, nullptr, nullptr),
                QDMI_SUCCESS);
    }
  }
}
