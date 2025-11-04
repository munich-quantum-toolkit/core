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
 * DDSIM QDMI Device - Error handling and invalid arguments
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

TEST(ErrorHandling, NullptrArguments) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                nullptr, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          nullptr, nullptr, QDMI_SITE_PROPERTY_INDEX, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                nullptr, nullptr, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                nullptr, QDMI_DEVICE_JOB_PROPERTY_ID, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                nullptr, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_cancel(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_check(nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(nullptr, 0),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(nullptr, QDMI_JOB_RESULT_MAX,
                                                  0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(ErrorHandling, GetResultsBeforeDone) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 64), QDMI_SUCCESS);
  // Before submit → invalid
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_HIST_KEYS, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_SUCCESS);
  // After submit but not necessarily done → still invalid or waits; contract
  // says invalid
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_HIST_KEYS, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_SUCCESS);
}

TEST(ErrorHandling, CustomAndMaxEnums) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  // Device property MAX invalid
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  // Device custom not supported
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);

  // Job property MAX invalid
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  // Job result MAX invalid
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 16), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(j.job, QDMI_JOB_RESULT_MAX, 0,
                                                  nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  // Custom results not supported
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST(ErrorHandling, MalformedProgramFailsForBothModes) {
  const qdmi_test::SessionGuard s{};
  // Sampling mode
  {
    const qdmi_test::JobGuard j{s.session};
    ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                    qdmi_test::MALFORMED_PROGRAM),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(j.job, 128), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
    QDMI_Job_Status js{};
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(j.job, &js), QDMI_SUCCESS);
    EXPECT_EQ(js, QDMI_JOB_STATUS_FAILED);
  }
  // Statevector mode
  {
    const qdmi_test::JobGuard j{s.session};
    ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                    qdmi_test::MALFORMED_PROGRAM),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
    QDMI_Job_Status js{};
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(j.job, &js), QDMI_SUCCESS);
    EXPECT_EQ(js, QDMI_JOB_STATUS_FAILED);
  }
}
