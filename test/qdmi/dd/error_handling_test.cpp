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

#include <cstddef>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

TEST(ErrorHandling, NullptrArguments) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_alloc(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
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
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_create_device_job(nullptr, nullptr),
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

  const qdmi_test::SessionGuard s{};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_create_device_job(s.session, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);

  const qdmi_test::JobGuard j{s.session};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_check(j.job, nullptr),
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

TEST(ErrorHandling, MaxEnums) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  MQT_DDSIM_QDMI_Device_Session session = nullptr;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  MQT_DDSIM_QDMI_device_session_free(session);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          s.session, nullptr, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, nullptr, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  constexpr QDMI_Program_Format maxFmt = QDMI_PROGRAM_FORMAT_MAX;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &maxFmt),
            QDMI_ERROR_INVALIDARGUMENT);

  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 16), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(j.job, QDMI_JOB_RESULT_MAX, 0,
                                                  nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(ErrorHandling, CustomEnums) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  MQT_DDSIM_QDMI_Device_Session session = nullptr;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM3, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM4, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM5, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  MQT_DDSIM_QDMI_device_session_free(session);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_CUSTOM2, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_CUSTOM3, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_CUSTOM4, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_CUSTOM5, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);

  const auto sites = qdmi_test::querySites(s.session);
  auto* const site = sites.front();
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          s.session, site, QDMI_SITE_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
      QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          s.session, site, QDMI_SITE_PROPERTY_CUSTOM2, 0, nullptr, nullptr),
      QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          s.session, site, QDMI_SITE_PROPERTY_CUSTOM3, 0, nullptr, nullptr),
      QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          s.session, site, QDMI_SITE_PROPERTY_CUSTOM4, 0, nullptr, nullptr),
      QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          s.session, site, QDMI_SITE_PROPERTY_CUSTOM5, 0, nullptr, nullptr),
      QDMI_ERROR_NOTSUPPORTED);

  const auto ops = qdmi_test::queryOperations(s.session);
  auto* const op = ops.front();
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, op, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, op, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM2, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, op, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM3, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, op, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM4, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, op, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM5, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM1, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM2, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM3, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM4, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM5, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);

  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 16), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_CUSTOM2, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_CUSTOM3, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_CUSTOM4, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_CUSTOM5, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST(ErrorHandling, BadState) {
  MQT_DDSIM_QDMI_Device_Session session = nullptr;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          session, nullptr, QDMI_SITE_PROPERTY_INDEX, 0, nullptr, nullptr),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                session, nullptr, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);

  MQT_DDSIM_QDMI_Device_Job job = nullptr;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_create_device_job(session, &job),
            QDMI_ERROR_BADSTATE);

  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  MQT_DDSIM_QDMI_device_job_cancel(j.job);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, 0, nullptr),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_ERROR_BADSTATE);
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
