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
 * DDSIM QDMI Device - Session lifecycle and parameters
 */
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

using testing::AnyOf;

TEST(SessionLifecycle, AllocAndInit) {
  const qdmi_test::SessionGuard s{}; // ctor does alloc+init
  ASSERT_NE(s.session, nullptr);
}

TEST(SessionLifecycle, InitBadStateAndInvalidArg) {
  // Allocate a fresh session but do not init twice via the guard
  MQT_DDSIM_QDMI_Device_Session session = nullptr;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_initialize(), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_init(session), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_init(session), QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_init(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  MQT_DDSIM_QDMI_device_session_free(session);
  MQT_DDSIM_QDMI_device_finalize();
}

TEST(SessionParameters, SetParameterBehaviors) {
  // Behavior depends on implementation; we assert allowed outcomes
  const qdmi_test::SessionGuard s{};
  MQT_DDSIM_QDMI_Device_Session uninit = nullptr;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&uninit), QDMI_SUCCESS);
  EXPECT_THAT(
      MQT_DDSIM_QDMI_device_session_set_parameter(
          uninit, QDMI_DEVICE_SESSION_PARAMETER_BASEURL, 20,
          "https://example.com"),
      AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED, QDMI_ERROR_INVALIDARGUMENT));
  // After init, setting a parameter returns BADSTATE
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                s.session, QDMI_DEVICE_SESSION_PARAMETER_BASEURL, 20,
                "https://example.com"),
            QDMI_ERROR_BADSTATE);
  // Invalid enum
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                s.session, QDMI_DEVICE_SESSION_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  // Null session invalid
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                nullptr, QDMI_DEVICE_SESSION_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  MQT_DDSIM_QDMI_device_session_free(uninit);
}

TEST(SessionParameters, WidthValidationAndImmutability) {
  const qdmi_test::SessionGuard initialized{2};
  MQT_DDSIM_QDMI_Device_Session session = nullptr;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  for (const std::string_view width : {
           "",
           "0",
           "-1",
           "+2",
           "2x",
           " 2",
           "65536",
           "999999999999999999999999999999999",
       }) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                  session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1,
                  width.size() + 1, width.data()),
              QDMI_ERROR_INVALIDARGUMENT);
  }
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1, 1, "2"),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1, 4, "2\0x"),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1, 6, "65535"),
            QDMI_SUCCESS);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_set_parameter(
          initialized.session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1, 2, "3"),
      QDMI_ERROR_BADSTATE);
  MQT_DDSIM_QDMI_device_session_free(session);
}

TEST(SessionParameters, WidthScopesSitesIndependently) {
  const qdmi_test::SessionGuard first{1};
  const qdmi_test::SessionGuard second{3};
  const auto sites = qdmi_test::querySites(second.session);
  ASSERT_EQ(sites.size(), 3);
  EXPECT_EQ(qdmi_test::querySites(first.session).size(), 1);
  size_t width = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                first.session, QDMI_DEVICE_PROPERTY_QUBITSNUM, sizeof(width),
                &width, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(width, 1);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                first.session, sites.back(), QDMI_SITE_PROPERTY_INDEX,
                sizeof(width), &width, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                second.session, sites.back(), QDMI_SITE_PROPERTY_INDEX,
                sizeof(width), &width, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(width, 2);
}

TEST(SessionParameters, WidthBoundsQASMAndStaticAndDynamicQIR) {
  for (const auto width : {1U, 2U}) {
    const qdmi_test::SessionGuard session{width};
    for (const auto shots : {0U, 8U}) {
      for (const auto& [format, code] : {
               std::pair{QDMI_PROGRAM_FORMAT_QASM3,
                         std::string(qdmi_test::QASM3_BELL_SAMPLING)},
               std::pair{QDMI_PROGRAM_FORMAT_QIRBASESTRING,
                         qdmi_test::getQIRProgram("BellPairStatic.ll")},
               std::pair{QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
                         qdmi_test::getQIRProgram("BellPairDynamic.ll")},
           }) {
        const qdmi_test::JobGuard job{session.session};
        ASSERT_EQ(qdmi_test::setProgram(job.job, format, code), QDMI_SUCCESS);
        ASSERT_EQ(qdmi_test::setShots(job.job, shots), QDMI_SUCCESS);
        ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
        QDMI_Job_Status status{};
        ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(job.job, &status),
                  QDMI_SUCCESS);
        EXPECT_EQ(status,
                  width == 1 ? QDMI_JOB_STATUS_FAILED : QDMI_JOB_STATUS_DONE)
            << "format=" << format << ", shots=" << shots;
      }
    }
  }
}
