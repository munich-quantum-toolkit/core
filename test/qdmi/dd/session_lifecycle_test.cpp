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
 * DDSIM QDMI Device - Session lifecycle and parameters
 */
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/device.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <qdmi/constants.h>

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
