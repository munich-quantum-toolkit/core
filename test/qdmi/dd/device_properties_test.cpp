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
 * DDSIM QDMI Device - Device, Site, and Operation Properties
 */
#include "helpers/test_utils.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using testing::AnyOf;

namespace {

std::vector<MQT_DDSIM_QDMI_Site>
querySites(MQT_DDSIM_QDMI_Device_Session session) {
  size_t size = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_SITES, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::vector<MQT_DDSIM_QDMI_Site> sites(size / sizeof(MQT_DDSIM_QDMI_Site));
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_SITES, size, sites.data(), nullptr),
      QDMI_SUCCESS);
  return sites;
}

std::vector<MQT_DDSIM_QDMI_Operation>
queryOperations(MQT_DDSIM_QDMI_Device_Session session) {
  size_t size = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_OPERATIONS, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::vector<MQT_DDSIM_QDMI_Operation> ops(size /
                                            sizeof(MQT_DDSIM_QDMI_Operation));
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_OPERATIONS, size, ops.data(), nullptr),
      QDMI_SUCCESS);
  return ops;
}

} // namespace

TEST(DeviceProperties, BasicStringsAndSizes) {
  const qdmi_test::SessionGuard s{};

  // Name
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::string name(size > 0 ? size - 1 : 0, '\0');
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          s.session, QDMI_DEVICE_PROPERTY_NAME, size, name.data(), nullptr),
      QDMI_SUCCESS);
  EXPECT_FALSE(name.empty());

  // Version
  size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_VERSION, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::string ver(size > 0 ? size - 1 : 0, '\0');
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          s.session, QDMI_DEVICE_PROPERTY_VERSION, size, ver.data(), nullptr),
      QDMI_SUCCESS);
  EXPECT_FALSE(ver.empty());

  // Library version
  size = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          s.session, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, 0, nullptr, &size),
      QDMI_SUCCESS);
  std::string lver(size > 0 ? size - 1 : 0, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, size,
                lver.data(), nullptr),
            QDMI_SUCCESS);
  EXPECT_FALSE(lver.empty());
}

TEST(DeviceProperties, UnitsAndScales) {
  const qdmi_test::SessionGuard s{};

  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_LENGTHUNIT, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::string len(size > 0 ? size - 1 : 0, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_LENGTHUNIT, size, len.data(),
                nullptr),
            QDMI_SUCCESS);
  EXPECT_THAT(len, AnyOf("nm", "um", "mm"));

  double lscale = 0.0;
  auto rc = MQT_DDSIM_QDMI_device_session_query_device_property(
      s.session, QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR, sizeof(double),
      &lscale, nullptr);
  EXPECT_THAT(rc, AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  if (rc == QDMI_SUCCESS) {
    EXPECT_GT(lscale, 0.0);
  }

  size = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          s.session, QDMI_DEVICE_PROPERTY_DURATIONUNIT, 0, nullptr, &size),
      QDMI_SUCCESS);
  std::string dur(size > 0 ? size - 1 : 0, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_DURATIONUNIT, size, dur.data(),
                nullptr),
            QDMI_SUCCESS);
  EXPECT_THAT(dur, AnyOf("ns", "us", "ms"));

  double dscale = 0.0;
  rc = MQT_DDSIM_QDMI_device_session_query_device_property(
      s.session, QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR, sizeof(double),
      &dscale, nullptr);
  EXPECT_THAT(rc, AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  if (rc == QDMI_SUCCESS) {
    EXPECT_GT(dscale, 0.0);
  }
}

TEST(DeviceProperties, SitesAndOperationsLists) {
  const qdmi_test::SessionGuard s{};

  auto sites = querySites(s.session);
  EXPECT_FALSE(sites.empty());

  auto ops = queryOperations(s.session);
  EXPECT_FALSE(ops.empty());
}

TEST(DeviceProperties, QubitsNumAvailable) {
  const qdmi_test::SessionGuard s{};
  size_t nq = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_QUBITSNUM, sizeof(size_t), &nq,
                nullptr),
            QDMI_SUCCESS);
  EXPECT_GT(nq, 0U);
}

TEST(DeviceProperties, InvalidAndCustomProperties) {
  const qdmi_test::SessionGuard s{};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                nullptr, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  // Custom properties not supported
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                s.session, QDMI_DEVICE_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST(SiteProperties, BasicAndCustom) {
  const qdmi_test::SessionGuard s{};
  const auto sites = querySites(s.session);
  const auto site = sites.front();

  size_t idx = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                s.session, site, QDMI_SITE_PROPERTY_INDEX, sizeof(size_t), &idx,
                nullptr),
            QDMI_SUCCESS);

  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          s.session, site, QDMI_SITE_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
      QDMI_ERROR_NOTSUPPORTED);
}

TEST(OperationProperties, BasicQueries) {
  const qdmi_test::SessionGuard s{};
  const auto ops = queryOperations(s.session);
  const auto op = ops.front();

  size_t nameSize = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, op, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, &nameSize),
            QDMI_SUCCESS);
  std::string name(nameSize > 0 ? nameSize - 1 : 0, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, op, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_NAME, nameSize, name.data(), nullptr),
            QDMI_SUCCESS);

  double fidelity = 0.0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, op, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(double), &fidelity,
                nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(fidelity, 1.0);

  size_t nparams = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                s.session, op, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sizeof(size_t), &nparams,
                nullptr),
            QDMI_SUCCESS);
  EXPECT_GE(nparams, 0U);

  auto rc = MQT_DDSIM_QDMI_device_session_query_operation_property(
      s.session, op, 0, nullptr, 0, nullptr, QDMI_OPERATION_PROPERTY_QUBITSNUM,
      0, nullptr, nullptr);
  EXPECT_THAT(rc, AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}
