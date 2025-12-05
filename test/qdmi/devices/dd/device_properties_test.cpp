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
#include "mqt_ddsim_qdmi/device.h"

#include <cstddef>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <qdmi/constants.h>
#include <string>
#include <vector>

using testing::AnyOf;

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

  const auto sites = qdmi_test::querySites(s.session);
  EXPECT_FALSE(sites.empty());

  const auto ops = qdmi_test::queryOperations(s.session);
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

TEST(SiteProperties, IndexAvailable) {
  const qdmi_test::SessionGuard s{};
  for (const auto sites = qdmi_test::querySites(s.session);
       auto* const site : sites) {
    size_t idx = 0;
    EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                  s.session, site, QDMI_SITE_PROPERTY_INDEX, sizeof(size_t),
                  &idx, nullptr),
              QDMI_SUCCESS);
    EXPECT_LT(idx, sites.size());
    // Expect the name property to not be supported
    EXPECT_EQ(
        MQT_DDSIM_QDMI_device_session_query_site_property(
            s.session, site, QDMI_SITE_PROPERTY_NAME, 0, nullptr, nullptr),
        QDMI_ERROR_NOTSUPPORTED);
  }
}

TEST(OperationProperties, BasicQueries) {
  const qdmi_test::SessionGuard s{};
  const auto sites = qdmi_test::querySites(s.session);
  for (const auto ops = qdmi_test::queryOperations(s.session);
       auto* const op : ops) {
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

    auto fidelity = 0.0;
    ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                  s.session, op, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(double), &fidelity,
                  nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(fidelity, 1.0);

    size_t nparams = 0;
    ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                  s.session, op, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sizeof(size_t),
                  &nparams, nullptr),
              QDMI_SUCCESS);
    EXPECT_GE(nparams, 0U);

    const auto rc = MQT_DDSIM_QDMI_device_session_query_operation_property(
        s.session, op, 0, nullptr, 0, nullptr,
        QDMI_OPERATION_PROPERTY_QUBITSNUM, 0, nullptr, nullptr);
    EXPECT_THAT(rc, AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
    if (rc == QDMI_SUCCESS) {
      size_t nqubits = 0;
      ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                    s.session, op, 0, nullptr, 0, nullptr,
                    QDMI_OPERATION_PROPERTY_QUBITSNUM, sizeof(size_t), &nqubits,
                    nullptr),
                QDMI_SUCCESS);
      EXPECT_GE(nqubits, 0U);

      if (nqubits > 0) {
        // get as many sites as needed
        std::vector<MQT_DDSIM_QDMI_Site> opSites{nqubits};
        for (size_t i = 0; i < nqubits; ++i) {
          opSites[i] = sites[i];
        }

        // query the fidelity with those sites
        ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                      s.session, op, nqubits, opSites.data(), 0, nullptr,
                      QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(double),
                      &fidelity, nullptr),
                  QDMI_SUCCESS);
        EXPECT_EQ(fidelity, 1.0);

        // remove one site and query again, which should be invalid
        opSites.pop_back();
        EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                      s.session, op, opSites.size(), opSites.data(), 0, nullptr,
                      QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(double),
                      &fidelity, nullptr),
                  QDMI_ERROR_INVALIDARGUMENT);
      }
    }
  }
}
