/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt_sc_qdmi/device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
// NOLINTNEXTLINE(misc-include-cleaner)
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
[[nodiscard]] auto querySites(MQT_SC_QDMI_Device_Session session)
    -> std::vector<MQT_SC_QDMI_Site> {
  size_t size = 0;
  if (MQT_SC_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_SITES, 0, nullptr, &size) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query sites");
  }
  if (size == 0) {
    throw std::runtime_error("No sites available");
  }
  std::vector<MQT_SC_QDMI_Site> sites(size / sizeof(MQT_SC_QDMI_Site));
  if (MQT_SC_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_SITES, size,
          static_cast<void*>(sites.data()), nullptr) != QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query sites");
  }
  return sites;
}
} // namespace

class ScQDMISpecificationTest : public ::testing::Test {
protected:
  MQT_SC_QDMI_Device_Session session = nullptr;

  void SetUp() override {
    ASSERT_EQ(MQT_SC_QDMI_device_initialize(), QDMI_SUCCESS)
        << "Failed to initialize the device";

    ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&session), QDMI_SUCCESS)
        << "Failed to allocate a session";

    ASSERT_EQ(MQT_SC_QDMI_device_session_init(session), QDMI_SUCCESS)
        << "Failed to initialize a session. Potential errors: Wrong or missing "
           "authentication information, device status is offline, or in "
           "maintenance. To provide credentials, take a look in " __FILE__
        << (__LINE__ - 4);
  }

  void TearDown() override {
    if (session != nullptr) {
      MQT_SC_QDMI_device_session_free(session);
      session = nullptr;
    }
    MQT_SC_QDMI_device_finalize();
  }
};

class ScQDMIJobSpecificationTest : public ScQDMISpecificationTest {
protected:
  MQT_SC_QDMI_Device_Job job = nullptr;

  void SetUp() override {
    ScQDMISpecificationTest::SetUp();
    ASSERT_EQ(MQT_SC_QDMI_device_session_create_device_job(session, &job),
              QDMI_SUCCESS)
        << "Failed to create a device job.";
  }

  void TearDown() override {
    if (job != nullptr) {
      MQT_SC_QDMI_device_job_free(job);
      job = nullptr;
    }
    ScQDMISpecificationTest::TearDown();
  }
};

TEST_F(ScQDMISpecificationTest, SessionAlloc) {
  EXPECT_EQ(MQT_SC_QDMI_device_session_alloc(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMISpecificationTest, SessionInit) {
  EXPECT_EQ(MQT_SC_QDMI_device_session_init(session), QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_SC_QDMI_device_session_init(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMISpecificationTest, SessionSetParameter) {
  MQT_SC_QDMI_Device_Session uninitializedSession = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&uninitializedSession),
            QDMI_SUCCESS);
  EXPECT_THAT(MQT_SC_QDMI_device_session_set_parameter(
                  uninitializedSession, QDMI_DEVICE_SESSION_PARAMETER_BASEURL,
                  20, "https://example.com"),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED,
                             QDMI_ERROR_INVALIDARGUMENT));
  EXPECT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_BASEURL, 20,
                "https://example.com"),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMISpecificationTest, JobCreate) {
  MQT_SC_QDMI_Device_Session uninitializedSession = nullptr;
  MQT_SC_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&uninitializedSession),
            QDMI_SUCCESS);
  EXPECT_EQ(
      MQT_SC_QDMI_device_session_create_device_job(uninitializedSession, &job),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_SC_QDMI_device_session_create_device_job(session, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_SC_QDMI_device_session_create_device_job(nullptr, &job),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(MQT_SC_QDMI_device_session_create_device_job(session, &job),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  MQT_SC_QDMI_device_job_free(job);
  MQT_SC_QDMI_device_session_free(uninitializedSession);
}

TEST_F(ScQDMISpecificationTest, JobSetParameter) {
  EXPECT_EQ(MQT_SC_QDMI_device_job_set_parameter(
                nullptr, QDMI_DEVICE_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMIJobSpecificationTest, JobSetParameter) {
  QDMI_Program_Format value = QDMI_PROGRAM_FORMAT_QASM2;
  EXPECT_THAT(MQT_SC_QDMI_device_job_set_parameter(
                  job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                  sizeof(QDMI_Program_Format), &value),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  EXPECT_EQ(MQT_SC_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMISpecificationTest, JobQueryProperty) {
  EXPECT_EQ(MQT_SC_QDMI_device_job_query_property(
                nullptr, QDMI_DEVICE_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMIJobSpecificationTest, JobQueryProperty) {
  EXPECT_THAT(MQT_SC_QDMI_device_job_query_property(
                  job, QDMI_DEVICE_JOB_PROPERTY_ID, 0, nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  EXPECT_EQ(MQT_SC_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMIJobSpecificationTest, QueryJobId) {
  size_t size = 0;
  const auto status = MQT_SC_QDMI_device_job_query_property(
      job, QDMI_DEVICE_JOB_PROPERTY_ID, 0, nullptr, &size);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  if (status == QDMI_ERROR_NOTSUPPORTED) {
    GTEST_SKIP() << "Job ID property is not supported by the device";
  }
  ASSERT_GT(size, 0);
  std::string id(size - 1, '\0');
  EXPECT_THAT(MQT_SC_QDMI_device_job_query_property(
                  job, QDMI_DEVICE_JOB_PROPERTY_ID, size, id.data(), nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(ScQDMISpecificationTest, JobSubmit) {
  EXPECT_EQ(MQT_SC_QDMI_device_job_submit(nullptr), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMIJobSpecificationTest, JobSubmit) {
  const auto status = MQT_SC_QDMI_device_job_submit(job);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(ScQDMISpecificationTest, JobCancel) {
  EXPECT_EQ(MQT_SC_QDMI_device_job_cancel(nullptr), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMIJobSpecificationTest, JobCancel) {
  const auto status = MQT_SC_QDMI_device_job_cancel(job);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_INVALIDARGUMENT,
                                     QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(ScQDMISpecificationTest, JobCheck) {
  EXPECT_EQ(MQT_SC_QDMI_device_job_check(nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMIJobSpecificationTest, JobCheck) {
  QDMI_Job_Status jobStatus = QDMI_JOB_STATUS_RUNNING;
  const auto status = MQT_SC_QDMI_device_job_check(job, &jobStatus);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(ScQDMISpecificationTest, JobWait) {
  EXPECT_EQ(MQT_SC_QDMI_device_job_wait(nullptr, 0),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMIJobSpecificationTest, JobWait) {
  const auto status = MQT_SC_QDMI_device_job_wait(job, 1);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED,
                                     QDMI_ERROR_TIMEOUT));
}

TEST_F(ScQDMISpecificationTest, JobGetResults) {
  EXPECT_EQ(MQT_SC_QDMI_device_job_get_results(nullptr, QDMI_JOB_RESULT_MAX, 0,
                                               nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMIJobSpecificationTest, JobGetResults) {
  EXPECT_THAT(MQT_SC_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_SHOTS, 0,
                                                 nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  EXPECT_EQ(MQT_SC_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_MAX, 0,
                                               nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(ScQDMISpecificationTest, QueryDeviceProperty) {
  MQT_SC_QDMI_Device_Session uninitializedSession = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&uninitializedSession),
            QDMI_SUCCESS);
  EXPECT_EQ(
      MQT_SC_QDMI_device_session_query_device_property(
          uninitializedSession, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_SC_QDMI_device_session_query_device_property(
                nullptr, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_SC_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_SC_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_COUPLINGMAP, 0, nullptr, nullptr),
            QDMI_SUCCESS);
}

TEST_F(ScQDMISpecificationTest, QuerySiteProperty) {
  MQT_SC_QDMI_Site site = querySites(session).front();
  EXPECT_EQ(
      MQT_SC_QDMI_device_session_query_site_property(
          session, nullptr, QDMI_SITE_PROPERTY_INDEX, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_SC_QDMI_device_session_query_site_property(
                nullptr, site, QDMI_SITE_PROPERTY_INDEX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_SC_QDMI_device_session_query_site_property(
                session, site, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(MQT_SC_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_NAME, 0, nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(ScQDMISpecificationTest, QueryDeviceName) {
  size_t size = 0;
  ASSERT_EQ(MQT_SC_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a name";
  std::string value(size - 1, '\0');
  ASSERT_EQ(
      MQT_SC_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_NAME, size, value.data(), nullptr),
      QDMI_SUCCESS)
      << "Devices must provide a name";
  EXPECT_FALSE(value.empty()) << "Devices must provide a name";
}

TEST_F(ScQDMISpecificationTest, QueryDeviceVersion) {
  size_t size = 0;
  ASSERT_EQ(MQT_SC_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_VERSION, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a version";
  std::string value(size - 1, '\0');
  ASSERT_EQ(
      MQT_SC_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_VERSION, size, value.data(), nullptr),
      QDMI_SUCCESS)
      << "Devices must provide a version";
  EXPECT_FALSE(value.empty()) << "Devices must provide a version";
}

TEST_F(ScQDMISpecificationTest, QueryDeviceLibraryVersion) {
  size_t size = 0;
  ASSERT_EQ(
      MQT_SC_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, 0, nullptr, &size),
      QDMI_SUCCESS)
      << "Devices must provide a library version";
  std::string value(size - 1, '\0');
  ASSERT_EQ(MQT_SC_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, size,
                value.data(), nullptr),
            QDMI_SUCCESS)
      << "Devices must provide a library version";
  EXPECT_FALSE(value.empty()) << "Devices must provide a library version";
}

TEST_F(ScQDMISpecificationTest, QuerySiteIndex) {
  size_t id = 0;
  EXPECT_NO_THROW(for (auto* site : querySites(session)) {
    EXPECT_EQ(MQT_SC_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_INDEX, sizeof(size_t), &id,
                  nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a site id";
  }) << "Devices must provide a list of sites";
}

TEST_F(ScQDMISpecificationTest, QueryDeviceQubitNum) {
  size_t numQubits = 0;
  EXPECT_EQ(MQT_SC_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_QUBITSNUM, sizeof(size_t),
                &numQubits, nullptr),
            QDMI_SUCCESS);
}
