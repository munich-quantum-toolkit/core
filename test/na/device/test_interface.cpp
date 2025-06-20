/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt_na_qdmi/device.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

class QDMISpecificationTest : public ::testing::Test {
protected:
  MQT_NA_QDMI_Device_Session session = nullptr;

  void SetUp() override {
    ASSERT_EQ(MQT_NA_QDMI_device_initialize(), QDMI_SUCCESS)
        << "Failed to initialize the device";

    ASSERT_EQ(MQT_NA_QDMI_device_session_alloc(&session), QDMI_SUCCESS)
        << "Failed to allocate a session";

    ASSERT_EQ(MQT_NA_QDMI_device_session_init(session), QDMI_SUCCESS)
        << "Failed to initialize a session. Potential errors: Wrong or missing "
           "authentication information, device status is offline, or in "
           "maintenance. To provide credentials, take a look in " __FILE__
        << (__LINE__ - 4);
  }

  void TearDown() override { MQT_NA_QDMI_device_finalize(); }
};

TEST_F(QDMISpecificationTest, SessionSetParameter) {
  ASSERT_EQ(MQT_NA_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMISpecificationTest, JobCreate) {
  MQT_NA_QDMI_Device_Job job = nullptr;
  ASSERT_NE(MQT_NA_QDMI_device_session_create_device_job(session, &job),
            QDMI_ERROR_NOTIMPLEMENTED);
  MQT_NA_QDMI_device_job_free(job);
}

TEST_F(QDMISpecificationTest, JobSetParameter) {
  MQT_NA_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_NA_QDMI_device_session_create_device_job(session, &job),
            QDMI_ERROR_NOTSUPPORTED);
  ASSERT_EQ(MQT_NA_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  MQT_NA_QDMI_device_job_free(job);
}

TEST_F(QDMISpecificationTest, JobQueryProperty) {
  MQT_NA_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_NA_QDMI_device_session_create_device_job(session, &job),
            QDMI_ERROR_NOTSUPPORTED);
  ASSERT_EQ(MQT_NA_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  MQT_NA_QDMI_device_job_free(job);
}

TEST_F(QDMISpecificationTest, JobSubmit) {
  MQT_NA_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_NA_QDMI_device_session_create_device_job(session, &job),
            QDMI_ERROR_NOTSUPPORTED);
  ASSERT_NE(MQT_NA_QDMI_device_job_submit(job), QDMI_ERROR_NOTIMPLEMENTED);
  MQT_NA_QDMI_device_job_free(job);
}

TEST_F(QDMISpecificationTest, JobCancel) {
  MQT_NA_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_NA_QDMI_device_session_create_device_job(session, &job),
            QDMI_ERROR_NOTSUPPORTED);
  ASSERT_NE(MQT_NA_QDMI_device_job_cancel(job), QDMI_ERROR_NOTIMPLEMENTED);
  MQT_NA_QDMI_device_job_free(job);
}

TEST_F(QDMISpecificationTest, JobCheck) {
  MQT_NA_QDMI_Device_Job job = nullptr;
  QDMI_Job_Status status = QDMI_JOB_STATUS_RUNNING;
  ASSERT_EQ(MQT_NA_QDMI_device_session_create_device_job(session, &job),
            QDMI_ERROR_NOTSUPPORTED);
  ASSERT_NE(MQT_NA_QDMI_device_job_check(job, &status),
            QDMI_ERROR_NOTIMPLEMENTED);
  MQT_NA_QDMI_device_job_free(job);
}

TEST_F(QDMISpecificationTest, JobWait) {
  MQT_NA_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_NA_QDMI_device_session_create_device_job(session, &job),
            QDMI_ERROR_NOTSUPPORTED);
  ASSERT_NE(MQT_NA_QDMI_device_job_wait(job, 0), QDMI_ERROR_NOTIMPLEMENTED);
  MQT_NA_QDMI_device_job_free(job);
}

TEST_F(QDMISpecificationTest, JobGetResults) {
  MQT_NA_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_NA_QDMI_device_session_create_device_job(session, &job),
            QDMI_ERROR_NOTSUPPORTED);
  ASSERT_EQ(MQT_NA_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_MAX, 0,
                                               nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  MQT_NA_QDMI_device_job_free(job);
}

TEST_F(QDMISpecificationTest, QueryDeviceProperty) {
  ASSERT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                nullptr, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMISpecificationTest, QuerySiteProperty) {
  ASSERT_EQ(MQT_NA_QDMI_device_session_query_site_property(
                nullptr, nullptr, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMISpecificationTest, QueryOperationProperty) {
  ASSERT_EQ(MQT_NA_QDMI_device_session_query_operation_property(
                nullptr, nullptr, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMISpecificationTest, QueryDeviceName) {
  size_t size = 0;
  ASSERT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a name";
  std::string value(size - 1, '\0');
  ASSERT_EQ(
      MQT_NA_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_NAME, size, value.data(), nullptr),
      QDMI_SUCCESS)
      << "Devices must provide a name";
  ASSERT_FALSE(value.empty()) << "Devices must provide a name";
}

TEST_F(QDMISpecificationTest, QueryDeviceVersion) {
  size_t size = 0;
  ASSERT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_VERSION, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a version";
  std::string value(size - 1, '\0');
  ASSERT_EQ(
      MQT_NA_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_VERSION, size, value.data(), nullptr),
      QDMI_SUCCESS)
      << "Devices must provide a version";
  ASSERT_FALSE(value.empty()) << "Devices must provide a version";
}

TEST_F(QDMISpecificationTest, QueryDeviceLibraryVersion) {
  size_t size = 0;
  ASSERT_EQ(
      MQT_NA_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, 0, nullptr, &size),
      QDMI_SUCCESS)
      << "Devices must provide a library version";
  std::string value(size - 1, '\0');
  ASSERT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, size,
                value.data(), nullptr),
            QDMI_SUCCESS)
      << "Devices must provide a library version";
  ASSERT_FALSE(value.empty()) << "Devices must provide a library version";
}

TEST_F(QDMISpecificationTest, QuerySiteIndex) {
  size_t size = 0;
  ASSERT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_SITES, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a list of sites";
  std::vector<MQT_NA_QDMI_Site> sites(size / sizeof(MQT_NA_QDMI_Site));
  ASSERT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_SITES, size,
                static_cast<void*>(sites.data()), nullptr),
            QDMI_SUCCESS)
      << "Devices must provide a list of sites";
  size_t id = 0;
  for (auto* site : sites) {
    ASSERT_EQ(MQT_NA_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_INDEX, sizeof(size_t), &id,
                  nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a site id";
  }
}

TEST_F(QDMISpecificationTest, QueryDeviceQubitNum) {
  size_t numQubits = 0;
  EXPECT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_QUBITSNUM, sizeof(size_t),
                &numQubits, nullptr),
            QDMI_SUCCESS);
}
