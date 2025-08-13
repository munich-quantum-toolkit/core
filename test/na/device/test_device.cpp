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
#include <cstdint>
#include <fstream>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace testing {
namespace {
auto stringConcat5(const std::string& a, const std::string& b,
                   const std::string& c, const std::string& d,
                   const std::string& e) -> std::string {
  std::stringstream ss;
  ss << a << b << c << d << e;
  return ss.str();
}
// NOLINTBEGIN(readability-identifier-naming,cppcoreguidelines-avoid-const-or-ref-data-members)
MATCHER_P2(IsBetween, a, b,
           stringConcat5(negation ? "isn't" : "is", " between ",
                         PrintToString(a), " and ", PrintToString(b))) {
  return a <= arg && arg <= b;
}
// NOLINTEND(readability-identifier-naming,cppcoreguidelines-avoid-const-or-ref-data-members)
} // namespace
} // namespace testing

namespace {
[[nodiscard]] auto querySites(MQT_NA_QDMI_Device_Session session)
    -> std::vector<MQT_NA_QDMI_Site> {
  size_t size = 0;
  if (MQT_NA_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_SITES, 0, nullptr, &size) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query sites");
  }
  if (size == 0) {
    throw std::runtime_error("No sites available");
  }
  std::vector<MQT_NA_QDMI_Site> sites(size / sizeof(MQT_NA_QDMI_Site));
  if (MQT_NA_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_SITES, size,
          static_cast<void*>(sites.data()), nullptr) != QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query sites");
  }
  return sites;
}
[[nodiscard]] auto queryOperations(MQT_NA_QDMI_Device_Session session)
    -> std::vector<MQT_NA_QDMI_Operation> {
  size_t size = 0;
  if (MQT_NA_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_OPERATIONS, 0, nullptr, &size) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query operations");
  }
  if (size == 0) {
    throw std::runtime_error("No operations available");
  }
  std::vector<MQT_NA_QDMI_Operation> operations(size /
                                                sizeof(MQT_NA_QDMI_Operation));
  if (MQT_NA_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_OPERATIONS, size,
          static_cast<void*>(operations.data()), nullptr) != QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query operations");
  }
  return operations;
}
} // namespace

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

  void TearDown() override {
    if (session != nullptr) {
      MQT_NA_QDMI_device_session_free(session);
      session = nullptr;
    }
    MQT_NA_QDMI_device_finalize();
  }
};

class QDMIJobSpecificationTest : public QDMISpecificationTest {
protected:
  MQT_NA_QDMI_Device_Job job = nullptr;

  void SetUp() override {
    QDMISpecificationTest::SetUp();
    ASSERT_THAT(MQT_NA_QDMI_device_session_create_device_job(session, &job),
                testing::AnyOf(QDMI_SUCCESS))
        << "Failed to create a device job";
  }

  void TearDown() override {
    if (job != nullptr) {
      MQT_NA_QDMI_device_job_free(job);
      job = nullptr;
    }
    QDMISpecificationTest::TearDown();
  }
};

TEST_F(QDMISpecificationTest, SessionAlloc) {
  EXPECT_EQ(MQT_NA_QDMI_device_session_alloc(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMISpecificationTest, SessionInit) {
  EXPECT_EQ(MQT_NA_QDMI_device_session_init(session), QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_NA_QDMI_device_session_init(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMISpecificationTest, SessionSetParameter) {
  MQT_NA_QDMI_Device_Session uninitializedSession = nullptr;
  ASSERT_EQ(MQT_NA_QDMI_device_session_alloc(&uninitializedSession),
            QDMI_SUCCESS);
  EXPECT_THAT(MQT_NA_QDMI_device_session_set_parameter(
                  uninitializedSession, QDMI_DEVICE_SESSION_PARAMETER_BASEURL,
                  20, "https://example.com"),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED,
                             QDMI_ERROR_INVALIDARGUMENT));
  EXPECT_EQ(MQT_NA_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_BASEURL, 20,
                "https://example.com"),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_NA_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMISpecificationTest, JobCreate) {
  MQT_NA_QDMI_Device_Session uninitializedSession = nullptr;
  MQT_NA_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_NA_QDMI_device_session_alloc(&uninitializedSession),
            QDMI_SUCCESS);
  EXPECT_EQ(
      MQT_NA_QDMI_device_session_create_device_job(uninitializedSession, &job),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_NA_QDMI_device_session_create_device_job(session, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_NA_QDMI_device_session_create_device_job(nullptr, &job),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(MQT_NA_QDMI_device_session_create_device_job(session, &job),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  MQT_NA_QDMI_device_job_free(job);
}

TEST_F(QDMISpecificationTest, JobSetParameter) {
  EXPECT_EQ(MQT_NA_QDMI_device_job_set_parameter(
                nullptr, QDMI_DEVICE_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMIJobSpecificationTest, JobSetParameter) {
  QDMI_Program_Format value = QDMI_PROGRAM_FORMAT_QASM2;
  EXPECT_THAT(MQT_NA_QDMI_device_job_set_parameter(
                  job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                  sizeof(QDMI_Program_Format), &value),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  EXPECT_EQ(MQT_NA_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMISpecificationTest, JobQueryProperty) {
  EXPECT_EQ(MQT_NA_QDMI_device_job_query_property(
                nullptr, QDMI_DEVICE_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMIJobSpecificationTest, JobQueryProperty) {
  EXPECT_THAT(MQT_NA_QDMI_device_job_query_property(
                  job, QDMI_DEVICE_JOB_PROPERTY_ID, 0, nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  EXPECT_EQ(MQT_NA_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMIJobSpecificationTest, QueryJobId) {
  size_t size = 0;
  const auto status = MQT_NA_QDMI_device_job_query_property(
      job, QDMI_DEVICE_JOB_PROPERTY_ID, 0, nullptr, &size);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  if (status == QDMI_ERROR_NOTSUPPORTED) {
    GTEST_SKIP() << "Job ID property is not supported by the device";
  }
  ASSERT_GT(size, 0);
  std::string id(size - 1, '\0');
  EXPECT_THAT(MQT_NA_QDMI_device_job_query_property(
                  job, QDMI_DEVICE_JOB_PROPERTY_ID, size, id.data(), nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(QDMISpecificationTest, JobSubmit) {
  EXPECT_EQ(MQT_NA_QDMI_device_job_submit(nullptr), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMIJobSpecificationTest, JobSubmit) {
  const auto status = MQT_NA_QDMI_device_job_submit(job);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(QDMISpecificationTest, JobCancel) {
  EXPECT_EQ(MQT_NA_QDMI_device_job_cancel(nullptr), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMIJobSpecificationTest, JobCancel) {
  const auto status = MQT_NA_QDMI_device_job_cancel(job);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_INVALIDARGUMENT,
                                     QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(QDMISpecificationTest, JobCheck) {
  EXPECT_EQ(MQT_NA_QDMI_device_job_check(nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMIJobSpecificationTest, JobCheck) {
  QDMI_Job_Status jobStatus = QDMI_JOB_STATUS_RUNNING;
  const auto status = MQT_NA_QDMI_device_job_check(job, &jobStatus);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(QDMISpecificationTest, JobWait) {
  EXPECT_EQ(MQT_NA_QDMI_device_job_wait(nullptr, 0),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMIJobSpecificationTest, JobWait) {
  const auto status = MQT_NA_QDMI_device_job_wait(job, 1);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED,
                                     QDMI_ERROR_TIMEOUT));
}

TEST_F(QDMISpecificationTest, JobGetResults) {
  EXPECT_EQ(MQT_NA_QDMI_device_job_get_results(nullptr, QDMI_JOB_RESULT_MAX, 0,
                                               nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMIJobSpecificationTest, JobGetResults) {
  EXPECT_THAT(MQT_NA_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_SHOTS, 0,
                                                 nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  EXPECT_EQ(MQT_NA_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_MAX, 0,
                                               nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(QDMISpecificationTest, QueryDeviceProperty) {
  MQT_NA_QDMI_Device_Session uninitializedSession = nullptr;
  ASSERT_EQ(MQT_NA_QDMI_device_session_alloc(&uninitializedSession),
            QDMI_SUCCESS);
  EXPECT_EQ(
      MQT_NA_QDMI_device_session_query_device_property(
          uninitializedSession, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                nullptr, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(
      MQT_NA_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_COUPLINGMAP, 0, nullptr, nullptr),
      testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(QDMISpecificationTest, QuerySiteProperty) {
  MQT_NA_QDMI_Site site = querySites(session).front();
  EXPECT_EQ(
      MQT_NA_QDMI_device_session_query_site_property(
          session, nullptr, QDMI_SITE_PROPERTY_INDEX, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_NA_QDMI_device_session_query_site_property(
                nullptr, site, QDMI_SITE_PROPERTY_INDEX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_NA_QDMI_device_session_query_site_property(
                session, site, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(MQT_NA_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_NAME, 0, nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(QDMISpecificationTest, QueryOperationProperty) {
  MQT_NA_QDMI_Operation operation = queryOperations(session).front();
  EXPECT_EQ(MQT_NA_QDMI_device_session_query_operation_property(
                nullptr, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_NA_QDMI_device_session_query_operation_property(
                session, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(MQT_NA_QDMI_device_session_query_operation_property(
                  session, operation, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_QUBITSNUM, 0, nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
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
  EXPECT_FALSE(value.empty()) << "Devices must provide a name";
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
  EXPECT_FALSE(value.empty()) << "Devices must provide a version";
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
  EXPECT_FALSE(value.empty()) << "Devices must provide a library version";
}

TEST_F(QDMISpecificationTest, QuerySiteIndex) {
  size_t id = 0;
  EXPECT_NO_THROW(for (auto* site : querySites(session)) {
    EXPECT_EQ(MQT_NA_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_INDEX, sizeof(size_t), &id,
                  nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a site id";
  }) << "Devices must provide a list of sites";
}

TEST_F(QDMISpecificationTest, QueryOperationName) {
  size_t nameSize = 0;
  EXPECT_NO_THROW(for (auto* operation : queryOperations(session)) {
    EXPECT_EQ(MQT_NA_QDMI_device_session_query_operation_property(
                  session, operation, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, &nameSize),
              QDMI_SUCCESS)
        << "Devices must provide a operation name";
    std::string name(nameSize - 1, '\0');
    EXPECT_EQ(MQT_NA_QDMI_device_session_query_operation_property(
                  session, operation, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_NAME, nameSize, name.data(), nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a operation name";
  }) << "Devices must provide a list of operations";
}

TEST_F(QDMISpecificationTest, QueryDeviceQubitNum) {
  size_t numQubits = 0;
  EXPECT_EQ(MQT_NA_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_QUBITSNUM, sizeof(size_t),
                &numQubits, nullptr),
            QDMI_SUCCESS);
}

class NADeviceTest : public QDMISpecificationTest {
protected:
  // NOLINTNEXTLINE(misc-include-cleaner)
  nlohmann::json json;

  void SetUp() override {
    QDMISpecificationTest::SetUp();
    // Open the file
    // NOLINTNEXTLINE(misc-include-cleaner)
    std::ifstream file(NA_DEVICE_JSON);
    ASSERT_TRUE(file.is_open()) << "Failed to open json file: " NA_DEVICE_JSON;

    // Parse the JSON file
    try {
      file >> json;
    } catch (const nlohmann::json::parse_error& e) {
      GTEST_FAIL() << "JSON parsing error: " << e.what();
    }
  }

  void TearDown() override { QDMISpecificationTest::TearDown(); }
};

TEST_F(NADeviceTest, QuerySiteData) {
  uint64_t module = 0;
  uint64_t subModule = 0;
  int64_t x = 0;
  int64_t y = 0;
  double t1 = 0;
  double t2 = 0;
  std::vector<MQT_NA_QDMI_Site> sites;
  EXPECT_NO_THROW(sites = querySites(session))
      << "Devices must provide a sites";
  EXPECT_EQ(sites.size(), 100);
  for (auto* site : sites) {
    EXPECT_EQ(MQT_NA_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_MODULEINDEX,
                  sizeof(uint64_t), &module, nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(MQT_NA_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_SUBMODULEINDEX,
                  sizeof(uint64_t), &subModule, nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(MQT_NA_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_XCOORDINATE,
                  sizeof(int64_t), &x, nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(MQT_NA_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_YCOORDINATE,
                  sizeof(int64_t), &y, nullptr),
              QDMI_SUCCESS);
    int64_t originX = 0;
    int64_t width = 0;
    ASSERT_NO_THROW(
        originX =
            json["traps"][module]["extent"]["origin"]["x"].get<int64_t>());
    ASSERT_NO_THROW(
        width =
            json["traps"][module]["extent"]["size"]["width"].get<int64_t>());
    EXPECT_THAT(x, ::testing::IsBetween(originX, originX + width));
    int64_t originY = 0;
    int64_t height = 0;
    ASSERT_NO_THROW(
        originY =
            json["traps"][module]["extent"]["origin"]["y"].get<int64_t>());
    ASSERT_NO_THROW(
        height =
            json["traps"][module]["extent"]["size"]["height"].get<int64_t>());
    EXPECT_THAT(y, ::testing::IsBetween(originY, height));

    EXPECT_EQ(
        MQT_NA_QDMI_device_session_query_site_property(
            session, site, QDMI_SITE_PROPERTY_T1, sizeof(double), &t1, nullptr),
        QDMI_SUCCESS);
    EXPECT_DOUBLE_EQ(t1, json["decoherenceTimes"]["t1"]);
    EXPECT_EQ(
        MQT_NA_QDMI_device_session_query_site_property(
            session, site, QDMI_SITE_PROPERTY_T2, sizeof(double), &t2, nullptr),
        QDMI_SUCCESS);
    EXPECT_DOUBLE_EQ(t2, json["decoherenceTimes"]["t2"]);
  }
}

TEST_F(NADeviceTest, QueryOperationData) {
  double duration = 0;
  double fidelity = 0;
  size_t numQubits = 0;
  size_t numParameters = 0;
  EXPECT_NO_THROW(for (auto* operation : queryOperations(session)) {
    EXPECT_THAT(MQT_NA_QDMI_device_session_query_operation_property(
                    session, operation, 0, nullptr, 0, nullptr,
                    QDMI_OPERATION_PROPERTY_DURATION, sizeof(double), &duration,
                    nullptr),
                testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
    EXPECT_THAT(MQT_NA_QDMI_device_session_query_operation_property(
                    session, operation, 0, nullptr, 0, nullptr,
                    QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(double), &fidelity,
                    nullptr),
                testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
    EXPECT_THAT(MQT_NA_QDMI_device_session_query_operation_property(
                    session, operation, 0, nullptr, 0, nullptr,
                    QDMI_OPERATION_PROPERTY_QUBITSNUM, sizeof(size_t),
                    &numQubits, nullptr),
                testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
    EXPECT_THAT(MQT_NA_QDMI_device_session_query_operation_property(
                    session, operation, 0, nullptr, 0, nullptr,
                    QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sizeof(size_t),
                    &numParameters, nullptr),
                testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  }) << "Devices must provide a list of operations";
}
