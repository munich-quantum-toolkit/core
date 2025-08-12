/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Driver.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <qdmi/client.h>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace qc {
class DriverTest : public ::testing::TestWithParam<std::string> {
protected:
  QDMI_Session session = nullptr;
  QDMI_Device device = nullptr;

  static void SetUpTestSuite() {
    qdmi::Driver::get().addDynamicDeviceLibrary(DYN_DEV_LIB, "MQT_NA_DYN");
  }

  void SetUp() override {
    const auto& deviceName = GetParam();

    ASSERT_EQ(QDMI_session_alloc(&session), QDMI_SUCCESS)
        << "Failed to allocate session.";

    ASSERT_EQ(QDMI_session_init(session), QDMI_SUCCESS)
        << "Failed to initialize session.";

    size_t devicesSize = 0;
    ASSERT_EQ(QDMI_session_query_session_property(session,
                                                  QDMI_SESSION_PROPERTY_DEVICES,
                                                  0, nullptr, &devicesSize),
              QDMI_SUCCESS)
        << "Failed to retrieve number of devices.";
    std::vector<QDMI_Device> devices(devicesSize / sizeof(QDMI_Device));
    ASSERT_EQ(QDMI_session_query_session_property(
                  session, QDMI_SESSION_PROPERTY_DEVICES, devicesSize,
                  static_cast<void*>(devices.data()), nullptr),
              QDMI_SUCCESS)
        << "Failed to retrieve devices.";

    for (auto* const dev : devices) {
      size_t namesSize = 0;
      ASSERT_EQ(QDMI_device_query_device_property(
                    dev, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, &namesSize),
                QDMI_SUCCESS)
          << "Failed to retrieve the length of the device's name.";
      std::string name(namesSize - 1, '\0');
      ASSERT_EQ(
          QDMI_device_query_device_property(dev, QDMI_DEVICE_PROPERTY_NAME,
                                            namesSize, name.data(), nullptr),
          QDMI_SUCCESS)
          << "Failed to retrieve the device's name.";

      ASSERT_FALSE(name.empty()) << "Device must provide a non-empty name.";

      if (name == deviceName) {
        device = dev;
        return;
      }
    }
    FAIL() << "Device with name '" << deviceName
           << "' not found in the session.";
  }

  void TearDown() override { QDMI_session_free(session); }
};

TEST_P(DriverTest, SessionSetParameterImplemented) {
  EXPECT_EQ(QDMI_session_set_parameter(session, QDMI_SESSION_PARAMETER_MAX, 0,
                                       nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_session_set_parameter`.";
}

TEST_P(DriverTest, JobCreateImplemented) {
  QDMI_Job job = nullptr;
  EXPECT_EQ(QDMI_device_create_job(device, &job), QDMI_SUCCESS)
      << "Devices must implement `QDMI_device_create_job`.";
  QDMI_job_free(job);
}

TEST_P(DriverTest, JobSetParameterImplemented) {
  QDMI_Job job = nullptr;
  EXPECT_EQ(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_job_set_parameter`.";
}

TEST_P(DriverTest, JobQueryPropertyImplemented) {
  QDMI_Job job = nullptr;
  EXPECT_EQ(
      QDMI_job_query_property(job, QDMI_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_job_set_parameter`.";
}

TEST_P(DriverTest, JobSubmitImplemented) {
  QDMI_Job job = nullptr;
  EXPECT_EQ(QDMI_job_submit(job), QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_job_submit`.";
}

TEST_P(DriverTest, JobCancelImplemented) {
  QDMI_Job job = nullptr;
  EXPECT_EQ(QDMI_job_cancel(job), QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_job_cancel`.";
}

TEST_P(DriverTest, JobCheckImplemented) {
  QDMI_Job job = nullptr;
  QDMI_Job_Status status = QDMI_JOB_STATUS_RUNNING;
  EXPECT_EQ(QDMI_job_check(job, &status), QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_job_check`.";
}

TEST_P(DriverTest, JobWaitImplemented) {
  QDMI_Job job = nullptr;
  EXPECT_EQ(QDMI_job_wait(job, 0), QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_job_wait`.";
}

TEST_P(DriverTest, JobGetResultsImplemented) {
  QDMI_Job job = nullptr;
  EXPECT_EQ(QDMI_job_get_results(job, QDMI_JOB_RESULT_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_job_get_results`.";
}

TEST_P(DriverTest, QueryDevicePropertyImplemented) {
  EXPECT_EQ(QDMI_device_query_device_property(device, QDMI_DEVICE_PROPERTY_MAX,
                                              0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_device_query_device_property`.";
}

TEST_P(DriverTest, QuerySitePropertyImplemented) {
  EXPECT_EQ(QDMI_device_query_site_property(
                device, nullptr, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_device_query_site_property`.";
}

TEST_P(DriverTest, QueryOperationPropertyImplemented) {
  EXPECT_EQ(QDMI_device_query_operation_property(
                device, nullptr, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "Devices must implement `QDMI_device_query_operation_property`.";
}

TEST_P(DriverTest, QueryDeviceVersion) {
  size_t size = 0;
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_VERSION, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a version.";
  std::string value(size - 1, '\0');
  ASSERT_EQ(QDMI_device_query_device_property(device,
                                              QDMI_DEVICE_PROPERTY_VERSION,
                                              size, value.data(), nullptr),
            QDMI_SUCCESS)
      << "Devices must provide a version.";
  EXPECT_FALSE(value.empty()) << "Devices must provide a version.";
}

TEST_P(DriverTest, QueryDeviceLibraryVersion) {
  size_t size = 0;
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a library version.";
  std::string value(size - 1, '\0');
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, size, value.data(),
                nullptr),
            QDMI_SUCCESS)
      << "Devices must provide a library version.";
  ASSERT_FALSE(value.empty()) << "Devices must provide a library version.";
}

TEST_P(DriverTest, QueryNumQubits) {
  size_t numQubits = 0;
  ASSERT_EQ(
      QDMI_device_query_device_property(device, QDMI_DEVICE_PROPERTY_QUBITSNUM,
                                        sizeof(size_t), &numQubits, nullptr),
      QDMI_SUCCESS)
      << "Devices must provide the number of qubits.";
  EXPECT_GT(numQubits, 0) << "Number of qubits must be greater than 0.";
}

TEST_P(DriverTest, QueryDeviceProperties) {
  EXPECT_EQ(QDMI_device_query_device_property(device, QDMI_DEVICE_PROPERTY_MAX,
                                              0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "The MAX property is not a valid value for any device.";
}

TEST_P(DriverTest, QuerySites) {
  size_t size = 0;
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_SITES, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a list of sites.";
  std::vector<QDMI_Site> sites(size / sizeof(QDMI_Site));
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_SITES, size,
                static_cast<void*>(sites.data()), nullptr),
            QDMI_SUCCESS)
      << "Failed to get sites.";
  std::unordered_set<size_t> ids;
  for (auto* site : sites) {
    uint64_t index = 0;
    EXPECT_EQ(
        QDMI_device_query_site_property(device, site, QDMI_SITE_PROPERTY_INDEX,
                                        sizeof(uint64_t), &index, nullptr),
        QDMI_SUCCESS)
        << "Devices must provide a site id";
    EXPECT_TRUE(ids.emplace(index).second)
        << "Device must provide unique site ids. Found duplicate id: " << index
        << ".";
    double t1 = 0;
    double t2 = 0;
    EXPECT_EQ(QDMI_device_query_site_property(device, site,
                                              QDMI_SITE_PROPERTY_T1,
                                              sizeof(double), &t1, nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a site T1 time.";
    EXPECT_GT(t1, 0) << "Devices must provide a site T1 time larger than 0.";
    EXPECT_EQ(QDMI_device_query_site_property(device, site,
                                              QDMI_SITE_PROPERTY_T2,
                                              sizeof(double), &t2, nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a site T2 time.";
    EXPECT_GT(t2, 0) << "Devices must provide a site T2 time larger than 0.";
    EXPECT_EQ(QDMI_device_query_site_property(
                  device, site, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
              QDMI_ERROR_INVALIDARGUMENT)
        << "The MAX property is not a valid value for any device.";
  }
}

TEST_P(DriverTest, QueryOperations) {
  size_t operationsSize = 0;
  ASSERT_EQ(QDMI_device_query_device_property(device,
                                              QDMI_DEVICE_PROPERTY_OPERATIONS,
                                              0, nullptr, &operationsSize),
            QDMI_SUCCESS)
      << "Failed to get the size to retrieve the operations.";
  std::vector<QDMI_Operation> operations(operationsSize /
                                         sizeof(QDMI_Operation));
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_OPERATIONS, operationsSize,
                static_cast<void*>(operations.data()), nullptr),
            QDMI_SUCCESS)
      << "Failed to retrieve the operations.";
  for (auto* const op : operations) {
    size_t namesSize = 0;
    ASSERT_EQ(QDMI_device_query_operation_property(
                  device, op, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, &namesSize),
              QDMI_SUCCESS)
        << "Failed to get the length of the operation's name.";
    std::string name(namesSize - 1, '\0');
    ASSERT_EQ(
        QDMI_device_query_operation_property(device, op, 0, nullptr, 0, nullptr,
                                             QDMI_OPERATION_PROPERTY_NAME,
                                             namesSize, name.data(), nullptr),
        QDMI_SUCCESS)
        << "Failed to retrieve the operation's name.";
    EXPECT_FALSE(name.empty())
        << "Device must provide a non-empty name for every operation.";

    size_t numParams = 0;
    ASSERT_EQ(QDMI_device_query_operation_property(
                  device, op, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sizeof(size_t),
                  &numParams, nullptr),
              QDMI_SUCCESS)
        << "Failed to query number of parameters for operation.";

    double duration = 0;
    double fidelity = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::vector<double> params(numParams);
    for (auto& param : params) {
      param = dis(gen);
    }
    EXPECT_THAT(QDMI_device_query_operation_property(
                    device, op, 0, nullptr, numParams, params.data(),
                    QDMI_OPERATION_PROPERTY_DURATION, sizeof(double), &duration,
                    nullptr),
                ::testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED))
        << "Failed to query duration for operation " << name << ".";
    EXPECT_THAT(QDMI_device_query_operation_property(
                    device, op, 0, nullptr, numParams, params.data(),
                    QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(double), &fidelity,
                    nullptr),
                ::testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED))
        << "Failed to query fidelity for operation " << name << ".";

    EXPECT_EQ(QDMI_device_query_operation_property(
                  device, op, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_MAX, 0, nullptr, nullptr),
              QDMI_ERROR_INVALIDARGUMENT)
        << "The MAX property is not a valid value for any device.";
  }
}

TEST_P(DriverTest, SessionInit) {
  EXPECT_EQ(QDMI_session_init(nullptr), QDMI_ERROR_INVALIDARGUMENT)
      << "`session == nullptr` is not a valid argument.";
  EXPECT_EQ(QDMI_session_init(session), QDMI_ERROR_BADSTATE)
      << "Session must return `BADSTATE` if it is initialized again.";
}

TEST_P(DriverTest, QuerySessionProperties) {
  EXPECT_EQ(QDMI_session_query_session_property(
                nullptr, QDMI_SESSION_PROPERTY_DEVICES, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "`session == nullptr` is not a valid argument.";
  EXPECT_EQ(QDMI_session_query_session_property(
                session, QDMI_SESSION_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "`prop >= QDMI_SESSION_PROPERTY_MAX` is not a valid argument.";

  // Must not query on an uninitialized session
  QDMI_Session uninitializedSession = nullptr;
  ASSERT_EQ(QDMI_session_alloc(&uninitializedSession), QDMI_SUCCESS);
  EXPECT_EQ(QDMI_session_query_session_property(uninitializedSession,
                                                QDMI_SESSION_PROPERTY_DEVICES,
                                                0, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);

  constexpr size_t size = sizeof(QDMI_Device) - 1;
  std::array<char, size> devices{};
  EXPECT_EQ(QDMI_session_query_session_property(
                session, QDMI_SESSION_PROPERTY_DEVICES, size,
                static_cast<void*>(devices.data()), nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "Device must return `INVALIDARGUMENT` if the buffer is too small.";
}

TEST_P(DriverTest, QueryNeedsCalibration) {
  size_t needsCalibration = 0;
  const auto ret = QDMI_device_query_device_property(
      device, QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION, sizeof(size_t),
      &needsCalibration, nullptr);
  EXPECT_EQ(ret, QDMI_SUCCESS);
  EXPECT_THAT(needsCalibration, ::testing::AnyOf(0, 1));
}

// Instantiate the test suite with different parameters
INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    DefaultDevices,
    // Test suite name
    DriverTest,
    // Parameters to test with
    ::testing::Values("MQT NA Default QDMI Device",
                      "MQT NA Dynamic QDMI Device"),
    [](const testing::TestParamInfo<std::string>& info) {
      std::string name = info.param;
      // Replace spaces with underscores for valid test names
      std::replace(name.begin(), name.end(), ' ', '_');
      // Remove parentheses for valid test names
      name.erase(std::remove(name.begin(), name.end(), '('), name.end());
      name.erase(std::remove(name.begin(), name.end(), ')'), name.end());
      return name;
    });
} // namespace qc
