/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt_ddsim_qdmi/device.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
[[nodiscard]] auto querySites(MQT_DDSIM_QDMI_Device_Session session)
    -> std::vector<MQT_DDSIM_QDMI_Site> {
  size_t size = 0;
  if (MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_SITES, 0, nullptr, &size) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query sites");
  }
  if (size == 0) {
    throw std::runtime_error("No sites available");
  }
  std::vector<MQT_DDSIM_QDMI_Site> sites(size / sizeof(MQT_DDSIM_QDMI_Site));
  if (MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_SITES, size,
          static_cast<void*>(sites.data()), nullptr) != QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query sites");
  }
  return sites;
}
[[nodiscard]] auto queryOperations(MQT_DDSIM_QDMI_Device_Session session)
    -> std::vector<MQT_DDSIM_QDMI_Operation> {
  size_t size = 0;
  if (MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_OPERATIONS, 0, nullptr, &size) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query operations");
  }
  if (size == 0) {
    throw std::runtime_error("No operations available");
  }
  std::vector<MQT_DDSIM_QDMI_Operation> operations(
      size / sizeof(MQT_DDSIM_QDMI_Operation));
  if (MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_OPERATIONS, size,
          static_cast<void*>(operations.data()), nullptr) != QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query operations");
  }
  return operations;
}
} // namespace

class DDSIMQDMISpecificationTest : public ::testing::Test {
protected:
  MQT_DDSIM_QDMI_Device_Session session = nullptr;

  void SetUp() override {
    ASSERT_EQ(MQT_DDSIM_QDMI_device_initialize(), QDMI_SUCCESS)
        << "Failed to initialize the device";

    ASSERT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&session), QDMI_SUCCESS)
        << "Failed to allocate a session";

    ASSERT_EQ(MQT_DDSIM_QDMI_device_session_init(session), QDMI_SUCCESS)
        << "Failed to initialize a session. Potential errors: Wrong or missing "
           "authentication information, device status is offline, or in "
           "maintenance. To provide credentials, take a look in " __FILE__
        << (__LINE__ - 4);
  }

  void TearDown() override {
    if (session != nullptr) {
      MQT_DDSIM_QDMI_device_session_free(session);
      session = nullptr;
    }
    MQT_DDSIM_QDMI_device_finalize();
  }
};

class DDSIMQDMIJobSpecificationTest : public DDSIMQDMISpecificationTest {
protected:
  MQT_DDSIM_QDMI_Device_Job job = nullptr;

  void SetUp() override {
    DDSIMQDMISpecificationTest::SetUp();
    ASSERT_EQ(MQT_DDSIM_QDMI_device_session_create_device_job(session, &job),
              QDMI_SUCCESS)
        << "Failed to create a device job.";
    // set program format to OpenQASM 3
    constexpr auto format = QDMI_PROGRAM_FORMAT_QASM3;
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                  job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                  sizeof(QDMI_Program_Format), &format),
              QDMI_SUCCESS)
        << "Failed to set program format to OpenQASM 3.";

    constexpr auto program = R"(
      OPENQASM 3;
      include "stdgates.inc";
      qubit[2] q;
      bit[2] c;
      h q[0];
      cx q[0], q[1];
      c = measure q;
    )";
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                  job, QDMI_DEVICE_JOB_PARAMETER_PROGRAM, strlen(program) + 1,
                  program),
              QDMI_SUCCESS)
        << "Failed to set program.";
  }

  void TearDown() override {
    if (job != nullptr) {
      MQT_DDSIM_QDMI_device_job_free(job);
      job = nullptr;
    }
    DDSIMQDMISpecificationTest::TearDown();
  }
};

TEST_F(DDSIMQDMISpecificationTest, SessionAlloc) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_alloc(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMISpecificationTest, SessionInit) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_init(session), QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_init(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMISpecificationTest, SessionSetParameter) {
  MQT_DDSIM_QDMI_Device_Session uninitializedSession = nullptr;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&uninitializedSession),
            QDMI_SUCCESS);
  EXPECT_THAT(MQT_DDSIM_QDMI_device_session_set_parameter(
                  uninitializedSession, QDMI_DEVICE_SESSION_PARAMETER_BASEURL,
                  20, "https://example.com"),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED,
                             QDMI_ERROR_INVALIDARGUMENT));
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_BASEURL, 20,
                "https://example.com"),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                nullptr, QDMI_DEVICE_SESSION_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMISpecificationTest, JobCreate) {
  MQT_DDSIM_QDMI_Device_Session uninitializedSession = nullptr;
  MQT_DDSIM_QDMI_Device_Job job = nullptr;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&uninitializedSession),
            QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_create_device_job(
                uninitializedSession, &job),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_create_device_job(session, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_create_device_job(nullptr, &job),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(MQT_DDSIM_QDMI_device_session_create_device_job(session, &job),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  MQT_DDSIM_QDMI_device_job_free(job);
}

TEST_F(DDSIMQDMISpecificationTest, JobSetParameter) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                nullptr, QDMI_DEVICE_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobSetParameter) {
  constexpr QDMI_Program_Format value = QDMI_PROGRAM_FORMAT_QASM2;
  EXPECT_THAT(MQT_DDSIM_QDMI_device_job_set_parameter(
                  job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                  sizeof(QDMI_Program_Format), &value),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  // The DDSIM devices does not support custom parameters
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM1, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM2, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM3, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM4, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM5, 0, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST_F(DDSIMQDMISpecificationTest, JobQueryProperty) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                nullptr, QDMI_DEVICE_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobQueryProperty) {
  EXPECT_THAT(MQT_DDSIM_QDMI_device_job_query_property(
                  job, QDMI_DEVICE_JOB_PROPERTY_ID, 0, nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));

  QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QASM3;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &format),
            QDMI_SUCCESS);
  // The set parameter value must coincide with the value returned for the
  // respective property
  size_t size = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &format, &size),
            QDMI_SUCCESS);
  EXPECT_EQ(size, sizeof(QDMI_Program_Format));
  EXPECT_EQ(format, QDMI_PROGRAM_FORMAT_QASM3);

  size_t shots = 5;
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          nullptr, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, 0, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job, QDMI_DEVICE_JOB_PARAMETER_MAX, sizeof(size_t), &shots),
            QDMI_ERROR_INVALIDARGUMENT);
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS);
  // The set parameter value must coincide with the value returned for the
  // respective property
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM, sizeof(size_t), &shots,
                nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(shots, 5);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  // The DDSIM devices does not support custom parameters
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_CUSTOM2, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_CUSTOM3, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_CUSTOM4, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_CUSTOM5, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST_F(DDSIMQDMIJobSpecificationTest, QueryJobId) {
  size_t size = 0;
  const auto status = MQT_DDSIM_QDMI_device_job_query_property(
      job, QDMI_DEVICE_JOB_PROPERTY_ID, 0, nullptr, &size);
  ASSERT_EQ(status, QDMI_SUCCESS);
  ASSERT_GT(size, 0);
  std::string id(size - 1, '\0');
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job, QDMI_DEVICE_JOB_PROPERTY_ID, size, id.data(), nullptr),
            QDMI_SUCCESS);
}

TEST_F(DDSIMQDMISpecificationTest, JobSubmit) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobSubmit) {
  const auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(DDSIMQDMISpecificationTest, JobCancel) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_cancel(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobCancel) {
  const auto status = MQT_DDSIM_QDMI_device_job_cancel(job);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_INVALIDARGUMENT,
                                     QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(DDSIMQDMISpecificationTest, JobCheck) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_check(nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobCheck) {
  QDMI_Job_Status jobStatus = QDMI_JOB_STATUS_RUNNING;
  const auto status = MQT_DDSIM_QDMI_device_job_check(job, &jobStatus);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_F(DDSIMQDMISpecificationTest, JobWait) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(nullptr, 0),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobWait) {
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobWaitTimeout) {
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 1);
  ASSERT_THAT(status, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_TIMEOUT));
}

TEST_F(DDSIMQDMISpecificationTest, JobGetResults) {
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(nullptr, QDMI_JOB_RESULT_MAX,
                                                  0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobGetResultsShots) {
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_SHOTS, 0,
                                                 nullptr, nullptr);
  ASSERT_EQ(status, QDMI_ERROR_NOTSUPPORTED);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobGetResultsHistogram) {
  constexpr size_t shots = 1024U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);

  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_HIST_KEYS, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::string keyList(size - 1, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_HIST_KEYS, size, keyList.data(), nullptr),
            QDMI_SUCCESS);
  std::vector<std::string> keyVec;
  std::string token;
  std::stringstream ss(keyList);
  while (std::getline(ss, token, ',')) {
    keyVec.emplace_back(token);
  }

  size_t valSize = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_HIST_VALUES, 0, nullptr, &valSize),
            QDMI_SUCCESS);
  ASSERT_EQ(valSize / sizeof(size_t), keyVec.size());

  std::vector<size_t> valVec(keyVec.size());
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_HIST_VALUES,
                                            valSize, valVec.data(), nullptr),
      QDMI_SUCCESS);

  size_t sum = 0;
  for (const auto& val : valVec) {
    sum += val;
  }
  ASSERT_EQ(sum, shots);

  std::unordered_map<std::string, size_t> results;
  for (size_t i = 0; i < keyVec.size(); ++i) {
    results[keyVec[i]] = valVec[i];
  }
  ASSERT_EQ(results.size(), keyVec.size());
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobGetResultsStateDense) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);

  size_t stateSize = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0, nullptr, &stateSize),
            QDMI_SUCCESS);
  const size_t vecLength = stateSize / sizeof(double);
  ASSERT_EQ(vecLength % 2, 0) << "State vector must contain pairs of values";

  std::vector<double> stateVector(vecLength);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, stateSize,
                stateVector.data(), nullptr),
            QDMI_SUCCESS);

  std::vector<std::complex<double>> complexStateVector;
  complexStateVector.reserve(vecLength / 2);
  for (size_t i = 0; i < stateVector.size(); i += 2) {
    complexStateVector.emplace_back(stateVector[i], stateVector[i + 1]);
  }

  // assert that the complex vector is normalized up to a certain tolerance
  double norm = 0;
  for (const auto& val : complexStateVector) {
    norm += std::norm(val);
  }
  ASSERT_NEAR(norm, 1, 1e-6);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobGetResultsStateSparse) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);

  size_t size = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 0, nullptr, &size),
      QDMI_SUCCESS);
  std::string keyList(size - 1, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, size,
                keyList.data(), nullptr),
            QDMI_SUCCESS);
  std::vector<std::string> keyVec;
  std::string token;
  std::stringstream ss(keyList);
  while (std::getline(ss, token, ',')) {
    keyVec.emplace_back(token);
  }

  size_t valSize = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, 0, nullptr, &valSize),
      QDMI_SUCCESS);
  ASSERT_EQ(valSize / 2 / sizeof(double), keyVec.size());

  std::vector<std::complex<double>> valVec(keyVec.size());
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, valSize,
                valVec.data(), nullptr),
            QDMI_SUCCESS);

  double norm = 0;
  for (const auto& val : valVec) {
    norm += std::norm(val);
  }
  ASSERT_NEAR(norm, 1, 1e-6);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobGetResultsProbsDense) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);

  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_DENSE, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::vector<double> probVector(size / sizeof(double));
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_DENSE,
                sizeof(double) * probVector.size(), probVector.data(), nullptr),
            QDMI_SUCCESS);

  double sum = 0;
  for (const auto& prob : probVector) {
    sum += prob;
  }
  ASSERT_NEAR(sum, 1.0, 1e-6);
}

TEST_F(DDSIMQDMIJobSpecificationTest, JobGetResultsProbsSparse) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);

  size_t size = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, 0, nullptr, &size),
      QDMI_SUCCESS);
  std::string keyList(size - 1, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, size,
                keyList.data(), nullptr),
            QDMI_SUCCESS);
  std::vector<std::string> keyVec;
  std::string token;
  std::stringstream ss(keyList);
  while (std::getline(ss, token, ',')) {
    keyVec.emplace_back(token);
  }

  size_t valSize = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, 0, nullptr,
                &valSize),
            QDMI_SUCCESS);
  ASSERT_EQ(valSize / sizeof(double), keyVec.size());

  std::vector<double> valVec(keyVec.size());
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, valSize,
                valVec.data(), nullptr),
            QDMI_SUCCESS);

  double sum = 0;
  for (const auto& val : valVec) {
    sum += val;
  }
  ASSERT_NEAR(sum, 1.0, 1e-6);
}

TEST_F(DDSIMQDMIJobSpecificationTest, GetResultsCornerCases) {
  constexpr size_t shots = 1024U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);

  // The MAX parameter is not a valid value for any device
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_MAX, 0,
                                                  nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_SHOTS, 0,
                                                  nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_DENSE, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, 0, nullptr,
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  // The DDSIM device does not support custom results
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_CUSTOM1,
                                                  0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_CUSTOM2,
                                                  0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_CUSTOM3,
                                                  0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_CUSTOM4,
                                                  0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_CUSTOM5,
                                                  0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST_F(DDSIMQDMIJobSpecificationTest, GetHistogramKeysBufferTooSmall) {
  constexpr size_t shots = 1024U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_HIST_KEYS, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::vector<char> buffer(size - 1); // Buffer too small
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_HIST_KEYS, buffer.size(), buffer.data(),
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, GetHistogramValuesBufferTooSmall) {
  constexpr size_t shots = 1024U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_HIST_VALUES, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::vector<char> buffer(size - 1); // Buffer too small
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_HIST_VALUES, buffer.size(), buffer.data(),
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, GetStateDenseBufferTooSmall) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::vector<char> buffer(size - 1); // Buffer too small
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, buffer.size(),
                buffer.data(), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, GetStateSparseKeysBufferTooSmall) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
  size_t size = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 0, nullptr, &size),
      QDMI_SUCCESS);
  std::vector<char> buffer(size - 1); // Buffer too small
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, buffer.size(),
                buffer.data(), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, GetStateSparseValuesBufferTooSmall) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
  size_t size = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, 0, nullptr, &size),
      QDMI_SUCCESS);
  std::vector<char> buffer(size - 1); // Buffer too small
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, buffer.size(),
                buffer.data(), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, GetProbsDenseBufferTooSmall) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_DENSE, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::vector<char> buffer(size - 1); // Buffer too small
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_DENSE, buffer.size(),
                buffer.data(), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, GetProbsSparseKeysBufferTooSmall) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
  size_t size = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, 0, nullptr, &size),
      QDMI_SUCCESS);
  std::vector<char> buffer(size - 1); // Buffer too small
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, buffer.size(),
                buffer.data(), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMIJobSpecificationTest, GetProbsSparseValuesBufferTooSmall) {
  constexpr size_t shots = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS)
      << "Failed to set shots.";
  auto status = MQT_DDSIM_QDMI_device_job_submit(job);
  ASSERT_EQ(status, QDMI_SUCCESS);
  status = MQT_DDSIM_QDMI_device_job_wait(job, 0);
  ASSERT_EQ(status, QDMI_SUCCESS);
  size_t size = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, 0, nullptr, &size),
      QDMI_SUCCESS);
  std::vector<char> buffer(size - 1); // Buffer too small
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, buffer.size(),
                buffer.data(), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSIMQDMISpecificationTest, QueryDeviceProperty) {
  MQT_DDSIM_QDMI_Device_Session uninitializedSession = nullptr;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_alloc(&uninitializedSession),
            QDMI_SUCCESS);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          uninitializedSession, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                nullptr, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_COUPLINGMAP, 0, nullptr, nullptr),
      testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));

  // The DDSIM device does not support custom properties
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_CUSTOM2, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_CUSTOM3, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_CUSTOM4, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_CUSTOM5, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST_F(DDSIMQDMISpecificationTest, QuerySiteProperty) {
  MQT_DDSIM_QDMI_Site site = querySites(session).front();
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_session_query_site_property(
          session, nullptr, QDMI_SITE_PROPERTY_INDEX, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                nullptr, site, QDMI_SITE_PROPERTY_INDEX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                session, site, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(MQT_DDSIM_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_NAME, 0, nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));

  // The DDSIM device does not support custom properties
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                session, site, QDMI_SITE_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                session, site, QDMI_SITE_PROPERTY_CUSTOM2, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                session, site, QDMI_SITE_PROPERTY_CUSTOM3, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                session, site, QDMI_SITE_PROPERTY_CUSTOM4, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                session, site, QDMI_SITE_PROPERTY_CUSTOM5, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST_F(DDSIMQDMISpecificationTest, QueryOperationProperty) {
  const MQT_DDSIM_QDMI_Operation operation = queryOperations(session).front();
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                nullptr, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                session, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(MQT_DDSIM_QDMI_device_session_query_operation_property(
                  session, operation, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_QUBITSNUM, 0, nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));

  // The DDSIM device does not support custom properties
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                session, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                session, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM2, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                session, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM3, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                session, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM4, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                session, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_CUSTOM5, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST_F(DDSIMQDMISpecificationTest, QueryDeviceName) {
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a name";
  std::string value(size - 1, '\0');
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_NAME, size, value.data(), nullptr),
      QDMI_SUCCESS)
      << "Devices must provide a name";
  EXPECT_FALSE(value.empty()) << "Devices must provide a name";
}

TEST_F(DDSIMQDMISpecificationTest, QueryDeviceVersion) {
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_VERSION, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a version";
  std::string value(size - 1, '\0');
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_VERSION, size, value.data(), nullptr),
      QDMI_SUCCESS)
      << "Devices must provide a version";
  EXPECT_FALSE(value.empty()) << "Devices must provide a version";
}

TEST_F(DDSIMQDMISpecificationTest, QueryDeviceLibraryVersion) {
  size_t size = 0;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, 0, nullptr, &size),
      QDMI_SUCCESS)
      << "Devices must provide a library version";
  std::string value(size - 1, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, size,
                value.data(), nullptr),
            QDMI_SUCCESS)
      << "Devices must provide a library version";
  EXPECT_FALSE(value.empty()) << "Devices must provide a library version";
}

TEST_F(DDSIMQDMISpecificationTest, QueryDeviceLengthUnit) {
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_LENGTHUNIT, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::string value(size - 1, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_LENGTHUNIT, size, value.data(),
                nullptr),
            QDMI_SUCCESS);
  EXPECT_THAT(value, testing::AnyOf("nm", "um", "mm"));
  double scaleFactor = 0.;
  const auto result = MQT_DDSIM_QDMI_device_session_query_device_property(
      session, QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR, sizeof(double),
      &scaleFactor, nullptr);
  EXPECT_THAT(result, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  if (result == QDMI_SUCCESS) {
    EXPECT_GT(scaleFactor, 0.);
  }
}

TEST_F(DDSIMQDMISpecificationTest, QueryDeviceDurationUnit) {
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_DURATIONUNIT, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::string value(size - 1, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_DURATIONUNIT, size, value.data(),
                nullptr),
            QDMI_SUCCESS);
  EXPECT_THAT(value, testing::AnyOf("ns", "us", "ms"));
  double scaleFactor = 0.;
  const auto result = MQT_DDSIM_QDMI_device_session_query_device_property(
      session, QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR, sizeof(double),
      &scaleFactor, nullptr);
  EXPECT_THAT(result, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  if (result == QDMI_SUCCESS) {
    EXPECT_GT(scaleFactor, 0.);
  }
}

TEST_F(DDSIMQDMISpecificationTest, QuerySiteIndex) {
  size_t id = 0;
  EXPECT_NO_THROW(for (auto* site : querySites(session)) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_site_property(
                  session, site, QDMI_SITE_PROPERTY_INDEX, sizeof(size_t), &id,
                  nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a site id";
  }) << "Devices must provide a list of sites";
}

TEST_F(DDSIMQDMISpecificationTest, QueryOperationName) {
  size_t nameSize = 0;
  EXPECT_NO_THROW(for (auto* operation : queryOperations(session)) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                  session, operation, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, &nameSize),
              QDMI_SUCCESS)
        << "Devices must provide a operation name";
    std::string name(nameSize - 1, '\0');
    EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                  session, operation, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_NAME, nameSize, name.data(), nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a operation name";
  }) << "Devices must provide a list of operations";
}

TEST_F(DDSIMQDMISpecificationTest, QueryOperationFidelity) {
  double fidelity = 0;
  EXPECT_NO_THROW(for (auto* operation : queryOperations(session)) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                  session, operation, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(double), &fidelity,
                  nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a operation fidelity";
    EXPECT_EQ(fidelity, 1.0) << "Operation fidelity must be between 0 and 1";
  }) << "Devices must provide a list of operations";
}

TEST_F(DDSIMQDMISpecificationTest, QueryOperationParameterNum) {
  size_t numParameters = 0;
  EXPECT_NO_THROW(for (auto* operation : queryOperations(session)) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_operation_property(
                  session, operation, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sizeof(size_t),
                  &numParameters, nullptr),
              QDMI_SUCCESS)
        << "Devices must provide a operation parameter number";
    EXPECT_GE(numParameters, 0)
        << "Operation parameter number must be non-negative";
  }) << "Devices must provide a list of operations";
}

TEST_F(DDSIMQDMISpecificationTest, QueryOperationQubitNum) {
  size_t numQubits = 0;
  EXPECT_NO_THROW(for (auto* operation : queryOperations(session)) {
    const auto result = MQT_DDSIM_QDMI_device_session_query_operation_property(
        session, operation, 0, nullptr, 0, nullptr,
        QDMI_OPERATION_PROPERTY_QUBITSNUM, sizeof(size_t), &numQubits, nullptr);
    EXPECT_THAT(result, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED))
        << "Devices must provide a operation qubit number or indicate that it "
           "is not supported";
    if (result == QDMI_SUCCESS) {
      EXPECT_GE(numQubits, 0) << "Operation qubit number must be non-negative";
    }
  }) << "Devices must provide a list of operations";
}

TEST_F(DDSIMQDMISpecificationTest, QueryDeviceQubitNum) {
  size_t numQubits = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_session_query_device_property(
                session, QDMI_DEVICE_PROPERTY_QUBITSNUM, sizeof(size_t),
                &numQubits, nullptr),
            QDMI_SUCCESS);
}
