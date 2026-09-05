/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt_sc_qdmi/device.h"
#include "qdmi/TestUtils.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp> // NOLINT(misc-include-cleaner)
#include <nlohmann/json_fwd.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] std::vector<MQT_SC_QDMI_Site>
querySites(MQT_SC_QDMI_Device_Session session) {
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

} // namespace

namespace {
constexpr auto CUSTOM_SC = R"({
  "schema-version": 1,
  "name": "Custom SC",
  "numQubits": 2,
  "durationUnit": {"unit": "ns", "scaleFactor": 0.5},
  "qubitProperties": {
    "defaults": {"t1": 100, "t2": 200},
    "overrides": [
      {"qubit": 0, "name": "QB1"},
      {"qubit": 1, "name": "QB2", "t1": 90}
    ]
  },
  "couplings": [[0, 1], [1, 0]],
  "operations": [{
    "name": "cz",
    "numParameters": 0,
    "numQubits": 2,
    "duration": 20,
    "fidelity": 0.9,
    "siteOverrides": [{"sites": [0, 1], "duration": 10, "fidelity": 0.95}]
  }]
})";

using mqt::test::ScopedEnvironmentVariable;

[[nodiscard]] MQT_SC_QDMI_Device_Session
initializedSession(const std::string_view configuration = CUSTOM_SC) {
  MQT_SC_QDMI_Device_Session session = nullptr;
  if (MQT_SC_QDMI_device_session_alloc(&session) != QDMI_SUCCESS) {
    throw std::runtime_error("Failed to allocate custom SC session");
  }
  if (MQT_SC_QDMI_device_session_set_parameter(
          session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1,
          configuration.size() + 1, configuration.data()) != QDMI_SUCCESS ||
      MQT_SC_QDMI_device_session_init(session) != QDMI_SUCCESS) {
    MQT_SC_QDMI_device_session_free(session);
    throw std::runtime_error("Failed to initialize custom SC session");
  }
  return session;
}

[[nodiscard]] std::string queryName(MQT_SC_QDMI_Device_Session session) {
  size_t size = 0;
  if (MQT_SC_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, &size) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query device name size");
  }
  std::string name(size, '\0');
  if (MQT_SC_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_NAME, size, name.data(), nullptr) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query device name");
  }
  name.resize(size - 1);
  return name;
}

[[nodiscard]] std::vector<MQT_SC_QDMI_Operation>
queryOperations(MQT_SC_QDMI_Device_Session session) {
  size_t size = 0;
  if (MQT_SC_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_OPERATIONS, 0, nullptr, &size) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query operations");
  }
  std::vector<MQT_SC_QDMI_Operation> operations(size /
                                                sizeof(MQT_SC_QDMI_Operation));
  if (MQT_SC_QDMI_device_session_query_device_property(
          session, QDMI_DEVICE_PROPERTY_OPERATIONS, size,
          static_cast<void*>(operations.data()), nullptr) != QDMI_SUCCESS) {
    throw std::runtime_error("Failed to retrieve operations");
  }
  return operations;
}

[[nodiscard]] std::string querySiteName(MQT_SC_QDMI_Device_Session session,
                                        MQT_SC_QDMI_Site site) {
  size_t size = 0;
  if (MQT_SC_QDMI_device_session_query_site_property(
          session, site, QDMI_SITE_PROPERTY_NAME, 0, nullptr, &size) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query site name size");
  }
  std::string name(size, '\0');
  if (MQT_SC_QDMI_device_session_query_site_property(
          session, site, QDMI_SITE_PROPERTY_NAME, size, name.data(), nullptr) !=
      QDMI_SUCCESS) {
    throw std::runtime_error("Failed to query site name");
  }
  name.resize(size - 1);
  return name;
}
} // namespace

TEST(ScRuntimeConfiguration, ValidatesRawParameterStringsAndRetry) {
  MQT_SC_QDMI_Device_Session session = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  EXPECT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1, 0, nullptr),
            QDMI_SUCCESS);
  constexpr std::array missingNul{'{', '}'};
  EXPECT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1,
                missingNul.size(), missingNul.data()),
            QDMI_ERROR_INVALIDARGUMENT);
  constexpr std::array embeddedNul{'{', '\0', '}', '\0'};
  EXPECT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1,
                embeddedNul.size(), embeddedNul.data()),
            QDMI_ERROR_INVALIDARGUMENT);
  constexpr auto file = std::to_array(SC_DEVICE_JSON);
  ASSERT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1,
                std::strlen(CUSTOM_SC) + 1, CUSTOM_SC),
            QDMI_SUCCESS);
  ASSERT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2, file.size(),
                file.data()),
            QDMI_SUCCESS);
  ASSERT_EQ(MQT_SC_QDMI_device_session_init(session), QDMI_SUCCESS);
  EXPECT_EQ(queryName(session), "Custom SC");
  MQT_SC_QDMI_device_session_free(session);

  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  constexpr auto malformed = std::to_array("{");
  ASSERT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1,
                malformed.size(), malformed.data()),
            QDMI_SUCCESS);
  testing::internal::CaptureStderr();
  const auto malformedStatus = MQT_SC_QDMI_device_session_init(session);
  const auto malformedDiagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(malformedStatus, QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(
      malformedDiagnostic,
      testing::AllOf(testing::HasSubstr("[mqt-core] [error]"),
                     testing::HasSubstr("Invalid SC device configuration from "
                                        "inline session configuration"),
                     testing::HasSubstr("invalid JSON")));
  ASSERT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1,
                std::strlen(CUSTOM_SC) + 1, CUSTOM_SC),
            QDMI_SUCCESS);
  EXPECT_EQ(MQT_SC_QDMI_device_session_init(session), QDMI_SUCCESS);
  EXPECT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1, 1, ""),
            QDMI_ERROR_BADSTATE);
  MQT_SC_QDMI_device_session_free(session);
}

TEST(ScRuntimeConfiguration, RejectsOperationOutsideCouplingMap) {
  auto configuration = nlohmann::json::parse(CUSTOM_SC);
  configuration["couplings"] = {{0, 1}};
  configuration["operations"][0]["sites"] = {{1, 0}};
  configuration["operations"][0]["siteOverrides"] = nlohmann::json::array();
  const auto serialized = configuration.dump();

  MQT_SC_QDMI_Device_Session session = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  ASSERT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1,
                serialized.size() + 1, serialized.c_str()),
            QDMI_SUCCESS);
  EXPECT_EQ(MQT_SC_QDMI_device_session_init(session),
            QDMI_ERROR_INVALIDARGUMENT);
  MQT_SC_QDMI_device_session_free(session);
}

TEST(ScRuntimeConfiguration, SessionsOwnIndependentModelsAndCalibration) {
  auto* custom = initializedSession();
  MQT_SC_QDMI_Device_Session bundled = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&bundled), QDMI_SUCCESS);
  ASSERT_EQ(MQT_SC_QDMI_device_session_init(bundled), QDMI_SUCCESS);

  size_t customQubits = 0;
  size_t bundledQubits = 0;
  ASSERT_EQ(MQT_SC_QDMI_device_session_query_device_property(
                custom, QDMI_DEVICE_PROPERTY_QUBITSNUM, sizeof(customQubits),
                &customQubits, nullptr),
            QDMI_SUCCESS);
  ASSERT_EQ(MQT_SC_QDMI_device_session_query_device_property(
                bundled, QDMI_DEVICE_PROPERTY_QUBITSNUM, sizeof(bundledQubits),
                &bundledQubits, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(customQubits, 2);
  EXPECT_EQ(bundledQubits, 100);

  const auto sites = querySites(custom);
  EXPECT_EQ(querySiteName(custom, sites[0]), "QB1");
  EXPECT_EQ(querySiteName(custom, sites[1]), "QB2");
  uint64_t t1 = 0;
  ASSERT_EQ(
      MQT_SC_QDMI_device_session_query_site_property(
          custom, sites[1], QDMI_SITE_PROPERTY_T1, sizeof(t1), &t1, nullptr),
      QDMI_SUCCESS);
  EXPECT_EQ(t1, 90);
  uint64_t t2 = 0;
  ASSERT_EQ(
      MQT_SC_QDMI_device_session_query_site_property(
          custom, sites[1], QDMI_SITE_PROPERTY_T2, sizeof(t2), &t2, nullptr),
      QDMI_SUCCESS);
  EXPECT_EQ(t2, 200);

  auto* const operation = queryOperations(custom).front();
  uint64_t duration = 0;
  double fidelity = 0.;
  ASSERT_EQ(MQT_SC_QDMI_device_session_query_operation_property(
                custom, operation, sites.size(), sites.data(), 0, nullptr,
                QDMI_OPERATION_PROPERTY_DURATION, sizeof(duration), &duration,
                nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(duration, 10);
  ASSERT_EQ(MQT_SC_QDMI_device_session_query_operation_property(
                custom, operation, sites.size(), sites.data(), 0, nullptr,
                QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(fidelity), &fidelity,
                nullptr),
            QDMI_SUCCESS);
  EXPECT_DOUBLE_EQ(fidelity, 0.95);

  const std::array reverseSites{sites[1], sites[0]};
  ASSERT_EQ(MQT_SC_QDMI_device_session_query_operation_property(
                custom, operation, reverseSites.size(), reverseSites.data(), 0,
                nullptr, QDMI_OPERATION_PROPERTY_DURATION, sizeof(duration),
                &duration, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(duration, 20);
  ASSERT_EQ(MQT_SC_QDMI_device_session_query_operation_property(
                custom, operation, reverseSites.size(), reverseSites.data(), 0,
                nullptr, QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(fidelity),
                &fidelity, nullptr),
            QDMI_SUCCESS);
  EXPECT_DOUBLE_EQ(fidelity, 0.9);

  auto* const bundledSite = querySites(bundled).front();
  EXPECT_EQ(
      MQT_SC_QDMI_device_session_query_site_property(
          bundled, bundledSite, QDMI_SITE_PROPERTY_NAME, 0, nullptr, nullptr),
      QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(
      MQT_SC_QDMI_device_session_query_site_property(
          custom, bundledSite, QDMI_SITE_PROPERTY_INDEX, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_SC_QDMI_device_session_query_operation_property(
                bundled, operation, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);

  MQT_SC_QDMI_device_session_free(bundled);
  MQT_SC_QDMI_device_session_free(custom);
}

TEST(ScRuntimeConfiguration, SelectsEnvironmentAndExplicitFileSources) {
  const ScopedEnvironmentVariable environmentJson(
      "MQT_CORE_QDMI_SC_CONFIG_JSON", CUSTOM_SC);

  MQT_SC_QDMI_Device_Session environmentSession = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&environmentSession),
            QDMI_SUCCESS);
  ASSERT_EQ(MQT_SC_QDMI_device_session_init(environmentSession), QDMI_SUCCESS);
  EXPECT_EQ(queryName(environmentSession), "Custom SC");
  MQT_SC_QDMI_device_session_free(environmentSession);

  MQT_SC_QDMI_Device_Session explicitFileSession = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&explicitFileSession),
            QDMI_SUCCESS);
  constexpr auto file = std::to_array(SC_DEVICE_JSON);
  ASSERT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                explicitFileSession, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2,
                file.size(), file.data()),
            QDMI_SUCCESS);
  ASSERT_EQ(MQT_SC_QDMI_device_session_init(explicitFileSession), QDMI_SUCCESS);
  EXPECT_EQ(queryName(explicitFileSession), "MQT SC Default QDMI Device");
  MQT_SC_QDMI_device_session_free(explicitFileSession);

  const ScopedEnvironmentVariable environmentFile(
      "MQT_CORE_QDMI_SC_CONFIG_FILE", SC_DEVICE_JSON);
  MQT_SC_QDMI_Device_Session conflictingEnvironmentSession = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&conflictingEnvironmentSession),
            QDMI_SUCCESS);
  testing::internal::CaptureStderr();
  const auto conflictingStatus =
      MQT_SC_QDMI_device_session_init(conflictingEnvironmentSession);
  const auto conflictingDiagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(conflictingStatus, QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_THAT(conflictingDiagnostic,
              testing::AllOf(
                  testing::HasSubstr("[mqt-core] [error]"),
                  testing::HasSubstr("Both MQT_CORE_QDMI_SC_CONFIG_JSON and "
                                     "MQT_CORE_QDMI_SC_CONFIG_FILE are set")));
  MQT_SC_QDMI_device_session_free(conflictingEnvironmentSession);
}

TEST(ScRuntimeConfiguration, MapsMissingExplicitFileToNotFound) {
  MQT_SC_QDMI_Device_Session session = nullptr;
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  constexpr auto missing =
      std::to_array("missing-sc-device-configuration.json");
  ASSERT_EQ(MQT_SC_QDMI_device_session_set_parameter(
                session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2, missing.size(),
                missing.data()),
            QDMI_SUCCESS);
  testing::internal::CaptureStderr();
  const auto status = MQT_SC_QDMI_device_session_init(session);
  const auto diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(status, QDMI_ERROR_NOTFOUND);
  EXPECT_THAT(
      diagnostic,
      testing::AllOf(
          testing::HasSubstr("[mqt-core] [error]"),
          testing::HasSubstr("QDMI device configuration "
                             "'missing-sc-device-configuration.json' does not "
                             "exist")));
  MQT_SC_QDMI_device_session_free(session);
}

TEST(ScRuntimeConfiguration, MissingCalibrationReturnsNotSupported) {
  auto configuration = nlohmann::json::parse(CUSTOM_SC);
  configuration["qubitProperties"]["defaults"] = nlohmann::json::object();
  configuration["qubitProperties"]["overrides"] = nlohmann::json::array();
  configuration["operations"][0].erase("duration");
  configuration["operations"][0].erase("fidelity");
  configuration["operations"][0]["siteOverrides"] = nlohmann::json::array();
  const auto serialized = configuration.dump();
  auto* session = initializedSession(serialized);
  const auto sites = querySites(session);
  auto* const operation = queryOperations(session).front();
  EXPECT_EQ(MQT_SC_QDMI_device_session_query_site_property(
                session, sites[0], QDMI_SITE_PROPERTY_T1, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(MQT_SC_QDMI_device_session_query_operation_property(
                session, operation, 2, sites.data(), 0, nullptr,
                QDMI_OPERATION_PROPERTY_DURATION, 0, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  MQT_SC_QDMI_device_session_free(session);
}

TEST(ScRuntimeConfiguration, InitializesIndependentSessionsConcurrently) {
  auto const initializeAndQuery = [] {
    auto* session = initializedSession();
    const auto name = queryName(session);
    MQT_SC_QDMI_device_session_free(session);
    return name;
  };
  auto first = std::async(std::launch::async, initializeAndQuery);
  auto second = std::async(std::launch::async, initializeAndQuery);
  EXPECT_EQ(first.get(), "Custom SC");
  EXPECT_EQ(second.get(), "Custom SC");
}

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
  MQT_SC_QDMI_device_session_free(uninitializedSession);
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
  MQT_SC_QDMI_device_session_free(uninitializedSession);
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
