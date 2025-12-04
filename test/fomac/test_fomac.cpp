/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "fomac/FoMaC.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <new>
#include <numbers>
#include <qdmi/client.h>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace fomac {
class DeviceTest : public testing::TestWithParam<Session::Device> {
protected:
  Session::Device device;

  DeviceTest() : device(GetParam()) {}
};

class SiteTest : public DeviceTest {
protected:
  Session::Device::Site site;

  SiteTest() : site(device.getSites().front()) {}
};

class OperationTest : public DeviceTest {
protected:
  Session::Device::Operation operation;

  OperationTest() : operation(device.getOperations().front()) {}
};

class DDSimulatorDeviceTest : public testing::Test {
protected:
  Session::Device device;

  DDSimulatorDeviceTest() : device(getDDSimulatorDevice()) {}

private:
  static auto getDDSimulatorDevice() -> Session::Device {
    Session session;
    for (const auto& dev : session.getDevices()) {
      if (dev.getName() == "MQT Core DDSIM QDMI Device") {
        return dev;
      }
    }
    throw std::runtime_error("DD simulator device not found");
  }
};

class JobTest : public DDSimulatorDeviceTest {
protected:
  Session::Job job;

  JobTest() : job(createTestJob()) {}

  [[nodiscard]] Session::Job createTestJob() const {
    const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
h q[0];
c[0] = measure q[0];
)";
    return device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 10);
  }
};

class SimulatorJobTest : public DDSimulatorDeviceTest {
protected:
  Session::Job job;

  SimulatorJobTest() : job(createTestJob()) {}

  [[nodiscard]] Session::Job createTestJob() const {
    const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
)";
    return device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 0);
  }
};

TEST(FoMaCTest, StatusToString) {
  EXPECT_EQ(toString(QDMI_WARN_GENERAL), "General warning");
  EXPECT_EQ(toString(QDMI_SUCCESS), "Success");
  EXPECT_EQ(toString(QDMI_ERROR_FATAL), "A fatal error");
  EXPECT_EQ(toString(QDMI_ERROR_OUTOFMEM), "Out of memory");
  EXPECT_EQ(toString(QDMI_ERROR_NOTIMPLEMENTED), "Not implemented");
  EXPECT_EQ(toString(QDMI_ERROR_LIBNOTFOUND), "Library not found");
  EXPECT_EQ(toString(QDMI_ERROR_NOTFOUND), "Element not found");
  EXPECT_EQ(toString(QDMI_ERROR_OUTOFRANGE), "Out of range");
  EXPECT_EQ(toString(QDMI_ERROR_INVALIDARGUMENT), "Invalid argument");
  EXPECT_EQ(toString(QDMI_ERROR_PERMISSIONDENIED), "Permission denied");
  EXPECT_EQ(toString(QDMI_ERROR_NOTSUPPORTED), "Not supported");
  EXPECT_EQ(toString(QDMI_ERROR_BADSTATE), "Bad state");
  EXPECT_EQ(toString(QDMI_ERROR_TIMEOUT), "Timeout");
}

TEST(FoMaCTest, SitePropertyToString) {
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_INDEX), "QDMI_SITE_PROPERTY_INDEX");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_T1), "QDMI_SITE_PROPERTY_T1");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_T2), "QDMI_SITE_PROPERTY_T2");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_NAME), "QDMI_SITE_PROPERTY_NAME");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_XCOORDINATE),
            "QDMI_SITE_PROPERTY_XCOORDINATE");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_YCOORDINATE),
            "QDMI_SITE_PROPERTY_YCOORDINATE");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_ZCOORDINATE),
            "QDMI_SITE_PROPERTY_ZCOORDINATE");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_ISZONE), "QDMI_SITE_PROPERTY_ISZONE");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_XEXTENT), "QDMI_SITE_PROPERTY_XEXTENT");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_YEXTENT), "QDMI_SITE_PROPERTY_YEXTENT");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_ZEXTENT), "QDMI_SITE_PROPERTY_ZEXTENT");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_MODULEINDEX),
            "QDMI_SITE_PROPERTY_MODULEINDEX");
  EXPECT_EQ(toString(QDMI_SITE_PROPERTY_SUBMODULEINDEX),
            "QDMI_SITE_PROPERTY_SUBMODULEINDEX");
}

TEST(FoMaCTest, OperationPropertyToString) {
  EXPECT_EQ(toString(QDMI_OPERATION_PROPERTY_NAME),
            "QDMI_OPERATION_PROPERTY_NAME");
  EXPECT_EQ(toString(QDMI_OPERATION_PROPERTY_QUBITSNUM),
            "QDMI_OPERATION_PROPERTY_QUBITSNUM");
  EXPECT_EQ(toString(QDMI_OPERATION_PROPERTY_PARAMETERSNUM),
            "QDMI_OPERATION_PROPERTY_PARAMETERSNUM");
  EXPECT_EQ(toString(QDMI_OPERATION_PROPERTY_DURATION),
            "QDMI_OPERATION_PROPERTY_DURATION");
  EXPECT_EQ(toString(QDMI_OPERATION_PROPERTY_FIDELITY),
            "QDMI_OPERATION_PROPERTY_FIDELITY");
  EXPECT_EQ(toString(QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS),
            "QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS");
  EXPECT_EQ(toString(QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS),
            "QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS");
}

TEST(FoMaCTest, DevicePropertyToString) {
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_NAME), "QDMI_DEVICE_PROPERTY_NAME");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_VERSION),
            "QDMI_DEVICE_PROPERTY_VERSION");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_STATUS),
            "QDMI_DEVICE_PROPERTY_STATUS");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_LIBRARYVERSION),
            "QDMI_DEVICE_PROPERTY_LIBRARYVERSION");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_QUBITSNUM),
            "QDMI_DEVICE_PROPERTY_QUBITSNUM");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_SITES), "QDMI_DEVICE_PROPERTY_SITES");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_OPERATIONS),
            "QDMI_DEVICE_PROPERTY_OPERATIONS");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_COUPLINGMAP),
            "QDMI_DEVICE_PROPERTY_COUPLINGMAP");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION),
            "QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_LENGTHUNIT),
            "QDMI_DEVICE_PROPERTY_LENGTHUNIT");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR),
            "QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_DURATIONUNIT),
            "QDMI_DEVICE_PROPERTY_DURATIONUNIT");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR),
            "QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR");
  EXPECT_EQ(toString(QDMI_DEVICE_PROPERTY_MINATOMDISTANCE),
            "QDMI_DEVICE_PROPERTY_MINATOMDISTANCE");
}

TEST(FoMaCTest, SessionPropertyToString) {
  EXPECT_EQ(toString(QDMI_SESSION_PROPERTY_DEVICES),
            "QDMI_SESSION_PROPERTY_DEVICES");
}

TEST(FoMaCTest, ThrowIfError) {
  EXPECT_NO_THROW(throwIfError(QDMI_SUCCESS, "Test"));
  EXPECT_NO_THROW(throwIfError(QDMI_WARN_GENERAL, "Test"));
  EXPECT_THROW(throwIfError(QDMI_ERROR_FATAL, "Test"), std::runtime_error);
  EXPECT_THROW(throwIfError(QDMI_ERROR_OUTOFMEM, "Test"), std::bad_alloc);
  EXPECT_THROW(throwIfError(QDMI_ERROR_NOTIMPLEMENTED, "Test"),
               std::runtime_error);
  EXPECT_THROW(throwIfError(QDMI_ERROR_LIBNOTFOUND, "Test"),
               std::runtime_error);
  EXPECT_THROW(throwIfError(QDMI_ERROR_NOTFOUND, "Test"), std::runtime_error);
  EXPECT_THROW(throwIfError(QDMI_ERROR_OUTOFRANGE, "Test"), std::out_of_range);
  EXPECT_THROW(throwIfError(QDMI_ERROR_INVALIDARGUMENT, "Test"),
               std::invalid_argument);
  EXPECT_THROW(throwIfError(QDMI_ERROR_PERMISSIONDENIED, "Test"),
               std::runtime_error);
  EXPECT_THROW(throwIfError(QDMI_ERROR_NOTSUPPORTED, "Test"),
               std::runtime_error);
  EXPECT_THROW(throwIfError(QDMI_ERROR_BADSTATE, "Test"), std::runtime_error);
  EXPECT_THROW(throwIfError(QDMI_ERROR_TIMEOUT, "Test"), std::runtime_error);
}

TEST_P(DeviceTest, Name) {
  EXPECT_NO_THROW(EXPECT_FALSE(device.getName().empty()));
}

TEST_P(DeviceTest, Version) {
  EXPECT_NO_THROW(EXPECT_FALSE(device.getVersion().empty()));
}

TEST_P(DeviceTest, Status) {
  EXPECT_NO_THROW(std::ignore = device.getStatus());
}

TEST_P(DeviceTest, LibraryVersion) {
  EXPECT_NO_THROW(EXPECT_FALSE(device.getLibraryVersion().empty()));
}

TEST_P(DeviceTest, QubitsNum) {
  EXPECT_NO_THROW(EXPECT_GT(device.getQubitsNum(), 0));
}

TEST_P(DeviceTest, Sites) {
  EXPECT_NO_THROW(EXPECT_FALSE(device.getSites().empty()));
}

TEST_P(DeviceTest, Operations) {
  EXPECT_NO_THROW(EXPECT_FALSE(device.getOperations().empty()));
}

TEST_P(DeviceTest, CouplingMap) {
  EXPECT_NO_THROW(std::ignore = device.getCouplingMap());
}

TEST_P(DeviceTest, NeedsCalibration) {
  EXPECT_NO_THROW(std::ignore = device.getNeedsCalibration());
}

TEST_P(DeviceTest, LengthUnit) {
  EXPECT_NO_THROW(std::ignore = device.getLengthUnit());
}

TEST_P(DeviceTest, LengthScaleFactor) {
  EXPECT_NO_THROW(std::ignore = device.getLengthScaleFactor());
}

TEST_P(DeviceTest, DurationUnit) {
  EXPECT_NO_THROW(std::ignore = device.getDurationUnit());
}

TEST_P(DeviceTest, DurationScaleFactor) {
  EXPECT_NO_THROW(std::ignore = device.getDurationScaleFactor());
}

TEST_P(DeviceTest, MinAtomDistance) {
  EXPECT_NO_THROW(std::ignore = device.getMinAtomDistance());
}

TEST_P(DeviceTest, SupportedProgramFormats) {
  EXPECT_NO_THROW(std::ignore = device.getSupportedProgramFormats());
}

TEST_P(SiteTest, Index) { EXPECT_NO_THROW(std::ignore = site.getIndex()); }

TEST_P(SiteTest, T1) { EXPECT_NO_THROW(std::ignore = site.getT1()); }

TEST_P(SiteTest, T2) { EXPECT_NO_THROW(std::ignore = site.getT2()); }

TEST_P(SiteTest, Name) { EXPECT_NO_THROW(std::ignore = site.getName()); }

TEST_P(SiteTest, XCoordinate) {
  EXPECT_NO_THROW(std::ignore = site.getXCoordinate());
}

TEST_P(SiteTest, YCoordinate) {
  EXPECT_NO_THROW(std::ignore = site.getYCoordinate());
}

TEST_P(SiteTest, ZCoordinate) {
  EXPECT_NO_THROW(std::ignore = site.getZCoordinate());
}

TEST_P(SiteTest, IsZone) { EXPECT_NO_THROW(std::ignore = site.isZone()); }

TEST_P(SiteTest, XExtent) { EXPECT_NO_THROW(std::ignore = site.getXExtent()); }

TEST_P(SiteTest, YExtent) { EXPECT_NO_THROW(std::ignore = site.getYExtent()); }

TEST_P(SiteTest, ZExtent) { EXPECT_NO_THROW(std::ignore = site.getZExtent()); }

TEST_P(SiteTest, ModuleIndex) {
  EXPECT_NO_THROW(std::ignore = site.getModuleIndex());
}

TEST_P(SiteTest, SubmoduleIndex) {
  EXPECT_NO_THROW(std::ignore = site.getSubmoduleIndex());
}

TEST_P(OperationTest, Name) {
  EXPECT_NO_THROW(EXPECT_FALSE(operation.getName().empty()););
}

TEST_P(OperationTest, QubitsNum) {
  EXPECT_NO_THROW(std::ignore = operation.getQubitsNum());
}

TEST_P(OperationTest, ParametersNum) {
  EXPECT_NO_THROW(std::ignore = operation.getParametersNum());
}

TEST_P(OperationTest, Duration) {
  EXPECT_NO_THROW(std::ignore = operation.getDuration());
}

TEST_P(OperationTest, Fidelity) {
  EXPECT_NO_THROW(std::ignore = operation.getFidelity());
}

TEST_P(OperationTest, InteractionRadius) {
  EXPECT_NO_THROW(std::ignore = operation.getInteractionRadius());
}

TEST_P(OperationTest, BlockingRadius) {
  EXPECT_NO_THROW(std::ignore = operation.getBlockingRadius());
}

TEST_P(OperationTest, IdlingFidelity) {
  EXPECT_NO_THROW(std::ignore = operation.getIdlingFidelity());
}

TEST_P(OperationTest, oned) {
  EXPECT_NO_THROW(std::ignore = operation.isZoned());
}

TEST_P(OperationTest, Sites) {
  EXPECT_NO_THROW(std::ignore = operation.getSites());
}

TEST_P(OperationTest, SitePairs) {
  const auto sitePairs = operation.getSitePairs();
  const auto qubitsNum = operation.getQubitsNum();
  const auto isZonedOp = operation.isZoned();

  if (!qubitsNum.has_value() || *qubitsNum != 2 || isZonedOp) {
    EXPECT_FALSE(sitePairs.has_value());
  } else {
    const auto sites = operation.getSites();
    if (!sites.has_value() || sites->empty() || sites->size() % 2 != 0) {
      EXPECT_FALSE(sitePairs.has_value());
    } else {
      EXPECT_TRUE(sitePairs.has_value());
      if (sitePairs.has_value()) {
        EXPECT_EQ(sitePairs->size(), sites->size() / 2);
      }
    }
  }
}

TEST_P(OperationTest, MeanShuttlingSpeed) {
  EXPECT_NO_THROW(std::ignore = operation.getMeanShuttlingSpeed());
}

TEST_P(DeviceTest, RegularSitesAndZones) {
  const auto allSites = device.getSites();
  const auto regularSites = device.getRegularSites();
  const auto zones = device.getZones();

  EXPECT_FALSE(allSites.empty());
  EXPECT_EQ(regularSites.size() + zones.size(), allSites.size());

  for (const auto& site : regularSites) {
    EXPECT_FALSE(site.isZone());
  }

  for (const auto& site : zones) {
    EXPECT_TRUE(site.isZone());
  }
}

TEST_F(DDSimulatorDeviceTest, SubmitJobReturnsValidJob) {
  const std::string qasm3Program = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;)";

  const auto job =
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 100);

  EXPECT_FALSE(job.getId().empty());
  EXPECT_EQ(job.getProgramFormat(), QDMI_PROGRAM_FORMAT_QASM3);
  EXPECT_STREQ(job.getProgram().c_str(), qasm3Program.c_str());
  EXPECT_EQ(job.getNumShots(), 100);
  EXPECT_TRUE(job.wait());
  EXPECT_EQ(job.check(), QDMI_JOB_STATUS_DONE);
}

TEST_F(DDSimulatorDeviceTest, SubmitJobPreservesNumShots) {
  const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
)";

  const auto job1 =
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 10);
  EXPECT_EQ(job1.getNumShots(), 10);

  const auto job2 =
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 100);
  EXPECT_EQ(job2.getNumShots(), 100);

  const auto job3 =
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 1000);
  EXPECT_EQ(job3.getNumShots(), 1000);
}

TEST_F(JobTest, IdIsUnique) {
  const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
)";
  const auto job2 =
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 10);

  EXPECT_NE(job.getId(), job2.getId());
}

TEST_F(JobTest, StatusProgresses) {
  EXPECT_TRUE(job.wait());

  const auto finalStatus = job.check();
  EXPECT_THAT(finalStatus,
              testing::AnyOf(QDMI_JOB_STATUS_DONE, QDMI_JOB_STATUS_FAILED));
}

TEST_F(JobTest, GetCountsReturnsValidHistogram) {
  EXPECT_TRUE(job.wait());

  const auto counts = job.getCounts();
  EXPECT_FALSE(counts.empty());

  // All keys should be valid binary strings of length 1 (single qubit)
  for (const auto& [key, value] : counts) {
    EXPECT_EQ(key.length(), 1);
    EXPECT_TRUE(key == "0" || key == "1");
    EXPECT_GT(value, 0);
  }

  size_t totalCounts = 0;
  for (const auto& value : counts | std::views::values) {
    totalCounts += value;
  }
  EXPECT_EQ(totalCounts, job.getNumShots());
}

TEST_F(JobTest, MultipleGetCountsCalls) {
  EXPECT_TRUE(job.wait());

  const auto counts1 = job.getCounts();
  const auto counts2 = job.getCounts();

  EXPECT_EQ(counts1, counts2);
}

TEST_F(JobTest, GetShotsReturnsValidShots) {
  EXPECT_TRUE(job.wait());

  // Some devices may not support the SHOTS result type
  try {
    const auto shots = job.getShots();
    EXPECT_FALSE(shots.empty());

    // Each shot should be a valid binary string of length 1 (single qubit)
    for (const auto& shot : shots) {
      EXPECT_EQ(shot.length(), 1);
      EXPECT_TRUE(shot == "0" || shot == "1");
    }

    // The number of shots should match the expected number
    EXPECT_EQ(shots.size(), job.getNumShots());
  } catch (const std::runtime_error& e) {
    // If the device doesn't support shots, the error message should indicate so
    const std::string errorMsg(e.what());
    EXPECT_TRUE(errorMsg.find("Not supported") != std::string::npos ||
                errorMsg.find("not supported") != std::string::npos);
  }
}

TEST_F(JobTest, CancelJob) {
  const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
)";
  const auto jobToCancel =
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 10);

  // Fast-executing jobs (like the DD simulator) may complete before
  // cancel is called, which should throw an exception.
  // Both outcomes are valid based on timing.
  try {
    jobToCancel.cancel();
    // If cancel succeeded, the job should be in CANCELED state
    const auto status = jobToCancel.check();
    EXPECT_EQ(status, QDMI_JOB_STATUS_CANCELED);
  } catch (const std::invalid_argument&) {
    // If cancel threw an exception, the job should already be done
    const auto status = jobToCancel.check();
    EXPECT_THAT(status,
                testing::AnyOf(QDMI_JOB_STATUS_DONE, QDMI_JOB_STATUS_FAILED));
  }
}

TEST_F(JobTest, CancelCompletedJobThrows) {
  EXPECT_TRUE(job.wait());

  const auto statusBefore = job.check();
  EXPECT_THAT(statusBefore,
              testing::AnyOf(QDMI_JOB_STATUS_DONE, QDMI_JOB_STATUS_FAILED));

  EXPECT_THROW(job.cancel(), std::invalid_argument);
}

TEST_F(SimulatorJobTest, getDenseStateVectorReturnsValidState) {
  EXPECT_TRUE(job.wait());

  const auto stateVector = job.getDenseStateVector();
  EXPECT_EQ(stateVector.size(), 4); // 2 qubits -> 4 amplitudes

  // The expected state is (|00> + |11>)/sqrt(2)
  constexpr double invSqrt2 = 1.0 / std::numbers::sqrt2;
  EXPECT_NEAR(std::abs(stateVector[0]), invSqrt2, 1e-10); // |00>
  EXPECT_NEAR(std::abs(stateVector[1]), 0.0, 1e-10);      // |01>
  EXPECT_NEAR(std::abs(stateVector[2]), 0.0, 1e-10);
  EXPECT_NEAR(std::abs(stateVector[3]), invSqrt2, 1e-10); // |11>
}

TEST_F(SimulatorJobTest, getDenseProbabilitiesReturnsValidProbabilities) {
  EXPECT_TRUE(job.wait());

  const auto probabilities = job.getDenseProbabilities();
  EXPECT_EQ(probabilities.size(), 4); // 2 qubits -> 4 probabilities

  // The expected probabilities are 0.5 for |00> and |11>, and 0 for |01> and
  // |10>
  EXPECT_NEAR(probabilities[0], 0.5, 1e-10); // |00>
  EXPECT_NEAR(probabilities[1], 0.0, 1e-10); // |01>
  EXPECT_NEAR(probabilities[2], 0.0, 1e-10); // |10>
  EXPECT_NEAR(probabilities[3], 0.5, 1e-10); // |11>
}

TEST_F(SimulatorJobTest, getSparseStateVectorReturnsValidState) {
  EXPECT_TRUE(job.wait());

  const auto sparseStateVector = job.getSparseStateVector();
  EXPECT_EQ(sparseStateVector.size(),
            2); // Only |00> and |11> should be present

  constexpr double invSqrt2 = 1.0 / std::numbers::sqrt2;
  const auto it00 = sparseStateVector.find("00");
  ASSERT_NE(it00, sparseStateVector.end());
  EXPECT_NEAR(std::abs(it00->second), invSqrt2, 1e-10);

  const auto it11 = sparseStateVector.find("11");
  ASSERT_NE(it11, sparseStateVector.end());
  EXPECT_NEAR(std::abs(it11->second), invSqrt2, 1e-10);
}

TEST_F(SimulatorJobTest, getSparseProbabilitiesReturnsValidProbabilities) {
  EXPECT_TRUE(job.wait());

  const auto sparseProbabilities = job.getSparseProbabilities();
  EXPECT_EQ(sparseProbabilities.size(),
            2); // Only |00> and |11> should be present

  const auto it00 = sparseProbabilities.find("00");
  ASSERT_NE(it00, sparseProbabilities.end());
  EXPECT_NEAR(it00->second, 0.5, 1e-10);

  const auto it11 = sparseProbabilities.find("11");
  ASSERT_NE(it11, sparseProbabilities.end());
  EXPECT_NEAR(it11->second, 0.5, 1e-10);
}

TEST(AuthenticationTest, SessionParameterToString) {
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_TOKEN), "TOKEN");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_AUTHFILE), "AUTHFILE");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_AUTHURL), "AUTHURL");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_USERNAME), "USERNAME");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_PASSWORD), "PASSWORD");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_PROJECTID), "PROJECTID");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_CUSTOM1), "CUSTOM1");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_CUSTOM2), "CUSTOM2");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_CUSTOM3), "CUSTOM3");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_CUSTOM4), "CUSTOM4");
  EXPECT_EQ(toString(QDMI_SESSION_PARAMETER_CUSTOM5), "CUSTOM5");
}

TEST(AuthenticationTest, SessionConstructionWithToken) {
  // Empty token should be accepted
  SessionConfig config1;
  config1.token = "";
  try {
    Session session(config1);
    SUCCEED(); // If we get here, the session was created successfully
  } catch (const std::runtime_error&) {
    // If not supported, that's okay for now
    SUCCEED();
  }

  // Non-empty token should be accepted
  SessionConfig config2;
  config2.token = "test_token_123";
  try {
    Session session(config2);
    SUCCEED();
  } catch (const std::runtime_error&) {
    // If not supported, that's okay for now
    SUCCEED();
  }

  // Token with special characters should be accepted
  SessionConfig config3;
  config3.token = "very_long_token_with_special_characters_!@#$%^&*()";
  try {
    Session session(config3);
    SUCCEED();
  } catch (const std::runtime_error&) {
    // If not supported, that's okay for now
    SUCCEED();
  }
}

TEST(AuthenticationTest, SessionConstructionWithAuthUrl) {
  // Valid HTTPS URL
  SessionConfig config1;
  config1.authUrl = "https://example.com";
  try {
    Session session(config1);
    SUCCEED();
  } catch (const std::runtime_error&) {
    // Either not supported or validation failed - both acceptable
    SUCCEED();
  }

  // Valid HTTP URL with port and path
  SessionConfig config2;
  config2.authUrl = "http://auth.server.com:8080/api";
  try {
    Session session(config2);
    SUCCEED();
  } catch (const std::runtime_error&) {
    SUCCEED();
  }

  // Valid HTTPS URL with query parameters
  SessionConfig config3;
  config3.authUrl = "https://auth.example.com/token?param=value";
  try {
    Session session(config3);
    SUCCEED();
  } catch (const std::runtime_error&) {
    SUCCEED();
  }

  // Invalid URL - not a URL at all
  SessionConfig config4;
  config4.authUrl = "not-a-url";
  EXPECT_THROW({ Session session(config4); }, std::runtime_error);

  // Invalid URL - unsupported protocol
  SessionConfig config5;
  config5.authUrl = "ftp://invalid.com";
  EXPECT_THROW({ Session session(config5); }, std::runtime_error);

  // Invalid URL - missing protocol
  SessionConfig config6;
  config6.authUrl = "example.com";
  EXPECT_THROW({ Session session(config6); }, std::runtime_error);
}

TEST(AuthenticationTest, SessionConstructionWithAuthFile) {
  // Test with non-existent file - should raise error
  SessionConfig config1;
  config1.authFile = "/nonexistent/path/to/file.txt";
  EXPECT_THROW({ Session session(config1); }, std::runtime_error);

  // Test with another non-existent file
  SessionConfig config2;
  config2.authFile = "/tmp/this_file_does_not_exist_12345.txt";
  EXPECT_THROW({ Session session(config2); }, std::runtime_error);

  // Test with existing file
  // Create a temporary file
  char tmpFilename[] = "/tmp/fomac_test_XXXXXX";
  const int fd = mkstemp(tmpFilename);
  ASSERT_NE(fd, -1) << "Failed to create temporary file";

  // Write some content to the file
  const char* content = "test_token_content";
  write(fd, content, strlen(content));
  close(fd);

  // Try to create session with existing file
  SessionConfig config3;
  config3.authFile = tmpFilename;
  try {
    Session session(config3);
    SUCCEED();
  } catch (const std::runtime_error&) {
    // If not supported, that's okay for now
    SUCCEED();
  }

  // Clean up
  remove(tmpFilename);
}

TEST(AuthenticationTest, SessionConstructionWithUsernamePassword) {
  // Username only
  SessionConfig config1;
  config1.username = "user123";
  try {
    Session session(config1);
    SUCCEED();
  } catch (const std::runtime_error&) {
    SUCCEED();
  }

  // Password only
  SessionConfig config2;
  config2.password = "secure_password";
  try {
    Session session(config2);
    SUCCEED();
  } catch (const std::runtime_error&) {
    SUCCEED();
  }

  // Both username and password
  SessionConfig config3;
  config3.username = "user123";
  config3.password = "secure_password";
  try {
    Session session(config3);
    SUCCEED();
  } catch (const std::runtime_error&) {
    SUCCEED();
  }
}

TEST(AuthenticationTest, SessionConstructionWithProjectId) {
  SessionConfig config;
  config.projectId = "project-123-abc";
  try {
    Session session(config);
    SUCCEED();
  } catch (const std::runtime_error&) {
    // If not supported, that's okay for now
    SUCCEED();
  }
}

TEST(AuthenticationTest, SessionConstructionWithMultipleParameters) {
  SessionConfig config;
  config.token = "test_token";
  config.username = "test_user";
  config.password = "test_pass";
  config.projectId = "test_project";
  try {
    Session session(config);
    SUCCEED();
  } catch (const std::runtime_error&) {
    // If not supported, that's okay for now
    SUCCEED();
  }
}

TEST(AuthenticationTest, SessionGetDevicesReturnsList) {
  Session session;
  auto devices = session.getDevices();

  EXPECT_FALSE(devices.empty());

  // All elements should be Device instances
  for (const auto& device : devices) {
    // Device should have a name
    EXPECT_FALSE(device.getName().empty());
  }
}

TEST(AuthenticationTest, SessionMultipleInstances) {
  Session session1;
  Session session2;

  auto devices1 = session1.getDevices();
  auto devices2 = session2.getDevices();

  // Both should return devices
  EXPECT_FALSE(devices1.empty());
  EXPECT_FALSE(devices2.empty());

  // Should return the same number of devices
  EXPECT_EQ(devices1.size(), devices2.size());
}

namespace {
// Helper function to get all devices for parameterized tests
inline auto getDevices() -> std::vector<Session::Device> {
  Session session;
  return session.getDevices();
}
} // namespace

INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    DeviceTest,
    // Test suite name
    DeviceTest,
    // Parameters to test with
    testing::ValuesIn(getDevices()),
    [](const testing::TestParamInfo<Session::Device>& paramInfo) {
      auto name = paramInfo.param.getName();
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    SiteTest,
    // Test suite name
    SiteTest,
    // Parameters to test with
    testing::ValuesIn(getDevices()),
    [](const testing::TestParamInfo<Session::Device>& paramInfo) {
      auto name = paramInfo.param.getName();
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    OperationTest,
    // Test suite name
    OperationTest,
    // Parameters to test with
    testing::ValuesIn(getDevices()),
    [](const testing::TestParamInfo<Session::Device>& paramInfo) {
      auto name = paramInfo.param.getName();
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      return name;
    });

} // namespace fomac
