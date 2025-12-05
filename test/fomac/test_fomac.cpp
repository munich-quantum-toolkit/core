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
#include <cstdlib>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <new>
#include <numbers>
#include <optional>
#include <qdmi/client.h>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fomac {
class DeviceTest : public testing::TestWithParam<FoMaC::Device> {
protected:
  FoMaC::Device device;

  DeviceTest() : device(GetParam()) {}
};

class SiteTest : public DeviceTest {
protected:
  std::vector<FoMaC::Device::Site> sites;

  void SetUp() override { sites = device.getSites(); }
};

class OperationTest : public DeviceTest {
protected:
  std::vector<FoMaC::Device::Operation> operations;

  void SetUp() override { operations = device.getOperations(); }
};

class DDSimulatorDeviceTest : public testing::Test {
protected:
  FoMaC::Device device;

  DDSimulatorDeviceTest() : device(getDDSimulatorDevice()) {}

private:
  static auto getDDSimulatorDevice() -> FoMaC::Device {
    for (const auto& dev : FoMaC::getDevices()) {
      if (dev.getName() == "MQT Core DDSIM QDMI Device") {
        return dev;
      }
    }
    throw std::runtime_error("DD simulator device not found");
  }
};

class JobTest : public DDSimulatorDeviceTest {
protected:
  FoMaC::Job job;

  JobTest() : job(createTestJob()) {}

  [[nodiscard]] FoMaC::Job createTestJob() const {
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
  FoMaC::Job job;

  SimulatorJobTest() : job(createTestJob()) {}

  [[nodiscard]] FoMaC::Job createTestJob() const {
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

TEST_P(SiteTest, Index) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getIndex());
  }
}

TEST_P(SiteTest, T1) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getT1());
  }
}

TEST_P(SiteTest, T2) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getT2());
  }
}

TEST_P(SiteTest, Name) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getName());
  }
}

TEST_P(SiteTest, XCoordinate) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getXCoordinate());
  }
}

TEST_P(SiteTest, YCoordinate) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getYCoordinate());
  }
}

TEST_P(SiteTest, ZCoordinate) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getZCoordinate());
  }
}

TEST_P(SiteTest, IsZone) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.isZone());
  }
}

TEST_P(SiteTest, XExtent) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getXExtent());
  }
}

TEST_P(SiteTest, YExtent) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getYExtent());
  }
}

TEST_P(SiteTest, ZExtent) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getZExtent());
  }
}

TEST_P(SiteTest, ModuleIndex) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getModuleIndex());
  }
}

TEST_P(SiteTest, SubmoduleIndex) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = site.getSubmoduleIndex());
  }
}

TEST_P(OperationTest, Name) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(EXPECT_FALSE(operation.getName().empty()));
  }
}

TEST_P(OperationTest, QubitsNum) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = operation.getQubitsNum());
  }
}

TEST_P(OperationTest, ParametersNum) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = operation.getParametersNum());
  }
}

TEST_P(OperationTest, Duration) {
  for (const auto& operation : operations) {
    const auto qubitsNum = operation.getQubitsNum();
    if (!qubitsNum.has_value()) {
      EXPECT_NO_THROW(std::ignore = operation.getDuration());
      continue;
    }
    const auto numQubits = *qubitsNum;
    if (numQubits == 1) {
      const auto sites = operation.getSites();
      if (!sites.has_value()) {
        EXPECT_NO_THROW(std::ignore = operation.getDuration());
        continue;
      }
      for (const auto& site : *sites) {
        EXPECT_NO_THROW(std::ignore = operation.getDuration({site}));
      }
      continue;
    }

    if (numQubits == 2) {
      const auto sitePairs = operation.getSitePairs();
      if (!sitePairs.has_value()) {
        EXPECT_NO_THROW(std::ignore = operation.getDuration());
        continue;
      }
      for (const auto& [site1, site2] : *sitePairs) {
        EXPECT_NO_THROW(std::ignore = operation.getDuration({site1, site2}));
      }
      continue;
    }

    EXPECT_NO_THROW(std::ignore = operation.getDuration());
  }
}

TEST_P(OperationTest, Fidelity) {
  for (const auto& operation : operations) {
    const auto qubitsNum = operation.getQubitsNum();
    if (!qubitsNum.has_value()) {
      EXPECT_NO_THROW(std::ignore = operation.getFidelity());
      continue;
    }
    const auto numQubits = *qubitsNum;
    if (numQubits == 1) {
      const auto sites = operation.getSites();
      if (!sites.has_value()) {
        EXPECT_NO_THROW(std::ignore = operation.getFidelity());
        continue;
      }
      for (const auto& site : *sites) {
        EXPECT_NO_THROW(std::ignore = operation.getFidelity({site}));
      }
      continue;
    }

    if (numQubits == 2) {
      const auto sitePairs = operation.getSitePairs();
      if (!sitePairs.has_value()) {
        EXPECT_NO_THROW(std::ignore = operation.getFidelity());
        continue;
      }
      for (const auto& [site1, site2] : *sitePairs) {
        EXPECT_NO_THROW(std::ignore = operation.getFidelity({site1, site2}));
      }
      continue;
    }

    EXPECT_NO_THROW(std::ignore = operation.getFidelity());
  }
}

TEST_P(OperationTest, InteractionRadius) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = operation.getInteractionRadius());
  }
}

TEST_P(OperationTest, BlockingRadius) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = operation.getBlockingRadius());
  }
}

TEST_P(OperationTest, IdlingFidelity) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = operation.getIdlingFidelity());
  }
}

TEST_P(OperationTest, IsZoned) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = operation.isZoned());
  }
}

TEST_P(OperationTest, Sites) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = operation.getSites());
  }
}

TEST_P(OperationTest, SitePairs) {
  for (const auto& operation : operations) {
    const auto sitePairs = operation.getSitePairs();
    const auto qubitsNum = operation.getQubitsNum();
    const auto isZonedOp = operation.isZoned();

    if (!qubitsNum.has_value() || *qubitsNum != 2 || isZonedOp) {
      EXPECT_FALSE(sitePairs.has_value());
      continue;
    }

    const auto sites = operation.getSites();
    if (!sites.has_value() || sites->empty() || sites->size() % 2 != 0) {
      EXPECT_FALSE(sitePairs.has_value());
      continue;
    }

    EXPECT_TRUE(sitePairs.has_value());
    if (sitePairs.has_value()) {
      EXPECT_EQ(sitePairs->size(), sites->size() / 2);
    }
  }
}

TEST_P(OperationTest, MeanShuttlingSpeed) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = operation.getMeanShuttlingSpeed());
  }
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

INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    DeviceTest,
    // Test suite name
    DeviceTest,
    // Parameters to test with
    testing::ValuesIn(FoMaC::getDevices()),
    [](const testing::TestParamInfo<FoMaC::Device>& paramInfo) {
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
    testing::ValuesIn(FoMaC::getDevices()),
    [](const testing::TestParamInfo<FoMaC::Device>& paramInfo) {
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
    testing::ValuesIn(FoMaC::getDevices()),
    [](const testing::TestParamInfo<FoMaC::Device>& paramInfo) {
      auto name = paramInfo.param.getName();
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      return name;
    });

} // namespace fomac
