/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/FoMaC.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fomac {
#define MAYBE_NOT_SUPPORTED(statement)                                         \
  try {                                                                        \
    statement;                                                                 \
  } catch (const std::runtime_error& exception) {                              \
    if (!std::string(exception.what()).ends_with("Not supported.")) {          \
      FAIL()                                                                   \
          << "Expected: " #statement                                           \
             " throws an exception of type std::runtime_error that ends with " \
             "\"Not supported.\"\n"                                            \
             "Actual: it throws std::runtime_error with a different message: " \
          << exception.what();                                                 \
    }                                                                          \
  } catch (const std::exception& exception) {                                  \
    FAIL()                                                                     \
        << "Expected: " #statement                                             \
           " throws an exception of type std::runtime_error that ends with "   \
           "\"Not supported.\"\n"                                              \
           "Actual: it throws a different type: "                              \
        << exception.what();                                                   \
  }
class DeviceTest : public testing::TestWithParam<Device> {
protected:
  Device device;

  DeviceTest() : device(GetParam()) {}
};

class SiteTest : public DeviceTest {
protected:
  Site site;

  SiteTest() : site(device.getSites().front()) {}
};

class OperationTest : public DeviceTest {
protected:
  Operation operation;

  OperationTest() : operation(device.getOperations().front()) {}
};

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
  MAYBE_NOT_SUPPORTED(std::ignore = device.getCouplingMap());
}

TEST_P(DeviceTest, NeedsCalibration) {
  MAYBE_NOT_SUPPORTED(std::ignore = device.getNeedsCalibration());
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
  MAYBE_NOT_SUPPORTED(std::ignore = device.getMinAtomDistance());
}

TEST_P(SiteTest, Index) { EXPECT_NO_THROW(std::ignore = site.getIndex()); }

TEST_P(SiteTest, T1) { MAYBE_NOT_SUPPORTED(std::ignore = site.getT1()); }

TEST_P(SiteTest, T2) { MAYBE_NOT_SUPPORTED(std::ignore = site.getT2()); }

TEST_P(SiteTest, Name) { MAYBE_NOT_SUPPORTED(std::ignore = site.getName()); }

TEST_P(SiteTest, XCoordinate) {
  MAYBE_NOT_SUPPORTED(std::ignore = site.getXCoordinate());
}

TEST_P(SiteTest, YCoordinate) {
  MAYBE_NOT_SUPPORTED(std::ignore = site.getYCoordinate());
}

TEST_P(SiteTest, ZCoordinate) {
  MAYBE_NOT_SUPPORTED(std::ignore = site.getZCoordinate());
}

TEST_P(SiteTest, IsZone) { MAYBE_NOT_SUPPORTED(std::ignore = site.isZone()); }

TEST_P(SiteTest, XExtent) {
  MAYBE_NOT_SUPPORTED(std::ignore = site.getXExtent());
}

TEST_P(SiteTest, YExtent) {
  MAYBE_NOT_SUPPORTED(std::ignore = site.getYExtent());
}

TEST_P(SiteTest, ZExtent) {
  MAYBE_NOT_SUPPORTED(std::ignore = site.getZExtent());
}

TEST_P(SiteTest, ModuleIndex) {
  MAYBE_NOT_SUPPORTED(std::ignore = site.getModuleIndex());
}

TEST_P(SiteTest, SubmoduleIndex) {
  MAYBE_NOT_SUPPORTED(std::ignore = site.getSubmoduleIndex());
}

TEST_P(OperationTest, Name) {
  EXPECT_NO_THROW(EXPECT_FALSE(operation.getName().empty()););
}

TEST_P(OperationTest, QubitsNum) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.getQubitsNum());
}

TEST_P(OperationTest, ParametersNum) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.getParametersNum());
}

TEST_P(OperationTest, Duration) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.getDuration());
}

TEST_P(OperationTest, Fidelity) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.getFidelity());
}

TEST_P(OperationTest, InteractionRadius) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.getInteractionRadius());
}

TEST_P(OperationTest, BlockingRadius) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.getBlockingRadius());
}

TEST_P(OperationTest, IdlingFidelity) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.getIdlingFidelity());
}

TEST_P(OperationTest, oned) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.isZoned());
}

TEST_P(OperationTest, Sites) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.getSites());
}

TEST_P(OperationTest, MeanShuttlingSpeed) {
  MAYBE_NOT_SUPPORTED(std::ignore = operation.getMeanShuttlingSpeed());
}

INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    DeviceTest,
    // Test suite name
    DeviceTest,
    // Parameters to test with
    testing::ValuesIn(FoMaC::getDevices()),
    [](const testing::TestParamInfo<Device>& paramInfo) {
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
    [](const testing::TestParamInfo<Device>& paramInfo) {
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
    [](const testing::TestParamInfo<Device>& paramInfo) {
      auto name = paramInfo.param.getName();
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      return name;
    });
} // namespace fomac
