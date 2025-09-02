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
#include <gtest/gtest.h>
#include <new>
#include <qdmi/client.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fomac {
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
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(toString(static_cast<QDMI_STATUS>(-999)), "Unknown status code");
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
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(toString(static_cast<QDMI_Site_Property>(-999)),
            "QDMI_SITE_PROPERTY_UNKNOWN");
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
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(toString(static_cast<QDMI_Operation_Property>(-999)),
            "QDMI_OPERATION_PROPERTY_UNKNOWN");
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
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(toString(static_cast<QDMI_Device_Property>(-999)),
            "QDMI_DEVICE_PROPERTY_UNKNOWN");
}

TEST(FoMaCTest, SessionPropertyToString) {
  EXPECT_EQ(toString(QDMI_SESSION_PROPERTY_DEVICES),
            "QDMI_SESSION_PROPERTY_DEVICES");
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(toString(static_cast<QDMI_Session_Property>(-999)),
            "QDMI_SESSION_PROPERTY_UNKNOWN");
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
  EXPECT_THROW(throwIfError(-999, "Test"), std::runtime_error);
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

TEST_P(OperationTest, MeanShuttlingSpeed) {
  EXPECT_NO_THROW(std::ignore = operation.getMeanShuttlingSpeed());
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
