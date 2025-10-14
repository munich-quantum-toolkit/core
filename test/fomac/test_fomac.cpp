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
#include <gtest/gtest.h>
#include <new>
#include <qdmi/client.h>
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
  FoMaC::Device::Site site;

  SiteTest() : site(device.getSites().front()) {}
};

class OperationTest : public DeviceTest {
protected:
  FoMaC::Device::Operation operation;

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

TEST_P(DeviceTest, CachingSites) {
  // Test that repeated calls to getSites() return equivalent data
  const auto sites1 = device.getSites();
  const auto sites2 = device.getSites();

  // Both calls should return the same number of sites
  EXPECT_EQ(sites1.size(), sites2.size());

  // Verify that the sites are equivalent (same indices)
  for (std::size_t i = 0; i < sites1.size(); ++i) {
    EXPECT_EQ(sites1[i].getIndex(), sites2[i].getIndex());
  }
}

TEST_P(DeviceTest, CachingOperations) {
  // Test that repeated calls to getOperations() return equivalent data
  const auto ops1 = device.getOperations();
  const auto ops2 = device.getOperations();

  // Both calls should return the same number of operations
  EXPECT_EQ(ops1.size(), ops2.size());

  // Verify that the operations are equivalent (same names)
  for (std::size_t i = 0; i < ops1.size(); ++i) {
    EXPECT_EQ(ops1[i].getName(), ops2[i].getName());
  }
}

TEST_P(DeviceTest, CachingCouplingMap) {
  // Test that repeated calls to getCouplingMap() return equivalent data
  const auto cm1 = device.getCouplingMap();
  const auto cm2 = device.getCouplingMap();

  // Both calls should have the same value (both present or both nullopt)
  EXPECT_EQ(cm1.has_value(), cm2.has_value());

  if (cm1.has_value() && cm2.has_value()) {
    // If present, they should have the same size and content
    EXPECT_EQ(cm1->size(), cm2->size());

    // Verify that the coupling pairs are equivalent
    for (std::size_t i = 0; i < cm1->size(); ++i) {
      EXPECT_EQ((*cm1)[i].first.getIndex(), (*cm2)[i].first.getIndex());
      EXPECT_EQ((*cm1)[i].second.getIndex(), (*cm2)[i].second.getIndex());
    }
  }
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
