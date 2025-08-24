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

#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>

namespace fomac {
class DeviceTest : public testing::Test {
protected:
  Device device;

  DeviceTest() : device(FoMaC::getDevices().front()) {}
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

TEST_F(DeviceTest, Name) {
  EXPECT_NO_THROW(EXPECT_FALSE(device.getName().empty()););
}

TEST_F(SiteTest, Index) { EXPECT_NO_THROW(std::ignore = site.getIndex();); }

TEST_F(OperationTest, Name) {
  EXPECT_NO_THROW(EXPECT_FALSE(operation.getName().empty()););
}
} // namespace fomac
