/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/GatePowering.h"

#include <gtest/gtest.h>
#include <llvm/ADT/StringRef.h>

#include <limits>

TEST(GatePoweringTest, RecognizesFixedGatePowerPeriods) {
  for (llvm::StringRef gate : {"x", "y", "z", "h", "ecr", "rccx", "swap"}) {
    SCOPED_TRACE(gate.str());
    EXPECT_EQ(mlir::mqt::getFixedGatePowerPeriod(gate), 2U);
  }
  for (llvm::StringRef gate : {"s", "sdg", "sx", "sxdg", "iswap"}) {
    SCOPED_TRACE(gate.str());
    EXPECT_EQ(mlir::mqt::getFixedGatePowerPeriod(gate), 4U);
  }
  for (llvm::StringRef gate : {"t", "tdg"}) {
    SCOPED_TRACE(gate.str());
    EXPECT_EQ(mlir::mqt::getFixedGatePowerPeriod(gate), 8U);
  }
}

TEST(GatePoweringTest, DoesNotAssignFixedPeriodsToOtherGates) {
  for (llvm::StringRef gate : {"rx", "ry", "rz", "p", "u", "r", "dcx", ""}) {
    SCOPED_TRACE(gate.str());
    EXPECT_EQ(mlir::mqt::getFixedGatePowerPeriod(gate), 0U);
  }
}

TEST(GatePoweringTest, recognizesIntegerExponents) {
  EXPECT_TRUE(mlir::mqt::isIntegerExponent(-2.0));
  EXPECT_TRUE(mlir::mqt::isIntegerExponent(0.0));
  EXPECT_FALSE(mlir::mqt::isIntegerExponent(0.5));
  EXPECT_FALSE(
      mlir::mqt::isIntegerExponent(std::numeric_limits<double>::infinity()));
}

TEST(GatePoweringTest, recognizesEvenIntegerExponents) {
  EXPECT_TRUE(mlir::mqt::isEvenExponent(-2.0));
  EXPECT_TRUE(mlir::mqt::isEvenExponent(0.0));
  EXPECT_FALSE(mlir::mqt::isEvenExponent(3.0));
  EXPECT_FALSE(mlir::mqt::isEvenExponent(2.5));
  EXPECT_FALSE(
      mlir::mqt::isEvenExponent(std::numeric_limits<double>::infinity()));
}
