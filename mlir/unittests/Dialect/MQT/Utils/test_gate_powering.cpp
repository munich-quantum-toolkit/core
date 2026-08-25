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

#include <limits>

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
