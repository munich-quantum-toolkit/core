/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Package.hpp"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <cstddef>
#include <memory>

namespace mqt::test {

/**
 * @brief Compare complete QCO function matrices, including global phase.
 *
 * This deliberately performs entry-by-entry matrix equality within @p
 * tolerance. It never quotients out a global phase, so an incorrect rewrite
 * becomes observable here and when the rewritten function is put under an
 * additional control.
 */
inline void expectFullUnitaryEqual(mlir::ModuleOp expectedModule,
                                   mlir::ModuleOp actualModule,
                                   const std::size_t numQubits,
                                   const double tolerance = 1e-12) {
  auto expectedFunc =
      *expectedModule.getBody()->getOps<mlir::func::FuncOp>().begin();
  auto actualFunc =
      *actualModule.getBody()->getOps<mlir::func::FuncOp>().begin();
  auto package = std::make_unique<dd::Package>(numQubits);
  const auto expected = mlir::qco::buildFunctionality(expectedFunc, *package);
  const auto actual = mlir::qco::buildFunctionality(actualFunc, *package);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));

  const auto expectedMatrix = expected->getMatrix(numQubits);
  const auto actualMatrix = actual->getMatrix(numQubits);
  ASSERT_EQ(expectedMatrix.size(), actualMatrix.size());
  for (std::size_t row = 0; row < expectedMatrix.size(); ++row) {
    ASSERT_EQ(expectedMatrix[row].size(), actualMatrix[row].size());
    for (std::size_t column = 0; column < expectedMatrix[row].size();
         ++column) {
      SCOPED_TRACE(testing::Message()
                   << "matrix entry (" << row << ", " << column << ")");
      EXPECT_LE(
          std::abs(expectedMatrix[row][column] - actualMatrix[row][column]),
          tolerance);
    }
  }
  package->decRef(*expected);
  package->decRef(*actual);
}

} // namespace mqt::test
