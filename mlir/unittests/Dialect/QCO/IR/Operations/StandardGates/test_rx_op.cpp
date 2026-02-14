/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/GateMatrixDefinitions.hpp"
#include "ir/operations/OpType.hpp"
#include "qco_programs.h"
#include "test_qco_ir.h"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCORXOpTest, QCOTest,
    testing::Values(QCOTestCase{"RX", rx, rx},
                    QCOTestCase{"SingleControlledRX", singleControlledRx,
                                singleControlledRx},
                    QCOTestCase{"MultipleControlledRX", multipleControlledRx,
                                multipleControlledRx},
                    QCOTestCase{"NestedControlledRX", nestedControlledRx,
                                multipleControlledRx},
                    QCOTestCase{"TrivialControlledRX", trivialControlledRx, rx},
                    QCOTestCase{"InverseRX", inverseRx, rx},
                    QCOTestCase{"InverseMultipleControlledRX",
                                inverseMultipleControlledRx,
                                multipleControlledRx}),
    printTestName);

TEST_F(QCOTest, RXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto rxOp = *funcOp.getBody().getOps<RXOp>().begin();
  const auto matrix = *rxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::RX, {0.123});

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
