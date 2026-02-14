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
    QCORXXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RXX", rxx, rxx},
        QCOTestCase{"SingleControlledRXX", singleControlledRxx,
                    singleControlledRxx},
        QCOTestCase{"MultipleControlledRXX", multipleControlledRxx,
                    multipleControlledRxx},
        QCOTestCase{"NestedControlledRXX", nestedControlledRxx,
                    multipleControlledRxx},
        QCOTestCase{"TrivialControlledRXX", trivialControlledRxx, rxx},
        QCOTestCase{"InverseRXX", inverseRxx, rxx},
        QCOTestCase{"InverseMultipleControlledRXX",
                    inverseMultipleControlledRxx, multipleControlledRxx}),
    printTestName);

TEST_F(QCOTest, RXXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rxx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto rxxOp = *funcOp.getBody().getOps<RXXOp>().begin();
  const auto matrix = *rxxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RXX, {0.123});

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
