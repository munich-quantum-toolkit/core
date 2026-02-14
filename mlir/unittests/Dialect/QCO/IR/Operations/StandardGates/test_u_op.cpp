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
    QCOUOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"U", u, u},
        QCOTestCase{"SingleControlledU", singleControlledU, singleControlledU},
        QCOTestCase{"MultipleControlledU", multipleControlledU,
                    multipleControlledU},
        QCOTestCase{"NestedControlledU", nestedControlledU,
                    multipleControlledU},
        QCOTestCase{"TrivialControlledU", trivialControlledU, u},
        QCOTestCase{"InverseU", inverseU, u},
        QCOTestCase{"InverseMultipleControlledU", inverseMultipleControlledU,
                    multipleControlledU}),
    printTestName);

TEST_F(QCOTest, UOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), u);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto uOp = *funcOp.getBody().getOps<UOp>().begin();
  const auto matrix = *uOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::U, {0.1, 0.2, 0.3});

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
