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
    QCOXXMinusYYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"XXMinusYY", MQT_NAMED_BUILDER(xxMinusYY),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"SingleControlledXXMinusYY",
                    MQT_NAMED_BUILDER(singleControlledXxMinusYY),
                    MQT_NAMED_BUILDER(singleControlledXxMinusYY)},
        QCOTestCase{"MultipleControlledXXMinusYY",
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCOTestCase{"NestedControlledXXMinusYY",
                    MQT_NAMED_BUILDER(nestedControlledXxMinusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCOTestCase{"TrivialControlledXXMinusYY",
                    MQT_NAMED_BUILDER(trivialControlledXxMinusYY),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"InverseXXMinusYY", MQT_NAMED_BUILDER(inverseXxMinusYY),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"InverseMultipleControlledXXMinusYY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledXxMinusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCOTestCase{"TwoXXMinusYYOppositePhase",
                    MQT_NAMED_BUILDER(twoXxMinusYYOppositePhase),
                    MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, XXMinusYYOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), xxMinusYY);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto xxMinusYYOp = *funcOp.getBody().getOps<XXMinusYYOp>().begin();
  const auto matrix = *xxMinusYYOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToTwoQubitGateMatrix(qc::OpType::XXminusYY, {0.123, 0.456});

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
