/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "dd/GateMatrixDefinitions.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "qco_programs.h"
#include "test_qco_ir.h"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCORYYOpTest, QCOTest,
    testing::Values(QCOTestCase{"RYY", MQT_NAMED_BUILDER(ryy),
                                MQT_NAMED_BUILDER(ryy)},
                    QCOTestCase{"SingleControlledRYY",
                                MQT_NAMED_BUILDER(singleControlledRyy),
                                MQT_NAMED_BUILDER(singleControlledRyy)},
                    QCOTestCase{"MultipleControlledRYY",
                                MQT_NAMED_BUILDER(multipleControlledRyy),
                                MQT_NAMED_BUILDER(multipleControlledRyy)},
                    QCOTestCase{"NestedControlledRYY",
                                MQT_NAMED_BUILDER(nestedControlledRyy),
                                MQT_NAMED_BUILDER(multipleControlledRyy)},
                    QCOTestCase{"TrivialControlledRYY",
                                MQT_NAMED_BUILDER(trivialControlledRyy),
                                MQT_NAMED_BUILDER(ryy)},
                    QCOTestCase{"InverseRYY", MQT_NAMED_BUILDER(inverseRyy),
                                MQT_NAMED_BUILDER(ryy)},
                    QCOTestCase{"InverseMultipleControlledRYY",
                                MQT_NAMED_BUILDER(inverseMultipleControlledRyy),
                                MQT_NAMED_BUILDER(multipleControlledRyy)},
                    QCOTestCase{"TwoRYYOppositePhase",
                                MQT_NAMED_BUILDER(twoRyyOppositePhase),
                                MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, RYYOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), ryy);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto ryyOp = *funcOp.getBody().getOps<RYYOp>().begin();
  const auto matrix = *ryyOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RYY, {0.123});

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
