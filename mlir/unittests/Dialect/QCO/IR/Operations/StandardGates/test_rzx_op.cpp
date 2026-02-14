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
    QCORZXOpTest, QCOTest,
    testing::Values(QCOTestCase{"RZX", MQT_NAMED_BUILDER(rzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QCOTestCase{"SingleControlledRZX",
                                MQT_NAMED_BUILDER(singleControlledRzx),
                                MQT_NAMED_BUILDER(singleControlledRzx)},
                    QCOTestCase{"MultipleControlledRZX",
                                MQT_NAMED_BUILDER(multipleControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCOTestCase{"NestedControlledRZX",
                                MQT_NAMED_BUILDER(nestedControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCOTestCase{"TrivialControlledRZX",
                                MQT_NAMED_BUILDER(trivialControlledRzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QCOTestCase{"InverseRZX", MQT_NAMED_BUILDER(inverseRzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QCOTestCase{"InverseMultipleControlledRZX",
                                MQT_NAMED_BUILDER(inverseMultipleControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCOTestCase{"TwoRZXOppositePhase",
                                MQT_NAMED_BUILDER(twoRzxOppositePhase),
                                MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, RZXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rzx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto rzxOp = *funcOp.getBody().getOps<RZXOp>().begin();
  const auto matrix = *rzxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RZX, {0.123});

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
