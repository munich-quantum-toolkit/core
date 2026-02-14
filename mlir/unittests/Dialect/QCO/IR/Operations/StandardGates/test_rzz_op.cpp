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
    QCORZZOpTest, QCOTest,
    testing::Values(QCOTestCase{"RZZ", MQT_NAMED_BUILDER(rzz),
                                MQT_NAMED_BUILDER(rzz)},
                    QCOTestCase{"SingleControlledRZZ",
                                MQT_NAMED_BUILDER(singleControlledRzz),
                                MQT_NAMED_BUILDER(singleControlledRzz)},
                    QCOTestCase{"MultipleControlledRZZ",
                                MQT_NAMED_BUILDER(multipleControlledRzz),
                                MQT_NAMED_BUILDER(multipleControlledRzz)},
                    QCOTestCase{"NestedControlledRZZ",
                                MQT_NAMED_BUILDER(nestedControlledRzz),
                                MQT_NAMED_BUILDER(multipleControlledRzz)},
                    QCOTestCase{"TrivialControlledRZZ",
                                MQT_NAMED_BUILDER(trivialControlledRzz),
                                MQT_NAMED_BUILDER(rzz)},
                    QCOTestCase{"InverseRZZ", MQT_NAMED_BUILDER(inverseRzz),
                                MQT_NAMED_BUILDER(rzz)},
                    QCOTestCase{"InverseMultipleControlledRZZ",
                                MQT_NAMED_BUILDER(inverseMultipleControlledRzz),
                                MQT_NAMED_BUILDER(multipleControlledRzz)},
                    QCOTestCase{"TwoRZZOppositePhase",
                                MQT_NAMED_BUILDER(twoRzzOppositePhase),
                                MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, RZZOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rzz);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto rzzOp = *funcOp.getBody().getOps<RZZOp>().begin();
  const auto matrix = *rzzOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RZZ, {0.123});

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
