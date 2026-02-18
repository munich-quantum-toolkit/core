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
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "qco_programs.h"
#include "test_qco_ir.h"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCORYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RY", MQT_NAMED_BUILDER(ry), MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"SingleControlledRY", MQT_NAMED_BUILDER(singleControlledRy),
                    MQT_NAMED_BUILDER(singleControlledRy)},
        QCOTestCase{"MultipleControlledRY",
                    MQT_NAMED_BUILDER(multipleControlledRy),
                    MQT_NAMED_BUILDER(multipleControlledRy)},
        QCOTestCase{"NestedControlledRY", MQT_NAMED_BUILDER(nestedControlledRy),
                    MQT_NAMED_BUILDER(multipleControlledRy)},
        QCOTestCase{"TrivialControlledRY",
                    MQT_NAMED_BUILDER(trivialControlledRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"InverseRY", MQT_NAMED_BUILDER(inverseRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"InverseMultipleControlledRY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRy),
                    MQT_NAMED_BUILDER(multipleControlledRy)},
        QCOTestCase{"TwoRYOppositePhase", MQT_NAMED_BUILDER(twoRyOppositePhase),
                    MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, RYOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), ry);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto ryOp = *funcOp.getBody().getOps<RYOp>().begin();
  const auto matrix = *ryOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::RY, {0.456});

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
