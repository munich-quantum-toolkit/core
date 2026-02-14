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
    QCORXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RX", MQT_NAMED_BUILDER(rx), MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"SingleControlledRX", MQT_NAMED_BUILDER(singleControlledRx),
                    MQT_NAMED_BUILDER(singleControlledRx)},
        QCOTestCase{"MultipleControlledRX",
                    MQT_NAMED_BUILDER(multipleControlledRx),
                    MQT_NAMED_BUILDER(multipleControlledRx)},
        QCOTestCase{"NestedControlledRX", MQT_NAMED_BUILDER(nestedControlledRx),
                    MQT_NAMED_BUILDER(multipleControlledRx)},
        QCOTestCase{"TrivialControlledRX",
                    MQT_NAMED_BUILDER(trivialControlledRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"InverseRX", MQT_NAMED_BUILDER(inverseRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"InverseMultipleControlledRX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRx),
                    MQT_NAMED_BUILDER(multipleControlledRx)},
        QCOTestCase{"TwoRXOppositePhase", MQT_NAMED_BUILDER(twoRxOppositePhase),
                    MQT_NAMED_BUILDER(emptyQCO)}));

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
