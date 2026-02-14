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

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOSWAPOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SWAP", MQT_NAMED_BUILDER(swap), MQT_NAMED_BUILDER(swap)},
        QCOTestCase{"SingleControlledSWAP",
                    MQT_NAMED_BUILDER(singleControlledSwap),
                    MQT_NAMED_BUILDER(singleControlledSwap)},
        QCOTestCase{"MultipleControlledSWAP",
                    MQT_NAMED_BUILDER(multipleControlledSwap),
                    MQT_NAMED_BUILDER(multipleControlledSwap)},
        QCOTestCase{"NestedControlledSWAP",
                    MQT_NAMED_BUILDER(nestedControlledSwap),
                    MQT_NAMED_BUILDER(multipleControlledSwap)},
        QCOTestCase{"TrivialControlledSWAP",
                    MQT_NAMED_BUILDER(trivialControlledSwap),
                    MQT_NAMED_BUILDER(swap)},
        QCOTestCase{"InverseSWAP", MQT_NAMED_BUILDER(inverseSwap),
                    MQT_NAMED_BUILDER(swap)},
        QCOTestCase{"InverseMultipleControlledSWAP",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSwap),
                    MQT_NAMED_BUILDER(multipleControlledSwap)},
        QCOTestCase{"TwoSWAP", MQT_NAMED_BUILDER(twoSwap),
                    MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, SWAPOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SWAPOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::SWAP);

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
