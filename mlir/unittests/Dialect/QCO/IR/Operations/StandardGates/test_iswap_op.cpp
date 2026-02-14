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

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOiSWAPOpTest, QCOTest,
    testing::Values(QCOTestCase{"iSWAP", MQT_NAMED_BUILDER(iswap),
                                MQT_NAMED_BUILDER(iswap)},
                    QCOTestCase{"SingleControllediSWAP",
                                MQT_NAMED_BUILDER(singleControlledIswap),
                                MQT_NAMED_BUILDER(singleControlledIswap)},
                    QCOTestCase{"MultipleControllediSWAP",
                                MQT_NAMED_BUILDER(multipleControlledIswap),
                                MQT_NAMED_BUILDER(multipleControlledIswap)},
                    QCOTestCase{"NestedControllediSWAP",
                                MQT_NAMED_BUILDER(nestedControlledIswap),
                                MQT_NAMED_BUILDER(multipleControlledIswap)},
                    QCOTestCase{"TrivialControllediSWAP",
                                MQT_NAMED_BUILDER(trivialControlledIswap),
                                MQT_NAMED_BUILDER(iswap)},
                    QCOTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(inverseIswap),
                                MQT_NAMED_BUILDER(inverseIswap)},
                    QCOTestCase{
                        "InverseMultipleControllediSWAP",
                        MQT_NAMED_BUILDER(inverseMultipleControlledIswap),
                        MQT_NAMED_BUILDER(inverseMultipleControlledIswap)}));

TEST_F(QCOTest, iSWAPOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = iSWAPOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::iSWAP);

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
