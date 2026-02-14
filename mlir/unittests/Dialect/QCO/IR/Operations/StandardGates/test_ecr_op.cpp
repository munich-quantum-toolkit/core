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
    QCOECROpTest, QCOTest,
    testing::Values(QCOTestCase{"ECR", ecr, ecr},
                    QCOTestCase{"SingleControlledECR", singleControlledEcr,
                                singleControlledEcr},
                    QCOTestCase{"MultipleControlledECR", multipleControlledEcr,
                                multipleControlledEcr},
                    QCOTestCase{"NestedControlledECR", nestedControlledEcr,
                                multipleControlledEcr},
                    QCOTestCase{"TrivialControlledECR", trivialControlledEcr,
                                ecr},
                    QCOTestCase{"InverseECR", inverseEcr, inverseEcr},
                    QCOTestCase{"InverseMultipleControlledECR",
                                inverseMultipleControlledEcr,
                                inverseMultipleControlledEcr}),
    printTestName);

TEST_F(QCOTest, ECROpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = ECROp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::ECR);

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
