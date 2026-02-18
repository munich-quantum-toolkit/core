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

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOECROpTest, QCOTest,
    testing::Values(QCOTestCase{"ECR", MQT_NAMED_BUILDER(ecr),
                                MQT_NAMED_BUILDER(ecr)},
                    QCOTestCase{"SingleControlledECR",
                                MQT_NAMED_BUILDER(singleControlledEcr),
                                MQT_NAMED_BUILDER(singleControlledEcr)},
                    QCOTestCase{"MultipleControlledECR",
                                MQT_NAMED_BUILDER(multipleControlledEcr),
                                MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCOTestCase{"NestedControlledECR",
                                MQT_NAMED_BUILDER(nestedControlledEcr),
                                MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCOTestCase{"TrivialControlledECR",
                                MQT_NAMED_BUILDER(trivialControlledEcr),
                                MQT_NAMED_BUILDER(ecr)},
                    QCOTestCase{"InverseECR", MQT_NAMED_BUILDER(inverseEcr),
                                MQT_NAMED_BUILDER(ecr)},
                    QCOTestCase{"InverseMultipleControlledECR",
                                MQT_NAMED_BUILDER(inverseMultipleControlledEcr),
                                MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCOTestCase{"TwoECR", MQT_NAMED_BUILDER(twoEcr),
                                MQT_NAMED_BUILDER(emptyQCO)}));

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
