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
    QCOZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Z", MQT_NAMED_BUILDER(z), MQT_NAMED_BUILDER(z)},
        QCOTestCase{"SingleControlledZ", MQT_NAMED_BUILDER(singleControlledZ),
                    MQT_NAMED_BUILDER(singleControlledZ)},
        QCOTestCase{"MultipleControlledZ",
                    MQT_NAMED_BUILDER(multipleControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"NestedControlledZ", MQT_NAMED_BUILDER(nestedControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"TrivialControlledZ", MQT_NAMED_BUILDER(trivialControlledZ),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"InverseZ", MQT_NAMED_BUILDER(inverseZ),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"InverseMultipleControlledZ",
                    MQT_NAMED_BUILDER(inverseMultipleControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"TwoZ", MQT_NAMED_BUILDER(twoZ),
                    MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, ZOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = ZOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Z);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
