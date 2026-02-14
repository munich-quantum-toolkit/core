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
    QCOXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"X", MQT_NAMED_BUILDER(x), MQT_NAMED_BUILDER(x)},
        QCOTestCase{"SingleControlledX", MQT_NAMED_BUILDER(singleControlledX),
                    MQT_NAMED_BUILDER(singleControlledX)},
        QCOTestCase{"MultipleControlledX",
                    MQT_NAMED_BUILDER(multipleControlledX),
                    MQT_NAMED_BUILDER(multipleControlledX)},
        QCOTestCase{"NestedControlledX", MQT_NAMED_BUILDER(nestedControlledX),
                    MQT_NAMED_BUILDER(multipleControlledX)},
        QCOTestCase{"TrivialControlledX", MQT_NAMED_BUILDER(trivialControlledX),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"InverseX", MQT_NAMED_BUILDER(inverseX),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"InverseMultipleControlledX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledX),
                    MQT_NAMED_BUILDER(multipleControlledX)},
        QCOTestCase{"TwoX", MQT_NAMED_BUILDER(twoX),
                    MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, XOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = XOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::X);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
