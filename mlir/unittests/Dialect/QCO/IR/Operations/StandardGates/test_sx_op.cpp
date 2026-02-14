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
    QCOSXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SX", MQT_NAMED_BUILDER(sx), MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"SingleControlledSX", MQT_NAMED_BUILDER(singleControlledSx),
                    MQT_NAMED_BUILDER(singleControlledSx)},
        QCOTestCase{"MultipleControlledSX",
                    MQT_NAMED_BUILDER(multipleControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"NestedControlledSX", MQT_NAMED_BUILDER(nestedControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"TrivialControlledSX",
                    MQT_NAMED_BUILDER(trivialControlledSx),
                    MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"InverseSX", MQT_NAMED_BUILDER(inverseSx),
                    MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"InverseMultipleControlledSX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)}));

TEST_F(QCOTest, SXOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SXOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::SX);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
