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
    QCOSXdgOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SXdg", MQT_NAMED_BUILDER(sxdg), MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"SingleControlledSXdg",
                    MQT_NAMED_BUILDER(singleControlledSxdg),
                    MQT_NAMED_BUILDER(singleControlledSxdg)},
        QCOTestCase{"MultipleControlledSXdg",
                    MQT_NAMED_BUILDER(multipleControlledSxdg),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)},
        QCOTestCase{"NestedControlledSXdg",
                    MQT_NAMED_BUILDER(nestedControlledSxdg),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)},
        QCOTestCase{"TrivialControlledSXdg",
                    MQT_NAMED_BUILDER(trivialControlledSxdg),
                    MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"InverseSXdg", MQT_NAMED_BUILDER(inverseSxdg),
                    MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"InverseMultipleControlledSXdg",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSxdg),
                    MQT_NAMED_BUILDER(multipleControlledSx)}));

TEST_F(QCOTest, SXdgOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SXdgOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::SXdg);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
