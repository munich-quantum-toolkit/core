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
    QCOYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Y", y, y},
        QCOTestCase{"SingleControlledY", singleControlledY, singleControlledY},
        QCOTestCase{"MultipleControlledY", multipleControlledY,
                    multipleControlledY},
        QCOTestCase{"NestedControlledY", nestedControlledY,
                    multipleControlledY},
        QCOTestCase{"TrivialControlledY", trivialControlledY, y},
        QCOTestCase{"InverseY", inverseY, y},
        QCOTestCase{"InverseMultipleControlledY", inverseMultipleControlledY,
                    multipleControlledY}),
    printTestName);

TEST_F(QCOTest, YOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = YOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Y);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
