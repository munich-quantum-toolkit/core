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
    QCODCXOpTest, QCOTest,
    testing::Values(QCOTestCase{"DCX", dcx, dcx},
                    QCOTestCase{"SingleControlledDCX", singleControlledDcx,
                                singleControlledDcx},
                    QCOTestCase{"MultipleControlledDCX", multipleControlledDcx,
                                multipleControlledDcx},
                    QCOTestCase{"NestedControlledDCX", nestedControlledDcx,
                                multipleControlledDcx},
                    QCOTestCase{"TrivialControlledDCX", trivialControlledDcx,
                                dcx},
                    QCOTestCase{"InverseDCX", inverseDcx, inverseDcx},
                    QCOTestCase{"InverseMultipleControlledDCX",
                                inverseMultipleControlledDcx,
                                inverseMultipleControlledDcx}),
    printTestName);

TEST_F(QCOTest, DCXOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = DCXOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::DCX);

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
