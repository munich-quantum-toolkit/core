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
    QCOSOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"S", MQT_NAMED_BUILDER(s), MQT_NAMED_BUILDER(s)},
        QCOTestCase{"SingleControlledS", MQT_NAMED_BUILDER(singleControlledS),
                    MQT_NAMED_BUILDER(singleControlledS)},
        QCOTestCase{"MultipleControlledS",
                    MQT_NAMED_BUILDER(multipleControlledS),
                    MQT_NAMED_BUILDER(multipleControlledS)},
        QCOTestCase{"NestedControlledS", MQT_NAMED_BUILDER(nestedControlledS),
                    MQT_NAMED_BUILDER(multipleControlledS)},
        QCOTestCase{"TrivialControlledS", MQT_NAMED_BUILDER(trivialControlledS),
                    MQT_NAMED_BUILDER(s)},
        QCOTestCase{"InverseS", MQT_NAMED_BUILDER(inverseS),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"InverseMultipleControlledS",
                    MQT_NAMED_BUILDER(inverseMultipleControlledS),
                    MQT_NAMED_BUILDER(multipleControlledSdg)},
        QCOTestCase{"SThenSdg", MQT_NAMED_BUILDER(sThenSdg),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"TwoS", MQT_NAMED_BUILDER(twoS), MQT_NAMED_BUILDER(z)}));

TEST_F(QCOTest, SOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::S);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
