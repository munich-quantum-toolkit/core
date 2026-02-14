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
    QCOIDOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Identity", MQT_NAMED_BUILDER(identity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"SingleControlledIdentity",
                    MQT_NAMED_BUILDER(singleControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"MultipleControlledIdentity",
                    MQT_NAMED_BUILDER(multipleControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"NestedControlledIdentity",
                    MQT_NAMED_BUILDER(nestedControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"TrivialControlledIdentity",
                    MQT_NAMED_BUILDER(trivialControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"InverseIdentity", MQT_NAMED_BUILDER(inverseIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"InverseMultipleControlledIdentity",
                    MQT_NAMED_BUILDER(inverseMultipleControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, IdOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = IdOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::I);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
