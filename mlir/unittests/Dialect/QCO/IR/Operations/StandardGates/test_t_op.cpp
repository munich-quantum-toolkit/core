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
    QCOTOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"T", MQT_NAMED_BUILDER(t_), MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"SingleControlledT", MQT_NAMED_BUILDER(singleControlledT),
                    MQT_NAMED_BUILDER(singleControlledT)},
        QCOTestCase{"MultipleControlledT",
                    MQT_NAMED_BUILDER(multipleControlledT),
                    MQT_NAMED_BUILDER(multipleControlledT)},
        QCOTestCase{"NestedControlledT", MQT_NAMED_BUILDER(nestedControlledT),
                    MQT_NAMED_BUILDER(multipleControlledT)},
        QCOTestCase{"TrivialControlledT", MQT_NAMED_BUILDER(trivialControlledT),
                    MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"InverseT", MQT_NAMED_BUILDER(inverseT),
                    MQT_NAMED_BUILDER(tdg)},
        QCOTestCase{"InverseMultipleControlledT",
                    MQT_NAMED_BUILDER(inverseMultipleControlledT),
                    MQT_NAMED_BUILDER(multipleControlledTdg)},
        QCOTestCase{"TThenTdg", MQT_NAMED_BUILDER(tThenTdg),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"TwoT", MQT_NAMED_BUILDER(twoT), MQT_NAMED_BUILDER(s)}));

TEST_F(QCOTest, TOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = TOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::T);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
