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
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "qco_programs.h"
#include "test_qco_ir.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOHOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"H", MQT_NAMED_BUILDER(h), MQT_NAMED_BUILDER(h)},
        QCOTestCase{"SingleControlledH", MQT_NAMED_BUILDER(singleControlledH),
                    MQT_NAMED_BUILDER(singleControlledH)},
        QCOTestCase{"MultipleControlledH",
                    MQT_NAMED_BUILDER(multipleControlledH),
                    MQT_NAMED_BUILDER(multipleControlledH)},
        QCOTestCase{"NestedControlledH", MQT_NAMED_BUILDER(nestedControlledH),
                    MQT_NAMED_BUILDER(multipleControlledH)},
        QCOTestCase{"TrivialControlledH", MQT_NAMED_BUILDER(trivialControlledH),
                    MQT_NAMED_BUILDER(h)},
        QCOTestCase{"InverseH", MQT_NAMED_BUILDER(inverseH),
                    MQT_NAMED_BUILDER(h)},
        QCOTestCase{"InverseMultipleControlledH",
                    MQT_NAMED_BUILDER(inverseMultipleControlledH),
                    MQT_NAMED_BUILDER(multipleControlledH)},
        QCOTestCase{"TwoH", MQT_NAMED_BUILDER(twoH),
                    MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, HOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = HOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::H);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
