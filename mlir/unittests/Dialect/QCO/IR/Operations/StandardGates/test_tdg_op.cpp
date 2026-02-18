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
    QCOTdgOpTest, QCOTest,
    testing::Values(QCOTestCase{"Tdg", MQT_NAMED_BUILDER(tdg),
                                MQT_NAMED_BUILDER(tdg)},
                    QCOTestCase{"SingleControlledTdg",
                                MQT_NAMED_BUILDER(singleControlledTdg),
                                MQT_NAMED_BUILDER(singleControlledTdg)},
                    QCOTestCase{"MultipleControlledTdg",
                                MQT_NAMED_BUILDER(multipleControlledTdg),
                                MQT_NAMED_BUILDER(multipleControlledTdg)},
                    QCOTestCase{"NestedControlledTdg",
                                MQT_NAMED_BUILDER(nestedControlledTdg),
                                MQT_NAMED_BUILDER(multipleControlledTdg)},
                    QCOTestCase{"TrivialControlledTdg",
                                MQT_NAMED_BUILDER(trivialControlledTdg),
                                MQT_NAMED_BUILDER(tdg)},
                    QCOTestCase{"InverseTdg", MQT_NAMED_BUILDER(inverseTdg),
                                MQT_NAMED_BUILDER(t_)},
                    QCOTestCase{"InverseMultipleControlledTdg",
                                MQT_NAMED_BUILDER(inverseMultipleControlledTdg),
                                MQT_NAMED_BUILDER(multipleControlledT)},
                    QCOTestCase{"TdgThenS", MQT_NAMED_BUILDER(tdgThenT),
                                MQT_NAMED_BUILDER(emptyQCO)},
                    QCOTestCase{"TwoTdg", MQT_NAMED_BUILDER(twoTdg),
                                MQT_NAMED_BUILDER(sdg)}));

TEST_F(QCOTest, TdgOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = TdgOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Tdg);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
