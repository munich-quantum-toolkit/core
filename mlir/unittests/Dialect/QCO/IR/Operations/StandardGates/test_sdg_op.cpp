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
    QCOSdgOpTest, QCOTest,
    testing::Values(QCOTestCase{"Sdg", MQT_NAMED_BUILDER(sdg),
                                MQT_NAMED_BUILDER(sdg)},
                    QCOTestCase{"SingleControlledSdg",
                                MQT_NAMED_BUILDER(singleControlledSdg),
                                MQT_NAMED_BUILDER(singleControlledSdg)},
                    QCOTestCase{"MultipleControlledSdg",
                                MQT_NAMED_BUILDER(multipleControlledSdg),
                                MQT_NAMED_BUILDER(multipleControlledSdg)},
                    QCOTestCase{"NestedControlledSdg",
                                MQT_NAMED_BUILDER(nestedControlledSdg),
                                MQT_NAMED_BUILDER(multipleControlledSdg)},
                    QCOTestCase{"TrivialControlledSdg",
                                MQT_NAMED_BUILDER(trivialControlledSdg),
                                MQT_NAMED_BUILDER(sdg)},
                    QCOTestCase{"InverseSdg", MQT_NAMED_BUILDER(inverseSdg),
                                MQT_NAMED_BUILDER(s)},
                    QCOTestCase{"InverseMultipleControlledSdg",
                                MQT_NAMED_BUILDER(inverseMultipleControlledSdg),
                                MQT_NAMED_BUILDER(multipleControlledS)},
                    QCOTestCase{"SdgThenS", MQT_NAMED_BUILDER(sdgThenS),
                                MQT_NAMED_BUILDER(emptyQCO)},
                    QCOTestCase{"TwoSdg", MQT_NAMED_BUILDER(twoSdg),
                                MQT_NAMED_BUILDER(z)}));

TEST_F(QCOTest, SdgOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SdgOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Sdg);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
