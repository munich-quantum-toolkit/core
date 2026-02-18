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
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOUOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"U", MQT_NAMED_BUILDER(u), MQT_NAMED_BUILDER(u)},
        QCOTestCase{"SingleControlledU", MQT_NAMED_BUILDER(singleControlledU),
                    MQT_NAMED_BUILDER(singleControlledU)},
        QCOTestCase{"MultipleControlledU",
                    MQT_NAMED_BUILDER(multipleControlledU),
                    MQT_NAMED_BUILDER(multipleControlledU)},
        QCOTestCase{"NestedControlledU", MQT_NAMED_BUILDER(nestedControlledU),
                    MQT_NAMED_BUILDER(multipleControlledU)},
        QCOTestCase{"TrivialControlledU", MQT_NAMED_BUILDER(trivialControlledU),
                    MQT_NAMED_BUILDER(u)},
        QCOTestCase{"InverseU", MQT_NAMED_BUILDER(inverseU),
                    MQT_NAMED_BUILDER(u)},
        QCOTestCase{"InverseMultipleControlledU",
                    MQT_NAMED_BUILDER(inverseMultipleControlledU),
                    MQT_NAMED_BUILDER(multipleControlledU)},
        QCOTestCase{"CanonicalizeUToP", MQT_NAMED_BUILDER(canonicalizeUToP),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"CanonicalizeUToRx", MQT_NAMED_BUILDER(canonicalizeUToRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"CanonicalizeUToRy", MQT_NAMED_BUILDER(canonicalizeUToRy),
                    MQT_NAMED_BUILDER(ry)}));

TEST_F(QCOTest, UOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), u);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto uOp = *funcOp.getBody().getOps<UOp>().begin();
  const auto matrix = *uOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::U, {0.1, 0.2, 0.3});

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
