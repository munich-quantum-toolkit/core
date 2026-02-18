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
    QCOU2OpTest, QCOTest,
    testing::Values(
        QCOTestCase{"U2", MQT_NAMED_BUILDER(u2), MQT_NAMED_BUILDER(u2)},
        QCOTestCase{"SingleControlledU2", MQT_NAMED_BUILDER(singleControlledU2),
                    MQT_NAMED_BUILDER(singleControlledU2)},
        QCOTestCase{"MultipleControlledU2",
                    MQT_NAMED_BUILDER(multipleControlledU2),
                    MQT_NAMED_BUILDER(multipleControlledU2)},
        QCOTestCase{"NestedControlledU2", MQT_NAMED_BUILDER(nestedControlledU2),
                    MQT_NAMED_BUILDER(multipleControlledU2)},
        QCOTestCase{"TrivialControlledU2",
                    MQT_NAMED_BUILDER(trivialControlledU2),
                    MQT_NAMED_BUILDER(u2)},
        QCOTestCase{"InverseU2", MQT_NAMED_BUILDER(inverseU2),
                    MQT_NAMED_BUILDER(u2)},
        QCOTestCase{"InverseMultipleControlledU2",
                    MQT_NAMED_BUILDER(inverseMultipleControlledU2),
                    MQT_NAMED_BUILDER(multipleControlledU2)},
        QCOTestCase{"CanonicalizeU2ToH", MQT_NAMED_BUILDER(canonicalizeU2ToH),
                    MQT_NAMED_BUILDER(h)},
        QCOTestCase{"CanonicalizeU2ToRx", MQT_NAMED_BUILDER(canonicalizeU2ToRx),
                    MQT_NAMED_BUILDER(rxPiOver2)},
        QCOTestCase{"CanonicalizeU2ToRy", MQT_NAMED_BUILDER(canonicalizeU2ToRy),
                    MQT_NAMED_BUILDER(ryPiOver2)}));

TEST_F(QCOTest, U2OpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), u2);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto u2Op = *funcOp.getBody().getOps<U2Op>().begin();
  const auto matrix = *u2Op.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::U2, {0.234, 0.567});

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
