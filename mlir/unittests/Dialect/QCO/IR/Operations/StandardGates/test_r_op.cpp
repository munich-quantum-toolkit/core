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
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOROpTest, QCOTest,
    testing::Values(
        QCOTestCase{"R", MQT_NAMED_BUILDER(r), MQT_NAMED_BUILDER(r)},
        QCOTestCase{"SingleControlledR", MQT_NAMED_BUILDER(singleControlledR),
                    MQT_NAMED_BUILDER(singleControlledR)},
        QCOTestCase{"MultipleControlledR",
                    MQT_NAMED_BUILDER(multipleControlledR),
                    MQT_NAMED_BUILDER(multipleControlledR)},
        QCOTestCase{"NestedControlledR", MQT_NAMED_BUILDER(nestedControlledR),
                    MQT_NAMED_BUILDER(multipleControlledR)},
        QCOTestCase{"TrivialControlledR", MQT_NAMED_BUILDER(trivialControlledR),
                    MQT_NAMED_BUILDER(r)},
        QCOTestCase{"InverseR", MQT_NAMED_BUILDER(inverseR),
                    MQT_NAMED_BUILDER(r)},
        QCOTestCase{"InverseMultipleControlledR",
                    MQT_NAMED_BUILDER(inverseMultipleControlledR),
                    MQT_NAMED_BUILDER(multipleControlledR)},
        QCOTestCase{"CanonicalizeRToRx", MQT_NAMED_BUILDER(canonicalizeRToRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"CanonicalizeRToRy", MQT_NAMED_BUILDER(canonicalizeRToRy),
                    MQT_NAMED_BUILDER(ry)}));

TEST_F(QCOTest, ROpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), r);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto rOp = *funcOp.getBody().getOps<ROp>().begin();
  const auto matrix = *rOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::R, {0.123, 0.456});
  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
