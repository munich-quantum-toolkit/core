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
    QCOPOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"P", MQT_NAMED_BUILDER(p), MQT_NAMED_BUILDER(p)},
        QCOTestCase{"SingleControlledP", MQT_NAMED_BUILDER(singleControlledP),
                    MQT_NAMED_BUILDER(singleControlledP)},
        QCOTestCase{"MultipleControlledP",
                    MQT_NAMED_BUILDER(multipleControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"NestedControlledP", MQT_NAMED_BUILDER(nestedControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"TrivialControlledP", MQT_NAMED_BUILDER(trivialControlledP),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"InverseP", MQT_NAMED_BUILDER(inverseP),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"InverseMultipleControlledP",
                    MQT_NAMED_BUILDER(inverseMultipleControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"TwoPOppositePhase", MQT_NAMED_BUILDER(twoPOppositePhase),
                    MQT_NAMED_BUILDER(emptyQCO)}));

TEST_F(QCOTest, POpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), p);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto pOp = *funcOp.getBody().getOps<POp>().begin();
  const auto matrix = *pOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::P, {0.123});

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
