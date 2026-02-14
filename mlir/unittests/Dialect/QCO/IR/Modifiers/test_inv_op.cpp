/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Operations.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "qco_programs.h"
#include "test_qco_ir.h"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOInvOpTest, QCOTest,
    testing::Values(QCOTestCase{"NestedInv", MQT_NAMED_BUILDER(nestedInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"TripleNestedInv",
                                MQT_NAMED_BUILDER(tripleNestedInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"InvControlSandwich",
                                MQT_NAMED_BUILDER(invCtrlSandwich),
                                MQT_NAMED_BUILDER(singleControlledRxx)}));

TEST_F(QCOTest, InverseIswapOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), inverseIswap);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  auto matrix = invOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto iswapdg = qc::StandardOperation({0, 1}, qc::OpType::iSWAPdg);
  const auto dd = std::make_unique<dd::Package>(2);
  const auto iswapdgDD = dd::getDD(iswapdg, *dd);
  const auto definition = iswapdgDD.getMatrix(2);

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix->isApprox(eigenDefinition));
}
