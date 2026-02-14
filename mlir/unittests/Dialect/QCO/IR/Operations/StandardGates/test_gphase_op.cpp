/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qco_programs.h"
#include "test_qco_ir.h"

#include <Eigen/Core>
#include <complex>
#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOGPhaseOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"GlobalPhase", MQT_NAMED_BUILDER(globalPhase),
                    MQT_NAMED_BUILDER(globalPhase)},
        QCOTestCase{"SingleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(singleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"MultipleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(multipleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"InverseGlobalPhase", MQT_NAMED_BUILDER(inverseGlobalPhase),
                    MQT_NAMED_BUILDER(globalPhase)},
        QCOTestCase{"InverseMultipleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(inverseMultipleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(multipleControlledGlobalPhase)}));

TEST_F(QCOTest, GPhaseOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), globalPhase);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto gPhaseOp = *funcOp.getBody().getOps<GPhaseOp>().begin();
  const auto matrix = *gPhaseOp.getUnitaryMatrix();

  // Get the definition
  const auto definition = std::polar(1.0, 0.123); // e^(i*0.123)

  // Convert it to an Eigen matrix
  Eigen::Matrix<std::complex<double>, 1, 1> eigenDefinition;
  eigenDefinition << definition;

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
