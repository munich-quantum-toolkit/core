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
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "qco_programs.h"

#include <Eigen/Core>
#include <complex>
#include <gtest/gtest.h>
#include <iosfwd>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <string>

using namespace mlir;
using namespace qco;

struct QCOTestCase {
  std::string name;
  mqt::test::NamedBuilder<QCOProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<QCOProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os, const QCOTestCase& info);
};

class QCOTest : public testing::TestWithParam<QCOTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

/// \name QCO/Modifiers/CtrlOp.cpp
/// @{
TEST_F(QCOTest, CXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), singleControlledX);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ctrlOp = *funcOp.getBody().getOps<CtrlOp>().begin();
  auto matrix = ctrlOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto cx = qc::StandardOperation(1, 0, qc::OpType::X);
  const auto dd = std::make_unique<dd::Package>(2);
  const auto cxDD = dd::getDD(cx, *dd);
  const auto definition = cxDD.getMatrix(2);

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
/// @}

/// \name QCO/Modifiers/InvOp.cpp
/// @{
TEST_F(QCOTest, InverseIswapOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), inverseIswap);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
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
/// @}

/// \name QCO/Operations/StandardGates/DcxOp.cpp
/// @{
TEST_F(QCOTest, DCXOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = DCXOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::DCX);

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/EcrOp.cpp
/// @{
TEST_F(QCOTest, ECROpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = ECROp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::ECR);

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/GphaseOp.cpp
/// @{
TEST_F(QCOTest, GPhaseOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), globalPhase);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
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
/// @}

/// \name QCO/Operations/StandardGates/HOp.cpp
/// @{
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
/// @}

/// \name QCO/Operations/StandardGates/IdOp.cpp
/// @{
TEST_F(QCOTest, IdOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = IdOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::I);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/IswapOp.cpp
/// @{
TEST_F(QCOTest, iSWAPOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = iSWAPOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::iSWAP);

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/POp.cpp
/// @{
TEST_F(QCOTest, POpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), p);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
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
/// @}

/// \name QCO/Operations/StandardGates/ROp.cpp
/// @{
TEST_F(QCOTest, ROpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), r);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
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
/// @}

/// \name QCO/Operations/StandardGates/RxOp.cpp
/// @{
TEST_F(QCOTest, RXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rxOp = *funcOp.getBody().getOps<RXOp>().begin();
  const auto matrix = *rxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::RX, {0.123});

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/RxxOp.cpp
/// @{
TEST_F(QCOTest, RXXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rxx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rxxOp = *funcOp.getBody().getOps<RXXOp>().begin();
  const auto matrix = *rxxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RXX, {0.123});

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/RyOp.cpp
/// @{
TEST_F(QCOTest, RYOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), ry);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ryOp = *funcOp.getBody().getOps<RYOp>().begin();
  const auto matrix = *ryOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::RY, {0.456});

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/RyyOp.cpp
/// @{
TEST_F(QCOTest, RYYOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), ryy);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ryyOp = *funcOp.getBody().getOps<RYYOp>().begin();
  const auto matrix = *ryyOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RYY, {0.123});

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/RzOp.cpp
/// @{
TEST_F(QCOTest, RZOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rz);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rzOp = *funcOp.getBody().getOps<RZOp>().begin();
  const auto matrix = *rzOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::RZ, {0.789});

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/RzxOp.cpp
/// @{
TEST_F(QCOTest, RZXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rzx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rzxOp = *funcOp.getBody().getOps<RZXOp>().begin();
  const auto matrix = *rzxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RZX, {0.123});

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/RzzOp.cpp
/// @{
TEST_F(QCOTest, RZZOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rzz);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rzzOp = *funcOp.getBody().getOps<RZZOp>().begin();
  const auto matrix = *rzzOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RZZ, {0.123});

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/SOp.cpp
/// @{
TEST_F(QCOTest, SOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::S);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/SdgOp.cpp
/// @{
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
/// @}

/// \name QCO/Operations/StandardGates/SwapOp.cpp
/// @{
TEST_F(QCOTest, SWAPOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SWAPOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::SWAP);

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/SxOp.cpp
/// @{
TEST_F(QCOTest, SXOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SXOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::SX);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/SxdgOp.cpp
/// @{
TEST_F(QCOTest, SXdgOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SXdgOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::SXdg);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/TOp.cpp
/// @{
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
/// @}

/// \name QCO/Operations/StandardGates/TdgOp.cpp
/// @{
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
/// @}

/// \name QCO/Operations/StandardGates/U2Op.cpp
/// @{
TEST_F(QCOTest, U2OpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), u2);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
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
/// @}

/// \name QCO/Operations/StandardGates/UOp.cpp
/// @{
TEST_F(QCOTest, UOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), u);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
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
/// @}

/// \name QCO/Operations/StandardGates/XOp.cpp
/// @{
TEST_F(QCOTest, XOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = XOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::X);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
TEST_F(QCOTest, XXMinusYYOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), xxMinusYY);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto xxMinusYYOp = *funcOp.getBody().getOps<XXMinusYYOp>().begin();
  const auto matrix = *xxMinusYYOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToTwoQubitGateMatrix(qc::OpType::XXminusYY, {0.123, 0.456});

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
TEST_F(QCOTest, XXPlusYYOp) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), xxPlusYY);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto xxPlusYYOp = *funcOp.getBody().getOps<XXPlusYYOp>().begin();
  const auto matrix = *xxPlusYYOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToTwoQubitGateMatrix(qc::OpType::XXplusYY, {0.123, 0.456});

  // Convert it to an Eigen matrix
  Eigen::Matrix4cd eigenDefinition;
  eigenDefinition << definition[0][0], definition[0][1], definition[0][2],
      definition[0][3], definition[1][0], definition[1][1], definition[1][2],
      definition[1][3], definition[2][0], definition[2][1], definition[2][2],
      definition[2][3], definition[3][0], definition[3][1], definition[3][2],
      definition[3][3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/YOp.cpp
/// @{
TEST_F(QCOTest, YOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = YOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Y);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}

/// \name QCO/Operations/StandardGates/ZOp.cpp
/// @{
TEST_F(QCOTest, ZOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = ZOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Z);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
/// @}
