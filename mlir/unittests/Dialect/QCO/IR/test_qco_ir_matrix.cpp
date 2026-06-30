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
#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <complex>
#include <cstddef>
#include <memory>
#include <string>

using namespace mlir;
using namespace qco;

[[nodiscard]] static Matrix2x2 matrix2FromFlat(const dd::GateMatrix& def) {
  return Matrix2x2::fromElements(def[0], def[1], def[2], def[3]);
}

template <typename Definition>
[[nodiscard]] static Matrix4x4
matrix4FromDefinition(const Definition& definition) {
  return Matrix4x4::fromElements(
      definition[0][0], definition[0][1], definition[0][2], definition[0][3],
      definition[1][0], definition[1][1], definition[1][2], definition[1][3],
      definition[2][0], definition[2][1], definition[2][2], definition[2][3],
      definition[3][0], definition[3][1], definition[3][2], definition[3][3]);
}

template <typename Fn>
[[nodiscard]] static Matrix4x4
expectedMatrixFromComputation(const Fn& build,
                              const std::size_t numQubits = 2) {
  qc::QuantumComputation comp;
  build(comp);
  const auto package = std::make_unique<dd::Package>(numQubits);
  return matrix4FromDefinition(
      dd::buildFunctionality(comp, *package).getMatrix(numQubits));
}

static void controlledXH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&](ValueRange targets) {
    auto wire = b.x(targets[0]);
    wire = b.h(wire);
    return SmallVector{wire};
  });
}

static void controlledInverseHT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&](ValueRange targets) {
    auto wire = b.inv({targets[0]}, [&](ValueRange innerTargets) {
      auto inner = b.h(innerTargets[0]);
      inner = b.t(inner);
      return SmallVector{inner};
    })[0];
    return SmallVector{wire};
  });
}

namespace {

struct QCOMatrixTestCase {
  std::string name;
  mqt::test::NamedBuilder<QCOProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<QCOProgramBuilder> referenceBuilder;
};

class QCOMatrixTest : public testing::TestWithParam<QCOMatrixTestCase> {
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

} // namespace

/// \name QCO/Modifiers/CtrlOp.cpp
/// @{
TEST_F(QCOMatrixTest, CXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), singleControlledX);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ctrlOp = *funcOp.getBody().getOps<CtrlOp>().begin();
  auto matrix = ctrlOp.getUnitaryMatrix();

  const Matrix4x4 expected =
      expectedMatrixFromComputation([](qc::QuantumComputation& comp) {
        comp.addQubitRegister(2, "q");
        comp.cx(1, 0);
      });

  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, ControlledHOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), singleControlledH);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ctrlOp = *funcOp.getBody().getOps<CtrlOp>().begin();
  auto matrix = ctrlOp.getUnitaryMatrix();
  ASSERT_TRUE(matrix);

  const Matrix4x4 expected =
      expectedMatrixFromComputation([](qc::QuantumComputation& comp) {
        comp.addQubitRegister(2, "q");
        comp.ch(1, 0);
      });

  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, ControlledXHOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), controlledXH);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ctrlOp = *funcOp.getBody().getOps<CtrlOp>().begin();
  auto matrix = ctrlOp.getUnitaryMatrix();
  ASSERT_TRUE(matrix);

  const Matrix4x4 expected =
      expectedMatrixFromComputation([](qc::QuantumComputation& comp) {
        comp.addQubitRegister(2, "q");
        comp.cx(1, 0);
        comp.ch(1, 0);
      });

  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, ControlledInverseHTOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), controlledInverseHT);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ctrlOp = *funcOp.getBody().getOps<CtrlOp>().begin();
  auto matrix = ctrlOp.getUnitaryMatrix();
  ASSERT_TRUE(matrix);

  const Matrix4x4 expected =
      expectedMatrixFromComputation([](qc::QuantumComputation& comp) {
        comp.addQubitRegister(2, "q");
        qc::CompoundOperation body;
        body.emplace_back<qc::StandardOperation>(1, 0, qc::OpType::H);
        body.emplace_back<qc::StandardOperation>(1, 0, qc::OpType::T);
        body.invert();
        comp.push_back(body);
      });

  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, ControlledHOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), singleControlledH);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ctrlOp = *funcOp.getBody().getOps<CtrlOp>().begin();
  auto matrix = ctrlOp.getUnitaryMatrix();

  const auto ch = qc::StandardOperation(1, 0, qc::OpType::H);
  const auto dd = std::make_unique<dd::Package>(2);
  const auto chDD = dd::getDD(ch, *dd);
  const auto definition = chDD.getMatrix(2);

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, ControlledXHOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), controlledXH);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ctrlOp = *funcOp.getBody().getOps<CtrlOp>().begin();
  auto matrix = ctrlOp.getUnitaryMatrix();
  ASSERT_TRUE(matrix);

  DynamicMatrix expected = DynamicMatrix::identity(4);
  expected.setBottomRightCorner(HOp::getUnitaryMatrix() *
                                XOp::getUnitaryMatrix());

  ASSERT_TRUE(matrix->isApprox(expected));
}
/// @}

/// \name QCO/Modifiers/InvOp.cpp
/// @{
TEST_F(QCOMatrixTest, InverseIswapOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), inverseIswap);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  auto matrix = invOp.getUnitaryMatrix();

  const Matrix4x4 expected =
      expectedMatrixFromComputation([](qc::QuantumComputation& comp) {
        comp.addQubitRegister(2, "q");
        comp.iswapdg(0, 1);
      });

  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, InverseTwoXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), inverseTwoX);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  const auto matrix = invOp.getUnitaryMatrix();
  ASSERT_TRUE(matrix);

  DynamicMatrix expected;
  expected.assignFrom(Matrix2x2::identity());
  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, InverseXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), inverseX);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  const auto matrix = invOp.getUnitaryMatrix();
  ASSERT_TRUE(matrix);

  DynamicMatrix expected;
  expected.assignFrom(XOp::getUnitaryMatrix());
  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, InverseSxOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), inverseSx);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  const auto matrix = invOp.getUnitaryMatrix();
  ASSERT_TRUE(matrix);

  DynamicMatrix expected;
  expected.assignFrom(SXdgOp::getUnitaryMatrix());
  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, InverseGphaseXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), inverseGphaseX);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  const auto matrix = invOp.getUnitaryMatrix();
  ASSERT_TRUE(matrix);

  const auto composeGlobal = std::polar(1.0, -0.123);
  const Matrix2x2 body = XOp::getUnitaryMatrix() * composeGlobal;

  ASSERT_TRUE(matrix->isApprox(DynamicMatrix::fromAdjoint(body)));
}

TEST_F(QCOMatrixTest, InverseGphaseBarrierOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), inverseGphaseBarrier);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  const auto matrix = invOp.getUnitaryMatrix();
  ASSERT_TRUE(matrix);

  const auto global = std::conj(std::polar(1.0, 0.123));
  DynamicMatrix expected;
  expected.assignFrom(Matrix2x2::fromElements(global, 0, 0, global));
  ASSERT_TRUE(matrix->isApprox(expected));
}

TEST_F(QCOMatrixTest, InverseTwoBarriersInInvOpMatrix) {
  auto moduleOp =
      QCOProgramBuilder::build(context.get(), inverseTwoBarriersInInv);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  EXPECT_FALSE(invOp.getUnitaryMatrix());
}

TEST_F(QCOMatrixTest, InvTwoOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), invTwo);
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  EXPECT_FALSE(invOp.getUnitaryMatrix());
}

TEST_F(QCOMatrixTest, InverseDynamicRzXOpMatrix) {
  constexpr auto mlirCode = R"(
    module {
      func.func @test(%theta: f64) -> !qco.qubit {
        %q_in = qco.alloc : !qco.qubit
        %q_out = qco.inv (%q = %q_in) {
          %q_1 = qco.rz(%theta) %q : !qco.qubit -> !qco.qubit
          %q_2 = qco.x %q_1 : !qco.qubit -> !qco.qubit
          qco.yield %q_2 : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %q_out : !qco.qubit
      }
    }
  )";

  auto moduleOp = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(moduleOp);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto invOp = *funcOp.getBody().getOps<InvOp>().begin();
  EXPECT_FALSE(invOp.getUnitaryMatrix());
}
/// @}

/// \name QCO/Operations/StandardGates/DcxOp.cpp
/// @{
TEST_F(QCOMatrixTest, DCXOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = DCXOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::DCX);

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/EcrOp.cpp
/// @{
TEST_F(QCOMatrixTest, ECROpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = ECROp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::ECR);

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/GphaseOp.cpp
/// @{
TEST_F(QCOMatrixTest, GPhaseOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), globalPhase);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto gPhaseOp = *funcOp.getBody().getOps<GPhaseOp>().begin();
  const auto matrix = *gPhaseOp.getUnitaryMatrix();

  // Get the definition
  const auto definition = std::polar(1.0, 0.123); // e^(i*0.123)

  const Matrix1x1 expected = Matrix1x1::fromElements(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/HOp.cpp
/// @{
TEST_F(QCOMatrixTest, HOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = HOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::H);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/IdOp.cpp
/// @{
TEST_F(QCOMatrixTest, IdOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = IdOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::I);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/IswapOp.cpp
/// @{
TEST_F(QCOMatrixTest, iSWAPOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = iSWAPOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::iSWAP);

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/POp.cpp
/// @{
TEST_F(QCOMatrixTest, POpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), p);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto pOp = *funcOp.getBody().getOps<POp>().begin();
  const auto matrix = *pOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::P, {0.123});

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/ROp.cpp
/// @{
TEST_F(QCOMatrixTest, ROpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), r);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rOp = *funcOp.getBody().getOps<ROp>().begin();
  const auto matrix = *rOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::R, {0.123, 0.456});
  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/RxOp.cpp
/// @{
TEST_F(QCOMatrixTest, RXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rxOp = *funcOp.getBody().getOps<RXOp>().begin();
  const auto matrix = *rxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::RX, {0.123});

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/RxxOp.cpp
/// @{
TEST_F(QCOMatrixTest, RXXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rxx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rxxOp = *funcOp.getBody().getOps<RXXOp>().begin();
  const auto matrix = *rxxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RXX, {0.123});

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/RyOp.cpp
/// @{
TEST_F(QCOMatrixTest, RYOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), ry);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ryOp = *funcOp.getBody().getOps<RYOp>().begin();
  const auto matrix = *ryOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::RY, {0.456});

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/RyyOp.cpp
/// @{
TEST_F(QCOMatrixTest, RYYOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), ryy);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto ryyOp = *funcOp.getBody().getOps<RYYOp>().begin();
  const auto matrix = *ryyOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RYY, {0.123});

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/RzOp.cpp
/// @{
TEST_F(QCOMatrixTest, RZOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rz);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rzOp = *funcOp.getBody().getOps<RZOp>().begin();
  const auto matrix = *rzOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::RZ, {0.789});

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/RzxOp.cpp
/// @{
TEST_F(QCOMatrixTest, RZXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rzx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rzxOp = *funcOp.getBody().getOps<RZXOp>().begin();
  const auto matrix = *rzxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RZX, {0.123});

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/RzzOp.cpp
/// @{
TEST_F(QCOMatrixTest, RZZOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), rzz);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto rzzOp = *funcOp.getBody().getOps<RZZOp>().begin();
  const auto matrix = *rzzOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::RZZ, {0.123});

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/SOp.cpp
/// @{
TEST_F(QCOMatrixTest, SOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::S);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/SdgOp.cpp
/// @{
TEST_F(QCOMatrixTest, SdgOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SdgOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Sdg);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/SwapOp.cpp
/// @{
TEST_F(QCOMatrixTest, SWAPOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SWAPOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToTwoQubitGateMatrix(qc::OpType::SWAP);

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/SxOp.cpp
/// @{
TEST_F(QCOMatrixTest, SXOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SXOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::SX);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/SxdgOp.cpp
/// @{
TEST_F(QCOMatrixTest, SXdgOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = SXdgOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::SXdg);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/TOp.cpp
/// @{
TEST_F(QCOMatrixTest, TOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = TOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::T);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/TdgOp.cpp
/// @{
TEST_F(QCOMatrixTest, TdgOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = TdgOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Tdg);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/U2Op.cpp
/// @{
TEST_F(QCOMatrixTest, U2OpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), u2);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto u2Op = *funcOp.getBody().getOps<U2Op>().begin();
  const auto matrix = *u2Op.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::U2, {0.234, 0.567});

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/UOp.cpp
/// @{
TEST_F(QCOMatrixTest, UOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), u);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto uOp = *funcOp.getBody().getOps<UOp>().begin();
  const auto matrix = *uOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToSingleQubitGateMatrix(qc::OpType::U, {0.1, 0.2, 0.3});

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/XOp.cpp
/// @{
TEST_F(QCOMatrixTest, XOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = XOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::X);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
TEST_F(QCOMatrixTest, XXMinusYYOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), xxMinusYY);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto xxMinusYYOp = *funcOp.getBody().getOps<XXMinusYYOp>().begin();
  const auto matrix = *xxMinusYYOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToTwoQubitGateMatrix(qc::OpType::XXminusYY, {0.123, 0.456});

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
TEST_F(QCOMatrixTest, XXPlusYYOp) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), xxPlusYY);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  auto xxPlusYYOp = *funcOp.getBody().getOps<XXPlusYYOp>().begin();
  const auto matrix = *xxPlusYYOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition =
      dd::opToTwoQubitGateMatrix(qc::OpType::XXplusYY, {0.123, 0.456});

  const Matrix4x4 expected = matrix4FromDefinition(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/YOp.cpp
/// @{
TEST_F(QCOMatrixTest, YOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = YOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Y);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}

/// \name QCO/Operations/StandardGates/ZOp.cpp
/// @{
TEST_F(QCOMatrixTest, ZOpMatrix) {
  // Get the (static) matrix from the operation
  const auto matrix = ZOp::getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Z);

  const Matrix2x2 expected = matrix2FromFlat(definition);

  ASSERT_TRUE(matrix.isApprox(expected));
}
/// @}
