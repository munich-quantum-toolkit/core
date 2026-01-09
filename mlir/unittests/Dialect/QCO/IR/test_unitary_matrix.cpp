/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace {

using namespace mlir;

class UnitaryMatrixTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<qco::QCODialect, arith::ArithDialect, cf::ControlFlowDialect,
                func::FuncDialect, scf::SCFDialect, LLVM::LLVMDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  /**
   * @brief Build expected QCO IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQCOIR(
      const std::function<void(qco::QCOProgramBuilder&)>& buildFunc) const {
    qco::QCOProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    return builder.finalize();
  }

  /**
   * @brief Get first operation of given type in a module containing a function
   *        as its first operation.
   */
  template <typename OpType>
  [[nodiscard]] OpType getFirstOp(ModuleOp moduleOp) {
    auto funcOp = llvm::dyn_cast<func::FuncOp>(
        moduleOp.getBody()->getOperations().front());
    assert(funcOp);

    auto ops = funcOp.getOps<OpType>();
    if (ops.empty()) {
      return nullptr;
    }

    return *ops.begin();
  }

  /**
   * @brief Get text representation of given module.
   */
  [[nodiscard]] static std::string toString(ModuleOp moduleOp) {
    std::string buffer;
    llvm::raw_string_ostream serializeStream{buffer};
    moduleOp->print(serializeStream);
    return serializeStream.str();
  }

private:
  std::unique_ptr<MLIRContext> context;
};

} // namespace

// ##################################################
// # Standard Gates Unitary Matrix Tests
// ##################################################

/**
 * @brief Test: Identity unitary matrix
 *
 * @details
 * Ensure the correct gate definition is returned.
 */
TEST_F(UnitaryMatrixTest, IdOpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    builder.id(reg[0]);
  });
  auto op = getFirstOp<qco::IdOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  EXPECT_EQ(op.getUnitaryMatrixDefinition(), utils::getMatrixId());
}

/**
 * @brief Test: X unitary matrix
 *
 * @details
 * Ensure the correct gate definition is returned.
 */
TEST_F(UnitaryMatrixTest, XOpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    builder.x(reg[0]);
  });
  auto op = getFirstOp<qco::XOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  EXPECT_EQ(op.getUnitaryMatrixDefinition(), utils::getMatrixX());
}

/**
 * @brief Test: Y unitary matrix
 *
 * @details
 * Ensure the correct gate definition is returned.
 */
TEST_F(UnitaryMatrixTest, YOpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    builder.y(reg[0]);
  });
  auto op = getFirstOp<qco::YOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  EXPECT_EQ(op.getUnitaryMatrixDefinition(), utils::getMatrixY());
}

/**
 * @brief Test: Z unitary matrix
 *
 * @details
 * Ensure the correct gate definition is returned.
 */
TEST_F(UnitaryMatrixTest, ZOpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    builder.z(reg[0]);
  });
  auto op = getFirstOp<qco::ZOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  EXPECT_EQ(op.getUnitaryMatrixDefinition(), utils::getMatrixZ());
}

/**
 * @brief Test: CX unitary matrix
 *
 * @details
 * Ensure the correct gate definition is returned.
 */
TEST_F(UnitaryMatrixTest, CtrlXOpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(2, "q");
    builder.cx(reg[0], reg[1]);
  });
  auto op = getFirstOp<qco::CtrlOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  const Eigen::MatrixXcd cxMatrix = Eigen::Matrix4cd{
      {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}};

  EXPECT_EQ(op.getUnitaryMatrixDefinition(), cxMatrix);
}

/**
 * @brief Test: CX unitary matrix, both orientations
 *
 * @details
 * Ensure the correct gate definition is returned and is equal for both
 * orientations.
 */
TEST_F(UnitaryMatrixTest, CtrlX10OpMatrix) {
  auto moduleOp01 = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(2, "q");
    builder.cx(reg[0], reg[1]);
  });
  auto moduleOp10 = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(2, "q");
    builder.cx(reg[1], reg[0]);
  });
  auto op01 = getFirstOp<qco::CtrlOp>(*moduleOp01);
  ASSERT_TRUE(op01) << toString(*moduleOp01);
  auto op10 = getFirstOp<qco::CtrlOp>(*moduleOp10);
  ASSERT_TRUE(op10) << toString(*moduleOp10);

  const Eigen::MatrixXcd cxMatrix = Eigen::Matrix4cd{
      {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}};

  EXPECT_EQ(op01.getUnitaryMatrixDefinition(), cxMatrix);
  EXPECT_EQ(op10.getUnitaryMatrixDefinition(), cxMatrix);
  EXPECT_EQ(op10.getUnitaryMatrixDefinition(),
            op01.getUnitaryMatrixDefinition());
}
