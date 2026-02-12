/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include "llvm/ADT/SmallVector.h"

#include <Eigen/Core>
#include <complex>
#include <functional>
#include <gtest/gtest.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <string>

namespace {

using namespace mlir;
using namespace std::complex_literals;

class QcoUnitaryMatrixTest : public testing::Test {
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
    if (!funcOp) {
      return nullptr;
    }

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
TEST_F(QcoUnitaryMatrixTest, IdOpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    builder.id(reg[0]);
  });
  auto op = getFirstOp<qco::IdOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  EXPECT_EQ(op.getUnitaryMatrix(), Eigen::Matrix2cd::Identity());
}

/**
 * @brief Test: X unitary matrix
 *
 * @details
 * Ensure the correct gate definition is returned.
 */
TEST_F(QcoUnitaryMatrixTest, XOpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    builder.x(reg[0]);
  });
  auto op = getFirstOp<qco::XOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  const Eigen::Matrix2cd expectedValue{{0, 1}, {1, 0}};
  EXPECT_EQ(op.getUnitaryMatrix(), expectedValue);
}

/**
 * @brief Test: U2 unitary matrix
 *
 * @details
 * Ensure the correct gate definition is returned.
 */
TEST_F(QcoUnitaryMatrixTest, U2OpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    builder.u2(0.2, 0.8, reg[0]);
  });
  auto op = getFirstOp<qco::U2Op>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  const Eigen::Matrix2cd expectedValue{
      {0.70710678 + 0.i, -0.49264604 - 0.50724736i},
      {0.69301172 + 0.14048043i, // NOLINT(modernize-use-std-numbers)
       0.38205142 + 0.59500984i}};
  const auto actualValue = op.getUnitaryMatrix();
  ASSERT_TRUE(actualValue);
  EXPECT_TRUE(actualValue->isApprox(expectedValue, 1e-8));
}

/**
 * @brief Test: CX unitary matrix
 *
 * @details
 * Ensure the correct gate definition is returned.
 */
TEST_F(QcoUnitaryMatrixTest, CtrlXOpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(2, "q");
    builder.cx(reg[0], reg[1]);
  });
  auto op = getFirstOp<qco::CtrlOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  const Eigen::MatrixXcd cxMatrix = Eigen::Matrix4cd{
      {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}};

  EXPECT_EQ(op.getUnitaryMatrix(), cxMatrix);
}

/**
 * @brief Test: CX unitary matrix, both orientations
 *
 * @details
 * Ensure the correct gate definition is returned and is equal for both
 * orientations.
 */
TEST_F(QcoUnitaryMatrixTest, CtrlX10OpMatrix) {
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

  EXPECT_EQ(op01.getUnitaryMatrix(), cxMatrix);
  EXPECT_EQ(op10.getUnitaryMatrix(), cxMatrix);
  EXPECT_EQ(op10.getUnitaryMatrix(), op01.getUnitaryMatrix());
}

/**
 * @brief Test: Inverse Iswap unitary matrix
 *
 * @details
 * Ensure the correct gate definition is returned.
 */
TEST_F(QcoUnitaryMatrixTest, InvIswapOpMatrix) {
  using namespace std::complex_literals;
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(2, "q");
    builder.inv(reg, [&](auto qubits) -> llvm::SmallVector<Value> {
      auto [q0, q1] = builder.iswap(qubits[0], qubits[1]);
      return {q0, q1};
    });
  });
  auto op = getFirstOp<qco::InvOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  const Eigen::MatrixXcd invIswapMatrix = Eigen::Matrix4cd{
      {{1, 0, 0, 0}, {0, 0, -1i, 0}, {0, -1i, 0, 0}, {0, 0, 0, 1}}};

  EXPECT_EQ(op.getUnitaryMatrix(), invIswapMatrix);
}
