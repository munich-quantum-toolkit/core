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

#include <gtest/gtest.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace {

using namespace mlir;

class QcoUnitaryOpInterfaceTest : public testing::Test {
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

TEST_F(QcoUnitaryOpInterfaceTest, getFastUnitaryMatrix2x2) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    reg[0] = builder.id(reg[0]);
    reg[0] = builder.rx(1.0, reg[0]);
    reg[0] = builder.u(0.2, 0.3, 0.4, reg[0]);
  });

  auto&& moduleOps = moduleOp->getBody()->getOperations();
  ASSERT_FALSE(moduleOps.empty());
  auto funcOp = llvm::dyn_cast<func::FuncOp>(moduleOps.begin());
  for (auto&& op : funcOp.getOps()) {
    auto unitaryOp = llvm::dyn_cast<qco::UnitaryOpInterface>(op);
    if (unitaryOp) {
      EXPECT_EQ(unitaryOp.getUnitaryMatrix(),
                unitaryOp.getFastUnitaryMatrix<Eigen::Matrix2cd>());
    }
  }
}

TEST_F(QcoUnitaryOpInterfaceTest, getFastUnitaryMatrix4x4) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(2, "q");
    std::tie(reg[0], reg[1]) = builder.rxx(2.0, reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.rzx(1.0, reg[0], reg[1]);
  });

  auto&& moduleOps = moduleOp->getBody()->getOperations();
  ASSERT_FALSE(moduleOps.empty());
  auto funcOp = llvm::dyn_cast<func::FuncOp>(moduleOps.begin());
  for (auto&& op : funcOp.getOps()) {
    auto unitaryOp = llvm::dyn_cast<qco::UnitaryOpInterface>(op);
    if (unitaryOp) {
      EXPECT_EQ(unitaryOp.getUnitaryMatrix(),
                unitaryOp.getFastUnitaryMatrix<Eigen::Matrix4cd>());
    }
  }
}

TEST_F(QcoUnitaryOpInterfaceTest, getFastUnitaryMatrixDynamic) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(2, "q");
    std::tie(reg[1], reg[0]) = builder.ch(reg[1], reg[0]);
  });

  auto&& moduleOps = moduleOp->getBody()->getOperations();
  ASSERT_FALSE(moduleOps.empty());
  auto funcOp = llvm::dyn_cast<func::FuncOp>(moduleOps.begin());
  for (auto&& op : funcOp.getOps()) {
    auto unitaryOp = llvm::dyn_cast<qco::UnitaryOpInterface>(op);
    if (unitaryOp) {
      EXPECT_EQ(unitaryOp.getUnitaryMatrix(),
                unitaryOp.getFastUnitaryMatrix<Eigen::MatrixXcd>());
    }
  }
}
