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
   * @brief Get first operation of given type in a module.
   */
  template <typename OpType>
  [[nodiscard]] static OpType getFirstOp(ModuleOp moduleOp) {
    auto&& moduleOperations = moduleOp.getBody()->getOperations();

    ASSERT_EQ(moduleOperations.size(), 1);
    auto&& funcOp = moduleOperations.front();
    auto&& concreteFuncOp = llvm::dyn_cast<func::FuncOp>(funcOp);
    ASSERT_TRUE(concreteFuncOp);

    auto funcOperations = concreteFuncOp.getBody().getOps<qco::IdOp>();
    ASSERT_EQ(std::distance(funcOperations.begin(), funcOperations.end()), 1);
    return *funcOperations.begin();
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
 * Ensure the correct gate definition is returned for a IdOp.
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
 * Ensure the correct gate definition is returned for a IdOp.
 */
TEST_F(UnitaryMatrixTest, XOpMatrix) {
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    builder.x(reg[0]);
  });
  auto op = getFirstOp<qco::XOp>(*moduleOp);
  ASSERT_TRUE(op) << toString(*moduleOp);

  EXPECT_EQ(op.getUnitaryMatrixDefinition(), utils::getMatrixId());
}
