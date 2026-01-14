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

TEST_F(QcoUnitaryOpInterfaceTest, getUnitaryMatrix2x2) {
  const auto expectedValues = std::array{
      Eigen::Matrix2cd{{1, 0}, {0, 1}},
      Eigen::Matrix2cd{{0.87758256, -0.47942554i}, {-0.47942554i, 0.87758256}},
      Eigen::Matrix2cd{{0.99500417, -0.09195267 - 0.03887696i},
                       {0.09537451 + 0.02950279i, 0.76102116 + 0.64099928i}}};
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    reg[0] = builder.id(reg[0]);
    reg[0] = builder.rx(1.0, reg[0]);
    reg[0] = builder.u(0.2, 0.3, 0.4, reg[0]);
  });

  auto&& moduleOps = moduleOp->getBody()->getOperations();
  ASSERT_FALSE(moduleOps.empty());
  auto funcOp = llvm::dyn_cast<func::FuncOp>(moduleOps.begin());

  llvm::SmallVector<Eigen::Matrix2cd> actualValues;
  for (auto&& op : funcOp.getOps()) {
    auto unitaryOp = llvm::dyn_cast<qco::UnitaryOpInterface>(op);
    if (unitaryOp) {
      auto matrix = unitaryOp.getUnitaryMatrix<Eigen::Matrix2cd>();
      ASSERT_TRUE(matrix) << toString(*moduleOp)
                          << "\nFailed to get matrix of gate "
                          << actualValues.size();
      actualValues.push_back(*matrix);
    }
  }

  ASSERT_EQ(actualValues.size(), expectedValues.size())
      << "Mismatch of size of actual and expected values";
  for (std::size_t i = 0; i < actualValues.size(); ++i) {
    EXPECT_TRUE(actualValues[i].isApprox(expectedValues.at(i), 1e-8))
        << "Wrong matrix at gate " << i;
  }
}
