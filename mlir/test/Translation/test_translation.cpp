/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"
#include "mlir/Dialect/MQTDyn/Translation/ImportQuantumComputation.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace {

mlir::MLIRContext* getMLIRContext() {
  mlir::DialectRegistry registry;
  registry.insert<mqt::ir::dyn::MQTDynDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();

  auto* context = new mlir::MLIRContext();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  return context;
}

class ImportTest : public ::testing::Test {
protected:
  mlir::MLIRContext* context = nullptr;

  void SetUp() override { context = getMLIRContext(); }

  void TearDown() override {
    delete context;
    context = nullptr;
  }
};

} // namespace

TEST_F(ImportTest, HOperation) {
  qc::QuantumComputation qc(1);
  qc.h(0);

  auto module = translateQuantumComputationToMLIR(*context, qc);

  std::string moduleStr;
  llvm::raw_string_ostream os(moduleStr);
  module->print(os);
  os.flush();

  auto expectedStr = R"(module {
  func.func @main() {
    %0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
    %1 = "mqtdyn.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    mqtdyn.h() %1
    return
  }
}
)";

  ASSERT_EQ(moduleStr, expectedStr);
}

TEST_F(ImportTest, RxOperation) {
  qc::QuantumComputation qc(1);
  qc.rx(0.5, 0);

  auto module = translateQuantumComputationToMLIR(*context, qc);

  std::string moduleStr;
  llvm::raw_string_ostream os(moduleStr);
  module->print(os);
  os.flush();

  auto expectedStr = R"(module {
  func.func @main() {
    %0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
    %1 = "mqtdyn.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    %cst = arith.constant 5.000000e-01 : f64
    mqtdyn.rx(%cst) %1
    return
  }
}
)";

  ASSERT_EQ(moduleStr, expectedStr);
}

TEST_F(ImportTest, SwapOperation) {
  qc::QuantumComputation qc(2);
  qc.swap(0, 1);

  auto module = translateQuantumComputationToMLIR(*context, qc);

  std::string moduleStr;
  llvm::raw_string_ostream os(moduleStr);
  module->print(os);
  os.flush();

  auto expectedStr = R"(module {
  func.func @main() {
    %0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
    %1 = "mqtdyn.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    %2 = "mqtdyn.extractQubit"(%0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    mqtdyn.swap() %1, %2
    return
  }
}
)";

  ASSERT_EQ(moduleStr, expectedStr);
}

TEST_F(ImportTest, CXOperation) {
  qc::QuantumComputation qc(2);
  qc.cx(0, 1);

  auto module = translateQuantumComputationToMLIR(*context, qc);

  std::string moduleStr;
  llvm::raw_string_ostream os(moduleStr);
  module->print(os);
  os.flush();

  auto expectedStr = R"(module {
  func.func @main() {
    %0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
    %1 = "mqtdyn.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    %2 = "mqtdyn.extractQubit"(%0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    mqtdyn.x() %2 ctrl %1
    return
  }
}
)";

  ASSERT_EQ(moduleStr, expectedStr);
}
