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

#include "llvm/ADT/StringRef.h"
#include "llvm/FileCheck/FileCheck.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

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

std::string getOutputString(mlir::OwningOpRef<mlir::ModuleOp>* module) {
  std::string outputString;
  llvm::raw_string_ostream os(outputString);
  (*module)->print(os);
  os.flush();
  return outputString;
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/d2b3e86321eaf954451e0a49534fa654dd67421e/llvm/unittests/MIR/MachineMetadata.cpp#L181
bool checkOutput(std::string checkString, std::string outputString) {
  auto checkBuffer = llvm::MemoryBuffer::getMemBuffer(checkString, "");
  auto outputBuffer =
      llvm::MemoryBuffer::getMemBuffer(outputString, "Output", false);

  llvm::SmallString<4096> checkFileBuffer;
  llvm::FileCheckRequest request;
  llvm::FileCheck fc(request);
  llvm::StringRef checkFileText =
      fc.CanonicalizeFile(*checkBuffer, checkFileBuffer);

  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(checkFileText, "CheckFile"),
      llvm::SMLoc());
  if (fc.readCheckFile(sm, checkFileText))
    return false;

  auto outputBufferBuffer = outputBuffer->getBuffer();
  sm.AddNewSourceBuffer(std::move(outputBuffer), llvm::SMLoc());
  return fc.checkInput(sm, outputBufferBuffer);
}

} // namespace

TEST_F(ImportTest, Allocation) {
  qc::QuantumComputation qc(3);

  auto module = translateQuantumComputationToMLIR(*context, qc);

  auto outputString = getOutputString(&module);
  auto checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
    CHECK: "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    CHECK: "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    CHECK: "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, HOperation) {
  qc::QuantumComputation qc(1);
  qc.h(0);

  auto module = translateQuantumComputationToMLIR(*context, qc);

  auto outputString = getOutputString(&module);
  auto checkString = "CHECK: mqtdyn.h()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, RxOperation) {
  qc::QuantumComputation qc(1);
  qc.rx(0.5, 0);

  auto module = translateQuantumComputationToMLIR(*context, qc);

  auto outputString = getOutputString(&module);
  auto checkString = R"(
    CHECK: %[[Cst:.*]] = arith.constant 5.000000e-01 : f64
    CHECK: mqtdyn.rx(%[[Cst]])
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, SwapOperation) {
  qc::QuantumComputation qc(2);
  qc.swap(0, 1);

  auto module = translateQuantumComputationToMLIR(*context, qc);

  auto outputString = getOutputString(&module);
  auto checkString = "CHECK: mqtdyn.swap()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, CXOperation) {
  qc::QuantumComputation qc(2);
  qc.cx(0, 1);

  auto module = translateQuantumComputationToMLIR(*context, qc);

  auto outputString = getOutputString(&module);
  auto checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
    CHECK: %[[Q_1:.*]] = "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    CHECK: %[[Q_2:.*]] = "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    CHECK: mqtdyn.x() %[[Q_2]] ctrl %[[Q_1]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}
