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
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/FileCheck/FileCheck.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <string>
#include <utility>

namespace {
class ImportTest : public ::testing::Test {
protected:
  std::unique_ptr<mlir::MLIRContext> context;

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mqt::ir::dyn::MQTDynDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();

    context = std::make_unique<mlir::MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  void TearDown() override {}
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
bool checkOutput(const std::string& checkString,
                 const std::string& outputString) {
  auto checkBuffer = llvm::MemoryBuffer::getMemBuffer(checkString, "");
  auto outputBuffer =
      llvm::MemoryBuffer::getMemBuffer(outputString, "Output", false);

  llvm::SmallString<4096> checkFileBuffer;
  const llvm::FileCheckRequest request;
  llvm::FileCheck fc(request);
  const llvm::StringRef checkFileText =
      fc.CanonicalizeFile(*checkBuffer, checkFileBuffer);

  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(checkFileText, "CheckFile"),
      llvm::SMLoc());
  if (fc.readCheckFile(sm, checkFileText)) {
    return false;
  }

  auto outputBufferBuffer = outputBuffer->getBuffer();
  sm.AddNewSourceBuffer(std::move(outputBuffer), llvm::SMLoc());
  return fc.checkInput(sm, outputBufferBuffer);
}

} // namespace

TEST_F(ImportTest, Allocation) {
  const qc::QuantumComputation qc(3);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
    CHECK: "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    CHECK: "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    CHECK: "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, I) {
  qc::QuantumComputation qc(1);
  qc.i(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.i()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, H) {
  qc::QuantumComputation qc(1);
  qc.h(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.h()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, X) {
  qc::QuantumComputation qc(1);
  qc.x(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.x()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Y) {
  qc::QuantumComputation qc(1);
  qc.y(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.y()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Z) {
  qc::QuantumComputation qc(1);
  qc.z(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.z()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, S) {
  qc::QuantumComputation qc(1);
  qc.s(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.s()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Sdg) {
  qc::QuantumComputation qc(1);
  qc.sdg(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.sdg()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, T) {
  qc::QuantumComputation qc(1);
  qc.t(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.t()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Tdg) {
  qc::QuantumComputation qc(1);
  qc.tdg(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.tdg()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, V) {
  qc::QuantumComputation qc(1);
  qc.v(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.v()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Vdg) {
  qc::QuantumComputation qc(1);
  qc.vdg(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.vdg()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Rx) {
  qc::QuantumComputation qc(1);
  qc.rx(0.5, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: mqtdyn.rx( static [5.000000e-01]) %[[Q:.*]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Swap) {
  qc::QuantumComputation qc(2);
  qc.swap(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtdyn.swap()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, CX) {
  qc::QuantumComputation qc(2);
  qc.cx(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
    CHECK: %[[Q_1:.*]] = "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    CHECK: %[[Q_2:.*]] = "mqtdyn.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    CHECK: mqtdyn.x() %[[Q_2]] ctrl %[[Q_1]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}
