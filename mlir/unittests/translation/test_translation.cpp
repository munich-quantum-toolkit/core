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
#include "ir/operations/Control.hpp"
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"
#include "mlir/Dialect/MQTRef/Translation/ImportQuantumComputation.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/FileCheck/FileCheck.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
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
    registry.insert<mqt::ir::ref::MQTRefDialect>();
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

using namespace qc;

TEST_F(ImportTest, EntryPoint) {
  const qc::QuantumComputation qc{};

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: func.func @main()
    CHECK: return
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, AllocationAndDeallocation) {
  const qc::QuantumComputation qc(3);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
    CHECK: "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: "mqtref.deallocQubitRegister"(%[[Reg]]) : (!mqtref.QubitRegister) -> ()
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Measure01) {
  qc::QuantumComputation qc(2, 2);
  qc.measure({0, 1}, {0, 1});

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Measure0) {
  qc::QuantumComputation qc(2, 2);
  qc.measure(0, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK-NOT: %[[M1:.*]] = mqtref.measure %[[Q1]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Reset01) {
  qc::QuantumComputation qc(2);
  qc.reset({0, 1});

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: "mqtref.reset"(%[[Q0]]) : (!mqtref.Qubit) -> ()
    CHECK: "mqtref.reset"(%[[Q1]]) : (!mqtref.Qubit) -> ()
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Reset0) {
  qc::QuantumComputation qc(2);
  qc.reset(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: "mqtref.reset"(%[[Q0]]) : (!mqtref.Qubit) -> ()
    CHECK-NOT: "mqtref.reset"(%[[Q1]]) : (!mqtref.Qubit) -> ()
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, I) {
  qc::QuantumComputation qc(1);
  qc.i(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.i()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, H) {
  qc::QuantumComputation qc(1);
  qc.h(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.h()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, X) {
  qc::QuantumComputation qc(1);
  qc.x(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.x()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, CX01) {
  qc::QuantumComputation qc(2);
  qc.cx(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.x() %[[Q1]] ctrl %[[Q0]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, CX10) {
  qc::QuantumComputation qc(2);
  qc.cx(1, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.x() %[[Q0]] ctrl %[[Q1]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, CX0N1) {
  qc::QuantumComputation qc(2);
  qc.cx(0_nc, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.x() %[[Q1]] nctrl %[[Q0]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, MCX012) {
  qc::QuantumComputation qc(3);
  qc.mcx({0, 1}, 2);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q2:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.x() %[[Q2]] ctrl %[[Q0]], %[[Q1]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, MCX0N2P1) {
  qc::QuantumComputation qc(3);
  qc.mcx({0_nc, 2}, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q2:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.x() %[[Q1]] ctrl %[[Q2]] nctrl %[[Q0]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, MCX2N1N0) {
  qc::QuantumComputation qc(3);
  qc.mcx({2_nc, 1_nc}, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q2:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.x() %[[Q0]] nctrl %[[Q1]], %[[Q2]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Y) {
  qc::QuantumComputation qc(1);
  qc.y(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.y()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Z) {
  qc::QuantumComputation qc(1);
  qc.z(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.z()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, S) {
  qc::QuantumComputation qc(1);
  qc.s(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.s()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Sdg) {
  qc::QuantumComputation qc(1);
  qc.sdg(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.sdg()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, T) {
  qc::QuantumComputation qc(1);
  qc.t(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.t()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Tdg) {
  qc::QuantumComputation qc(1);
  qc.tdg(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.tdg()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, V) {
  qc::QuantumComputation qc(1);
  qc.v(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.v()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Vdg) {
  qc::QuantumComputation qc(1);
  qc.vdg(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.vdg()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, U) {
  qc::QuantumComputation qc(1);
  qc.u(0.1, 0.2, 0.3, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString =
      "CHECK: mqtref.u( static [1.000000e-01, 2.000000e-01, 3.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, U2) {
  qc::QuantumComputation qc(1);
  qc.u2(0.1, 0.2, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString =
      "CHECK: mqtref.u2( static [1.000000e-01, 2.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, P) {
  qc::QuantumComputation qc(1);
  qc.p(0.1, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.p( static [1.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, SX) {
  qc::QuantumComputation qc(1);
  qc.sx(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.sx()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, SXdg) {
  qc::QuantumComputation qc(1);
  qc.sxdg(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.sxdg()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Rx) {
  qc::QuantumComputation qc(1);
  qc.rx(0.1, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.rx( static [1.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Ry) {
  qc::QuantumComputation qc(1);
  qc.ry(0.1, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.ry( static [1.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Rz) {
  qc::QuantumComputation qc(1);
  qc.rz(0.1, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.rz( static [1.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, SWAP01) {
  qc::QuantumComputation qc(2);
  qc.swap(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.swap() %[[Q0]], %[[Q1]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, SWAP10) {
  qc::QuantumComputation qc(2);
  qc.swap(1, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.swap() %[[Q1]], %[[Q0]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, iSWAP) {
  qc::QuantumComputation qc(2);
  qc.iswap(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.iswap()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, iSWAPdg) {
  qc::QuantumComputation qc(2);
  qc.iswapdg(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.iswapdg()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Peres) {
  qc::QuantumComputation qc(2);
  qc.peres(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.peres()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Peresdg) {
  qc::QuantumComputation qc(2);
  qc.peresdg(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.peresdg()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, DCX) {
  qc::QuantumComputation qc(2);
  qc.dcx(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.dcx()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, ECR) {
  qc::QuantumComputation qc(2);
  qc.ecr(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.ecr()";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, RXX) {
  qc::QuantumComputation qc(2);
  qc.rxx(0.1, 0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.rxx( static [1.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, CRXX) {
  qc::QuantumComputation qc(3);
  qc.crxx(0.1, 0, 1, 2);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q2:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.rxx( static [1.000000e-01]) %[[Q1]], %[[Q2]] ctrl %[[Q0]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, RYY) {
  qc::QuantumComputation qc(2);
  qc.ryy(0.1, 0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.ryy( static [1.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, RZZ) {
  qc::QuantumComputation qc(2);
  qc.rzz(0.1, 0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.rzz( static [1.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, RZX) {
  qc::QuantumComputation qc(2);
  qc.rzx(0.1, 0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = "CHECK: mqtref.rzx( static [1.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, XXminusYY) {
  qc::QuantumComputation qc(2);
  qc.xx_minus_yy(0.1, 0.2, 0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString =
      "CHECK: mqtref.xxminusyy( static [1.000000e-01, 2.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, XXplusYY) {
  qc::QuantumComputation qc(2);
  qc.xx_plus_yy(0.1, 0.2, 0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString =
      "CHECK: mqtref.xxplusyy( static [1.000000e-01, 2.000000e-01])";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, GHZ) {
  qc::QuantumComputation qc(3, 3);
  qc.h(0);
  qc.cx(0, 1);
  qc.cx(0, 2);
  qc.measure({0, 1, 2}, {0, 1, 2});

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Reg:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
    CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: %[[Q2:.*]] = "mqtref.extractQubit"(%[[Reg]]) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
    CHECK: mqtref.h() %[[Q0]]
    CHECK: mqtref.x() %[[Q1]] ctrl %[[Q0]]
    CHECK: mqtref.x() %[[Q2]] ctrl %[[Q0]]
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
    CHECK: %[[M2:.*]] = mqtref.measure %[[Q2]]
    CHECK: "mqtref.deallocQubitRegister"(%[[Reg]]) : (!mqtref.QubitRegister) -> ()
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}
