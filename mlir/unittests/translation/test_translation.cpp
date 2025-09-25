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
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"
#include "mlir/Dialect/MQTRef/Translation/ImportQuantumComputation.h"

#include <cstddef>
#include <functional>
#include <gtest/gtest.h>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/FileCheck/FileCheck.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace {

using namespace qc;

class ImportTest : public ::testing::Test {
protected:
  std::unique_ptr<mlir::MLIRContext> context;

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mqt::ir::ref::MQTRefDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();

    context = std::make_unique<mlir::MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  void runPasses(const mlir::ModuleOp module) const {
    mlir::PassManager passManager(context.get());
    passManager.addPass(mlir::createCanonicalizerPass());
    passManager.addPass(mlir::createMem2Reg());
    passManager.addPass(mlir::createRemoveDeadValuesPass());
    if (passManager.run(module).failed()) {
      FAIL() << "Failed to run passes";
    }
  }

  void TearDown() override {}
};

// ##################################################
// # Helper functions
// ##################################################

std::string getOutputString(mlir::OwningOpRef<mlir::ModuleOp>* module) {
  std::string outputString;
  llvm::raw_string_ostream os(outputString);
  (*module)->print(os);
  os.flush();
  return outputString;
}

std::string formatTargets(std::initializer_list<size_t> targets) {
  std::string s;
  bool first = true;
  for (auto t : targets) {
    if (!first) {
      s += ", ";
    }
    first = false;
    s += "%[[Q" + std::to_string(t) + "]]";
  }
  return s;
}

std::string formatParams(std::initializer_list<double> params) {
  if (params.size() == 0) {
    return "";
  }
  std::ostringstream os;
  os.setf(std::ios::scientific);
  os << std::setprecision(6);
  bool first = true;
  os << "static [";
  for (const double p : params) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << p;
  }
  os << "]";
  return os.str();
}

std::string getCheckStringOperation(const char* op,
                                    std::initializer_list<size_t> targets) {
  return std::string("CHECK: mqtref.") + op + "() " + formatTargets(targets);
}

std::string
getCheckStringOperationParams(const char* op,
                              std::initializer_list<double> params,
                              std::initializer_list<size_t> targets) {
  return std::string("CHECK: mqtref.") + op + "(" + formatParams(params) +
         ") " + formatTargets(targets);
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

// ##################################################
// # Basic tests
// ##################################################

TEST_F(ImportTest, EntryPoint) {
  const QuantumComputation qc{};

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: func.func @main() attributes {passthrough = ["entry_point"]}
    CHECK: return
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, AllocationAndDeallocation) {
  const QuantumComputation qc(3, 2);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<3x!mqtref.Qubit>
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<3x!mqtref.Qubit>
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<3x!mqtref.Qubit>
    CHECK: %[[I2:.*]] = arith.constant 2 : index
    CHECK: %[[Q2:.*]] = memref.load %[[Qreg]][%[[I2]]] : memref<3x!mqtref.Qubit>
    CHECK: %[[Creg:.*]] = memref.alloca() : memref<2xi1>
    CHECK: memref.dealloc %[[Qreg]] : memref<3x!mqtref.Qubit>
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Measure01) {
  QuantumComputation qc(2, 2);
  qc.measure({0, 1}, {0, 1});

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtref.Qubit>
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[Creg:.*]] = memref.alloca() : memref<2xi1>
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: memref.store %[[M0]], %[[Creg]][%[[I0]]] : memref<2xi1>
    CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: memref.store %[[M1]], %[[Creg]][%[[I1]]] : memref<2xi1>
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Measure0) {
  QuantumComputation qc(2, 2);
  qc.measure(0, 0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtref.Qubit>
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[Creg:.*]] = memref.alloca() : memref<2xi1>
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: memref.store %[[M0]], %[[Creg]][%[[I0]]] : memref<2xi1>
    CHECK-NOT: mqtref.measure %[[Q1]]
    CHECK-NOT: arith.constant 1 : index
    CHECK-NOT: memref.store  %[[ANY:.*]], %[[Creg]][%[[ANY:.*]]] : memref<2xi1>
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Reset01) {
  QuantumComputation qc(2);
  qc.reset({0, 1});

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtref.Qubit>
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtref.Qubit>
    CHECK: mqtref.reset %[[Q0]]
    CHECK: mqtref.reset %[[Q1]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, Reset0) {
  QuantumComputation qc(2);
  qc.reset(0);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtref.Qubit>
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtref.Qubit>
    CHECK: mqtref.reset %[[Q0]]
    CHECK-NOT: mqtref.reset %[[Q1]]
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

// ##################################################
// # Test unitary operations
// ##################################################

struct TestCaseUnitary {
  std::string name; // Name of the test case
  size_t numQubits;
  std::function<void(QuantumComputation&)> build;
  std::string checkStringOperation;
};

std::ostream& operator<<(std::ostream& os, const TestCaseUnitary& testCase) {
  return os << testCase.name;
}

std::string getCheckStringTestCaseUnitary(const TestCaseUnitary& testCase) {
  std::string result;

  // Add entry point
  result +=
      "CHECK: func.func @main() attributes {passthrough = [\"entry_point\"]}\n";

  // Add register allocation
  result += "CHECK: %[[Qreg:.*]] = memref.alloc() : memref<" +
            std::to_string(testCase.numQubits) + "x!mqtref.Qubit>\n";

  // Add qubit extraction
  for (size_t i = 0; i < testCase.numQubits; ++i) {
    result += "CHECK: %[[I" + std::to_string(i) + ":.*]] = arith.constant " +
              std::to_string(i) + " : index\n";
    result += "CHECK: %[[Q" + std::to_string(i) +
              ":.*]] = memref.load %[[Qreg]][%[[I" + std::to_string(i) +
              ":.*]]] : memref<" + std::to_string(testCase.numQubits) +
              "x!mqtref.Qubit>\n";
  }

  // Add operation-specific check
  result += testCase.checkStringOperation;
  result += "\n";

  // Add dealocation
  result += "CHECK: memref.dealloc %[[Qreg]] : memref<" +
            std::to_string(testCase.numQubits) + "x!mqtref.Qubit>\n";

  // Add return
  result += "CHECK: return\n";

  return result;
}

class OperationTestUnitary
    : public ImportTest,
      public ::testing::WithParamInterface<TestCaseUnitary> {};

TEST_P(OperationTestUnitary, EmitsExpectedOperation) {
  const auto& param = GetParam();

  QuantumComputation qc(param.numQubits);
  param.build(qc);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto checkString = getCheckStringTestCaseUnitary(param);

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

INSTANTIATE_TEST_SUITE_P(
    Operations, OperationTestUnitary,
    ::testing::Values(
        // Barrier
        TestCaseUnitary{
            .name = "Barrier",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.barrier(0); },
            .checkStringOperation = getCheckStringOperation("barrier", {0})},
        // 1-qubit gates without parameters
        TestCaseUnitary{
            .name = "I",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.i(0); },
            .checkStringOperation = getCheckStringOperation("i", {0})},
        TestCaseUnitary{
            .name = "H",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.h(0); },
            .checkStringOperation = getCheckStringOperation("h", {0})},
        TestCaseUnitary{
            .name = "X",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.x(0); },
            .checkStringOperation = getCheckStringOperation("x", {0})},
        TestCaseUnitary{
            .name = "Y",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.y(0); },
            .checkStringOperation = getCheckStringOperation("y", {0})},
        TestCaseUnitary{
            .name = "Z",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.z(0); },
            .checkStringOperation = getCheckStringOperation("z", {0})},
        TestCaseUnitary{
            .name = "S",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.s(0); },
            .checkStringOperation = getCheckStringOperation("s", {0})},
        TestCaseUnitary{
            .name = "Sdg",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.sdg(0); },
            .checkStringOperation = getCheckStringOperation("sdg", {0})},
        TestCaseUnitary{
            .name = "T",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.t(0); },
            .checkStringOperation = getCheckStringOperation("t", {0})},
        TestCaseUnitary{
            .name = "Tdg",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.tdg(0); },
            .checkStringOperation = getCheckStringOperation("tdg", {0})},
        TestCaseUnitary{
            .name = "V",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.v(0); },
            .checkStringOperation = getCheckStringOperation("v", {0})},
        TestCaseUnitary{
            .name = "Vdg",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.vdg(0); },
            .checkStringOperation = getCheckStringOperation("vdg", {0})},
        // 1-qubit gates with parameters
        TestCaseUnitary{
            .name = "U",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.u(0.1, 0.2, 0.3, 0); },
            .checkStringOperation =
                getCheckStringOperationParams("u", {0.1, 0.2, 0.3}, {0})},
        TestCaseUnitary{
            .name = "U2",
            .numQubits = 1,
            .build = [](QuantumComputation& qc) { qc.u2(0.1, 0.2, 0); },
            .checkStringOperation =
                getCheckStringOperationParams("u2", {0.1, 0.2}, {0})},
        TestCaseUnitary{.name = "Phase",
                        .numQubits = 1,
                        .build = [](QuantumComputation& qc) { qc.p(0.1, 0); },
                        .checkStringOperation =
                            getCheckStringOperationParams("p", {0.1}, {0})},
        TestCaseUnitary{.name = "RX",
                        .numQubits = 1,
                        .build = [](QuantumComputation& qc) { qc.rx(0.1, 0); },
                        .checkStringOperation =
                            getCheckStringOperationParams("rx", {0.1}, {0})},
        TestCaseUnitary{.name = "RY",
                        .numQubits = 1,
                        .build = [](QuantumComputation& qc) { qc.ry(0.1, 0); },
                        .checkStringOperation =
                            getCheckStringOperationParams("ry", {0.1}, {0})},
        TestCaseUnitary{.name = "RZ",
                        .numQubits = 1,
                        .build = [](QuantumComputation& qc) { qc.rz(0.1, 0); },
                        .checkStringOperation =
                            getCheckStringOperationParams("rz", {0.1}, {0})},
        // 2-qubit gates without parameters
        TestCaseUnitary{
            .name = "iSWAP",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.iswap(0, 1); },
            .checkStringOperation = getCheckStringOperation("iswap", {0, 1})},
        TestCaseUnitary{
            .name = "iSWAPdg",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.iswapdg(0, 1); },
            .checkStringOperation = getCheckStringOperation("iswapdg", {0, 1})},
        TestCaseUnitary{
            .name = "Peres",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.peres(0, 1); },
            .checkStringOperation = getCheckStringOperation("peres", {0, 1})},
        TestCaseUnitary{
            .name = "Peresdg",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.peresdg(0, 1); },
            .checkStringOperation = getCheckStringOperation("peresdg", {0, 1})},
        TestCaseUnitary{
            .name = "DCX",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.dcx(0, 1); },
            .checkStringOperation = getCheckStringOperation("dcx", {0, 1})},
        TestCaseUnitary{
            .name = "ECR",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.ecr(0, 1); },
            .checkStringOperation = getCheckStringOperation("ecr", {0, 1})},
        // 2-qubit gates with parameters
        TestCaseUnitary{
            .name = "RXX",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.rxx(0.1, 0, 1); },
            .checkStringOperation = getCheckStringOperationParams("rxx", {0.1},
                                                                  {0, 1})},
        TestCaseUnitary{
            .name = "RYY",
            .numQubits = 2,

            .build = [](QuantumComputation& qc) { qc.ryy(0.1, 0, 1); },
            .checkStringOperation = getCheckStringOperationParams("ryy", {0.1},
                                                                  {0, 1})},
        TestCaseUnitary{
            .name = "RZZ",
            .numQubits = 2,

            .build = [](QuantumComputation& qc) { qc.rzz(0.1, 0, 1); },
            .checkStringOperation = getCheckStringOperationParams("rzz", {0.1},
                                                                  {0, 1})},
        TestCaseUnitary{
            .name = "RZX",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.rzx(0.1, 0, 1); },
            .checkStringOperation = getCheckStringOperationParams("rzx", {0.1},
                                                                  {0, 1})},
        TestCaseUnitary{
            .name = "XX_MINUS_YY",
            .numQubits = 2,
            .build =
                [](QuantumComputation& qc) { qc.xx_minus_yy(0.1, 0.2, 0, 1); },
            .checkStringOperation = getCheckStringOperationParams(
                "xx_minus_yy", {0.1, 0.2}, {0, 1})},
        TestCaseUnitary{
            .name = "XX_PLUS_YY",
            .numQubits = 2,
            .build =
                [](QuantumComputation& qc) { qc.xx_plus_yy(0.1, 0.2, 0, 1); },
            .checkStringOperation = getCheckStringOperationParams(
                "xx_plus_yy", {0.1, 0.2}, {0, 1})},
        // Controlled gates
        TestCaseUnitary{
            .name = "CX_0P_1",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.cx(0, 1); },
            .checkStringOperation = "CHECK: mqtref.x() %[[Q1]] ctrl %[[Q0]]"},
        TestCaseUnitary{
            .name = "CX_1P_0",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.cx(1, 0); },
            .checkStringOperation = "CHECK: mqtref.x() %[[Q0]] ctrl %[[Q1]]"},
        TestCaseUnitary{
            .name = "CX_0N_1",
            .numQubits = 2,
            .build = [](QuantumComputation& qc) { qc.cx(0_nc, 1); },
            .checkStringOperation = "CHECK: mqtref.x() %[[Q1]] nctrl %[[Q0]]"},
        TestCaseUnitary{
            .name = "MCX_0P1P_2",
            .numQubits = 3,
            .build = [](QuantumComputation& qc) { qc.mcx({0, 1}, 2); },
            .checkStringOperation =
                "CHECK: mqtref.x() %[[Q2]] ctrl %[[Q0]], %[[Q1]]"},
        TestCaseUnitary{
            .name = "MCX_0N2P_1",
            .numQubits = 3,
            .build = [](QuantumComputation& qc) { qc.mcx({0_nc, 2}, 1); },
            .checkStringOperation =
                "CHECK: mqtref.x() %[[Q1]] ctrl %[[Q2]] nctrl %[[Q0]]"},
        TestCaseUnitary{
            .name = "MCX_2N1N_0",
            .numQubits = 3,
            .build = [](QuantumComputation& qc) { qc.mcx({2_nc, 1_nc}, 0); },
            .checkStringOperation =
                "CHECK: mqtref.x() %[[Q0]] nctrl %[[Q1]], %[[Q2]]"}));

// ##################################################
// # Test register-controlled if-else operations
// ##################################################

struct TestCaseIfRegister {
  std::string name; // Name of the test case
  ComparisonKind comparisonKind;
  std::string predicate;
};

std::ostream& operator<<(std::ostream& os, const TestCaseIfRegister& testCase) {
  return os << testCase.name;
}

std::string
getCheckStringTestCaseIfRegister(const TestCaseIfRegister& testCase) {
  std::string result;

  result += R"(
    CHECK: func.func @main() attributes {passthrough = ["entry_point"]}
    CHECK: %[[Exp:.*]] = arith.constant 1 : i64
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<1x!mqtref.Qubit>
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<1x!mqtref.Qubit>
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: %[[C0:.*]] = arith.extui %[[M0]] : i1 to i64
  )";

  // Add comparison
  result += "CHECK: %[[Cnd0:.*]] = arith.cmpi " + testCase.predicate +
            ", %[[C0]], %[[Exp]] : i64\n";

  result += R"(
    CHECK: scf.if %[[Cnd0]] {
    CHECK: mqtref.x() %[[Q0]]
    CHECK: }
    CHECK: memref.dealloc %[[Qreg]] : memref<1x!mqtref.Qubit>
    CHECK: return
  )";

  return result;
}

class OperationTestIfRegister
    : public ImportTest,
      public ::testing::WithParamInterface<TestCaseIfRegister> {};

TEST_P(OperationTestIfRegister, EmitsExpectedOperation) {
  const auto& param = GetParam();

  QuantumComputation qc(1);
  const auto& creg = qc.addClassicalRegister(1);
  qc.measure(0, 0);
  qc.if_(X, 0, creg, 1U, param.comparisonKind);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);
  runPasses(module.get());

  const auto outputString = getOutputString(&module);
  const auto checkString = getCheckStringTestCaseIfRegister(param);

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

INSTANTIATE_TEST_SUITE_P(
    Operations, OperationTestIfRegister,
    ::testing::Values(
        TestCaseIfRegister{
            .name = "Eq", .comparisonKind = Eq, .predicate = "eq"},
        TestCaseIfRegister{
            .name = "Neq", .comparisonKind = Neq, .predicate = "ne"},
        TestCaseIfRegister{
            .name = "Lt", .comparisonKind = Lt, .predicate = "ult"},
        TestCaseIfRegister{
            .name = "Leq", .comparisonKind = Leq, .predicate = "ule"},
        TestCaseIfRegister{
            .name = "Geq", .comparisonKind = Geq, .predicate = "uge"},
        TestCaseIfRegister{
            .name = "Gt", .comparisonKind = Gt, .predicate = "ugt"}));

TEST_F(ImportTest, IfRegisterEq2) {
  QuantumComputation qc(2);
  const auto& creg = qc.addClassicalRegister(2);
  qc.measure({0, 1}, {0, 1});
  qc.if_(X, 0, creg, 2U, Eq);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);
  runPasses(module.get());

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Exp:.*]] = arith.constant 2 : i64
    CHECK: %[[Sum0:.*]] = arith.constant 0 : i64
    CHECK: %[[I2:.*]] = arith.constant 2 : index
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtref.Qubit>
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[Creg:.*]] = memref.alloca() : memref<2xi1>
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: memref.store %[[M0]], %[[Creg]][%[[I0]]] : memref<2xi1>
    CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
    CHECK: memref.store %[[M1]], %[[Creg]][%[[I1]]] : memref<2xi1>
    CHECK: %[[Sum1:.*]] = scf.for %[[Ii:.*]] = %[[I0]] to %[[I2]] step %[[I1]] iter_args(%[[Sumi:.*]] = %[[Sum0]]) -> (i64) {
    CHECK: %[[Bi:.*]] = memref.load %[[Creg]][%[[Ii]]] : memref<2xi1>
    CHECK: %[[Ci:.*]] = arith.extui %[[Bi:.*]] : i1 to i64
    CHECK: %[[Indi:.*]] = arith.index_cast %[[Ii]] : index to i64
    CHECK: %[[Shli:.*]] = arith.shli %[[Ci]], %[[Indi]] : i64
    CHECK: %[[Sumj:.*]] = arith.addi %[[Sumi]], %[[Shli]] : i64
    CHECK: scf.yield %[[Sumj]] : i64
    CHECK: }
    CHECK: %[[Cnd0:.*]] = arith.cmpi eq, %[[Sum1]], %[[Exp]] : i64
    CHECK: scf.if %[[Cnd0]] {
    CHECK: mqtref.x() %[[Q0]]
    CHECK: }
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, IfElseRegister) {
  QuantumComputation qc(2);
  const auto& creg = qc.addClassicalRegister(2);
  qc.measure({0, 1}, {0, 1});
  qc.ifElse(std::make_unique<StandardOperation>(0, X),
            std::make_unique<StandardOperation>(0, Y), creg, 2U, Eq);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);
  runPasses(module.get());

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: %[[Exp:.*]] = arith.constant 2 : i64
    CHECK: %[[Sum0:.*]] = arith.constant 0 : i64
    CHECK: %[[I2:.*]] = arith.constant 2 : index
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtref.Qubit>
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[Creg:.*]] = memref.alloca() : memref<2xi1>
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: memref.store %[[M0]], %[[Creg]][%[[I0]]] : memref<2xi1>
    CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
    CHECK: memref.store %[[M1]], %[[Creg]][%[[I1]]] : memref<2xi1>
    CHECK: %[[Sum1:.*]] = scf.for %[[Ii:.*]] = %[[I0]] to %[[I2]] step %[[I1]] iter_args(%[[Sumi:.*]] = %[[Sum0]]) -> (i64) {
    CHECK: %[[Bi:.*]] = memref.load %[[Creg]][%[[Ii]]] : memref<2xi1>
    CHECK: %[[Ci:.*]] = arith.extui %[[Bi:.*]] : i1 to i64
    CHECK: %[[Indi:.*]] = arith.index_cast %[[Ii]] : index to i64
    CHECK: %[[Shli:.*]] = arith.shli %[[Ci]], %[[Indi]] : i64
    CHECK: %[[Sumj:.*]] = arith.addi %[[Sumi]], %[[Shli]] : i64
    CHECK: scf.yield %[[Sumj]] : i64
    CHECK: }
    CHECK: %[[Cnd0:.*]] = arith.cmpi eq, %[[Sum1]], %[[Exp]] : i64
    CHECK: scf.if %[[Cnd0:.*]] {
    CHECK: mqtref.x() %[[Q0]]
    CHECK: } else {
    CHECK: mqtref.y() %[[Q0]]
    CHECK: }
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

// ##################################################
// # Test bit-controlled if-else operations
// ##################################################

struct TestCaseIfBit {
  std::string name; // Name of the test case
  ComparisonKind comparisonKind;
  bool expectedValueInput;
  bool expectedValueOutput;
};

std::ostream& operator<<(std::ostream& os, const TestCaseIfBit& testCase) {
  return os << testCase.name;
}

std::string getCheckStringTestCaseIfBit(const TestCaseIfBit& testCase) {
  std::string result;

  result += R"(
    CHECK: func.func @main() attributes {passthrough = ["entry_point"]}
  )";

  if (!testCase.expectedValueOutput) {
    result += "CHECK: %[[Exp0:.*]] = arith.constant false\n";
  }

  result += R"(
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<1x!mqtref.Qubit>
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<1x!mqtref.Qubit>
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
  )";

  if (!testCase.expectedValueOutput) {
    result += R"(
      CHECK: %[[Cnd0:.*]] = arith.cmpi eq, %[[M0]], %[[Exp0]] : i1
      CHECK: scf.if %[[Cnd0]] {
    )";
  } else {
    result += "CHECK: scf.if %[[M0]] {\n";
  }

  result += R"(
    CHECK: mqtref.x() %[[Q0]]
    CHECK: }
    CHECK: memref.dealloc %[[Qreg]] : memref<1x!mqtref.Qubit>
    CHECK: return
  )";

  return result;
}

class OperationTestIfBit : public ImportTest,
                           public ::testing::WithParamInterface<TestCaseIfBit> {
};

TEST_P(OperationTestIfBit, EmitsExpectedOperation) {
  const auto& param = GetParam();

  QuantumComputation qc(1, 1);
  qc.measure(0, 0);
  qc.if_(X, 0, 0, param.expectedValueInput, param.comparisonKind);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);
  runPasses(module.get());

  const auto outputString = getOutputString(&module);
  const auto checkString = getCheckStringTestCaseIfBit(param);

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

INSTANTIATE_TEST_SUITE_P(
    Operations, OperationTestIfBit,
    ::testing::Values(TestCaseIfBit{.name = "EqTrue",
                                    .comparisonKind = Eq,
                                    .expectedValueInput = true,
                                    .expectedValueOutput = true},
                      TestCaseIfBit{.name = "EqFalse",
                                    .comparisonKind = Eq,
                                    .expectedValueInput = false,
                                    .expectedValueOutput = false},
                      TestCaseIfBit{.name = "NeqTrue",
                                    .comparisonKind = Neq,
                                    .expectedValueInput = true,
                                    .expectedValueOutput = false},
                      TestCaseIfBit{.name = "NeqFalse",
                                    .comparisonKind = Neq,
                                    .expectedValueInput = false,
                                    .expectedValueOutput = true}));

struct TestCaseIfElseBit {
  std::string name; // Name of the test case
  ComparisonKind comparisonKind;
  bool expectedValueInput;
  std::string thenOperation;
  std::string elseOperation;
};

std::ostream& operator<<(std::ostream& os, const TestCaseIfElseBit& testCase) {
  return os << testCase.name;
}

std::string getCheckStringTestCaseIfElseBit(const TestCaseIfElseBit& testCase) {
  std::string result;

  result += R"(
    CHECK: func.func @main() attributes {passthrough = ["entry_point"]}
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<1x!mqtref.Qubit>
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<1x!mqtref.Qubit>
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: scf.if %[[M0]] {
  )";

  result += "CHECK: mqtref." + testCase.thenOperation + "() %[[Q0]]\n";
  result += "} else {\n";
  result += "CHECK: mqtref." + testCase.elseOperation + "() %[[Q0]]\n";

  result += R"(
    CHECK: }
    CHECK: memref.dealloc %[[Qreg]] : memref<1x!mqtref.Qubit>
    CHECK: return
  )";

  return result;
}

class OperationTestIfElseBit
    : public ImportTest,
      public ::testing::WithParamInterface<TestCaseIfElseBit> {};

TEST_P(OperationTestIfElseBit, EmitsExpectedOperation) {
  const auto& param = GetParam();

  QuantumComputation qc(1, 1);
  qc.measure(0, 0);
  qc.ifElse(std::make_unique<StandardOperation>(0, X),
            std::make_unique<StandardOperation>(0, Y), 0,
            param.expectedValueInput, param.comparisonKind);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);
  runPasses(module.get());

  const auto outputString = getOutputString(&module);
  const auto checkString = getCheckStringTestCaseIfElseBit(param);

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

INSTANTIATE_TEST_SUITE_P(
    Operations, OperationTestIfElseBit,
    ::testing::Values(TestCaseIfElseBit{.name = "EqTrue",
                                        .comparisonKind = Eq,
                                        .expectedValueInput = true,
                                        .thenOperation = "x",
                                        .elseOperation = "y"},
                      TestCaseIfElseBit{.name = "EqFalse",
                                        .comparisonKind = Eq,
                                        .expectedValueInput = false,
                                        .thenOperation = "y",
                                        .elseOperation = "x"},
                      TestCaseIfElseBit{.name = "NeqTrue",
                                        .comparisonKind = Neq,
                                        .expectedValueInput = true,
                                        .thenOperation = "y",
                                        .elseOperation = "x"},
                      TestCaseIfElseBit{.name = "NeqFalse",
                                        .comparisonKind = Neq,
                                        .expectedValueInput = false,
                                        .thenOperation = "x",
                                        .elseOperation = "y"}));

// ##################################################
// # Test full programs
// ##################################################

TEST_F(ImportTest, GHZ) {
  QuantumComputation qc(3, 3);
  qc.h(0);
  qc.cx(0, 1);
  qc.cx(0, 2);
  qc.measure({0, 1, 2}, {0, 1, 2});

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto outputString = getOutputString(&module);
  const auto* checkString = R"(
    CHECK: func.func @main() attributes {passthrough = ["entry_point"]}
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<3x!mqtref.Qubit>
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<3x!mqtref.Qubit>
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<3x!mqtref.Qubit>
    CHECK: %[[I2:.*]] = arith.constant 2 : index
    CHECK: %[[Q2:.*]] = memref.load %[[Qreg]][%[[I2]]] : memref<3x!mqtref.Qubit>
    CHECK: %[[Creg:.*]] = memref.alloca() : memref<3xi1>
    CHECK: mqtref.h() %[[Q0]]
    CHECK: mqtref.x() %[[Q1]] ctrl %[[Q0]]
    CHECK: mqtref.x() %[[Q2]] ctrl %[[Q0]]
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: memref.store %[[M0]], %[[Creg]][%[[I0]]] : memref<3xi1>
    CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: memref.store %[[M1]], %[[Creg]][%[[I1]]] : memref<3xi1>
    CHECK: %[[M2:.*]] = mqtref.measure %[[Q2]]
    CHECK: %[[I2:.*]] = arith.constant 2 : index
    CHECK: memref.store %[[M2]], %[[Creg]][%[[I2]]] : memref<3xi1>
    CHECK: memref.dealloc %[[Qreg]] : memref<3x!mqtref.Qubit>
    CHECK: return
  )";

  ASSERT_TRUE(checkOutput(checkString, outputString));
}

TEST_F(ImportTest, MultipleClassicalRegistersMeasureStores) {
  QuantumComputation qc(2, 0);
  qc.addClassicalRegister(1, "c0");
  qc.addClassicalRegister(1, "c1");
  qc.measure({0, 1}, {0, 1});

  auto module = translateQuantumComputationToMLIR(context.get(), qc);
  // We do not run passes here; pattern should match raw allocation and stores

  const auto output = getOutputString(&module);
  const std::string check = R"(
    CHECK: func.func @main() attributes {passthrough = ["entry_point"]}
    CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtref.Qubit>
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtref.Qubit>
    CHECK: %[[CregA:.*]] = memref.alloca() : memref<1xi1>
    CHECK: %[[CregB:.*]] = memref.alloca() : memref<1xi1>
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: %[[I0A:.*]] = arith.constant 0 : index
    CHECK: memref.store %[[M0]], %[[CregA]][%[[I0A]]] : memref<1xi1>
    CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
    CHECK: %[[I0B:.*]] = arith.constant 0 : index
    CHECK: memref.store %[[M1]], %[[CregB]][%[[I0B]]] : memref<1xi1>
    CHECK: memref.dealloc %[[Qreg]] : memref<2x!mqtref.Qubit>
    CHECK: return
  )";

  ASSERT_TRUE(checkOutput(check, output));
}

TEST_F(ImportTest, MultipleQuantumRegistersCX) {
  QuantumComputation qc(0, 0);
  qc.addQubitRegister(1, "q0");
  qc.addQubitRegister(1, "q1");
  qc.cx(0, 1);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto output = getOutputString(&module);
  const std::string check = R"(
    CHECK: func.func @main() attributes {passthrough = ["entry_point"]}
    CHECK: %[[QregA:.*]] = memref.alloc() : memref<1x!mqtref.Qubit>
    CHECK: %[[I0A:.*]] = arith.constant 0 : index
    CHECK: %[[Q0A:.*]] = memref.load %[[QregA]][%[[I0A]]] : memref<1x!mqtref.Qubit>
    CHECK: %[[QregB:.*]] = memref.alloc() : memref<1x!mqtref.Qubit>
    CHECK: %[[I0B:.*]] = arith.constant 0 : index
    CHECK: %[[Q0B:.*]] = memref.load %[[QregB]][%[[I0B]]] : memref<1x!mqtref.Qubit>
    CHECK: mqtref.x() %[[Q0B]] ctrl %[[Q0A]]
    CHECK: memref.dealloc %[[QregA]] : memref<1x!mqtref.Qubit>
    CHECK: memref.dealloc %[[QregB]] : memref<1x!mqtref.Qubit>
    CHECK: return
  )";

  ASSERT_TRUE(checkOutput(check, output));
}

} // namespace
