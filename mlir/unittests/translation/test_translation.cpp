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
#include <llvm/Support/LogicalResult.h>
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

// Small helpers to reduce FileCheck string duplication
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
  os << " static [";
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

std::string unitaryCheck(const char* op,
                         std::initializer_list<size_t> targets) {
  return std::string("CHECK: mqtref.") + op + "() " + formatTargets(targets);
}

std::string unitaryParamCheck(const char* op,
                              std::initializer_list<double> params,
                              std::initializer_list<size_t> targets) {
  return std::string("CHECK: mqtref.") + op + "(" + formatParams(params) +
         ") " + formatTargets(targets);
}

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

  void TearDown() override {}

  void runPassPipeline(const mlir::ModuleOp module) const {
    // Run passes
    mlir::PassManager passManager(context.get());
    passManager.addPass(mlir::createCanonicalizerPass());
    passManager.addPass(mlir::createMem2Reg());
    passManager.addPass(mlir::createRemoveDeadValuesPass());
    if (failed(passManager.run(module))) {
      FAIL() << "Failed to run passes";
    }
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

// Test case structure for all operations
struct TestCase {
  // Test case name for better identification
  std::string name;
  // Basic circuit configuration
  size_t numQubits;
  size_t numClbits;
  // Circuit construction function
  std::function<void(QuantumComputation&)> build;
  // Operation-specific FileCheck string (used for non-if/else tests)
  std::string operationCheck;

  // If/IfElse-specific configuration (optional)
  bool isIfElse = false;
  // Register-based condition configuration
  size_t cregSize = 0; // classical register size used in condition (0 means not
                       // using register)
  size_t expected = 0; // expected value for comparison (for register-based)
  // Bit-based condition configuration
  bool useBit = false;     // whether to use a single classical bit as condition
  size_t controlBit = 0;   // index of the classical bit used as condition
  bool expectedBit = true; // expected boolean value for comparison

  std::string cmpMnemonic; // e.g., eq, ne, ult, ule, ugt, uge
  bool withElse = false;   // whether an else branch exists
  std::string thenOp;      // mnemonic for then operation, e.g., "x"
  std::string elseOp;      // mnemonic for else operation

  // Whether to run the canonicalization pipeline (needed for if/else)
  bool runPasses = false;
};

// Output operator for googletest parameter printing
std::ostream& operator<<(std::ostream& os, const TestCase& tc) {
  return os << tc.name;
}

// Creates a FileCheck string with qubit allocation and checks
std::string createFileCheckString(const TestCase& testCase) {
  // Always start by checking the entry function and attribute
  const std::string prologue =
      "CHECK: func.func @main() attributes {passthrough = [\"entry_point\"]}\n";

  // If this is an If/IfElse test, generate the specialized checks
  if (testCase.isIfElse) {
    std::string s = prologue;

    // Register-based If/Else
    if (testCase.cregSize > 0) {
      // Expected value constant (i64)
      s += "CHECK: %[[Exp:.*]] = arith.constant " +
           std::to_string(testCase.expected) + " : i64\n";

      if (testCase.cregSize == 1) {
        // One-bit classical register case
        s += "CHECK: %[[Reg:.*]] = \"mqtref.allocQubitRegister\"() <{size_attr "
             "= " +
             std::to_string(testCase.numQubits) +
             " : i64}> : () -> !mqtref.QubitRegister\n";
        s += "CHECK: %[[Q0:.*]] = \"mqtref.extractQubit\"(%[[Reg]]) "
             "<{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> "
             "!mqtref.Qubit\n";
        // For creg-based condition with size 1 we allow a simplified path that
        // directly extends the measurement to i64 and compares against i64.
        s += "CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]\n";
        s += "CHECK: %[[C0:.*]] = arith.extui %[[M0]] : i1 to i64\n";
        s += "CHECK: %[[Cnd0:.*]] = arith.cmpi " + testCase.cmpMnemonic +
             ", %[[C0]], %[[Exp]] : i64\n";
        if (testCase.withElse) {
          s += "CHECK: scf.if %[[Cnd0:.*]] {\n";
          s += "CHECK: mqtref." + testCase.thenOp + "() %[[Q0]]\n";
          s += "CHECK: } else {\n";
          s += "CHECK: mqtref." + testCase.elseOp + "() %[[Q0]]\n";
          s += "CHECK: }\n";
        } else {
          s += "CHECK: scf.if %[[Cnd0]] {\n";
          s += "CHECK: mqtref." + testCase.thenOp + "() %[[Q0]]\n";
          s += "CHECK: }\n";
        }
      } else {
        // Multi-bit classical register case (matching existing tests for size
        // 2)
        const auto n = testCase.cregSize;
        s += "CHECK: %[[Sum0:.*]] = arith.constant 0 : i64\n";
        s += "CHECK: %[[I" + std::to_string(n) + ":.*]] = arith.constant " +
             std::to_string(n) + " : index\n";
        s += "CHECK: %[[I1:.*]] = arith.constant 1 : index\n";
        s += "CHECK: %[[I0:.*]] = arith.constant 0 : index\n";
        s += "CHECK: %[[Reg:.*]] = \"mqtref.allocQubitRegister\"() <{size_attr "
             "= " +
             std::to_string(testCase.numQubits) +
             " : i64}> : () -> !mqtref.QubitRegister\n";
        // Extract first two qubits (tests only use 2 qubits)
        s += "CHECK: %[[Q0:.*]] = \"mqtref.extractQubit\"(%[[Reg]]) "
             "<{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> "
             "!mqtref.Qubit\n";
        s += "CHECK: %[[Q1:.*]] = \"mqtref.extractQubit\"(%[[Reg]]) "
             "<{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> "
             "!mqtref.Qubit\n";
        s += "CHECK: %[[Mem:.*]] = memref.alloca() : memref<" +
             std::to_string(n) + "xi1>\n";
        s += "CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]\n";
        s += "CHECK: memref.store %[[M0]], %[[Mem]][%[[I0]]] : memref<" +
             std::to_string(n) + "xi1>\n";
        s += "CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]\n";
        s += "CHECK: memref.store %[[M1]], %[[Mem]][%[[I1]]] : memref<" +
             std::to_string(n) + "xi1>\n";
        s += "CHECK: %[[Sum1:.*]] = scf.for %[[Ii:.*]] = %[[I0]] to %[[I" +
             std::to_string(n) +
             "]] step %[[I1]] iter_args(%[[Sumi:.*]] = %[[Sum0]]) -> (i64) {\n";
        s += "CHECK: %[[Bi:.*]] = memref.load %[[Mem]][%[[Ii]]] : memref<" +
             std::to_string(n) + "xi1>\n";
        s += "CHECK: %[[Ci:.*]] = arith.extui %[[Bi:.*]] : i1 to i64\n";
        s += "CHECK: %[[Indi:.*]] = arith.index_cast %[[Ii]] : index to i64\n";
        s += "CHECK: %[[Shli:.*]] = arith.shli %[[Ci]], %[[Indi]] : i64\n";
        s += "CHECK: %[[Sumj:.*]] = arith.addi %[[Sumi]], %[[Shli]] : i64\n";
        s += "CHECK: scf.yield %[[Sumj]] : i64\n";
        s += "CHECK: }\n";
        s += "CHECK: %[[Cnd0:.*]] = arith.cmpi " + testCase.cmpMnemonic +
             ", %[[Sum1]], %[[Exp]] : i64\n";
        if (testCase.withElse) {
          s += "CHECK: scf.if %[[Cnd0:.*]] {\n";
          s += "CHECK: mqtref." + testCase.thenOp + "() %[[Q0]]\n";
          s += "CHECK: } else {\n";
          s += "CHECK: mqtref." + testCase.elseOp + "() %[[Q0]]\n";
          s += "CHECK: }\n";
        } else {
          s += "CHECK: scf.if %[[Cnd0]] {\n";
          s += "CHECK: mqtref." + testCase.thenOp + "() %[[Q0]]\n";
          s += "CHECK: }\n";
        }
      }
    } else if (testCase.useBit) {
      // Bit-based If/Else (i1 compare)
      s += "CHECK: %[[Reg:.*]] = \"mqtref.allocQubitRegister\"() <{size_attr "
           "= " +
           std::to_string(testCase.numQubits) +
           " : i64}> : () -> !mqtref.QubitRegister\n";
      s += "CHECK: %[[Q0:.*]] = \"mqtref.extractQubit\"(%[[Reg]]) <{index_attr "
           "= 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit\n";

      // After canonicalization/mem2reg, the bit memory may be promoted away.
      // We only require the measurement and the boolean comparison.
      s += "CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]\n";
      // Condition may be directly the measurement or a comparison; just check
      // for the if.

      if (testCase.withElse) {
        s += "CHECK: scf.if %\n";
        s += "CHECK: mqtref." + testCase.thenOp + "() %[[Q0]]\n";
        s += "CHECK: } else {\n";
        s += "CHECK: mqtref." + testCase.elseOp + "() %[[Q0]]\n";
        s += "CHECK: }\n";
      } else {
        s += "CHECK: scf.if %\n";
        s += "CHECK: mqtref." + testCase.thenOp + "() %[[Q0]]\n";
        s += "CHECK: }\n";
      }
    }

    // Common epilogue for If/Else tests: dealloc and return
    s += "CHECK: \"mqtref.deallocQubitRegister\"(%[[Reg]]) : "
         "(!mqtref.QubitRegister) -> ()\n";
    s += "CHECK: return\n";
    return s;
  }

  // Default: non-if/else operation tests
  std::string result = prologue;

  result +=
      "CHECK: %[[Reg:.*]] = \"mqtref.allocQubitRegister\"() <{size_attr = " +
      std::to_string(testCase.numQubits) +
      " : i64}> : () -> !mqtref.QubitRegister\n";

  // Generate qubit extraction checks
  for (size_t i = 0; i < testCase.numQubits; ++i) {
    result += "CHECK: %[[Q" + std::to_string(i) +
              ":.*]] = \"mqtref.extractQubit\"(%[[Reg]]) "
              "<{index_attr = " +
              std::to_string(i) +
              " : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit\n";
  }

  if (testCase.numClbits > 0) {
    result += "CHECK: %[[Mem:.*]] = memref.alloca() : memref<" +
              std::to_string(testCase.numClbits) + "xi1>\n";
  }

  // Add the operation-specific check
  result += testCase.operationCheck;

  // Common epilogue: dealloc and return
  result += "\nCHECK: \"mqtref.deallocQubitRegister\"(%[[Reg]]) : "
            "(!mqtref.QubitRegister) -> ()\n";
  result += "CHECK: return\n";

  return result;
}

// Param fixture that reuses ImportTest's context/setup.
class OperationTest : public ImportTest,
                      public ::testing::WithParamInterface<TestCase> {};

TEST_P(OperationTest, EmitsExpectedOperation) {
  const auto& param = GetParam();
  QuantumComputation qc(param.numQubits, param.numClbits);
  param.build(qc);

  auto module = translateQuantumComputationToMLIR(context.get(), qc);
  if (param.runPasses) {
    runPassPipeline(module.get());
  }
  const auto outputString = getOutputString(&module);

  // Create the FileCheck string with appropriate checks
  const std::string checkString = createFileCheckString(param);
  ASSERT_TRUE(checkOutput(checkString, outputString));
}

// Test cases for various quantum operations
INSTANTIATE_TEST_SUITE_P(
    Operations, OperationTest,
    ::testing::Values(
        // 1-qubit, no-parameter gates.
        TestCase{.name = "Identity",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.i(0); },
                 .operationCheck = unitaryCheck("i", {0})},
        TestCase{.name = "Hadamard",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.h(0); },
                 .operationCheck = unitaryCheck("h", {0})},
        TestCase{.name = "PauliX",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.x(0); },
                 .operationCheck = unitaryCheck("x", {0})},
        TestCase{.name = "PauliY",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.y(0); },
                 .operationCheck = unitaryCheck("y", {0})},
        TestCase{.name = "PauliZ",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.z(0); },
                 .operationCheck = unitaryCheck("z", {0})},
        TestCase{.name = "S",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.s(0); },
                 .operationCheck = unitaryCheck("s", {0})},
        TestCase{.name = "Sdg",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.sdg(0); },
                 .operationCheck = unitaryCheck("sdg", {0})},
        TestCase{.name = "T",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.t(0); },
                 .operationCheck = unitaryCheck("t", {0})},
        TestCase{.name = "Tdg",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.tdg(0); },
                 .operationCheck = unitaryCheck("tdg", {0})},
        TestCase{.name = "V",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.v(0); },
                 .operationCheck = unitaryCheck("v", {0})},
        TestCase{.name = "Vdg",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.vdg(0); },
                 .operationCheck = unitaryCheck("vdg", {0})},
        TestCase{.name = "SX",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.sx(0); },
                 .operationCheck = unitaryCheck("sx", {0})},
        TestCase{.name = "SXdg",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.sxdg(0); },
                 .operationCheck = unitaryCheck("sxdg", {0})},
        // 1-qubit, parameterized gates.
        TestCase{
            .name = "U3",
            .numQubits = 1,
            .numClbits = 0,
            .build = [](QuantumComputation& qc) { qc.u(0.1, 0.2, 0.3, 0); },
            .operationCheck = unitaryParamCheck("u", {0.1, 0.2, 0.3}, {0})},
        TestCase{.name = "U2",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.u2(0.1, 0.2, 0); },
                 .operationCheck = unitaryParamCheck("u2", {0.1, 0.2}, {0})},
        TestCase{.name = "Phase",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.p(0.1, 0); },
                 .operationCheck = unitaryParamCheck("p", {0.1}, {0})},
        TestCase{.name = "RX",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.rx(0.1, 0); },
                 .operationCheck = unitaryParamCheck("rx", {0.1}, {0})},
        TestCase{.name = "RY",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.ry(0.1, 0); },
                 .operationCheck = unitaryParamCheck("ry", {0.1}, {0})},
        TestCase{.name = "RZ",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.rz(0.1, 0); },
                 .operationCheck = unitaryParamCheck("rz", {0.1}, {0})},
        // 2-qubit, no-parameter gates.
        TestCase{.name = "iSWAP",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.iswap(0, 1); },
                 .operationCheck = unitaryCheck("iswap", {0, 1})},
        TestCase{.name = "iSWAPdg",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.iswapdg(0, 1); },
                 .operationCheck = unitaryCheck("iswapdg", {0, 1})},
        TestCase{.name = "Peres",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.peres(0, 1); },
                 .operationCheck = unitaryCheck("peres", {0, 1})},
        TestCase{.name = "Peresdg",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.peresdg(0, 1); },
                 .operationCheck = unitaryCheck("peresdg", {0, 1})},
        TestCase{.name = "DCX",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.dcx(0, 1); },
                 .operationCheck = unitaryCheck("dcx", {0, 1})},
        TestCase{.name = "ECR",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.ecr(0, 1); },
                 .operationCheck = unitaryCheck("ecr", {0, 1})},
        // 2-qubit, parameterized gates.
        TestCase{.name = "RXX",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.rxx(0.1, 0, 1); },
                 .operationCheck = unitaryParamCheck("rxx", {0.1}, {0, 1})},
        TestCase{.name = "RYY",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.ryy(0.1, 0, 1); },
                 .operationCheck = unitaryParamCheck("ryy", {0.1}, {0, 1})},
        TestCase{.name = "RZZ",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.rzz(0.1, 0, 1); },
                 .operationCheck = unitaryParamCheck("rzz", {0.1}, {0, 1})},
        TestCase{.name = "RZX",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.rzx(0.1, 0, 1); },
                 .operationCheck = unitaryParamCheck("rzx", {0.1}, {0, 1})},
        TestCase{
            .name = "XX_MINUS_YY",
            .numQubits = 2,
            .numClbits = 0,
            .build =
                [](QuantumComputation& qc) { qc.xx_minus_yy(0.1, 0.2, 0, 1); },
            .operationCheck = unitaryParamCheck("xx_minus_yy", {0.1, 0.2},
                                                {0, 1})},
        TestCase{
            .name = "XX_PLUS_YY",
            .numQubits = 2,
            .numClbits = 0,
            .build =
                [](QuantumComputation& qc) { qc.xx_plus_yy(0.1, 0.2, 0, 1); },
            .operationCheck = unitaryParamCheck("xx_plus_yy", {0.1, 0.2},
                                                {0, 1})},
        // Controlled gates
        TestCase{.name = "CX_0_1",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.cx(0, 1); },
                 .operationCheck = "CHECK: mqtref.x() %[[Q1]] ctrl %[[Q0]]"},
        TestCase{.name = "CX_1_0",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.cx(1, 0); },
                 .operationCheck = "CHECK: mqtref.x() %[[Q0]] ctrl %[[Q1]]"},
        TestCase{.name = "CX_0N_1",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.cx(0_nc, 1); },
                 .operationCheck = "CHECK: mqtref.x() %[[Q1]] nctrl %[[Q0]]"},
        // Multi-controlled tests
        TestCase{.name = "MCX_01_2",
                 .numQubits = 3,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.mcx({0, 1}, 2); },
                 .operationCheck =
                     "CHECK: mqtref.x() %[[Q2]] ctrl %[[Q0]], %[[Q1]]"},
        TestCase{.name = "MCX_0N2_1",
                 .numQubits = 3,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.mcx({0_nc, 2}, 1); },
                 .operationCheck =
                     "CHECK: mqtref.x() %[[Q1]] ctrl %[[Q2]] nctrl %[[Q0]]"},
        TestCase{
            .name = "MCX_2N1N_0",
            .numQubits = 3,
            .numClbits = 0,
            .build = [](QuantumComputation& qc) { qc.mcx({2_nc, 1_nc}, 0); },
            .operationCheck =
                "CHECK: mqtref.x() %[[Q0]] nctrl %[[Q1]], %[[Q2]]"},
        // Measurements
        TestCase{
            .name = "Measure_Multiple",
            .numQubits = 2,
            .numClbits = 2,
            .build = [](QuantumComputation& qc) { qc.measure({0, 1}, {1, 0}); },
            .operationCheck = R"(
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: %[[I1:.*]] = arith.constant 1 : index
    CHECK: memref.store %[[M0]], %[[Mem]][%[[I1]]] : memref<2xi1>
    CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: memref.store %[[M1]], %[[Mem]][%[[I0]]] : memref<2xi1>
  )"},
        TestCase{.name = "Measure_Single",
                 .numQubits = 2,
                 .numClbits = 2,
                 .build = [](QuantumComputation& qc) { qc.measure(0, 0); },
                 .operationCheck = R"(
    CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
    CHECK: %[[I0:.*]] = arith.constant 0 : index
    CHECK: memref.store %[[M0]], %[[Mem]][%[[I0]]] : memref<2xi1>
    CHECK-NOT: mqtref.measure %[[Q1]]
    CHECK-NOT: arith.constant 1 : index
    CHECK-NOT: memref.store  %[[ANY:.*]], %[[Mem]][%[[ANY:.*]]] : memref<2xi1>
  )"},
        // Reset
        TestCase{.name = "Reset_Multiple",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.reset({0, 1}); },
                 .operationCheck = R"(
    CHECK: "mqtref.reset"(%[[Q0]]) : (!mqtref.Qubit) -> ()
    CHECK: "mqtref.reset"(%[[Q1]]) : (!mqtref.Qubit) -> ()
  )"},
        TestCase{.name = "Reset_Single",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build = [](QuantumComputation& qc) { qc.reset(0); },
                 .operationCheck = R"(
    CHECK: "mqtref.reset"(%[[Q0]]) : (!mqtref.Qubit) -> ()
    CHECK-NOT: "mqtref.reset"(%[[Q1]]) : (!mqtref.Qubit) -> ()
  )"}));

INSTANTIATE_TEST_SUITE_P(
    IfElse, OperationTest,
    ::testing::Values(
        // Register-based If/Else tests
        TestCase{.name = "IfRegisterEq1",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build =
                     [](QuantumComputation& qc) {
                       const auto& creg = qc.addClassicalRegister(1);
                       qc.measure(0, 0);
                       qc.if_(X, 0, creg, 1U, Eq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 1,
                 .expected = 1,
                 .useBit = false,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "eq",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfRegisterEq2",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build =
                     [](QuantumComputation& qc) {
                       const auto& creg = qc.addClassicalRegister(2);
                       qc.measure({0, 1}, {0, 1});
                       qc.if_(X, 0, creg, 2U, Eq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 2,
                 .expected = 2,
                 .useBit = false,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "eq",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfRegisterNeq",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build =
                     [](QuantumComputation& qc) {
                       const auto& creg = qc.addClassicalRegister(1);
                       qc.measure(0, 0);
                       qc.if_(X, 0, creg, 1U, Neq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 1,
                 .expected = 1,
                 .useBit = false,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "ne",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfRegisterLt",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build =
                     [](QuantumComputation& qc) {
                       const auto& creg = qc.addClassicalRegister(1);
                       qc.measure(0, 0);
                       qc.if_(X, 0, creg, 1U, Lt);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 1,
                 .expected = 1,
                 .useBit = false,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "ult",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfRegisterLeq",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build =
                     [](QuantumComputation& qc) {
                       const auto& creg = qc.addClassicalRegister(1);
                       qc.measure(0, 0);
                       qc.if_(X, 0, creg, 1U, Leq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 1,
                 .expected = 1,
                 .useBit = false,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "ule",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfRegisterGt",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build =
                     [](QuantumComputation& qc) {
                       const auto& creg = qc.addClassicalRegister(1);
                       qc.measure(0, 0);
                       qc.if_(X, 0, creg, 1U, Gt);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 1,
                 .expected = 1,
                 .useBit = false,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "ugt",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfRegisterGeq",
                 .numQubits = 1,
                 .numClbits = 0,
                 .build =
                     [](QuantumComputation& qc) {
                       const auto& creg = qc.addClassicalRegister(1);
                       qc.measure(0, 0);
                       qc.if_(X, 0, creg, 1U, Geq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 1,
                 .expected = 1,
                 .useBit = false,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "uge",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfElseRegister",
                 .numQubits = 2,
                 .numClbits = 0,
                 .build =
                     [](QuantumComputation& qc) {
                       const auto& creg = qc.addClassicalRegister(2);
                       qc.measure({0, 1}, {0, 1});
                       qc.ifElse(std::make_unique<StandardOperation>(0, X),
                                 std::make_unique<StandardOperation>(0, Y),
                                 creg, 2U, Eq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 2,
                 .expected = 2,
                 .useBit = false,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "eq",
                 .withElse = true,
                 .thenOp = "x",
                 .elseOp = "y",
                 .runPasses = true},
        // Bit-based If/Else tests (single bit, eq/ne with true/false)
        TestCase{.name = "IfBitEqTrue",
                 .numQubits = 1,
                 .numClbits = 1,
                 .build =
                     [](QuantumComputation& qc) {
                       qc.measure(0, 0);
                       qc.if_(X, 0, 0, true, Eq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 0,
                 .expected = 0,
                 .useBit = true,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "eq",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfBitEqFalse",
                 .numQubits = 1,
                 .numClbits = 1,
                 .build =
                     [](QuantumComputation& qc) {
                       qc.measure(0, 0);
                       qc.if_(X, 0, 0, false, Eq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 0,
                 .expected = 0,
                 .useBit = true,
                 .controlBit = 0,
                 .expectedBit = false,
                 .cmpMnemonic = "eq",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfBitNeqTrue",
                 .numQubits = 1,
                 .numClbits = 1,
                 .build =
                     [](QuantumComputation& qc) {
                       qc.measure(0, 0);
                       qc.if_(X, 0, 0, true, Neq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 0,
                 .expected = 0,
                 .useBit = true,
                 .controlBit = 0,
                 .expectedBit = false,
                 .cmpMnemonic = "eq",
                 .withElse = false,
                 .thenOp = "x",
                 .elseOp = "",
                 .runPasses = true},
        TestCase{.name = "IfElseBitEqTrue",
                 .numQubits = 1,
                 .numClbits = 1,
                 .build =
                     [](QuantumComputation& qc) {
                       qc.measure(0, 0);
                       qc.ifElse(std::make_unique<StandardOperation>(0, X),
                                 std::make_unique<StandardOperation>(0, Y), 0,
                                 true, Eq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 0,
                 .expected = 0,
                 .useBit = true,
                 .controlBit = 0,
                 .expectedBit = true,
                 .cmpMnemonic = "eq",
                 .withElse = true,
                 .thenOp = "x",
                 .elseOp = "y",
                 .runPasses = true},
        TestCase{.name = "IfElseBitNeqFalse",
                 .numQubits = 1,
                 .numClbits = 1,
                 .build =
                     [](QuantumComputation& qc) {
                       qc.measure(0, 0);
                       qc.ifElse(std::make_unique<StandardOperation>(0, X),
                                 std::make_unique<StandardOperation>(0, Y), 0,
                                 false, Neq);
                     },
                 .operationCheck = "",
                 .isIfElse = true,
                 .cregSize = 0,
                 .expected = 0,
                 .useBit = true,
                 .controlBit = 0,
                 .expectedBit = false,
                 .cmpMnemonic = "ne",
                 .withElse = true,
                 .thenOp = "x",
                 .elseOp = "y",
                 .runPasses = true}),
    [](const ::testing::TestParamInfo<TestCase>& info) {
      return info.param.name;
    });

} // namespace
