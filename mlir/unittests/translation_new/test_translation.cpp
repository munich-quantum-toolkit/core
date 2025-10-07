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
// # Test full programs
// ##################################################

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
