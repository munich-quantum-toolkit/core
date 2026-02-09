/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"

#include <cstddef>
#include <functional>
#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>
#include <numbers>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace mlir;

//===----------------------------------------------------------------------===//
// Stage Verification Utility
//===----------------------------------------------------------------------===//

/**
 * @brief Verify a stage matches expected module
 *
 * @param stageName Human-readable stage name for error messages
 * @param actualIR String IR from CompilationRecord
 * @param expectedModule Expected module to compare against
 * @return True if modules match, false otherwise with diagnostic output
 */
[[nodiscard]] static testing::AssertionResult
verify(const std::string& stageName, const std::string& actualIR,
       ModuleOp expectedModule) {
  // Parse the actual IR string into a ModuleOp
  const auto actualModule =
      parseSourceString<ModuleOp>(actualIR, expectedModule.getContext());
  if (!actualModule) {
    return testing::AssertionFailure()
           << stageName << " failed to parse actual IR";
  }

  if (!areModulesEquivalentWithPermutations(*actualModule, expectedModule)) {
    std::ostringstream msg;
    msg << stageName << " IR does not match expected structure\n\n";

    std::string expectedStr;
    llvm::raw_string_ostream expectedStream(expectedStr);
    expectedModule.print(expectedStream);
    expectedStream.flush();

    msg << "=== EXPECTED IR ===\n" << expectedStr << "\n\n";
    msg << "=== ACTUAL IR ===\n" << actualIR << "\n";

    return testing::AssertionFailure() << msg.str();
  }

  return testing::AssertionSuccess();
}

//===----------------------------------------------------------------------===//
// Stage Expectation Builder
//===----------------------------------------------------------------------===//

/**
 * @brief Helper to build expected IR for multiple stages at once
 *
 * @details
 * Reduces boilerplate by allowing specification of which stages should
 * match which expected IR.
 */
struct StageExpectations {
  ModuleOp qcImport;
  ModuleOp qcoConversion;
  ModuleOp optimization;
  ModuleOp qcConversion;
  ModuleOp qirConversion;
};

//===----------------------------------------------------------------------===//
// Base Test Fixture
//===----------------------------------------------------------------------===//

/**
 * @brief Base test fixture for end-to-end compiler pipeline tests
 *
 * @details
 * Provides a configured MLIR context with all necessary dialects loaded
 * and utility methods for creating quantum circuits and running the
 * compilation pipeline.
 */
class CompilerPipelineTest : public testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;
  QuantumCompilerConfig config;
  CompilationRecord record;

  OwningOpRef<ModuleOp> emptyQC;
  OwningOpRef<ModuleOp> emptyQCO;
  OwningOpRef<ModuleOp> emptyQIR;

  void SetUp() override {
    // Register all dialects needed for the full compilation pipeline
    DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                    cf::ControlFlowDialect, func::FuncDialect, scf::SCFDialect,
                    LLVM::LLVMDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();

    // Enable QIR conversion and recording by default
    config.convertToQIR = true;
    config.recordIntermediates = true;
    config.printIRAfterAllStages =
        true; /// TODO: Change back after everything is running

    emptyQC = buildQCIR([](mlir::qc::QCProgramBuilder&) {});
    emptyQCO = buildQCOIR([](qco::QCOProgramBuilder&) {});
    emptyQIR = buildQIR([](qir::QIRProgramBuilder&) {});
  }

  //===--------------------------------------------------------------------===//
  // Quantum Circuit Construction and Import
  //===--------------------------------------------------------------------===//

  /**
   * @brief Pretty print quantum computation with ASCII art borders
   *
   * @param qc The quantum computation to print
   */
  static void
  prettyPrintQuantumComputation(const ::qc::QuantumComputation& comp) {
    llvm::errs() << "\n";
    printBoxTop();

    // Print header
    printBoxLine("Initial Quantum Computation");

    printBoxMiddle();

    // Print internal representation
    printBoxLine("Internal Representation:");

    // Capture the internal representation
    std::ostringstream internalRepr;
    internalRepr << comp;
    const std::string internalStr = internalRepr.str();

    // Print with line wrapping
    printBoxText(internalStr);

    printBoxMiddle();

    // Print OpenQASM3 representation
    printBoxLine("OpenQASM3 Representation:");
    printBoxLine("");

    const auto qasmStr = comp.toQASM();

    // Print with line wrapping
    printBoxText(qasmStr);

    printBoxBottom();
    llvm::errs().flush();
  }

  /**
   * @brief Import a QuantumComputation into QC dialect
   */
  [[nodiscard]] OwningOpRef<ModuleOp>
  importQuantumCircuit(const ::qc::QuantumComputation& qc) const {
    if (config.printIRAfterAllStages) {
      prettyPrintQuantumComputation(qc);
    }
    return translateQuantumComputationToQC(context.get(), qc);
  }

  /**
   * @brief Run the compiler pipeline with the current configuration
   */
  [[nodiscard]] LogicalResult runPipeline(const ModuleOp module) {
    const QuantumCompilerPipeline pipeline(config);
    return pipeline.runPipeline(module, &record);
  }

  //===--------------------------------------------------------------------===//
  // Expected IR Builder Methods
  //===--------------------------------------------------------------------===//

  /**
   * @brief Build expected QC IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQCIR(
      const std::function<void(mlir::qc::QCProgramBuilder&)>& buildFunc) const {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  /**
   * @brief Build expected QCO IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQCOIR(
      const std::function<void(qco::QCOProgramBuilder&)>& buildFunc) const {
    qco::QCOProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  /**
   * @brief Build expected QIR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQIR(
      const std::function<void(qir::QIRProgramBuilder&)>& buildFunc) const {
    qir::QIRProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  //===--------------------------------------------------------------------===//
  // Stage Verification Methods
  //===--------------------------------------------------------------------===//

  /**
   * @brief Verify all stages match their expected IR
   *
   * @details
   * Simplifies test writing by checking all stages with one call.
   * Stages without expectations are skipped.
   */
  void verifyAllStages(const StageExpectations& expectations) const {
    if (expectations.qcImport != nullptr) {
      EXPECT_TRUE(
          verify("QC Import", record.afterInitialCanon, expectations.qcImport));
    }

    if (expectations.qcoConversion != nullptr) {
      EXPECT_TRUE(verify("QCO Conversion", record.afterQCOCanon,
                         expectations.qcoConversion));
    }

    if (expectations.optimization != nullptr) {
      EXPECT_TRUE(verify("Optimization", record.afterOptimizationCanon,
                         expectations.optimization));
    }

    if (expectations.qcConversion != nullptr) {
      EXPECT_TRUE(verify("QC Conversion", record.afterQCCanon,
                         expectations.qcConversion));
    }

    if (expectations.qirConversion != nullptr) {
      EXPECT_TRUE(verify("QIR Conversion", record.afterQIRCanon,
                         expectations.qirConversion));
    }
  }

  void TearDown() override {
    // Verify all stages were recorded (basic sanity check)
    EXPECT_FALSE(record.afterQCImport.empty())
        << "QC import stage was not recorded";
    EXPECT_FALSE(record.afterInitialCanon.empty())
        << "Initial canonicalization stage was not recorded";
    EXPECT_FALSE(record.afterQCOConversion.empty())
        << "QCO conversion stage was not recorded";
    EXPECT_FALSE(record.afterQCOCanon.empty())
        << "QCO canonicalization stage was not recorded";
    EXPECT_FALSE(record.afterOptimization.empty())
        << "Optimization stage was not recorded";
    EXPECT_FALSE(record.afterOptimizationCanon.empty())
        << "Optimization canonicalization stage was not recorded";
    EXPECT_FALSE(record.afterQCConversion.empty())
        << "QC conversion stage was not recorded";
    EXPECT_FALSE(record.afterQCCanon.empty())
        << "QC canonicalization stage was not recorded";

    if (config.convertToQIR) {
      EXPECT_FALSE(record.afterQIRConversion.empty())
          << "QIR conversion stage was not recorded";
      EXPECT_FALSE(record.afterQIRCanon.empty())
          << "QIR canonicalization stage was not recorded";
    }
  }
};

//===----------------------------------------------------------------------===//
// Test Cases
//===----------------------------------------------------------------------===//

// ##################################################
// # Empty Circuit Tests
// ##################################################

/**
 * @brief Test: Empty circuit compilation
 *
 * @details
 * Verifies that an empty circuit compiles correctly through all stages,
 * producing empty but valid IR at each stage.
 */
TEST_F(CompilerPipelineTest, EmptyCircuit) {
  // Create empty circuit
  const ::qc::QuantumComputation comp;

  // Import to QC dialect
  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);

  // Run compilation
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  // Verify all stages
  verifyAllStages({
      .qcImport = emptyQC.get(),
      .qcoConversion = emptyQCO.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

// ##################################################
// # Quantum Register Allocation Tests
// ##################################################

/**
 * @brief Test: Single qubit register allocation
 *
 * @details
 * Since the register is unused, it should be removed during canonicalization
 * in the QCO dialect.
 */
TEST_F(CompilerPipelineTest, SingleQubitRegister) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcExpected = buildQCIR(
      [](mlir::qc::QCProgramBuilder& b) { b.allocQubitRegister(1, "q"); });
  const auto qcoExpected = buildQCOIR(
      [](qco::QCOProgramBuilder& b) { b.allocQubitRegister(1, "q"); });

  verifyAllStages({
      .qcImport = qcExpected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multi-qubit register allocation
 */
TEST_F(CompilerPipelineTest, MultiQubitRegister) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcExpected = buildQCIR(
      [](mlir::qc::QCProgramBuilder& b) { b.allocQubitRegister(3, "q"); });
  const auto qcoExpected = buildQCOIR(
      [](qco::QCOProgramBuilder& b) { b.allocQubitRegister(3, "q"); });

  verifyAllStages({
      .qcImport = qcExpected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multiple quantum registers
 */
TEST_F(CompilerPipelineTest, MultipleQuantumRegisters) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.addQubitRegister(3, "aux");

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcExpected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    b.allocQubitRegister(2, "q");
    b.allocQubitRegister(3, "aux");
  });
  const auto qcoExpected = buildQCOIR([](qco::QCOProgramBuilder& b) {
    b.allocQubitRegister(2, "q");
    b.allocQubitRegister(3, "aux");
  });

  verifyAllStages({
      .qcImport = qcExpected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Large qubit register allocation
 */
TEST_F(CompilerPipelineTest, LargeQubitRegister) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(100, "q");

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());
}

// ##################################################
// # Classical Register Allocation Tests
// ##################################################

/**
 * @brief Test: Single classical bit register
 *
 * @details
 * Since the register is unused, it should be removed during canonicalization.
 */
TEST_F(CompilerPipelineTest, SingleClassicalBitRegister) {
  ::qc::QuantumComputation comp;
  comp.addClassicalRegister(1, "c");

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    std::ignore = b.allocClassicalBitRegister(1, "c");
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = emptyQCO.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multi-bit classical register
 *
 * @details
 * Since the register is unused, it should be removed during canonicalization.
 */
TEST_F(CompilerPipelineTest, MultiBitClassicalRegister) {
  ::qc::QuantumComputation comp;
  comp.addClassicalRegister(5, "c");

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    std::ignore = b.allocClassicalBitRegister(5, "c");
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = emptyQCO.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multiple classical registers
 *
 * @details
 * Since the registers are unused, they should be removed during
 * canonicalization.
 */
TEST_F(CompilerPipelineTest, MultipleClassicalRegisters) {
  ::qc::QuantumComputation comp;
  comp.addClassicalRegister(3, "c");
  comp.addClassicalRegister(2, "d");

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    std::ignore = b.allocClassicalBitRegister(3, "c");
    std::ignore = b.allocClassicalBitRegister(2, "d");
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = emptyQCO.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Large classical bit register
 */
TEST_F(CompilerPipelineTest, LargeClassicalBitRegister) {
  ::qc::QuantumComputation comp;
  comp.addClassicalRegister(128, "c");

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());
}

// ##################################################
// # Reset Operation Tests
// ##################################################

/**
 * @brief Test: Single reset in single qubit circuit
 *
 * @details
 * Since the reset directly follows an allocation, it should be removed during
 * canonicalization.
 */
TEST_F(CompilerPipelineTest, SingleResetInSingleQubitCircuit) {
  ::qc::QuantumComputation comp(1);
  comp.reset(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    const auto q = b.allocQubitRegister(1, "q");
    b.reset(q[0]);
  });
  const auto qcoExpected = buildQCOIR([](qco::QCOProgramBuilder& b) {
    const auto q = b.allocQubitRegister(1, "q");
    b.reset(q[0]);
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Consecutive reset operations
 *
 * @details
 * Since reset is idempotent, consecutive resets should be reduced to a single
 * reset during canonicalization. Since that single reset directly follows an
 * allocation, it should be removed as well.
 */
TEST_F(CompilerPipelineTest, ConsecutiveResetOperations) {
  ::qc::QuantumComputation comp(1);
  comp.reset(0);
  comp.reset(0);
  comp.reset(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    const auto q = b.allocQubitRegister(1, "q");
    b.reset(q[0]);
    b.reset(q[0]);
    b.reset(q[0]);
  });

  const auto qcoExpected = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto q = b.allocQubitRegister(1, "q");
    q[0] = b.reset(q[0]);
    q[0] = b.reset(q[0]);
    q[0] = b.reset(q[0]);
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Separate resets in two qubit system
 */
TEST_F(CompilerPipelineTest, SeparateResetsInTwoQubitSystem) {
  ::qc::QuantumComputation comp(2);
  comp.reset(0);
  comp.reset(1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    const auto q = b.allocQubitRegister(2, "q");
    b.reset(q[0]);
    b.reset(q[1]);
  });

  const auto qcoExpected = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto q = b.allocQubitRegister(2, "q");
    q[0] = b.reset(q[0]);
    q[1] = b.reset(q[1]);
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

// ##################################################
// # Measure Operation Tests
// ##################################################

/**
 * @brief Test: Single measurement to single bit
 */
TEST_F(CompilerPipelineTest, SingleMeasurementToSingleBit) {
  ::qc::QuantumComputation comp(1, 1);
  comp.measure(0, 0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
  });

  const auto qcoExpected = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = qcoExpected.get(),
      .qcConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

/**
 * @brief Test: Repeated measurement to same bit
 */
TEST_F(CompilerPipelineTest, RepeatedMeasurementToSameBit) {
  ::qc::QuantumComputation comp(1, 1);
  comp.measure(0, 0);
  comp.measure(0, 0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[0]);
  });

  const auto qcoExpected = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    q[0] = b.measure(q[0], c[0]);
    q[0] = b.measure(q[0], c[0]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[0]);
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = qcoExpected.get(),
      .qcConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

/**
 * @brief Test: Repeated measurement on separate bits
 */
TEST_F(CompilerPipelineTest, RepeatedMeasurementOnSeparateBits) {
  ::qc::QuantumComputation comp(1);
  comp.addClassicalRegister(3);
  comp.measure(0, 0);
  comp.measure(0, 1);
  comp.measure(0, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(3);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[1]);
    b.measure(q[0], c[2]);
  });

  const auto qcoExpected = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(3);
    q[0] = b.measure(q[0], c[0]);
    q[0] = b.measure(q[0], c[1]);
    q[0] = b.measure(q[0], c[2]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(3);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[1]);
    b.measure(q[0], c[2]);
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = qcoExpected.get(),
      .qcConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

/**
 * @brief Test: Multiple classical registers with measurements
 */
TEST_F(CompilerPipelineTest, MultipleClassicalRegistersAndMeasurements) {
  ::qc::QuantumComputation comp(2);
  const auto& c1 = comp.addClassicalRegister(1, "c1");
  const auto& c2 = comp.addClassicalRegister(1, "c2");
  comp.measure(0, c1[0]);
  comp.measure(1, c2[0]);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto& creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto& creg2 = b.allocClassicalBitRegister(1, "c2");
    b.measure(q[0], creg1[0]);
    b.measure(q[1], creg2[0]);
  });

  const auto qcoExpected = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto& creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto& creg2 = b.allocClassicalBitRegister(1, "c2");
    q[0] = b.measure(q[0], creg1[0]);
    q[1] = b.measure(q[1], creg2[0]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto& creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto& creg2 = b.allocClassicalBitRegister(1, "c2");
    b.measure(q[0], creg1[0]);
    b.measure(q[1], creg2[0]);
  });

  verifyAllStages({
      .qcImport = expected.get(),
      .qcoConversion = qcoExpected.get(),
      .optimization = qcoExpected.get(),
      .qcConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

// ##################################################
// # Temporary Unitary Operation Tests
// ##################################################

TEST_F(CompilerPipelineTest, GPhase) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.gphase(1.0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.gphase(1.0);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.gphase(1.0);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.gphase(1.0);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, CGPhase) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.cgphase(1.0, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.p(1.0, reg[0]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.p(1.0, reg[0]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.p(1.0, reg[0]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCGPhase) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.mcgphase(1.0, {reg[0], reg[1]});
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cp(1.0, reg[0], reg[1]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cp(1.0, reg[0], reg[1]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.cp(1.0, reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, Id) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.i(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.id(reg[0]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.id(reg[0]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

TEST_F(CompilerPipelineTest, CId) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.ci(0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cid(reg[0], reg[1]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cid(reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = emptyQCO.get(),
      .qcConversion = emptyQC.get(),
      .qirConversion = emptyQIR.get(),
  });
}

TEST_F(CompilerPipelineTest, X) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.x(0);
  comp.x(0);
  comp.x(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.x(q);
    b.x(q);
    b.x(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.x(q);
    q = b.x(q);
    b.x(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.x(reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.x(reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.x(reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, CX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.cx(0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cx(reg[0], reg[1]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cx(reg[0], reg[1]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.cx(reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, CX3) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.cx(0, 1);
  comp.cx(1, 0);
  comp.cx(0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.cx(q0, q1);
    b.cx(q1, q0);
    b.cx(q0, q1);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    std::tie(q0, q1) = b.cx(q0, q1);
    std::tie(q1, q0) = b.cx(q1, q0);
    std::tie(q0, q1) = b.cx(q0, q1);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.cx(q0, q1);
    b.cx(q1, q0);
    b.cx(q0, q1);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.mcx({0, 1}, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcx({reg[0], reg[1]}, reg[2]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcx({reg[0], reg[1]}, reg[2]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.mcx({reg[0], reg[1]}, reg[2]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCXNested) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.ctrl(reg[0], [&] { b.cx(reg[1], reg[2]); });
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcx({reg[0], reg[1]}, reg[2]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcx({reg[0], reg[1]}, reg[2]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.mcx({reg[0], reg[1]}, reg[2]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCXTrivial) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.mcx({}, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.x(reg[0]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.x(reg[0]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.x(reg[0]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, InvX) {
  const auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.inv([&] { b.x(reg[0]); });
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.inv([&] { b.x(reg[0]); });
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.x(reg[0]);
  });
  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.x(reg[0]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.x(reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, InvRx) {
  const auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.inv([&] { b.rx(0.5, reg[0]); });
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.inv([&] { b.rx(0.5, reg[0]); });
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.rx(-0.5, reg[0]);
  });
  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.rx(-0.5, reg[0]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.rx(-0.5, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, NestedInvs) {
  const auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.inv([&] { b.inv([&] { b.iswap(reg[0], reg[1]); }); });
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    // The following will be canonicalized to eliminate the nested invs.
    b.inv({reg[0], reg[1]}, [&](ValueRange outerQubits) {
      return SmallVector<Value>{b.inv(outerQubits, [&](ValueRange innerQubits) {
        auto [q0, q1] = b.iswap(innerQubits[0], innerQubits[1]);
        return llvm::SmallVector<Value>{q0, q1};
      })};
    });
  });
  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.iswap(reg[0], reg[1]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.iswap(reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, NestedInvCtrlS) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.inv([&] { b.ctrl(reg[0], [&] { b.s(reg[1]); }); });
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.ctrl(reg[0], [&] { b.inv([&] { b.s(reg[1]); }); });
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.csdg(reg[0], reg[1]);
  });
  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.csdg(reg[0], reg[1]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.csdg(reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, Y) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.y(0);
  comp.y(0);
  comp.y(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.y(q);
    b.y(q);
    b.y(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.y(q);
    q = b.y(q);
    b.y(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.y(reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.y(reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.y(reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, Z) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.z(0);
  comp.z(0);
  comp.z(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.z(q);
    b.z(q);
    b.z(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.z(q);
    q = b.z(q);
    b.z(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.z(reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.z(reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.z(reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, H) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.h(0);
  comp.h(0);
  comp.h(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.h(q);
    b.h(q);
    b.h(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.h(q);
    q = b.h(q);
    b.h(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.h(reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.h(reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.h(reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, S) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.s(0);
  comp.sdg(0);
  comp.s(0);
  comp.s(0);
  comp.s(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.s(q);
    b.sdg(q);
    b.s(q);
    b.s(q);
    b.s(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.s(q);
    q = b.sdg(q);
    q = b.s(q);
    q = b.s(q);
    q = b.s(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.z(q);
    b.s(q);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.z(q);
    b.s(q);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    const auto q = reg[0];
    b.z(q);
    b.s(q);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, Sdg) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.sdg(0);
  comp.s(0);
  comp.sdg(0);
  comp.sdg(0);
  comp.sdg(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.sdg(q);
    b.s(q);
    b.sdg(q);
    b.sdg(q);
    b.sdg(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.sdg(q);
    q = b.s(q);
    q = b.sdg(q);
    q = b.sdg(q);
    q = b.sdg(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.z(q);
    b.sdg(q);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.z(q);
    b.sdg(q);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    const auto q = reg[0];
    b.z(q);
    b.sdg(q);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, T) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.t(0);
  comp.tdg(0);
  comp.t(0);
  comp.t(0);
  comp.t(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.t(q);
    b.tdg(q);
    b.t(q);
    b.t(q);
    b.t(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.t(q);
    q = b.tdg(q);
    q = b.t(q);
    q = b.t(q);
    b.t(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.s(q);
    q = b.t(q);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.s(q);
    b.t(q);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    const auto q = reg[0];
    b.s(q);
    b.t(q);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, Tdg) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.tdg(0);
  comp.t(0);
  comp.tdg(0);
  comp.tdg(0);
  comp.tdg(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.tdg(q);
    b.t(q);
    b.tdg(q);
    b.tdg(q);
    b.tdg(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.tdg(q);
    q = b.t(q);
    q = b.tdg(q);
    q = b.tdg(q);
    b.tdg(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.sdg(q);
    b.tdg(q);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.sdg(q);
    b.tdg(q);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    const auto q = reg[0];
    b.sdg(q);
    b.tdg(q);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, SX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.sx(0);
  comp.sxdg(0);
  comp.sx(0);
  comp.sx(0);
  comp.sx(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.sx(q);
    b.sxdg(q);
    b.sx(q);
    b.sx(q);
    b.sx(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.sx(q);
    q = b.sxdg(q);
    q = b.sx(q);
    q = b.sx(q);
    q = b.sx(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.x(q);
    b.sx(q);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.x(q);
    b.sx(q);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    const auto q = reg[0];
    b.x(q);
    b.sx(q);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, SXdg) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.sxdg(0);
  comp.sx(0);
  comp.sxdg(0);
  comp.sxdg(0);
  comp.sxdg(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.sxdg(q);
    b.sx(q);
    b.sxdg(q);
    b.sxdg(q);
    b.sxdg(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.sxdg(q);
    q = b.sx(q);
    q = b.sxdg(q);
    q = b.sxdg(q);
    q = b.sxdg(q);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    q = b.x(q);
    b.sxdg(q);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.x(q);
    b.sxdg(q);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    const auto q = reg[0];
    b.x(q);
    b.sxdg(q);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, RX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.rx(1.0, 0);
  comp.rx(0.5, 0);
  comp.rx(1.0, 1);
  comp.rx(-1.0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.rx(1.0, q0);
    b.rx(0.5, q0);
    b.rx(1.0, q1);
    b.rx(-1.0, q1);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    q0 = b.rx(1.0, q0);
    b.rx(0.5, q0);
    q1 = b.rx(1.0, q1);
    b.rx(-1.0, q1);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.rx(1.5, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.rx(1.5, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.rx(1.5, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, CRX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.crx(1.0, 0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.crx(1.0, reg[0], reg[1]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.crx(1.0, reg[0], reg[1]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.crx(1.0, reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCRX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.mcrx(1.0, {0, 1}, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcrx(1.0, {reg[0], reg[1]}, reg[2]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcrx(1.0, {reg[0], reg[1]}, reg[2]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.mcrx(1.0, {reg[0], reg[1]}, reg[2]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, RY) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.ry(1.0, 0);
  comp.ry(0.5, 0);
  comp.ry(1.0, 1);
  comp.ry(-1.0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.ry(1.0, q0);
    b.ry(0.5, q0);
    b.ry(1.0, q1);
    b.ry(-1.0, q1);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    q0 = b.ry(1.0, q0);
    b.ry(0.5, q0);
    q1 = b.ry(1.0, q1);
    b.ry(-1.0, q1);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.ry(1.5, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.ry(1.5, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.ry(1.5, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, RZ) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.rz(1.0, 0);
  comp.rz(0.5, 0);
  comp.rz(1.0, 1);
  comp.rz(-1.0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.rz(1.0, q0);
    b.rz(0.5, q0);
    b.rz(1.0, q1);
    b.rz(-1.0, q1);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    q0 = b.rz(1.0, q0);
    b.rz(0.5, q0);
    q1 = b.rz(1.0, q1);
    b.rz(-1.0, q1);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.rz(1.5, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.rz(1.5, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.rz(1.5, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, P) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.p(1.0, 0);
  comp.p(0.5, 0);
  comp.p(1.0, 1);
  comp.p(-1.0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.p(1.0, q0);
    b.p(0.5, q0);
    b.p(1.0, q1);
    b.p(-1.0, q1);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    q0 = b.p(1.0, q0);
    b.p(0.5, q0);
    q1 = b.p(1.0, q1);
    b.p(-1.0, q1);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.p(1.5, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.p(1.5, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.p(1.5, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, R) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.r(1.0, 0.5, 0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.r(1.0, 0.5, reg[0]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.r(1.0, 0.5, reg[0]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.r(1.0, 0.5, reg[0]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, RToRX) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.r(1.0, 0.0, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.r(1.0, 0.0, reg[0]);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.r(1.0, 0.0, reg[0]);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.rx(1.0, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.rx(1.0, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.rx(1.0, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, RToRY) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.r(1.0, std::numbers::pi / 2, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.r(1.0, std::numbers::pi / 2, reg[0]);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.r(1.0, std::numbers::pi / 2, reg[0]);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.ry(1.0, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.ry(1.0, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.ry(1.0, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, CR) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.cr(1.0, 0.5, 0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cr(1.0, 0.5, reg[0], reg[1]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cr(1.0, 0.5, reg[0], reg[1]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.cr(1.0, 0.5, reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCR) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.mcr(1.0, 0.5, {0, 1}, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcr(1.0, 0.5, {reg[0], reg[1]}, reg[2]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcr(1.0, 0.5, {reg[0], reg[1]}, reg[2]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.mcr(1.0, 0.5, {reg[0], reg[1]}, reg[2]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, U2) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.u2(1.0, 0.5, 0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(1.0, 0.5, reg[0]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(1.0, 0.5, reg[0]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.u2(1.0, 0.5, reg[0]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, U2ToH) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(0.0, std::numbers::pi, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(0.0, std::numbers::pi, reg[0]);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(0.0, std::numbers::pi, reg[0]);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.h(reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.h(reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.h(reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, U2ToRX) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(-std::numbers::pi / 2.0, std::numbers::pi / 2.0, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(-std::numbers::pi / 2.0, std::numbers::pi / 2.0, reg[0]);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(-std::numbers::pi / 2.0, std::numbers::pi / 2.0, reg[0]);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.rx(std::numbers::pi / 2.0, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.rx(std::numbers::pi / 2.0, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.rx(std::numbers::pi / 2.0, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, U2ToRY) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(0.0, 0.0, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(0.0, 0.0, reg[0]);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u2(0.0, 0.0, reg[0]);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.ry(std::numbers::pi / 2.0, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.ry(std::numbers::pi / 2.0, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.ry(std::numbers::pi / 2.0, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, U) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.u(1.0, 0.5, 0.2, 0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(1.0, 0.5, 0.2, reg[0]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(1.0, 0.5, 0.2, reg[0]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.u(1.0, 0.5, 0.2, reg[0]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, UToP) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(0.0, 0.0, 1.0, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(0.0, 0.0, 1.0, reg[0]);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(0.0, 0.0, 1.0, reg[0]);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.p(1.0, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.p(1.0, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.p(1.0, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, UToRX) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(1.0, -std::numbers::pi / 2.0, std::numbers::pi / 2.0, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(1.0, -std::numbers::pi / 2.0, std::numbers::pi / 2.0, reg[0]);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(1.0, -std::numbers::pi / 2.0, std::numbers::pi / 2.0, reg[0]);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.rx(1.0, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.rx(1.0, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.rx(1.0, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, UToRY) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(1.0, 0.0, 0.0, reg[0]);
  });

  ASSERT_TRUE(runPipeline(input.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(1.0, 0.0, 0.0, reg[0]);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.u(1.0, 0.0, 0.0, reg[0]);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.ry(1.0, reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.ry(1.0, reg[0]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    b.ry(1.0, reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, CU) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.cu(1.0, 0.5, 0.2, 0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cu(1.0, 0.5, 0.2, reg[0], reg[1]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.cu(1.0, 0.5, 0.2, reg[0], reg[1]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.cu(1.0, 0.5, 0.2, reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCU) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.mcu(1.0, 0.5, 0.2, {0, 1}, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcu(1.0, 0.5, 0.2, {reg[0], reg[1]}, reg[2]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.mcu(1.0, 0.5, 0.2, {reg[0], reg[1]}, reg[2]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.mcu(1.0, 0.5, 0.2, {reg[0], reg[1]}, reg[2]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, SWAP) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.swap(0, 1);
  comp.swap(0, 1);
  comp.swap(0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.swap(q0, q1);
    b.swap(q0, q1);
    b.swap(q0, q1);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    std::tie(q0, q1) = b.swap(q0, q1);
    std::tie(q0, q1) = b.swap(q0, q1);
    b.swap(q0, q1);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.swap(reg[0], reg[1]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.swap(reg[0], reg[1]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.swap(reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, CSWAP) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.cswap(0, 1, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.cswap(reg[0], reg[1], reg[2]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.cswap(reg[0], reg[1], reg[2]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.cswap(reg[0], reg[1], reg[2]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCSWAP) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(4, "q");
  comp.mcswap({0, 1}, 2, 3);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4, "q");
    b.mcswap({reg[0], reg[1]}, reg[2], reg[3]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4, "q");
    b.mcswap({reg[0], reg[1]}, reg[2], reg[3]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4);
    b.mcswap({reg[0], reg[1]}, reg[2], reg[3]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, iSWAP) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.iswap(0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.iswap(reg[0], reg[1]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.iswap(reg[0], reg[1]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.iswap(reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, DCX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.dcx(0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.dcx(reg[0], reg[1]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.dcx(reg[0], reg[1]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.dcx(reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, ECR) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.ecr(0, 1);
  comp.ecr(0, 1);
  comp.ecr(0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.ecr(q0, q1);
    b.ecr(q0, q1);
    b.ecr(q0, q1);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    std::tie(q0, q1) = b.ecr(q0, q1);
    std::tie(q0, q1) = b.ecr(q0, q1);
    b.ecr(q0, q1);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.ecr(reg[0], reg[1]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    b.ecr(reg[0], reg[1]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.ecr(reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, RXX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.rxx(1.0, 0, 1);
  comp.rxx(0.5, 0, 1);
  comp.rxx(1.0, 1, 2);
  comp.rxx(-1.0, 1, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    const auto q2 = reg[2];
    b.rxx(1.0, q0, q1);
    b.rxx(0.5, q0, q1);
    b.rxx(1.0, q1, q2);
    b.rxx(-1.0, q1, q2);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    auto q2 = reg[2];
    std::tie(q0, q1) = b.rxx(1.0, q0, q1);
    std::tie(q0, q1) = b.rxx(0.5, q0, q1);
    std::tie(q1, q2) = b.rxx(1.0, q1, q2);
    b.rxx(-1.0, q1, q2);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.rxx(1.5, reg[0], reg[1]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.rxx(1.5, reg[0], reg[1]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.rxx(1.5, reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, CRXX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.crxx(1.0, 0, 1, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.crxx(1.0, reg[0], reg[1], reg[2]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.crxx(1.0, reg[0], reg[1], reg[2]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.crxx(1.0, reg[0], reg[1], reg[2]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCRXX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(4, "q");
  comp.mcrxx(1.0, {0, 1}, 2, 3);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4, "q");
    b.mcrxx(1.0, {reg[0], reg[1]}, reg[2], reg[3]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4, "q");
    b.mcrxx(1.0, {reg[0], reg[1]}, reg[2], reg[3]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4);
    b.mcrxx(1.0, {reg[0], reg[1]}, reg[2], reg[3]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, RYY) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.ryy(1.0, 0, 1);
  comp.ryy(0.5, 0, 1);
  comp.ryy(1.0, 1, 2);
  comp.ryy(-1.0, 1, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    const auto q2 = reg[2];
    b.ryy(1.0, q0, q1);
    b.ryy(0.5, q0, q1);
    b.ryy(1.0, q1, q2);
    b.ryy(-1.0, q1, q2);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    auto q2 = reg[2];
    std::tie(q0, q1) = b.ryy(1.0, q0, q1);
    std::tie(q0, q1) = b.ryy(0.5, q0, q1);
    std::tie(q1, q2) = b.ryy(1.0, q1, q2);
    b.ryy(-1.0, q1, q2);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.ryy(1.5, reg[0], reg[1]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.ryy(1.5, reg[0], reg[1]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.ryy(1.5, reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, RZX) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.rzx(1.0, 0, 1);
  comp.rzx(0.5, 0, 1);
  comp.rzx(1.0, 1, 2);
  comp.rzx(-1.0, 1, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    const auto q2 = reg[2];
    b.rzx(1.0, q0, q1);
    b.rzx(0.5, q0, q1);
    b.rzx(1.0, q1, q2);
    b.rzx(-1.0, q1, q2);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    auto q2 = reg[2];
    std::tie(q0, q1) = b.rzx(1.0, q0, q1);
    std::tie(q0, q1) = b.rzx(0.5, q0, q1);
    std::tie(q1, q2) = b.rzx(1.0, q1, q2);
    b.rzx(-1.0, q1, q2);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.rzx(1.5, reg[0], reg[1]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.rzx(1.5, reg[0], reg[1]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.rzx(1.5, reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, RZZ) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.rzz(1.0, 0, 1);
  comp.rzz(0.5, 0, 1);
  comp.rzz(1.0, 1, 2);
  comp.rzz(-1.0, 1, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    const auto q2 = reg[2];
    b.rzz(1.0, q0, q1);
    b.rzz(0.5, q0, q1);
    b.rzz(1.0, q1, q2);
    b.rzz(-1.0, q1, q2);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    auto q2 = reg[2];
    std::tie(q0, q1) = b.rzz(1.0, q0, q1);
    std::tie(q0, q1) = b.rzz(0.5, q0, q1);
    std::tie(q1, q2) = b.rzz(1.0, q1, q2);
    b.rzz(-1.0, q1, q2);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.rzz(1.5, reg[0], reg[1]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.rzz(1.5, reg[0], reg[1]);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.rzz(1.5, reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, XXPlusYY) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.xx_plus_yy(1.0, 0.5, 0, 1);
  comp.xx_plus_yy(0.5, 0.5, 0, 1);
  comp.xx_plus_yy(1.0, 1.0, 0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.xx_plus_yy(1.0, 0.5, q0, q1);
    b.xx_plus_yy(0.5, 0.5, q0, q1);
    b.xx_plus_yy(1.0, 1.0, q0, q1);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    std::tie(q0, q1) = b.xx_plus_yy(1.0, 0.5, q0, q1);
    std::tie(q0, q1) = b.xx_plus_yy(0.5, 0.5, q0, q1);
    b.xx_plus_yy(1.0, 1.0, q0, q1);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    std::tie(q0, q1) = b.xx_plus_yy(1.5, 0.5, q0, q1);
    b.xx_plus_yy(1.0, 1.0, q0, q1);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.xx_plus_yy(1.5, 0.5, q0, q1);
    b.xx_plus_yy(1.0, 1.0, q0, q1);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.xx_plus_yy(1.5, 0.5, q0, q1);
    b.xx_plus_yy(1.0, 1.0, q0, q1);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, CXXPlusYY) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.cxx_plus_yy(1.0, 0.5, 0, 1, 2);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.cxx_plus_yy(1.0, 0.5, reg[0], reg[1], reg[2]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.cxx_plus_yy(1.0, 0.5, reg[0], reg[1], reg[2]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3);
    b.cxx_plus_yy(1.0, 0.5, reg[0], reg[1], reg[2]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, MCXXPlusYY) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(4, "q");
  comp.mcxx_plus_yy(1.0, 0.5, {0, 1}, 2, 3);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4, "q");
    b.mcxx_plus_yy(1.0, 0.5, {reg[0], reg[1]}, reg[2], reg[3]);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4, "q");
    b.mcxx_plus_yy(1.0, 0.5, {reg[0], reg[1]}, reg[2], reg[3]);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4);
    b.mcxx_plus_yy(1.0, 0.5, {reg[0], reg[1]}, reg[2], reg[3]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

TEST_F(CompilerPipelineTest, XXMinusYY) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.xx_minus_yy(1.0, 0.5, 0, 1);
  comp.xx_minus_yy(0.5, 0.5, 0, 1);
  comp.xx_minus_yy(1.0, 1.0, 0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.xx_minus_yy(1.0, 0.5, q0, q1);
    b.xx_minus_yy(0.5, 0.5, q0, q1);
    b.xx_minus_yy(1.0, 1.0, q0, q1);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    std::tie(q0, q1) = b.xx_minus_yy(1.0, 0.5, q0, q1);
    std::tie(q0, q1) = b.xx_minus_yy(0.5, 0.5, q0, q1);
    b.xx_minus_yy(1.0, 1.0, q0, q1);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    std::tie(q0, q1) = b.xx_minus_yy(1.5, 0.5, q0, q1);
    b.xx_minus_yy(1.0, 1.0, q0, q1);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.xx_minus_yy(1.5, 0.5, q0, q1);
    b.xx_minus_yy(1.0, 1.0, q0, q1);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.xx_minus_yy(1.5, 0.5, q0, q1);
    b.xx_minus_yy(1.0, 1.0, q0, q1);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, Barrier1) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.barrier(0);
  comp.barrier(0);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.barrier(q);
    b.barrier(q);
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto qubitsOut = b.barrier(reg[0]);
    b.barrier(qubitsOut[0]);
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.barrier(reg[0]);
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    b.barrier(reg[0]);
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = emptyQIR.get(),
  });
}

TEST_F(CompilerPipelineTest, Barrier2) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(3, "q");
  comp.barrier({0, 1});
  comp.barrier({1, 2});

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qcInit = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    const auto q2 = reg[2];
    b.barrier({q0, q1});
    b.barrier({q1, q2});
  });
  const auto qcoInit = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    auto qubitsOut = b.barrier({reg[0], reg[1]});
    b.barrier({qubitsOut[1], reg[2]});
  });
  const auto qcoOpt = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.barrier(reg[0]);
    b.barrier({reg[1], reg[2]});
  });
  const auto qcOpt = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    b.barrier(reg[0]);
    b.barrier({reg[1], reg[2]});
  });

  verifyAllStages({
      .qcImport = qcInit.get(),
      .qcoConversion = qcoInit.get(),
      .optimization = qcoOpt.get(),
      .qcConversion = qcOpt.get(),
      .qirConversion = emptyQIR.get(),
  });
}

TEST_F(CompilerPipelineTest, Bell) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(2, "q");
  comp.h(0);
  comp.cx(0, 1);

  const auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto qc = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.h(q0);
    b.cx(q0, q1);
  });
  const auto qco = buildQCOIR([](qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    q0 = b.h(q0);
    b.cx(q0, q1);
  });
  const auto qir = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    b.h(reg[0]);
    b.cx(reg[0], reg[1]);
  });

  verifyAllStages({
      .qcImport = qc.get(),
      .qcoConversion = qco.get(),
      .optimization = qco.get(),
      .qcConversion = qc.get(),
      .qirConversion = qir.get(),
  });
}

} // namespace
