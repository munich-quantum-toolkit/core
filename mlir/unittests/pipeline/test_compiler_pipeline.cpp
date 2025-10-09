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
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Quartz/Builder/QuartzProgramBuilder.h"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"
#include "mlir/Dialect/Quartz/Translation/TranslateQuantumComputationToQuartz.h"
#include "mlir/Support/PrettyPrinting.h"
#include "mlir/Support/TestUtils.h"

#include <gtest/gtest.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Transforms/Passes.h>
#include <string>

namespace {

using namespace mlir;

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

  void SetUp() override {
    // Register all dialects needed for the full compilation pipeline
    DialectRegistry registry;
    registry.insert<quartz::QuartzDialect>();
    registry.insert<flux::FluxDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<LLVM::LLVMDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();

    // Enable QIR conversion by default
    config.convertToQIR = true;
    config.recordIntermediates = true;
    config.printIRAfterAllStages = true;
  }

  /**
   * @brief Pretty print quantum computation with ASCII art borders
   *
   * @param qc The quantum computation to print
   */
  static void prettyPrintQuantumComputation(const qc::QuantumComputation& qc) {
    llvm::errs() << "\n";
    printBoxTop();

    // Print header
    printBoxLine("Initial Quantum Computation");

    printBoxMiddle();

    // Print internal representation
    printBoxLine("Internal Representation:");

    // Capture the internal representation
    std::ostringstream internalRepr;
    internalRepr << qc;
    const std::string internalStr = internalRepr.str();

    // Print with line wrapping
    printBoxText(internalStr);

    printBoxMiddle();

    // Print OpenQASM3 representation
    printBoxLine("OpenQASM3 Representation:");
    printBoxLine("");

    const auto qasmStr = qc.toQASM();

    // Print with line wrapping
    printBoxText(qasmStr);

    printBoxBottom();
    llvm::errs().flush();
  }

  /**
   * @brief Import a QuantumComputation into Quartz dialect
   */
  OwningOpRef<ModuleOp>
  importQuantumCircuit(const qc::QuantumComputation& qc) const {
    if (config.printIRAfterAllStages) {
      prettyPrintQuantumComputation(qc);
    }
    return translateQuantumComputationToQuartz(context.get(), qc);
  }

  /**
   * @brief Run the compiler pipeline with specified configuration
   */
  static LogicalResult runCompiler(ModuleOp module,
                                   const QuantumCompilerConfig& config,
                                   CompilationRecord* record = nullptr) {
    const QuantumCompilerPipeline pipeline(config);
    return pipeline.runPipeline(module, record);
  }

  /**
   * @brief Clone a module for comparison purposes
   *
   * @details
   * Creates a deep copy of the module so we can compare it later
   * without worrying about in-place mutations.
   */
  static OwningOpRef<ModuleOp> cloneModule(ModuleOp module) {
    return module.clone();
  }

  /**
   * @brief Parse IR string back into a module
   *
   * @details
   * Useful for reconstructing modules from CompilationRecord strings.
   */
  OwningOpRef<ModuleOp> parseModule(const std::string& irString) const {
    return mlir::parseSourceString<ModuleOp>(irString,
                                             ParserConfig(context.get()));
  }

  /**
   * @brief Check if IR contains a specific pattern (for quick checks)
   */
  static bool irContains(const std::string& ir, const std::string& pattern) {
    return ir.find(pattern) != std::string::npos;
  }

  /**
   * @brief Build expected Quartz IR programmatically for comparison
   */
  OwningOpRef<ModuleOp> buildExpectedQuartzIR(
      const std::function<void(quartz::QuartzProgramBuilder&)>& buildFunc)
      const {
    quartz::QuartzProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    return builder.finalize();
  }

  /**
   * @brief Apply canonicalization to a module (for building expected IR)
   */
  LogicalResult applyCanonicalization(ModuleOp module) const {
    PassManager pm(context.get());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createRemoveDeadValuesPass());
    return pm.run(module);
  }

  /**
   * @brief Verify module at a specific stage matches expected
   */
  void verifyStageMatchesExpected(const std::string& stageName,
                                  const std::string& actualIR,
                                  ModuleOp expectedModule) const {
    const auto actualModule = parseModule(actualIR);
    ASSERT_TRUE(actualModule) << "Failed to parse " << stageName << " IR";

    EXPECT_TRUE(modulesAreEquivalent(actualModule.get(), expectedModule))
        << stageName << " IR does not match expected structure";
  }

  void TearDown() override {}
};

// ##################################################
// # Empty Circuit Tests
// ##################################################

/**
 * @brief Test: Empty circuit construction
 *
 * @details
 * Verifies that an empty QuantumComputation() can be imported and compiled
 * without errors. Checks multiple stages of the pipeline.
 */
TEST_F(CompilerPipelineTest, EmptyCircuit) {
  // Create empty circuit
  const qc::QuantumComputation qc;

  // Import to Quartz dialect
  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR for initial Quartz import
  const auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        // Empty circuit - just initialize
      });
  ASSERT_TRUE(expectedQuartzInitial);

  // Run compilation
  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify Quartz import stage
  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());

  // Verify canonicalized Quartz stage
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzInitial.get());

  // Verify final Quartz stage (after round-trip through Flux)
  verifyStageMatchesExpected("Final Quartz Canonicalization",
                             record.afterQuartzCanon,
                             expectedQuartzInitial.get());

  // Verify the IR is valid at all stages
  EXPECT_FALSE(record.afterQuartzImport.empty());
  EXPECT_FALSE(record.afterFluxConversion.empty());
  EXPECT_FALSE(record.afterQuartzConversion.empty());

  // QIR stages should also be present
  EXPECT_FALSE(record.afterQIRConversion.empty());
  EXPECT_FALSE(record.afterQIRCanon.empty());
}

// ##################################################
// # Quantum Register Allocation Tests
// ##################################################

/**
 * @brief Test: Single qubit register allocation
 *
 * @details
 * Tests addQubitRegister with a single qubit. Verifies import and
 * canonicalized stages.
 */
TEST_F(CompilerPipelineTest, SingleQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(1, "q");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR for initial import
  const auto expectedQuartzInitial = buildExpectedQuartzIR(
      [](quartz::QuartzProgramBuilder& b) { b.allocQubitRegister(1, "q"); });
  ASSERT_TRUE(expectedQuartzInitial);

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify multiple stages
  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Final Quartz Canonicalization",
                             record.afterQuartzCanon,
                             expectedQuartzInitial.get());
}

/**
 * @brief Test: Multi-qubit register allocation
 */
TEST_F(CompilerPipelineTest, MultiQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(3, "q");

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial = buildExpectedQuartzIR(
      [](quartz::QuartzProgramBuilder& b) { b.allocQubitRegister(3, "q"); });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());
}

/**
 * @brief Test: Multiple quantum registers
 */
TEST_F(CompilerPipelineTest, MultipleQuantumRegisters) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(2, "q");
  qc.addQubitRegister(3, "aux");

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        b.allocQubitRegister(2, "q");
        b.allocQubitRegister(3, "aux");
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());
}

/**
 * @brief Test: Large qubit register allocation
 */
TEST_F(CompilerPipelineTest, LargeQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(100, "q");

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify compilation succeeded and produced valid IR at all stages
  EXPECT_FALSE(record.afterQuartzImport.empty());
  EXPECT_FALSE(record.afterFluxConversion.empty());
  EXPECT_FALSE(record.afterQuartzCanon.empty());
  EXPECT_FALSE(record.afterQIRCanon.empty());
}

// ##################################################
// # Classical Register Allocation Tests
// ##################################################

/**
 * @brief Test: Single classical bit register
 */
TEST_F(CompilerPipelineTest, SingleClassicalBitRegister) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(1, "c");

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        b.allocClassicalBitRegister(1, "c");
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());
}

/**
 * @brief Test: Multi-bit classical register
 */
TEST_F(CompilerPipelineTest, MultiBitClassicalRegister) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(5, "c");

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        b.allocClassicalBitRegister(5, "c");
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());
}

/**
 * @brief Test: Multiple classical registers
 */
TEST_F(CompilerPipelineTest, MultipleClassicalRegisters) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(3, "c");
  qc.addClassicalRegister(2, "result");

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        b.allocClassicalBitRegister(3, "c");
        b.allocClassicalBitRegister(2, "result");
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());
}

/**
 * @brief Test: Large classical bit register
 */
TEST_F(CompilerPipelineTest, LargeClassicalBitRegister) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(128, "c");

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify compilation succeeded
  EXPECT_FALSE(record.afterQuartzImport.empty());
  EXPECT_FALSE(record.afterQuartzCanon.empty());
  EXPECT_FALSE(record.afterQIRCanon.empty());
}

// ##################################################
// # Reset Operation Tests
// ##################################################

/**
 * @brief Test: Single reset in single qubit circuit
 */
TEST_F(CompilerPipelineTest, SingleResetInSingleQubitCircuit) {
  qc::QuantumComputation qc(1);
  qc.reset(0);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q = b.staticQubit(0);
        b.reset(q);
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());

  // Verify Flux conversion contains flux.reset
  EXPECT_TRUE(irContains(record.afterFluxConversion, "flux.reset"));
}

/**
 * @brief Test: Consecutive reset operations
 */
TEST_F(CompilerPipelineTest, ConsecutiveResetOperations) {
  qc::QuantumComputation qc(1);
  qc.reset(0);
  qc.reset(0);
  qc.reset(0);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q = b.staticQubit(0);
        b.reset(q);
        b.reset(q);
        b.reset(q);
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());
}

/**
 * @brief Test: Separate resets in two qubit system
 */
TEST_F(CompilerPipelineTest, SeparateResetsInTwoQubitSystem) {
  qc::QuantumComputation qc(2);
  qc.reset(0);
  qc.reset(1);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q0 = b.staticQubit(0);
        auto q1 = b.staticQubit(1);
        b.reset(q0);
        b.reset(q1);
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());
}

// ##################################################
// # Measure Operation Tests
// ##################################################

/**
 * @brief Test: Single measurement to single bit
 */
TEST_F(CompilerPipelineTest, SingleMeasurementToSingleBit) {
  qc::QuantumComputation qc(1);
  qc.measure(0, 0);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q = b.staticQubit(0);
        auto c = b.allocClassicalBitRegister(1);
        b.measure(q, c, 0);
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());

  // Verify Flux conversion contains flux.measure
  EXPECT_TRUE(irContains(record.afterFluxConversion, "flux.measure"));
}

/**
 * @brief Test: Repeated measurement to same bit
 */
TEST_F(CompilerPipelineTest, RepeatedMeasurementToSameBit) {
  qc::QuantumComputation qc(1);
  qc.measure(0, 0);
  qc.reset(0);
  qc.measure(0, 0);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q = b.staticQubit(0);
        auto c = b.allocClassicalBitRegister(1);
        b.measure(q, c, 0);
        b.reset(q);
        b.measure(q, c, 0);
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());
}

/**
 * @brief Test: Repeated measurement on separate bits
 */
TEST_F(CompilerPipelineTest, RepeatedMeasurementOnSeparateBits) {
  qc::QuantumComputation qc(1);
  qc.addClassicalRegister(3);
  qc.measure(0, 0);
  qc.reset(0);
  qc.measure(0, 1);
  qc.reset(0);
  qc.measure(0, 2);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q = b.staticQubit(0);
        auto c = b.allocClassicalBitRegister(3);
        b.measure(q, c, 0);
        b.reset(q);
        b.measure(q, c, 1);
        b.reset(q);
        b.measure(q, c, 2);
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());
}

/**
 * @brief Test: Multiple classical registers with measurements
 */
TEST_F(CompilerPipelineTest, MultipleClassicalRegistersAndMeasurements) {
  qc::QuantumComputation qc(2);
  qc.addClassicalRegister(2, "c1");
  qc.addClassicalRegister(2, "c2");
  qc.measure(0, 0);
  qc.measure(1, 1);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify all stages completed
  EXPECT_FALSE(record.afterQuartzImport.empty());
  EXPECT_FALSE(record.afterFluxConversion.empty());
  EXPECT_FALSE(record.afterQuartzCanon.empty());
  EXPECT_FALSE(record.afterQIRCanon.empty());
}

// ##################################################
// # Combined Feature Tests
// ##################################################

/**
 * @brief Test: Quantum and classical registers with operations
 */
TEST_F(CompilerPipelineTest, QuantumClassicalRegistersWithOperations) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(3, "q");
  qc.addClassicalRegister(3, "c");

  // Reset all qubits
  qc.reset(0);
  qc.reset(1);
  qc.reset(2);

  // Measure all qubits
  qc.measure(0, 0);
  qc.measure(1, 1);
  qc.measure(2, 2);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Build expected IR
  auto expectedQuartzInitial =
      buildExpectedQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q = b.allocQubitRegister(3, "q");
        auto c = b.allocClassicalBitRegister(3, "c");
        b.reset(q[0]);
        b.reset(q[1]);
        b.reset(q[2]);
        b.measure(q[0], c, 0);
        b.measure(q[1], c, 1);
        b.measure(q[2], c, 2);
      });
  ASSERT_TRUE(expectedQuartzInitial);

  auto expectedQuartzCanon = cloneModule(expectedQuartzInitial.get());
  ASSERT_TRUE(succeeded(applyCanonicalization(expectedQuartzCanon.get())));

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  verifyStageMatchesExpected("Quartz Import", record.afterQuartzImport,
                             expectedQuartzInitial.get());
  verifyStageMatchesExpected("Initial Canonicalization",
                             record.afterInitialCanon,
                             expectedQuartzCanon.get());

  // Verify conversions to other dialects succeeded
  EXPECT_TRUE(irContains(record.afterFluxConversion, "flux"));
  EXPECT_TRUE(irContains(record.afterQIRConversion, "llvm"));
}

/**
 * @brief Test: End-to-end pipeline with all stages
 */
TEST_F(CompilerPipelineTest, EndToEndPipelineAllStages) {
  qc::QuantumComputation qc(2);
  qc.addClassicalRegister(2);
  qc.reset(0);
  qc.reset(1);
  qc.measure(0, 0);
  qc.measure(1, 1);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify each stage produces non-empty output
  EXPECT_FALSE(record.afterQuartzImport.empty());
  EXPECT_FALSE(record.afterInitialCanon.empty());
  EXPECT_FALSE(record.afterFluxConversion.empty());
  EXPECT_FALSE(record.afterFluxCanon.empty());
  EXPECT_FALSE(record.afterOptimization.empty());
  EXPECT_FALSE(record.afterOptimizationCanon.empty());
  EXPECT_FALSE(record.afterQuartzConversion.empty());
  EXPECT_FALSE(record.afterQuartzCanon.empty());
  EXPECT_FALSE(record.afterQIRConversion.empty());
  EXPECT_FALSE(record.afterQIRCanon.empty());

  // Verify dialect transitions
  EXPECT_TRUE(irContains(record.afterQuartzImport, "quartz"));
  EXPECT_TRUE(irContains(record.afterFluxConversion, "flux"));
  EXPECT_TRUE(irContains(record.afterQuartzConversion, "quartz"));
  EXPECT_TRUE(irContains(record.afterQIRConversion, "llvm"));
}

/**
 * @brief Test: Complex circuit with interleaved operations
 */
TEST_F(CompilerPipelineTest, ComplexInterleavedOperations) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(4, "q");
  qc.addClassicalRegister(4, "c1");
  qc.addClassicalRegister(2, "c2");

  // Interleaved operations
  qc.reset(0);
  qc.measure(0, 0);
  qc.reset(1);
  qc.reset(2);
  qc.measure(1, 1);
  qc.measure(2, 2);
  qc.reset(3);
  qc.measure(3, 3);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify all pipeline stages succeeded
  EXPECT_FALSE(record.afterQuartzImport.empty());
  EXPECT_FALSE(record.afterFluxConversion.empty());
  EXPECT_FALSE(record.afterQuartzCanon.empty());
  EXPECT_FALSE(record.afterQIRCanon.empty());

  // Verify operations are present in appropriate dialects
  EXPECT_TRUE(irContains(record.afterQuartzImport, "quartz.reset"));
  EXPECT_TRUE(irContains(record.afterQuartzImport, "quartz.measure"));
  EXPECT_TRUE(irContains(record.afterFluxConversion, "flux.reset"));
  EXPECT_TRUE(irContains(record.afterFluxConversion, "flux.measure"));
}

/**
 * @brief Test: Scalability with large mixed operations
 */
TEST_F(CompilerPipelineTest, ScalabilityLargeMixedOperations) {
  constexpr size_t NUM_QUBITS = 50;

  qc::QuantumComputation qc;
  qc.addQubitRegister(NUM_QUBITS, "q");
  qc.addClassicalRegister(NUM_QUBITS, "c");

  // Add operations for all qubits
  for (size_t i = 0; i < NUM_QUBITS; ++i) {
    qc.reset(i);
    qc.measure(i, i);
  }

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify compilation succeeded and produced valid output at all stages
  EXPECT_FALSE(record.afterQuartzImport.empty());
  EXPECT_FALSE(record.afterFluxConversion.empty());
  EXPECT_FALSE(record.afterQuartzCanon.empty());
  EXPECT_FALSE(record.afterQIRCanon.empty());
}

} // namespace
