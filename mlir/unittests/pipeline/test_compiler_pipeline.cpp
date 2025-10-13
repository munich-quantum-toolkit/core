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
#include "mlir/Dialect/Flux/Builder/FluxProgramBuilder.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
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
#include <sstream>
#include <string>

namespace {

using namespace mlir;

//===----------------------------------------------------------------------===//
// Stage Verification Utility
//===----------------------------------------------------------------------===//

/**
 * @brief Helper to verify a compilation stage matches expected IR
 *
 * @details
 * Provides detailed error messages including actual and expected IR when
 * verification fails.
 */
class StageVerifier {
public:
  explicit StageVerifier(MLIRContext* ctx) : context(ctx) {}

  /**
   * @brief Verify a stage matches expected module
   *
   * @param stageName Human-readable stage name for error messages
   * @param actualIR String IR from CompilationRecord
   * @param expectedModule Expected module to compare against
   * @return True if modules match, false otherwise with diagnostic output
   */
  [[nodiscard]] ::testing::AssertionResult
  verify(const std::string& stageName, const std::string& actualIR,
         ModuleOp expectedModule) const {
    // Parse actual IR
    const auto actualModule =
        parseSourceString<ModuleOp>(actualIR, ParserConfig(context));
    if (!actualModule) {
      return ::testing::AssertionFailure()
             << "Failed to parse " << stageName << " IR\n"
             << "Raw IR string:\n"
             << actualIR;
    }

    // Compare modules
    if (!modulesAreEquivalent(actualModule.get(), expectedModule)) {
      std::ostringstream msg;
      msg << stageName << " IR does not match expected structure\n\n";

      msg << "=== EXPECTED IR ===\n";
      std::string expectedStr;
      llvm::raw_string_ostream expectedStream(expectedStr);
      expectedModule.print(expectedStream);
      msg << expectedStr << "\n\n";

      msg << "=== ACTUAL IR ===\n";
      msg << actualIR << "\n\n";

      msg << "=== DIFFERENCE ===\n";
      msg << "Modules are structurally different. ";
      msg << "Check operation types, attributes, and structure.\n";

      return ::testing::AssertionFailure() << msg.str();
    }

    return ::testing::AssertionSuccess();
  }

private:
  MLIRContext* context;
};

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
  ModuleOp quartzImport;
  ModuleOp initialCanon;
  ModuleOp fluxConversion;
  ModuleOp fluxCanon;
  ModuleOp optimization;
  ModuleOp optimizationCanon;
  ModuleOp quartzConversion;
  ModuleOp quartzCanon;
  ModuleOp qirConversion;
  ModuleOp qirCanon;
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
  std::unique_ptr<StageVerifier> verifier;
  QuantumCompilerConfig config;
  CompilationRecord record;

  OwningOpRef<ModuleOp> emptyQuartz;
  OwningOpRef<ModuleOp> emptyFlux;
  OwningOpRef<ModuleOp> emptyQIR;

  void SetUp() override {
    // Register all dialects needed for the full compilation pipeline
    DialectRegistry registry;
    registry
        .insert<quartz::QuartzDialect, flux::FluxDialect, arith::ArithDialect,
                cf::ControlFlowDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect, LLVM::LLVMDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();

    verifier = std::make_unique<StageVerifier>(context.get());

    // Enable QIR conversion and recording by default
    config.convertToQIR = true;
    config.recordIntermediates = true;
    config.printIRAfterAllStages =
        false; /// TODO: Change back after everything is running

    emptyQuartz = buildQuartzIR([](quartz::QuartzProgramBuilder&) {});
    emptyFlux = buildFluxIR([](flux::FluxProgramBuilder&) {});
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
  [[nodiscard]] OwningOpRef<ModuleOp>
  importQuantumCircuit(const qc::QuantumComputation& qc) const {
    if (config.printIRAfterAllStages) {
      prettyPrintQuantumComputation(qc);
    }
    return translateQuantumComputationToQuartz(context.get(), qc);
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
   * @brief Build expected Quartz IR programmatically
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQuartzIR(
      const std::function<void(quartz::QuartzProgramBuilder&)>& buildFunc)
      const {
    quartz::QuartzProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    return builder.finalize();
  }

  /**
   * @brief Build expected Flux IR programmatically
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildFluxIR(
      const std::function<void(flux::FluxProgramBuilder&)>& buildFunc) const {
    flux::FluxProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    return builder.finalize();
  }

  /**
   * @brief Build expected QIR programmatically
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQIR(
      const std::function<void(qir::QIRProgramBuilder&)>& buildFunc) const {
    qir::QIRProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    return builder.finalize();
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
    if (expectations.quartzImport) {
      EXPECT_TRUE(verifier->verify("Quartz Import", record.afterQuartzImport,
                                   expectations.quartzImport));
    }

    if (expectations.initialCanon) {
      EXPECT_TRUE(verifier->verify("Initial Canonicalization",
                                   record.afterInitialCanon,
                                   expectations.initialCanon));
    }

    if (expectations.fluxConversion) {
      EXPECT_TRUE(verifier->verify("Flux Conversion",
                                   record.afterFluxConversion,
                                   expectations.fluxConversion));
    }

    if (expectations.fluxCanon) {
      EXPECT_TRUE(verifier->verify("Flux Canonicalization",
                                   record.afterFluxCanon,
                                   expectations.fluxCanon));
    }

    if (expectations.optimization) {
      EXPECT_TRUE(verifier->verify("Optimization", record.afterOptimization,
                                   expectations.optimization));
    }

    if (expectations.optimizationCanon) {
      EXPECT_TRUE(verifier->verify("Optimization Canonicalization",
                                   record.afterOptimizationCanon,
                                   expectations.optimizationCanon));
    }

    if (expectations.quartzConversion) {
      EXPECT_TRUE(verifier->verify("Quartz Conversion",
                                   record.afterQuartzConversion,
                                   expectations.quartzConversion));
    }

    if (expectations.quartzCanon) {
      EXPECT_TRUE(verifier->verify("Quartz Canonicalization",
                                   record.afterQuartzCanon,
                                   expectations.quartzCanon));
    }

    if (expectations.qirConversion) {
      EXPECT_TRUE(verifier->verify("QIR Conversion", record.afterQIRConversion,
                                   expectations.qirConversion));
    }

    if (expectations.qirCanon) {
      EXPECT_TRUE(verifier->verify("QIR Canonicalization", record.afterQIRCanon,
                                   expectations.qirCanon));
    }
  }

  /**
   * @brief Verify a single stage
   */
  void verifyStage(const std::string& stageName, const std::string& actualIR,
                   const ModuleOp expectedModule) const {
    EXPECT_TRUE(verifier->verify(stageName, actualIR, expectedModule));
  }

  void TearDown() override {
    // Verify all stages were recorded (basic sanity check)
    EXPECT_FALSE(record.afterQuartzImport.empty())
        << "Quartz import stage was not recorded";
    EXPECT_FALSE(record.afterInitialCanon.empty())
        << "Initial canonicalization stage was not recorded";
    EXPECT_FALSE(record.afterFluxConversion.empty())
        << "Flux conversion stage was not recorded";
    EXPECT_FALSE(record.afterFluxCanon.empty())
        << "Flux canonicalization stage was not recorded";
    EXPECT_FALSE(record.afterOptimization.empty())
        << "Optimization stage was not recorded";
    EXPECT_FALSE(record.afterOptimizationCanon.empty())
        << "Optimization canonicalization stage was not recorded";
    EXPECT_FALSE(record.afterQuartzConversion.empty())
        << "Quartz conversion stage was not recorded";
    EXPECT_FALSE(record.afterQuartzCanon.empty())
        << "Quartz canonicalization stage was not recorded";

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
  const qc::QuantumComputation qc;

  // Import to Quartz dialect
  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Run compilation
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  // Verify all stages
  verifyAllStages({
      .quartzImport = emptyQuartz.get(),
      .initialCanon = emptyQuartz.get(),
      .fluxConversion = emptyFlux.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
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
 * in the Flux dialect.
 */
TEST_F(CompilerPipelineTest, SingleQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(1, "q");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto quartzExpected = buildQuartzIR(
      [](quartz::QuartzProgramBuilder& b) { b.allocQubitRegister(1, "q"); });
  const auto fluxExpected = buildFluxIR(
      [](flux::FluxProgramBuilder& b) { b.allocQubitRegister(1, "q"); });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .initialCanon = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multi-qubit register allocation
 */
TEST_F(CompilerPipelineTest, MultiQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(3, "q");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto quartzExpected = buildQuartzIR(
      [](quartz::QuartzProgramBuilder& b) { b.allocQubitRegister(3, "q"); });
  const auto fluxExpected = buildFluxIR(
      [](flux::FluxProgramBuilder& b) { b.allocQubitRegister(3, "q"); });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .initialCanon = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multiple quantum registers
 */
TEST_F(CompilerPipelineTest, MultipleQuantumRegisters) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(2, "q");
  qc.addQubitRegister(3, "aux");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto quartzExpected =
      buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
        b.allocQubitRegister(2, "q");
        b.allocQubitRegister(3, "aux");
      });
  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    b.allocQubitRegister(2, "q");
    b.allocQubitRegister(3, "aux");
  });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .initialCanon = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
  });
}

/**
 * @brief Test: Large qubit register allocation
 */
TEST_F(CompilerPipelineTest, LargeQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(100, "q");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));
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
  qc::QuantumComputation qc;
  qc.addClassicalRegister(1, "c");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    b.allocClassicalBitRegister(1, "c");
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = emptyQuartz.get(),
      .fluxConversion = emptyFlux.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multi-bit classical register
 *
 * @details
 * Since the register is unused, it should be removed during canonicalization.
 */
TEST_F(CompilerPipelineTest, MultiBitClassicalRegister) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(5, "c");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    b.allocClassicalBitRegister(5, "c");
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = emptyQuartz.get(),
      .fluxConversion = emptyFlux.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
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
  qc::QuantumComputation qc;
  qc.addClassicalRegister(3, "c");
  qc.addClassicalRegister(2, "d");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    b.allocClassicalBitRegister(3, "c");
    b.allocClassicalBitRegister(2, "d");
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = emptyQuartz.get(),
      .fluxConversion = emptyFlux.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
  });
}

/**
 * @brief Test: Large classical bit register
 */
TEST_F(CompilerPipelineTest, LargeClassicalBitRegister) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(128, "c");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));
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
  qc::QuantumComputation qc(1);
  qc.reset(0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    const auto q = b.allocQubitRegister(1, "q");
    b.reset(q[0]);
  });
  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    const auto q = b.allocQubitRegister(1, "q");
    b.reset(q[0]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
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
  qc::QuantumComputation qc(1);
  qc.reset(0);
  qc.reset(0);
  qc.reset(0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    const auto q = b.allocQubitRegister(1, "q");
    b.reset(q[0]);
    b.reset(q[0]);
    b.reset(q[0]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1, "q");
    q[0] = b.reset(q[0]);
    q[0] = b.reset(q[0]);
    q[0] = b.reset(q[0]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
  });
}

/**
 * @brief Test: Separate resets in two qubit system
 */
TEST_F(CompilerPipelineTest, SeparateResetsInTwoQubitSystem) {
  qc::QuantumComputation qc(2);
  qc.reset(0);
  qc.reset(1);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    const auto q = b.allocQubitRegister(2, "q");
    b.reset(q[0]);
    b.reset(q[1]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(2, "q");
    q[0] = b.reset(q[0]);
    q[1] = b.reset(q[1]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .optimizationCanon = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .quartzCanon = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
      .qirCanon = emptyQIR.get(),
  });
}

// ##################################################
// # Measure Operation Tests
// ##################################################

/**
 * @brief Test: Single measurement to single bit
 */
TEST_F(CompilerPipelineTest, SingleMeasurementToSingleBit) {
  qc::QuantumComputation qc(1, 1);
  qc.measure(0, 0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c, 0);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c, 0);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    b.measure(q[0], 0);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .optimizationCanon = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .quartzCanon = expected.get(),
      .qirConversion = qirExpected.get(),
      .qirCanon = qirExpected.get(),
  });
}

/**
 * @brief Test: Repeated measurement to same bit
 */
TEST_F(CompilerPipelineTest, RepeatedMeasurementToSameBit) {
  qc::QuantumComputation qc(1);
  qc.measure(0, 0);
  qc.measure(0, 0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c, 0);
    b.measure(q[0], c, 0);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto c = b.allocClassicalBitRegister(1);
    q[0] = b.measure(q[0], c, 0);
    q[0] = b.measure(q[0], c, 0);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    b.measure(q[0], 0);
    b.measure(q[0], 0);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .optimizationCanon = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .quartzCanon = expected.get(),
      .qirConversion = qirExpected.get(),
      .qirCanon = qirExpected.get(),
  });
}

/**
 * @brief Test: Repeated measurement on separate bits
 */
TEST_F(CompilerPipelineTest, RepeatedMeasurementOnSeparateBits) {
  qc::QuantumComputation qc(1);
  qc.addClassicalRegister(3);
  qc.measure(0, 0);
  qc.measure(0, 1);
  qc.measure(0, 2);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto c = b.allocClassicalBitRegister(3);
    b.measure(q[0], c, 0);
    b.measure(q[0], c, 1);
    b.measure(q[0], c, 2);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto c = b.allocClassicalBitRegister(3);
    q[0] = b.measure(q[0], c, 0);
    q[0] = b.measure(q[0], c, 1);
    q[0] = b.measure(q[0], c, 2);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    b.measure(q[0], 0);
    b.measure(q[0], 1);
    b.measure(q[0], 2);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .optimizationCanon = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .quartzCanon = expected.get(),
      .qirConversion = qirExpected.get(),
      .qirCanon = qirExpected.get(),
  });
}

/**
 * @brief Test: Multiple classical registers with measurements
 */
TEST_F(CompilerPipelineTest, MultipleClassicalRegistersAndMeasurements) {
  qc::QuantumComputation qc(2);
  auto& c1 = qc.addClassicalRegister(1, "c1");
  auto& c2 = qc.addClassicalRegister(1, "c2");
  qc.measure(0, c1[0]);
  qc.measure(1, c2[0]);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runPipeline(module.get())));

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto creg2 = b.allocClassicalBitRegister(1, "c2");
    b.measure(q[0], creg1, 0);
    b.measure(q[1], creg2, 0);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto creg2 = b.allocClassicalBitRegister(1, "c2");
    q[0] = b.measure(q[0], creg1, 0);
    q[1] = b.measure(q[1], creg2, 0);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    b.measure(q[0], 0);
    b.measure(q[1], 1);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .initialCanon = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .fluxCanon = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .optimizationCanon = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .quartzCanon = expected.get(),
      .qirConversion = qirExpected.get(),
      .qirCanon = qirExpected.get(),
  });
}

} // namespace
