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

#include <functional>
#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>
#include <sstream>
#include <string>

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
[[nodiscard]] testing::AssertionResult verify(const std::string& stageName,
                                              const std::string& actualIR,
                                              ModuleOp expectedModule) {
  // Parse the actual IR string into a ModuleOp
  const auto actualModule =
      parseSourceString<ModuleOp>(actualIR, expectedModule.getContext());
  if (!actualModule) {
    return testing::AssertionFailure()
           << stageName << " failed to parse actual IR";
  }

  if (!OperationEquivalence::isEquivalentTo(
          actualModule.get(), expectedModule,
          OperationEquivalence::ignoreValueEquivalence, nullptr,
          OperationEquivalence::IgnoreLocations |
              OperationEquivalence::IgnoreDiscardableAttrs |
              OperationEquivalence::IgnoreProperties)) {
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
  ModuleOp quartzImport;
  ModuleOp fluxConversion;
  ModuleOp optimization;
  ModuleOp quartzConversion;
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

    // Enable QIR conversion and recording by default
    config.convertToQIR = true;
    config.recordIntermediates = true;
    config.printIRAfterAllStages =
        true; /// TODO: Change back after everything is running

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
   * @brief Run canonicalization
   */
  static void runCanonicalizationPasses(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (pm.run(module).failed()) {
      llvm::errs() << "Failed to run canonicalization passes\n";
    }
  }

  /**
   * @brief Build expected Quartz IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQuartzIR(
      const std::function<void(quartz::QuartzProgramBuilder&)>& buildFunc)
      const {
    quartz::QuartzProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  /**
   * @brief Build expected Flux IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildFluxIR(
      const std::function<void(flux::FluxProgramBuilder&)>& buildFunc) const {
    flux::FluxProgramBuilder builder(context.get());
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
    if (expectations.quartzImport != nullptr) {
      EXPECT_TRUE(verify("Quartz Import", record.afterInitialCanon,
                         expectations.quartzImport));
    }

    if (expectations.fluxConversion != nullptr) {
      EXPECT_TRUE(verify("Flux Conversion", record.afterFluxCanon,
                         expectations.fluxConversion));
    }

    if (expectations.optimization != nullptr) {
      EXPECT_TRUE(verify("Optimization", record.afterOptimizationCanon,
                         expectations.optimization));
    }

    if (expectations.quartzConversion != nullptr) {
      EXPECT_TRUE(verify("Quartz Conversion", record.afterQuartzCanon,
                         expectations.quartzConversion));
    }

    if (expectations.qirConversion != nullptr) {
      EXPECT_TRUE(verify("QIR Conversion", record.afterQIRCanon,
                         expectations.qirConversion));
    }
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
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  // Verify all stages
  verifyAllStages({
      .quartzImport = emptyQuartz.get(),
      .fluxConversion = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
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
 * in the Flux dialect.
 */
TEST_F(CompilerPipelineTest, SingleQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(1, "q");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzExpected = buildQuartzIR(
      [](quartz::QuartzProgramBuilder& b) { b.allocQubitRegister(1, "q"); });
  const auto fluxExpected = buildFluxIR(
      [](flux::FluxProgramBuilder& b) { b.allocQubitRegister(1, "q"); });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
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
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzExpected = buildQuartzIR(
      [](quartz::QuartzProgramBuilder& b) { b.allocQubitRegister(3, "q"); });
  const auto fluxExpected = buildFluxIR(
      [](flux::FluxProgramBuilder& b) { b.allocQubitRegister(3, "q"); });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
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
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

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
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
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
  qc::QuantumComputation qc;
  qc.addClassicalRegister(1, "c");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    b.allocClassicalBitRegister(1, "c");
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
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
  qc::QuantumComputation qc;
  qc.addClassicalRegister(5, "c");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    b.allocClassicalBitRegister(5, "c");
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
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
  qc::QuantumComputation qc;
  qc.addClassicalRegister(3, "c");
  qc.addClassicalRegister(2, "d");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    b.allocClassicalBitRegister(3, "c");
    b.allocClassicalBitRegister(2, "d");
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
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
  qc::QuantumComputation qc(1);
  qc.reset(0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

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
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
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
  qc::QuantumComputation qc(1);
  qc.reset(0);
  qc.reset(0);
  qc.reset(0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

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
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
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
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

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
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
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
  qc::QuantumComputation qc(1, 1);
  qc.measure(0, 0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    b.measure(q[0], 0);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

/**
 * @brief Test: Repeated measurement to same bit
 */
TEST_F(CompilerPipelineTest, RepeatedMeasurementToSameBit) {
  qc::QuantumComputation qc(1, 1);
  qc.measure(0, 0);
  qc.measure(0, 0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[0]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    q[0] = b.measure(q[0], c[0]);
    q[0] = b.measure(q[0], c[0]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    b.measure(q[0], 0);
    b.measure(q[0], 0);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .qirConversion = qirExpected.get(),
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
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(3);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[1]);
    b.measure(q[0], c[2]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(3);
    q[0] = b.measure(q[0], c[0]);
    q[0] = b.measure(q[0], c[1]);
    q[0] = b.measure(q[0], c[2]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    b.measure(q[0], 0);
    b.measure(q[0], 1);
    b.measure(q[0], 2);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

/**
 * @brief Test: Multiple classical registers with measurements
 */
TEST_F(CompilerPipelineTest, MultipleClassicalRegistersAndMeasurements) {
  qc::QuantumComputation qc(2);
  const auto& c1 = qc.addClassicalRegister(1, "c1");
  const auto& c2 = qc.addClassicalRegister(1, "c2");
  qc.measure(0, c1[0]);
  qc.measure(1, c2[0]);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto& creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto& creg2 = b.allocClassicalBitRegister(1, "c2");
    b.measure(q[0], creg1[0]);
    b.measure(q[1], creg2[0]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto& creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto& creg2 = b.allocClassicalBitRegister(1, "c2");
    q[0] = b.measure(q[0], creg1[0]);
    q[1] = b.measure(q[1], creg2[0]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    b.measure(q[0], "c1", 0);
    b.measure(q[1], "c2", 0);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

// ##################################################
// # Temporary Unit Tests
// ##################################################

TEST_F(CompilerPipelineTest, X) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(1, "q");
  qc.x(0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzExpected =
      buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q = b.allocQubitRegister(1, "q");
        b.x(q[0]);
      });
  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1, "q");
    b.x(q[0]);
  });
  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    b.x(q[0]);
  });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = quartzExpected.get(),
      .qirConversion = qirExpected.get(),
  });
}

} // namespace
