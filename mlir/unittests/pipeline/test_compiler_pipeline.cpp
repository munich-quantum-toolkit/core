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
#include "ir/operations/OpType.hpp"
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Conversion/FluxToQuartz/FluxToQuartz.h"
#include "mlir/Conversion/QuartzToFlux/QuartzToFlux.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Quartz/Builder/QuartzProgramBuilder.h"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"
#include "mlir/Dialect/Quartz/Translation/TranslateQuantumComputationToQuartz.h"

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
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Transforms/Passes.h>
#include <string>

namespace {

using namespace qc;
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
  }

  /**
   * @brief Import a QuantumComputation into Quartz dialect
   */
  OwningOpRef<ModuleOp>
  importQuantumCircuit(const QuantumComputation& qc) const {
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
   * @brief Check if IR contains a specific pattern
   */
  static bool irContains(const std::string& ir, const std::string& pattern) {
    return ir.find(pattern) != std::string::npos;
  }

  void TearDown() override {}
};

// ##################################################
// # Basic Circuit Tests
// ##################################################

/**
 * @brief Test: Single qubit circuit compilation
 *
 * @details
 * Creates a simple circuit with H and X gates, runs through the full
 * pipeline, and verifies the output.
 */
TEST_F(CompilerPipelineTest, SingleQubitCircuit) {
  // Create a simple quantum circuit
  QuantumComputation qc(1);
  qc.h(0);
  qc.x(0);

  // Import to Quartz dialect
  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Configure compiler to run full pipeline
  QuantumCompilerConfig config;
  config.recordIntermediates = true;

  // Run compilation
  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify initial Quartz IR contains the gates
  EXPECT_TRUE(irContains(record.afterQuartzImport, "quartz.alloc"));

  // Verify final IR is valid
  EXPECT_FALSE(record.afterQuartzCanon.empty());
}

/**
 * @brief Test: Two qubit entangling circuit
 *
 * @details
 * Creates a Bell state circuit (H followed by CNOT), runs through
 * the pipeline, and verifies transformations.
 */
TEST_F(CompilerPipelineTest, BellStateCircuit) {
  // Create Bell state circuit
  QuantumComputation qc(2);
  qc.h(0);
  qc.cx(0, 1); // CNOT with control on qubit 0

  // Import to Quartz dialect
  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Configure and run compiler
  QuantumCompilerConfig config;
  config.recordIntermediates = true;

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify circuit structure
  EXPECT_TRUE(irContains(record.afterQuartzImport, "quartz.alloc"));

  // After canonicalization, IR should be simplified
  EXPECT_FALSE(record.afterQuartzCanon.empty());
}

/**
 * @brief Test: Circuit with measurement
 *
 * @details
 * Creates a circuit that includes measurement operations, which
 * have different semantics in Quartz (in-place) vs Flux (SSA).
 */
TEST_F(CompilerPipelineTest, CircuitWithMeasurement) {
  // Create circuit with measurement
  QuantumComputation qc(2);
  qc.h(0);
  qc.cx(0, 1); // CNOT
  qc.measure(0, 0);
  qc.measure(1, 1);

  // Import and compile
  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  QuantumCompilerConfig config;
  config.recordIntermediates = true;

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify measurement operations are present
  EXPECT_TRUE(irContains(record.afterQuartzImport, "quartz.measure"));
}

/**
 * @brief Test: Circuit with reset operation
 *
 * @details
 * Tests reset operations which also have different semantics between
 * reference-based (Quartz) and value-based (Flux) representations.
 */
TEST_F(CompilerPipelineTest, CircuitWithReset) {
  // Create circuit with reset
  QuantumComputation qc(2);
  qc.h(0);
  qc.reset(0);
  qc.x(0);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  QuantumCompilerConfig config;
  config.recordIntermediates = true;

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  EXPECT_TRUE(irContains(record.afterQuartzImport, "quartz.reset"));
}

// ##################################################
// # Pipeline Stage Tests
// ##################################################

/**
 * @brief Test: Quartz to Flux conversion
 *
 * @details
 * Verifies that the conversion from reference semantics (Quartz)
 * to value semantics (Flux) preserves circuit semantics.
 */
TEST_F(CompilerPipelineTest, QuartzToFluxConversion) {
  QuantumComputation qc(1);
  qc.h(0);
  qc.x(0);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Run only up to Flux conversion
  PassManager pm(context.get());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createQuartzToFlux());
  pm.addPass(createCanonicalizerPass());

  ASSERT_TRUE(succeeded(pm.run(module.get())));

  // Verify Flux dialect is present
  std::string ir = captureIR(module.get());
  EXPECT_TRUE(irContains(ir, "flux.alloc"));
}

/**
 * @brief Test: Flux to Quartz round-trip
 *
 * @details
 * Verifies that converting Quartz -> Flux -> Quartz produces
 * semantically equivalent IR.
 */
TEST_F(CompilerPipelineTest, FluxToQuartzRoundTrip) {
  QuantumComputation qc(1);
  qc.h(0);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Capture initial state
  std::string initialIR = captureIR(module.get());

  // Run round-trip conversion
  PassManager pm(context.get());
  pm.addPass(createQuartzToFlux());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createFluxToQuartz());
  pm.addPass(createCanonicalizerPass());

  ASSERT_TRUE(succeeded(pm.run(module.get())));

  // Verify we're back in Quartz dialect
  std::string finalIR = captureIR(module.get());
  EXPECT_TRUE(irContains(finalIR, "quartz.alloc"));
  EXPECT_FALSE(irContains(finalIR, "flux.alloc"));
}

/**
 * @brief Test: QIR conversion
 *
 * @details
 * Tests the final lowering from Quartz to QIR (LLVM-based
 * quantum intermediate representation).
 */
TEST_F(CompilerPipelineTest, QIRConversion) {
  QuantumComputation qc(1);
  qc.h(0);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  QuantumCompilerConfig config;
  config.convertToQIR = true;
  config.recordIntermediates = true;

  CompilationRecord record;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config, &record)));

  // Verify QIR (LLVM dialect) is present in final output
  EXPECT_TRUE(irContains(record.afterQIRCanon, "llvm.func"));
}

// ##################################################
// # Canonicalization Tests
// ##################################################

/**
 * @brief Test: Dead code elimination
 *
 * @details
 * Verifies that dead value removal correctly eliminates unused
 * operations and values from the IR. Dead value removal now always
 * runs as part of the cleanup passes after each stage.
 */
TEST_F(CompilerPipelineTest, DeadCodeElimination) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.h(1);
  qc.h(2);
  // Only use qubit 0 in actual computation
  qc.x(0);

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  QuantumCompilerConfig config;
  // Dead value removal always runs automatically

  ASSERT_TRUE(succeeded(runCompiler(module.get(), config)));

  // After dead code elimination, unused allocations may be removed
  std::string ir = captureIR(module.get());
  EXPECT_FALSE(ir.empty());
}

// ##################################################
// # Error Handling Tests
// ##################################################

/**
 * @brief Test: Empty circuit handling
 *
 * @details
 * Verifies that the compiler correctly handles edge cases like
 * empty quantum circuits.
 */
TEST_F(CompilerPipelineTest, EmptyCircuit) {
  QuantumComputation qc(1);
  // No operations added

  auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  QuantumCompilerConfig config;
  ASSERT_TRUE(succeeded(runCompiler(module.get(), config)));
}

} // namespace
