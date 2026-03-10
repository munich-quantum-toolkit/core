/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "ir/QuantumComputation.hpp"
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"
#include "qir_programs.h"
#include "quantum_computation_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

#include <cstdlib>
#include <iosfwd>
#include <memory>
#include <string>

namespace mqt::test::compiler {

using QCProgramBuilderFn = NamedBuilder<mlir::qc::QCProgramBuilder>;
using QIRProgramBuilderFn = NamedBuilder<mlir::qir::QIRProgramBuilder>;
using QuantumComputationBuilderFn = NamedBuilder<::qc::QuantumComputation>;

struct CompilerPipelineTestCase {
  std::string name;
  QuantumComputationBuilderFn quantumComputationBuilder;
  QCProgramBuilderFn qcProgramBuilder;
  QCProgramBuilderFn qcReferenceBuilder;
  QIRProgramBuilderFn qirReferenceBuilder;
  bool startFromQuantumComputation = true;
  bool convertToQIR = true;

  friend std::ostream& operator<<(std::ostream& os,
                                  const CompilerPipelineTestCase& info) {
    os << "CompilerPipeline{" << info.name << ", original=";
    if (info.startFromQuantumComputation) {
      os << displayName(info.quantumComputationBuilder.name);
    } else {
      os << displayName(info.qcProgramBuilder.name);
    }
    os << ", qcReference=" << displayName(info.qcReferenceBuilder.name);
    if (info.convertToQIR) {
      os << ", qirReference=" << displayName(info.qirReferenceBuilder.name);
    }
    return os << "}";
  }
};

class CompilerPipelineTest
    : public testing::TestWithParam<CompilerPipelineTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                    mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::func::FuncDialect, mlir::scf::SCFDialect,
                    mlir::LLVM::LLVMDialect>();
    context = std::make_unique<mlir::MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
  buildQCReference(const QCProgramBuilderFn builder) const {
    auto module = mlir::qc::QCProgramBuilder::build(context.get(), builder.fn);
    runCanonicalizationPasses(module.get());
    return module;
  }

  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
  buildQIRReference(const QIRProgramBuilderFn builder) const {
    auto module =
        mlir::qir::QIRProgramBuilder::build(context.get(), builder.fn);
    runCanonicalizationPasses(module.get());
    return module;
  }

  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
  parseRecordedModule(const std::string& ir) const {
    return mlir::parseSourceString<mlir::ModuleOp>(ir, context.get());
  }

  static void runPipeline(const mlir::ModuleOp module, const bool convertToQIR,
                          mlir::CompilationRecord& record) {
    mlir::QuantumCompilerConfig config;
    config.convertToQIR = convertToQIR;
    config.recordIntermediates = true;
    config.printIRAfterAllStages = true;

    mlir::QuantumCompilerPipeline pipeline(config);
    ASSERT_TRUE(pipeline.runPipeline(module, &record).succeeded());
  }

  void expectEquivalent(const std::string& stage, const std::string& ir,
                        const mlir::ModuleOp expected) const {
    auto actual = parseRecordedModule(ir);
    ASSERT_TRUE(actual) << stage << " failed to parse";
    EXPECT_TRUE(mlir::verify(*actual).succeeded());
    EXPECT_TRUE(mlir::verify(expected).succeeded());
    EXPECT_TRUE(areModulesEquivalentWithPermutations(actual.get(), expected));
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

// ##################################################
// # Rotation Merge Pass Test
// ##################################################

/**
 * @brief Test: Rotation merging pass is invoked during optimization stage
 *
 * @details
 * The merged U gate parameters are computed via floating-point arithmetic
 * that is not bit-identical across platforms, so we cannot use
 * verifyAllStages with hardcoded expected values. Instead, we compare
 * the optimization output with and without the pass enabled.
 * Correctness of the pass is tested in a dedicated test.
 */
TEST_F(CompilerPipelineTest, RotationGateMergingPass) {
  ::qc::QuantumComputation comp;
  comp.addQubitRegister(1, "q");
  comp.rz(1.0, 0);
  comp.rx(1.0, 0);

  // Run with merging enabled
  config.mergeRotationGates = true;

  auto module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());
  const auto withMerging = record.afterOptimization;

  // Run with merging disabled
  config.mergeRotationGates = false;
  record = {};

  module = importQuantumCircuit(comp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());
  const auto withoutMerging = record.afterOptimization;

  // The outputs must differ, proving the pass ran and transformed the IR
  EXPECT_NE(withMerging, withoutMerging);
}
} // namespace
TEST_P(CompilerPipelineTest, EndToEndPipeline) {
  const auto& testCase = GetParam();
  const auto name = " (" + testCase.name + ")";
  DeferredPrinter printer;

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (testCase.startFromQuantumComputation) {
    ASSERT_TRUE(testCase.quantumComputationBuilder);
    ::qc::QuantumComputation comp;
    testCase.quantumComputationBuilder.fn(comp);

    module = mlir::translateQuantumComputationToQC(context.get(), comp);
    ASSERT_TRUE(module);
    printer.record(module.get(), "QC Import" + name);
  } else {
    ASSERT_TRUE(testCase.qcProgramBuilder);
    module = mlir::qc::QCProgramBuilder::build(context.get(),
                                               testCase.qcProgramBuilder.fn);
    ASSERT_TRUE(module);
    printer.record(module.get(), "QC Input" + name);
  }
  EXPECT_TRUE(mlir::verify(*module).succeeded());

  mlir::CompilationRecord record;
  runPipeline(module.get(), testCase.convertToQIR, record);

  ASSERT_TRUE(testCase.qcReferenceBuilder);
  auto qcReference = buildQCReference(testCase.qcReferenceBuilder);
  ASSERT_TRUE(qcReference);
  printer.record(qcReference.get(), "Reference QC IR" + name);

  expectEquivalent("Final QC", record.afterQCCanon, qcReference.get());
  auto finalQC = parseRecordedModule(record.afterQCCanon);
  ASSERT_TRUE(finalQC);
  printer.record(finalQC.get(), "Final QC IR" + name);

  if (testCase.convertToQIR) {
    ASSERT_TRUE(testCase.qirReferenceBuilder);

    auto qirReference = buildQIRReference(testCase.qirReferenceBuilder);
    ASSERT_TRUE(qirReference);
    printer.record(qirReference.get(), "Reference QIR IR" + name);

    expectEquivalent("Final QIR", record.afterQIRCanon, qirReference.get());
    auto finalQIR = parseRecordedModule(record.afterQIRCanon);
    ASSERT_TRUE(finalQIR);
    printer.record(finalQIR.get(), "Final QIR IR" + name);
  }
}

INSTANTIATE_TEST_SUITE_P(
    QuantumComputationPipelineProgramsTest, CompilerPipelineTest,
    testing::Values(
        CompilerPipelineTestCase{
            "StaticQubits", nullptr, MQT_NAMED_BUILDER(mlir::qc::staticQubits),
            MQT_NAMED_BUILDER(mlir::qc::staticQubits),
            MQT_NAMED_BUILDER(mlir::qir::staticQubits), false},
        CompilerPipelineTestCase{"AllocQubit",
                                 MQT_NAMED_BUILDER(qc::allocQubit), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::allocQubit),
                                 MQT_NAMED_BUILDER(mlir::qir::allocQubit)},
        CompilerPipelineTestCase{
            "AllocQubitRegister", MQT_NAMED_BUILDER(qc::allocQubitRegister),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::allocQubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::allocQubitRegister)},
        CompilerPipelineTestCase{
            "AllocMultipleQubitRegisters",
            MQT_NAMED_BUILDER(qc::allocMultipleQubitRegisters), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::allocMultipleQubitRegisters),
            MQT_NAMED_BUILDER(mlir::qir::allocMultipleQubitRegisters)},
        CompilerPipelineTestCase{
            "AllocLargeRegister", MQT_NAMED_BUILDER(qc::allocLargeRegister),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::allocLargeRegister),
            MQT_NAMED_BUILDER(mlir::qir::allocLargeRegister)},
        CompilerPipelineTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(mlir::qir::singleMeasurementToSingleBit)},
        CompilerPipelineTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(mlir::qir::repeatedMeasurementToSameBit)},
        CompilerPipelineTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(mlir::qir::repeatedMeasurementToDifferentBits)},
        CompilerPipelineTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements),
            nullptr,
            MQT_NAMED_BUILDER(
                mlir::qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(
                mlir::qir::multipleClassicalRegistersAndMeasurements)},
        CompilerPipelineTestCase{
            "ResetQubitAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::resetQubitAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qir::resetQubitAfterSingleOp)},
        CompilerPipelineTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qir::resetMultipleQubitsAfterSingleOp)},
        CompilerPipelineTestCase{
            "RepeatedResetAfterSingleOp",
            MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::resetQubitAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qir::resetQubitAfterSingleOp)},
        CompilerPipelineTestCase{"GlobalPhase",
                                 MQT_NAMED_BUILDER(qc::globalPhase), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::globalPhase),
                                 MQT_NAMED_BUILDER(mlir::qir::globalPhase)},
        CompilerPipelineTestCase{"Identity", MQT_NAMED_BUILDER(qc::identity),
                                 nullptr, MQT_NAMED_BUILDER(mlir::qc::emptyQC),
                                 MQT_NAMED_BUILDER(mlir::qir::emptyQIR)},
        CompilerPipelineTestCase{
            "SingleControlledIdentity",
            MQT_NAMED_BUILDER(qc::singleControlledIdentity), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::emptyQC),
            MQT_NAMED_BUILDER(mlir::qir::emptyQIR)},
        CompilerPipelineTestCase{
            "MultipleControlledIdentity",
            MQT_NAMED_BUILDER(qc::multipleControlledIdentity), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::emptyQC),
            MQT_NAMED_BUILDER(mlir::qir::emptyQIR)},
        CompilerPipelineTestCase{"X", MQT_NAMED_BUILDER(qc::x), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::x),
                                 MQT_NAMED_BUILDER(mlir::qir::x)},
        CompilerPipelineTestCase{
            "SingleControlledX", MQT_NAMED_BUILDER(qc::singleControlledX),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledX),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledX)},
        CompilerPipelineTestCase{
            "MultipleControlledX", MQT_NAMED_BUILDER(qc::multipleControlledX),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledX),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledX)},
        CompilerPipelineTestCase{"Y", MQT_NAMED_BUILDER(qc::y), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::y),
                                 MQT_NAMED_BUILDER(mlir::qir::y)},
        CompilerPipelineTestCase{
            "SingleControlledY", MQT_NAMED_BUILDER(qc::singleControlledY),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledY),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledY)},
        CompilerPipelineTestCase{
            "MultipleControlledY", MQT_NAMED_BUILDER(qc::multipleControlledY),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledY),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledY)},
        CompilerPipelineTestCase{"Z", MQT_NAMED_BUILDER(qc::z), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::z),
                                 MQT_NAMED_BUILDER(mlir::qir::z)},
        CompilerPipelineTestCase{
            "SingleControlledZ", MQT_NAMED_BUILDER(qc::singleControlledZ),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledZ),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledZ)},
        CompilerPipelineTestCase{
            "MultipleControlledZ", MQT_NAMED_BUILDER(qc::multipleControlledZ),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledZ),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledZ)},
        CompilerPipelineTestCase{"H", MQT_NAMED_BUILDER(qc::h), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::h),
                                 MQT_NAMED_BUILDER(mlir::qir::h)},
        CompilerPipelineTestCase{
            "SingleControlledH", MQT_NAMED_BUILDER(qc::singleControlledH),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledH),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledH)},
        CompilerPipelineTestCase{
            "MultipleControlledH", MQT_NAMED_BUILDER(qc::multipleControlledH),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledH),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledH)},
        CompilerPipelineTestCase{"S", MQT_NAMED_BUILDER(qc::s), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::s),
                                 MQT_NAMED_BUILDER(mlir::qir::s)},
        CompilerPipelineTestCase{
            "SingleControlledS", MQT_NAMED_BUILDER(qc::singleControlledS),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledS),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledS)},
        CompilerPipelineTestCase{
            "MultipleControlledS", MQT_NAMED_BUILDER(qc::multipleControlledS),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledS),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledS)},
        CompilerPipelineTestCase{"Sdg", MQT_NAMED_BUILDER(qc::sdg), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::sdg),
                                 MQT_NAMED_BUILDER(mlir::qir::sdg)},
        CompilerPipelineTestCase{
            "SingleControlledSdg", MQT_NAMED_BUILDER(qc::singleControlledSdg),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledSdg),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSdg)},
        CompilerPipelineTestCase{
            "MultipleControlledSdg",
            MQT_NAMED_BUILDER(qc::multipleControlledSdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSdg),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSdg)},
        CompilerPipelineTestCase{"T", MQT_NAMED_BUILDER(qc::t_), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::t_),
                                 MQT_NAMED_BUILDER(mlir::qir::t_)},
        CompilerPipelineTestCase{
            "SingleControlledT", MQT_NAMED_BUILDER(qc::singleControlledT),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledT),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledT)},
        CompilerPipelineTestCase{
            "MultipleControlledT", MQT_NAMED_BUILDER(qc::multipleControlledT),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledT),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledT)},
        CompilerPipelineTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::tdg),
                                 MQT_NAMED_BUILDER(mlir::qir::tdg)},
        CompilerPipelineTestCase{
            "SingleControlledTdg", MQT_NAMED_BUILDER(qc::singleControlledTdg),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledTdg),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledTdg)},
        CompilerPipelineTestCase{
            "MultipleControlledTdg",
            MQT_NAMED_BUILDER(qc::multipleControlledTdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledTdg),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledTdg)},
        CompilerPipelineTestCase{"SX", MQT_NAMED_BUILDER(qc::sx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::sx),
                                 MQT_NAMED_BUILDER(mlir::qir::sx)},
        CompilerPipelineTestCase{
            "SingleControlledSX", MQT_NAMED_BUILDER(qc::singleControlledSx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledSx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSx)},
        CompilerPipelineTestCase{
            "MultipleControlledSX", MQT_NAMED_BUILDER(qc::multipleControlledSx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledSx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSx)},
        CompilerPipelineTestCase{"SXdg", MQT_NAMED_BUILDER(qc::sxdg), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::sxdg),
                                 MQT_NAMED_BUILDER(mlir::qir::sxdg)},
        CompilerPipelineTestCase{
            "SingleControlledSXdg", MQT_NAMED_BUILDER(qc::singleControlledSxdg),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledSxdg),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSxdg)},
        CompilerPipelineTestCase{
            "MultipleControlledSXdg",
            MQT_NAMED_BUILDER(qc::multipleControlledSxdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSxdg),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSxdg)},
        CompilerPipelineTestCase{"RX", MQT_NAMED_BUILDER(qc::rx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rx),
                                 MQT_NAMED_BUILDER(mlir::qir::rx)},
        CompilerPipelineTestCase{
            "SingleControlledRX", MQT_NAMED_BUILDER(qc::singleControlledRx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRx)},
        CompilerPipelineTestCase{
            "MultipleControlledRX", MQT_NAMED_BUILDER(qc::multipleControlledRx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledRx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRx)},
        CompilerPipelineTestCase{"RY", MQT_NAMED_BUILDER(qc::ry), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::ry),
                                 MQT_NAMED_BUILDER(mlir::qir::ry)},
        CompilerPipelineTestCase{
            "SingleControlledRY", MQT_NAMED_BUILDER(qc::singleControlledRy),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRy),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRy)},
        CompilerPipelineTestCase{
            "MultipleControlledRY", MQT_NAMED_BUILDER(qc::multipleControlledRy),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledRy),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRy)},
        CompilerPipelineTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rz),
                                 MQT_NAMED_BUILDER(mlir::qir::rz)},
        CompilerPipelineTestCase{
            "SingleControlledRZ", MQT_NAMED_BUILDER(qc::singleControlledRz),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRz),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRz)},
        CompilerPipelineTestCase{
            "MultipleControlledRZ", MQT_NAMED_BUILDER(qc::multipleControlledRz),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledRz),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRz)},
        CompilerPipelineTestCase{"P", MQT_NAMED_BUILDER(qc::p), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::p),
                                 MQT_NAMED_BUILDER(mlir::qir::p)},
        CompilerPipelineTestCase{
            "SingleControlledP", MQT_NAMED_BUILDER(qc::singleControlledP),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledP),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledP)},
        CompilerPipelineTestCase{
            "MultipleControlledP", MQT_NAMED_BUILDER(qc::multipleControlledP),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledP),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledP)},
        CompilerPipelineTestCase{"R", MQT_NAMED_BUILDER(qc::r), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::r),
                                 MQT_NAMED_BUILDER(mlir::qir::r)},
        CompilerPipelineTestCase{
            "SingleControlledR",
            MQT_NAMED_BUILDER(qc::singleControlledR), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledR),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledR)},
        CompilerPipelineTestCase{
            "MultipleControlledR", MQT_NAMED_BUILDER(qc::multipleControlledR),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledR),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledR)},
        CompilerPipelineTestCase{"U2", MQT_NAMED_BUILDER(qc::u2), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::u2),
                                 MQT_NAMED_BUILDER(mlir::qir::u2)},
        CompilerPipelineTestCase{
            "SingleControlledU2", MQT_NAMED_BUILDER(qc::singleControlledU2),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledU2),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledU2)},
        CompilerPipelineTestCase{
            "MultipleControlledU2", MQT_NAMED_BUILDER(qc::multipleControlledU2),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledU2),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledU2)},
        CompilerPipelineTestCase{"U", MQT_NAMED_BUILDER(qc::u), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::u),
                                 MQT_NAMED_BUILDER(mlir::qir::u)},
        CompilerPipelineTestCase{
            "SingleControlledU",
            MQT_NAMED_BUILDER(qc::singleControlledU), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledU),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledU)},
        CompilerPipelineTestCase{
            "MultipleControlledU", MQT_NAMED_BUILDER(qc::multipleControlledU),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledU),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledU)},
        CompilerPipelineTestCase{"SWAP", MQT_NAMED_BUILDER(qc::swap), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::swap),
                                 MQT_NAMED_BUILDER(mlir::qir::swap)},
        CompilerPipelineTestCase{
            "SingleControlledSWAP", MQT_NAMED_BUILDER(qc::singleControlledSwap),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledSwap),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSwap)},
        CompilerPipelineTestCase{
            "MultipleControlledSWAP",
            MQT_NAMED_BUILDER(qc::multipleControlledSwap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSwap),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSwap)},
        CompilerPipelineTestCase{"iSWAP", MQT_NAMED_BUILDER(qc::iswap), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::iswap),
                                 MQT_NAMED_BUILDER(mlir::qir::iswap)},
        CompilerPipelineTestCase{
            "SingleControllediSWAP",
            MQT_NAMED_BUILDER(qc::singleControlledIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledIswap),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledIswap)},
        CompilerPipelineTestCase{
            "MultipleControllediSWAP",
            MQT_NAMED_BUILDER(qc::multipleControlledIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledIswap),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledIswap)},
        CompilerPipelineTestCase{
            "InverseISWAP", MQT_NAMED_BUILDER(qc::inverseIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::inverseIswap), nullptr, true, false},
        CompilerPipelineTestCase{
            "InverseMultiControlledISWAP",
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::inverseMultipleControlledIswap),
            nullptr, true, false},
        CompilerPipelineTestCase{"DCX", MQT_NAMED_BUILDER(qc::dcx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::dcx),
                                 MQT_NAMED_BUILDER(mlir::qir::dcx)},
        CompilerPipelineTestCase{
            "SingleControlledDCX", MQT_NAMED_BUILDER(qc::singleControlledDcx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledDcx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledDcx)},
        CompilerPipelineTestCase{
            "MultipleControlledDCX",
            MQT_NAMED_BUILDER(qc::multipleControlledDcx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledDcx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledDcx)},
        CompilerPipelineTestCase{"ECR", MQT_NAMED_BUILDER(qc::ecr), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::ecr),
                                 MQT_NAMED_BUILDER(mlir::qir::ecr)},
        CompilerPipelineTestCase{
            "SingleControlledECR", MQT_NAMED_BUILDER(qc::singleControlledEcr),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledEcr),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledEcr)},
        CompilerPipelineTestCase{
            "MultipleControlledECR",
            MQT_NAMED_BUILDER(qc::multipleControlledEcr), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledEcr),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledEcr)},
        CompilerPipelineTestCase{"RXX", MQT_NAMED_BUILDER(qc::rxx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rxx),
                                 MQT_NAMED_BUILDER(mlir::qir::rxx)},
        CompilerPipelineTestCase{
            "SingleControlledRXX", MQT_NAMED_BUILDER(qc::singleControlledRxx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRxx)},
        CompilerPipelineTestCase{
            "MultipleControlledRXX",
            MQT_NAMED_BUILDER(qc::multipleControlledRxx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRxx)},
        CompilerPipelineTestCase{
            "TripleControlledRXX", MQT_NAMED_BUILDER(qc::tripleControlledRxx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::tripleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qir::tripleControlledRxx)},
        CompilerPipelineTestCase{"RYY", MQT_NAMED_BUILDER(qc::ryy), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::ryy),
                                 MQT_NAMED_BUILDER(mlir::qir::ryy)},
        CompilerPipelineTestCase{
            "SingleControlledRYY", MQT_NAMED_BUILDER(qc::singleControlledRyy),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRyy),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRyy)},
        CompilerPipelineTestCase{
            "MultipleControlledRYY",
            MQT_NAMED_BUILDER(qc::multipleControlledRyy), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRyy),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRyy)},
        CompilerPipelineTestCase{"RZX", MQT_NAMED_BUILDER(qc::rzx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rzx),
                                 MQT_NAMED_BUILDER(mlir::qir::rzx)},
        CompilerPipelineTestCase{
            "SingleControlledRZX", MQT_NAMED_BUILDER(qc::singleControlledRzx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRzx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRzx)},
        CompilerPipelineTestCase{
            "MultipleControlledRZX",
            MQT_NAMED_BUILDER(qc::multipleControlledRzx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRzx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRzx)},
        CompilerPipelineTestCase{"RZZ", MQT_NAMED_BUILDER(qc::rzz), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rzz),
                                 MQT_NAMED_BUILDER(mlir::qir::rzz)},
        CompilerPipelineTestCase{
            "SingleControlledRZZ", MQT_NAMED_BUILDER(qc::singleControlledRzz),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRzz),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRzz)},
        CompilerPipelineTestCase{
            "MultipleControlledRZZ",
            MQT_NAMED_BUILDER(qc::multipleControlledRzz), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRzz),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRzz)},
        CompilerPipelineTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
                                 nullptr, MQT_NAMED_BUILDER(mlir::qc::xxPlusYY),
                                 MQT_NAMED_BUILDER(mlir::qir::xxPlusYY)},
        CompilerPipelineTestCase{
            "SingleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXxPlusYY),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledXxPlusYY)},
        CompilerPipelineTestCase{
            "MultipleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledXxPlusYY),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledXxPlusYY)},
        CompilerPipelineTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qc::xxMinusYY),
                                 nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::xxMinusYY),
                                 MQT_NAMED_BUILDER(mlir::qir::xxMinusYY)},
        CompilerPipelineTestCase{
            "SingleControlledXXMinusYY",
            MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXxMinusYY),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledXxMinusYY)},
        CompilerPipelineTestCase{
            "MultipleControlledXXMinusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledXxMinusYY),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledXxMinusYY)}));

} // namespace mqt::test::compiler
