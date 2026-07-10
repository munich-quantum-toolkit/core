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
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
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
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <cstdlib>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>

namespace mqt::test::compiler {

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::qco;
using namespace mlir::qir;

using QCProgramBuilderFn = NamedMLIRBuilder<QCProgramBuilder>;
using QIRProgramBuilderFn = NamedMLIRBuilder<QIRProgramBuilder>;
using QuantumComputationBuilderFn = NamedBuilder<::qc::QuantumComputation>;

namespace {

struct CompilerPipelineTestCase {
  std::string name;
  QuantumComputationBuilderFn quantumComputationBuilder;
  QCProgramBuilderFn qcProgramBuilder;
  QCProgramBuilderFn qcReferenceBuilder;
  QIRProgramBuilderFn qirReferenceBuilder;
  bool startFromQuantumComputation = true;
  bool convertToQIR = true;

  friend std::ostream& operator<<(std::ostream& os,
                                  const CompilerPipelineTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
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

class CompilerPipelineTest
    : public testing::TestWithParam<CompilerPipelineTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<QCDialect, QCODialect, qtensor::QTensorDialect,
                arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect, LLVM::LLVMDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  buildQCReference(const QCProgramBuilderFn builder) const {
    auto module = mqt::test::buildMLIRProgram(context.get(), builder);
    EXPECT_TRUE(runQCCleanupPipeline(module.get()).succeeded());
    return module;
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  buildQIRReference(const QIRProgramBuilderFn builder) const {
    auto module = mqt::test::buildMLIRProgram(
        context.get(), builder, QIRProgramBuilder::Profile::Adaptive);
    EXPECT_TRUE(runQIRCleanupPipeline(module.get(), true).succeeded());
    return module;
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  parseRecordedModule(const std::string& ir) const {
    return parseSourceString<ModuleOp>(ir, context.get());
  }

  static void runPipeline(const ModuleOp module, const bool convertToQIR,
                          const bool disableMergeSingleQubitRotationGates,
                          const bool enableHadamardLifting,
                          CompilationRecord& record) {
    QuantumCompilerConfig config;
    config.convertToQIRAdaptive = convertToQIR;
    config.disableMergeSingleQubitRotationGates =
        disableMergeSingleQubitRotationGates;
    config.enableHadamardLifting = enableHadamardLifting;
    config.recordIntermediates = true;
    config.printIRAfterAllStages = true;

    QuantumCompilerPipeline pipeline(config);
    ASSERT_TRUE(pipeline.runPipeline(module, &record).succeeded());
  }

  void expectEquivalent(const std::string& stage, const std::string& ir,
                        const ModuleOp expected) const {
    auto actual = parseRecordedModule(ir);
    ASSERT_TRUE(actual) << stage << " failed to parse";
    EXPECT_TRUE(verify(*actual).succeeded());
    EXPECT_TRUE(verify(expected).succeeded());
    EXPECT_TRUE(areModulesEquivalentWithPermutations(actual.get(), expected));
  }
};

} // namespace

TEST_P(CompilerPipelineTest, EndToEndPipeline) {
  const auto& testCase = GetParam();
  const auto name = " (" + testCase.name + ")";
  DeferredPrinter printer;

  OwningOpRef<ModuleOp> module;
  if (testCase.startFromQuantumComputation) {
    ASSERT_TRUE(testCase.quantumComputationBuilder);
    ::qc::QuantumComputation comp;
    testCase.quantumComputationBuilder.fn(comp);

    module = translateQuantumComputationToQC(context.get(), comp);
    ASSERT_TRUE(module);
    printer.record(module.get(), "QC Import" + name);
  } else {
    ASSERT_TRUE(testCase.qcProgramBuilder);
    module =
        mqt::test::buildMLIRProgram(context.get(), testCase.qcProgramBuilder);
    ASSERT_TRUE(module);
    printer.record(module.get(), "QC Input" + name);
  }
  EXPECT_TRUE(verify(*module).succeeded());

  CompilationRecord record;
  runPipeline(module.get(), testCase.convertToQIR, false, false, record);

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

/**
 * @brief Test: Rotation merging pass is invoked during the optimization stage
 *
 * @details
 * The merged U gate parameters are computed via floating-point arithmetic
 * that is not bit-identical across platforms, so we cannot use
 * verifyAllStages with hardcoded expected values. Instead, we run the
 * pipeline once with the pass enabled and compare afterQCOCanon against
 * afterOptimization to verify the pass transformed the IR.
 * Correctness of the pass is tested in a dedicated test.
 */
TEST_F(CompilerPipelineTest, RotationGateMergingPass) {
  auto module =
      QCProgramBuilder::build(context.get(), [&](QCProgramBuilder& b) {
        auto q = b.allocQubit();
        b.rz(1.0, q);
        b.rx(1.0, q);
        return b.measure(q);
      });
  ASSERT_TRUE(module);

  CompilationRecord record;
  runPipeline(module.get(), false, false, false, record);

  // The outputs must differ, proving the pass ran and transformed the IR
  EXPECT_NE(record.afterQCOCanon, record.afterOptimization);
}

/**
 * @brief Test: Hadamard lifting pass is invoked during the optimization stage
 *
 * We run the pipeline with enabled Hadamard lifting and check whether the
 * outputs differ, i.e. that the pipeline ran and changed the IR.
 * Correctness of the pass is tested in a dedicated test.
 */
TEST_F(CompilerPipelineTest, HadamardLiftingPass) {
  auto module =
      QCProgramBuilder::build(context.get(), [&](QCProgramBuilder& b) {
        auto q = b.allocQubit();
        b.x(q);
        b.h(q);
        return b.measure(q);
      });
  ASSERT_TRUE(module);

  CompilationRecord record;
  runPipeline(module.get(), false, true, true, record);

  // The outputs must differ, proving the pass ran and transformed the IR
  EXPECT_NE(record.afterQCOCanon, record.afterOptimization);
}

INSTANTIATE_TEST_SUITE_P(
    QuantumComputationPipelineProgramsTest, CompilerPipelineTest,
    testing::Values(
        CompilerPipelineTestCase{
            "StaticQubits", nullptr, MQT_NAMED_BUILDER(mlir::qc::staticQubits),
            MQT_NAMED_BUILDER(mlir::qc::staticQubits),
            MQT_NAMED_BUILDER(mlir::qir::staticQubits), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithOps", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithOps),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithOps),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithOps), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithParametricOps", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithParametricOps),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithParametricOps),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithParametricOps), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithTwoTargetOps", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithTwoTargetOps),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithTwoTargetOps),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithTwoTargetOps), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithCtrl", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithCtrl),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithCtrl),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithCtrl), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithInv", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithInv),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithInv),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithInv), false},
        CompilerPipelineTestCase{
            "AllocQubit", MQT_NAMED_BUILDER(::qc::allocQubit), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::alloc1QubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::alloc1QubitRegister<true>)},
        CompilerPipelineTestCase{
            "AllocQubitRegister", MQT_NAMED_BUILDER(::qc::allocQubitRegister),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::allocQubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::allocQubitRegister<true>)},
        CompilerPipelineTestCase{
            "AllocMultipleQubitRegisters",
            MQT_NAMED_BUILDER(::qc::allocMultipleQubitRegisters), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::allocMultipleQubitRegisters),
            MQT_NAMED_BUILDER(mlir::qir::allocMultipleQubitRegisters<true>)},
        CompilerPipelineTestCase{
            "AllocLargeRegister", MQT_NAMED_BUILDER(::qc::allocLargeRegister),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::allocLargeRegister),
            MQT_NAMED_BUILDER(mlir::qir::allocQubitRegister<true>)},
        CompilerPipelineTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(::qc::singleMeasurementToSingleBit), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(mlir::qir::singleMeasurementToSingleBit<true>)},
        CompilerPipelineTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(::qc::repeatedMeasurementToSameBit), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(mlir::qir::repeatedMeasurementToSameBit<true>)},
        CompilerPipelineTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(::qc::repeatedMeasurementToDifferentBits),
            nullptr,
            MQT_NAMED_BUILDER(mlir::qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(
                mlir::qir::repeatedMeasurementToDifferentBits<true>)},
        CompilerPipelineTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(::qc::multipleClassicalRegistersAndMeasurements),
            nullptr,
            MQT_NAMED_BUILDER(
                mlir::qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(
                mlir::qir::multipleClassicalRegistersAndMeasurements<true>)},
        CompilerPipelineTestCase{
            "MeasurementWithoutRegisters", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::measurementWithoutRegisters),
            MQT_NAMED_BUILDER(mlir::qc::measurementWithoutRegisters),
            MQT_NAMED_BUILDER(mlir::qir::measurementWithoutRegisters<true>),
            false},
        CompilerPipelineTestCase{
            "ResetQubitAfterSingleOp",
            MQT_NAMED_BUILDER(::qc::resetQubitAfterSingleOp), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::resetQubitAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qir::resetQubitAfterSingleOp<true>)},
        CompilerPipelineTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(::qc::resetMultipleQubitsAfterSingleOp), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(
                mlir::qir::resetMultipleQubitsAfterSingleOp<true>)},
        CompilerPipelineTestCase{
            "RepeatedResetAfterSingleOp",
            MQT_NAMED_BUILDER(::qc::repeatedResetAfterSingleOp), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::resetQubitAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qir::resetQubitAfterSingleOp<true>)},
        CompilerPipelineTestCase{
            "GlobalPhase", MQT_NAMED_BUILDER(::qc::globalPhase), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::globalPhase),
            MQT_NAMED_BUILDER(mlir::qir::globalPhase<true>)},
        CompilerPipelineTestCase{
            "Identity", MQT_NAMED_BUILDER(::qc::identity), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::alloc1QubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::alloc1QubitRegister<true>)},
        CompilerPipelineTestCase{
            "SingleControlledIdentity",
            MQT_NAMED_BUILDER(::qc::singleControlledIdentity), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::allocQubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::allocQubitRegister<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledIdentity",
            MQT_NAMED_BUILDER(::qc::multipleControlledIdentity), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::alloc3QubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::alloc3QubitRegister<true>)},
        CompilerPipelineTestCase{"X", MQT_NAMED_BUILDER(::qc::x), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::x),
                                 MQT_NAMED_BUILDER(mlir::qir::x<true>)},
        CompilerPipelineTestCase{
            "SingleControlledX", MQT_NAMED_BUILDER(::qc::singleControlledX),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledX),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledX<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledX", MQT_NAMED_BUILDER(::qc::multipleControlledX),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledX),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledX<true>)},
        CompilerPipelineTestCase{"Y", MQT_NAMED_BUILDER(::qc::y), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::y),
                                 MQT_NAMED_BUILDER(mlir::qir::y<true>)},
        CompilerPipelineTestCase{
            "SingleControlledY", MQT_NAMED_BUILDER(::qc::singleControlledY),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledY),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledY<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledY", MQT_NAMED_BUILDER(::qc::multipleControlledY),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledY),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledY<true>)},
        CompilerPipelineTestCase{"Z", MQT_NAMED_BUILDER(::qc::z), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::z),
                                 MQT_NAMED_BUILDER(mlir::qir::z<true>)},
        CompilerPipelineTestCase{
            "SingleControlledZ", MQT_NAMED_BUILDER(::qc::singleControlledZ),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledZ),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledZ<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledZ", MQT_NAMED_BUILDER(::qc::multipleControlledZ),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledZ),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledZ<true>)},
        CompilerPipelineTestCase{"H", MQT_NAMED_BUILDER(::qc::h), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::h),
                                 MQT_NAMED_BUILDER(mlir::qir::h<true>)},
        CompilerPipelineTestCase{
            "SingleControlledH", MQT_NAMED_BUILDER(::qc::singleControlledH),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledH),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledH<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledH", MQT_NAMED_BUILDER(::qc::multipleControlledH),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledH),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledH<true>)},
        CompilerPipelineTestCase{
            "HWithoutRegister", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::hWithoutRegister),
            MQT_NAMED_BUILDER(mlir::qc::hWithoutRegister),
            MQT_NAMED_BUILDER(mlir::qir::hWithoutRegister<true>), false},
        CompilerPipelineTestCase{"S", MQT_NAMED_BUILDER(::qc::s), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::s),
                                 MQT_NAMED_BUILDER(mlir::qir::s<true>)},
        CompilerPipelineTestCase{
            "SingleControlledS", MQT_NAMED_BUILDER(::qc::singleControlledS),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledS),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledS<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledS", MQT_NAMED_BUILDER(::qc::multipleControlledS),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledS),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledS<true>)},
        CompilerPipelineTestCase{"Sdg", MQT_NAMED_BUILDER(::qc::sdg), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::sdg),
                                 MQT_NAMED_BUILDER(mlir::qir::sdg<true>)},
        CompilerPipelineTestCase{
            "SingleControlledSdg", MQT_NAMED_BUILDER(::qc::singleControlledSdg),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledSdg),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSdg<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledSdg",
            MQT_NAMED_BUILDER(::qc::multipleControlledSdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSdg),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSdg<true>)},
        CompilerPipelineTestCase{"T", MQT_NAMED_BUILDER(::qc::t_), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::t_),
                                 MQT_NAMED_BUILDER(mlir::qir::t_<true>)},
        CompilerPipelineTestCase{
            "SingleControlledT", MQT_NAMED_BUILDER(::qc::singleControlledT),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledT),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledT<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledT", MQT_NAMED_BUILDER(::qc::multipleControlledT),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledT),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledT<true>)},
        CompilerPipelineTestCase{"Tdg", MQT_NAMED_BUILDER(::qc::tdg), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::tdg),
                                 MQT_NAMED_BUILDER(mlir::qir::tdg<true>)},
        CompilerPipelineTestCase{
            "SingleControlledTdg", MQT_NAMED_BUILDER(::qc::singleControlledTdg),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledTdg),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledTdg<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledTdg",
            MQT_NAMED_BUILDER(::qc::multipleControlledTdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledTdg),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledTdg<true>)},
        CompilerPipelineTestCase{"SX", MQT_NAMED_BUILDER(::qc::sx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::sx),
                                 MQT_NAMED_BUILDER(mlir::qir::sx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledSX", MQT_NAMED_BUILDER(::qc::singleControlledSx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledSx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledSX",
            MQT_NAMED_BUILDER(::qc::multipleControlledSx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSx<true>)},
        CompilerPipelineTestCase{"SXdg", MQT_NAMED_BUILDER(::qc::sxdg), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::sxdg),
                                 MQT_NAMED_BUILDER(mlir::qir::sxdg<true>)},
        CompilerPipelineTestCase{
            "SingleControlledSXdg",
            MQT_NAMED_BUILDER(::qc::singleControlledSxdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledSxdg),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSxdg<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledSXdg",
            MQT_NAMED_BUILDER(::qc::multipleControlledSxdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSxdg),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSxdg<true>)},
        CompilerPipelineTestCase{"RX", MQT_NAMED_BUILDER(::qc::rx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rx),
                                 MQT_NAMED_BUILDER(mlir::qir::rx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRX", MQT_NAMED_BUILDER(::qc::singleControlledRx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRX",
            MQT_NAMED_BUILDER(::qc::multipleControlledRx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRx<true>)},
        CompilerPipelineTestCase{"RY", MQT_NAMED_BUILDER(::qc::ry), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::ry),
                                 MQT_NAMED_BUILDER(mlir::qir::ry<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRY", MQT_NAMED_BUILDER(::qc::singleControlledRy),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRy),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRy<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRY",
            MQT_NAMED_BUILDER(::qc::multipleControlledRy), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRy),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRy<true>)},
        CompilerPipelineTestCase{"RZ", MQT_NAMED_BUILDER(::qc::rz), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rz),
                                 MQT_NAMED_BUILDER(mlir::qir::rz<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRZ", MQT_NAMED_BUILDER(::qc::singleControlledRz),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRz),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRz<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRZ",
            MQT_NAMED_BUILDER(::qc::multipleControlledRz), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRz),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRz<true>)},
        CompilerPipelineTestCase{"P", MQT_NAMED_BUILDER(::qc::p), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::p),
                                 MQT_NAMED_BUILDER(mlir::qir::p<true>)},
        CompilerPipelineTestCase{
            "SingleControlledP",
            MQT_NAMED_BUILDER(::qc::singleControlledP), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledP),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledP<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledP", MQT_NAMED_BUILDER(::qc::multipleControlledP),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledP),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledP<true>)},
        CompilerPipelineTestCase{"R", MQT_NAMED_BUILDER(::qc::r), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::r),
                                 MQT_NAMED_BUILDER(mlir::qir::r<true>)},
        CompilerPipelineTestCase{
            "SingleControlledR",
            MQT_NAMED_BUILDER(::qc::singleControlledR), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledR),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledR<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledR", MQT_NAMED_BUILDER(::qc::multipleControlledR),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledR),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledR<true>)},
        CompilerPipelineTestCase{"U2", MQT_NAMED_BUILDER(::qc::u2), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::u2),
                                 MQT_NAMED_BUILDER(mlir::qir::u2<true>)},
        CompilerPipelineTestCase{
            "SingleControlledU2", MQT_NAMED_BUILDER(::qc::singleControlledU2),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledU2),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledU2<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledU2",
            MQT_NAMED_BUILDER(::qc::multipleControlledU2), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledU2),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledU2<true>)},
        CompilerPipelineTestCase{"U", MQT_NAMED_BUILDER(::qc::u), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::u),
                                 MQT_NAMED_BUILDER(mlir::qir::u<true>)},
        CompilerPipelineTestCase{
            "SingleControlledU",
            MQT_NAMED_BUILDER(::qc::singleControlledU), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledU),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledU<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledU", MQT_NAMED_BUILDER(::qc::multipleControlledU),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledU),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledU<true>)},
        CompilerPipelineTestCase{"SWAP", MQT_NAMED_BUILDER(::qc::swap), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::swap),
                                 MQT_NAMED_BUILDER(mlir::qir::swap<true>)},
        CompilerPipelineTestCase{
            "SingleControlledSWAP",
            MQT_NAMED_BUILDER(::qc::singleControlledSwap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledSwap),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSwap<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledSWAP",
            MQT_NAMED_BUILDER(::qc::multipleControlledSwap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSwap),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSwap<true>)},
        CompilerPipelineTestCase{"iSWAP", MQT_NAMED_BUILDER(::qc::iswap),
                                 nullptr, MQT_NAMED_BUILDER(mlir::qc::iswap),
                                 MQT_NAMED_BUILDER(mlir::qir::iswap<true>)},
        CompilerPipelineTestCase{
            "SingleControllediSWAP",
            MQT_NAMED_BUILDER(::qc::singleControlledIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledIswap),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledIswap<true>)},
        CompilerPipelineTestCase{
            "MultipleControllediSWAP",
            MQT_NAMED_BUILDER(::qc::multipleControlledIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledIswap),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledIswap<true>)},
        CompilerPipelineTestCase{
            "InverseISWAP", MQT_NAMED_BUILDER(::qc::inverseIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::inverseIswap), nullptr, true, false},
        CompilerPipelineTestCase{
            "InverseMultiControlledISWAP",
            MQT_NAMED_BUILDER(::qc::inverseMultipleControlledIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::inverseMultipleControlledIswap),
            nullptr, true, false},
        CompilerPipelineTestCase{"DCX", MQT_NAMED_BUILDER(::qc::dcx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::dcx),
                                 MQT_NAMED_BUILDER(mlir::qir::dcx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledDCX", MQT_NAMED_BUILDER(::qc::singleControlledDcx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledDcx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledDcx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledDCX",
            MQT_NAMED_BUILDER(::qc::multipleControlledDcx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledDcx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledDcx<true>)},
        CompilerPipelineTestCase{"ECR", MQT_NAMED_BUILDER(::qc::ecr), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::ecr),
                                 MQT_NAMED_BUILDER(mlir::qir::ecr<true>)},
        CompilerPipelineTestCase{
            "SingleControlledECR", MQT_NAMED_BUILDER(::qc::singleControlledEcr),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledEcr),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledEcr<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledECR",
            MQT_NAMED_BUILDER(::qc::multipleControlledEcr), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledEcr),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledEcr<true>)},
        CompilerPipelineTestCase{"RXX", MQT_NAMED_BUILDER(::qc::rxx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rxx),
                                 MQT_NAMED_BUILDER(mlir::qir::rxx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRXX", MQT_NAMED_BUILDER(::qc::singleControlledRxx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRxx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRXX",
            MQT_NAMED_BUILDER(::qc::multipleControlledRxx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRxx<true>)},
        CompilerPipelineTestCase{
            "TripleControlledRXX", MQT_NAMED_BUILDER(::qc::tripleControlledRxx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::tripleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qir::tripleControlledRxx<true>)},
        CompilerPipelineTestCase{"RYY", MQT_NAMED_BUILDER(::qc::ryy), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::ryy),
                                 MQT_NAMED_BUILDER(mlir::qir::ryy<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRYY", MQT_NAMED_BUILDER(::qc::singleControlledRyy),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRyy),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRyy<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRYY",
            MQT_NAMED_BUILDER(::qc::multipleControlledRyy), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRyy),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRyy<true>)},
        CompilerPipelineTestCase{"RZX", MQT_NAMED_BUILDER(::qc::rzx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rzx),
                                 MQT_NAMED_BUILDER(mlir::qir::rzx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRZX", MQT_NAMED_BUILDER(::qc::singleControlledRzx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRzx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRzx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRZX",
            MQT_NAMED_BUILDER(::qc::multipleControlledRzx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRzx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRzx<true>)},
        CompilerPipelineTestCase{"RZZ", MQT_NAMED_BUILDER(::qc::rzz), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rzz),
                                 MQT_NAMED_BUILDER(mlir::qir::rzz<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRZZ", MQT_NAMED_BUILDER(::qc::singleControlledRzz),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRzz),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRzz<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRZZ",
            MQT_NAMED_BUILDER(::qc::multipleControlledRzz), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRzz),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRzz<true>)},
        CompilerPipelineTestCase{"XXPlusYY", MQT_NAMED_BUILDER(::qc::xxPlusYY),
                                 nullptr, MQT_NAMED_BUILDER(mlir::qc::xxPlusYY),
                                 MQT_NAMED_BUILDER(mlir::qir::xxPlusYY<true>)},
        CompilerPipelineTestCase{
            "SingleControlledXXPlusYY",
            MQT_NAMED_BUILDER(::qc::singleControlledXxPlusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXxPlusYY),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledXxPlusYY<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledXXPlusYY",
            MQT_NAMED_BUILDER(::qc::multipleControlledXxPlusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledXxPlusYY),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledXxPlusYY<true>)},
        CompilerPipelineTestCase{"XXMinusYY",
                                 MQT_NAMED_BUILDER(::qc::xxMinusYY), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::xxMinusYY),
                                 MQT_NAMED_BUILDER(mlir::qir::xxMinusYY<true>)},
        CompilerPipelineTestCase{
            "SingleControlledXXMinusYY",
            MQT_NAMED_BUILDER(::qc::singleControlledXxMinusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXxMinusYY),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledXxMinusYY<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledXXMinusYY",
            MQT_NAMED_BUILDER(::qc::multipleControlledXxMinusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledXxMinusYY),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledXxMinusYY<true>)},
        CompilerPipelineTestCase{"CtrlTwo", MQT_NAMED_BUILDER(::qc::ctrlTwo),
                                 nullptr, MQT_NAMED_BUILDER(mlir::qc::ctrlTwo),
                                 MQT_NAMED_BUILDER(mlir::qir::ctrlTwo<true>)}));

} // namespace mqt::test::compiler
