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
#include <iosfwd>
#include <memory>
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
    config.printIRAfterAllStages = irPrintingForced();

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
