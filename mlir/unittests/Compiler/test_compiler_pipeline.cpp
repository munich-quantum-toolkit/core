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
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"
#include "qc_programs.h"
#include "qir_programs.h"
#include "quantum_computation_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <string>

namespace {

using QCProgramBuilderFn =
    llvm::function_ref<void(mlir::qc::QCProgramBuilder&)>;
using QIRProgramBuilderFn =
    llvm::function_ref<void(mlir::qir::QIRProgramBuilder&)>;
using QuantumComputationBuilderFn =
    llvm::function_ref<void(::qc::QuantumComputation&)>;

struct CompilerPipelineTestCase {
  std::string name;
  QuantumComputationBuilderFn quantumComputationBuilder = nullptr;
  QCProgramBuilderFn qcProgramBuilder = nullptr;
  QCProgramBuilderFn qcReferenceBuilder = nullptr;
  QIRProgramBuilderFn qirReferenceBuilder = nullptr;
  bool startFromQuantumComputation = true;
  bool convertToQIR = true;
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
    auto module = mlir::qc::QCProgramBuilder::build(context.get(), builder);
    runCanonicalizationPasses(module.get());
    return module;
  }

  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
  buildQIRReference(const QIRProgramBuilderFn builder) const {
    auto module = mlir::qir::QIRProgramBuilder::build(context.get(), builder);
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
} // namespace

static std::string
printTestName(const testing::TestParamInfo<CompilerPipelineTestCase>& info) {
  return info.param.name;
}

TEST_P(CompilerPipelineTest, EndToEndPipeline) {
  const auto& testCase = GetParam();
  const auto name = " (" + testCase.name + ")";

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (testCase.startFromQuantumComputation) {
    ASSERT_NE(testCase.quantumComputationBuilder, nullptr);
    ::qc::QuantumComputation comp;
    testCase.quantumComputationBuilder(comp);

    module = mlir::translateQuantumComputationToQC(context.get(), comp);
    ASSERT_TRUE(module);
    mlir::printProgram(module.get(), "QC Import" + name, llvm::errs());
  } else {
    ASSERT_NE(testCase.qcProgramBuilder, nullptr);
    module = mlir::qc::QCProgramBuilder::build(context.get(),
                                               testCase.qcProgramBuilder);
    ASSERT_TRUE(module);
    mlir::printProgram(module.get(), "QC Input" + name, llvm::errs());
  }
  EXPECT_TRUE(mlir::verify(*module).succeeded());

  mlir::CompilationRecord record;
  runPipeline(module.get(), testCase.convertToQIR, record);

  ASSERT_NE(testCase.qcReferenceBuilder, nullptr);
  auto qcReference = buildQCReference(testCase.qcReferenceBuilder);
  ASSERT_TRUE(qcReference);
  mlir::printProgram(qcReference.get(), "Reference QC IR" + name, llvm::errs());

  expectEquivalent("Final QC", record.afterQCCanon, qcReference.get());
  auto finalQC = parseRecordedModule(record.afterQCCanon);
  ASSERT_TRUE(finalQC);
  mlir::printProgram(finalQC.get(), "Final QC IR" + name, llvm::errs());

  if (testCase.convertToQIR) {
    ASSERT_NE(testCase.qirReferenceBuilder, nullptr);

    auto qirReference = buildQIRReference(testCase.qirReferenceBuilder);
    ASSERT_TRUE(qirReference);
    mlir::printProgram(qirReference.get(), "Reference QIR IR" + name,
                       llvm::errs());

    expectEquivalent("Final QIR", record.afterQIRCanon, qirReference.get());
    auto finalQIR = parseRecordedModule(record.afterQIRCanon);
    ASSERT_TRUE(finalQIR);
    mlir::printProgram(finalQIR.get(), "Final QIR IR" + name, llvm::errs());
  }
}

INSTANTIATE_TEST_SUITE_P(
    QuantumComputationPipelineProgramsTest, CompilerPipelineTest,
    testing::Values(
        CompilerPipelineTestCase{"StaticQubits", nullptr,
                                 mlir::qc::staticQubits, mlir::qc::staticQubits,
                                 mlir::qir::staticQubits, false},
        CompilerPipelineTestCase{"AllocQubit", qc::allocQubit, nullptr,
                                 mlir::qc::allocQubit, mlir::qir::allocQubit},
        CompilerPipelineTestCase{"AllocQubitRegister", qc::allocQubitRegister,
                                 nullptr, mlir::qc::allocQubitRegister,
                                 mlir::qir::allocQubitRegister},
        CompilerPipelineTestCase{"AllocMultipleQubitRegisters",
                                 qc::allocMultipleQubitRegisters, nullptr,
                                 mlir::qc::allocMultipleQubitRegisters,
                                 mlir::qir::allocMultipleQubitRegisters},
        CompilerPipelineTestCase{"AllocLargeRegister", qc::allocLargeRegister,
                                 nullptr, mlir::qc::allocLargeRegister,
                                 mlir::qir::allocLargeRegister},
        CompilerPipelineTestCase{"SingleMeasurementToSingleBit",
                                 qc::singleMeasurementToSingleBit, nullptr,
                                 mlir::qc::singleMeasurementToSingleBit,
                                 mlir::qir::singleMeasurementToSingleBit},
        CompilerPipelineTestCase{"RepeatedMeasurementToSameBit",
                                 qc::repeatedMeasurementToSameBit, nullptr,
                                 mlir::qc::repeatedMeasurementToSameBit,
                                 mlir::qir::repeatedMeasurementToSameBit},
        CompilerPipelineTestCase{"RepeatedMeasurementToDifferentBits",
                                 qc::repeatedMeasurementToDifferentBits,
                                 nullptr,
                                 mlir::qc::repeatedMeasurementToDifferentBits,
                                 mlir::qir::repeatedMeasurementToDifferentBits},
        CompilerPipelineTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            qc::multipleClassicalRegistersAndMeasurements, nullptr,
            mlir::qc::multipleClassicalRegistersAndMeasurements,
            mlir::qir::multipleClassicalRegistersAndMeasurements},
        CompilerPipelineTestCase{"ResetQubitAfterSingleOp",
                                 qc::resetQubitAfterSingleOp, nullptr,
                                 mlir::qc::resetQubitAfterSingleOp,
                                 mlir::qir::resetQubitAfterSingleOp},
        CompilerPipelineTestCase{"ResetMultipleQubitsAfterSingleOp",
                                 qc::resetMultipleQubitsAfterSingleOp, nullptr,
                                 mlir::qc::resetMultipleQubitsAfterSingleOp,
                                 mlir::qir::resetMultipleQubitsAfterSingleOp},
        CompilerPipelineTestCase{"RepeatedResetAfterSingleOp",
                                 qc::repeatedResetAfterSingleOp, nullptr,
                                 mlir::qc::resetQubitAfterSingleOp,
                                 mlir::qir::resetQubitAfterSingleOp},
        CompilerPipelineTestCase{"GlobalPhase", qc::globalPhase, nullptr,
                                 mlir::qc::globalPhase, mlir::qir::globalPhase},
        CompilerPipelineTestCase{"Identity", qc::identity, nullptr,
                                 mlir::qc::emptyQC, mlir::qir::emptyQIR},
        CompilerPipelineTestCase{"SingleControlledIdentity",
                                 qc::singleControlledIdentity, nullptr,
                                 mlir::qc::emptyQC, mlir::qir::emptyQIR},
        CompilerPipelineTestCase{"MultipleControlledIdentity",
                                 qc::multipleControlledIdentity, nullptr,
                                 mlir::qc::emptyQC, mlir::qir::emptyQIR},
        CompilerPipelineTestCase{"X", qc::x, nullptr, mlir::qc::x,
                                 mlir::qir::x},
        CompilerPipelineTestCase{"SingleControlledX", qc::singleControlledX,
                                 nullptr, mlir::qc::singleControlledX,
                                 mlir::qir::singleControlledX},
        CompilerPipelineTestCase{"MultipleControlledX", qc::multipleControlledX,
                                 nullptr, mlir::qc::multipleControlledX,
                                 mlir::qir::multipleControlledX},
        CompilerPipelineTestCase{"Y", qc::y, nullptr, mlir::qc::y,
                                 mlir::qir::y},
        CompilerPipelineTestCase{"SingleControlledY", qc::singleControlledY,
                                 nullptr, mlir::qc::singleControlledY,
                                 mlir::qir::singleControlledY},
        CompilerPipelineTestCase{"MultipleControlledY", qc::multipleControlledY,
                                 nullptr, mlir::qc::multipleControlledY,
                                 mlir::qir::multipleControlledY},
        CompilerPipelineTestCase{"Z", qc::z, nullptr, mlir::qc::z,
                                 mlir::qir::z},
        CompilerPipelineTestCase{"SingleControlledZ", qc::singleControlledZ,
                                 nullptr, mlir::qc::singleControlledZ,
                                 mlir::qir::singleControlledZ},
        CompilerPipelineTestCase{"MultipleControlledZ", qc::multipleControlledZ,
                                 nullptr, mlir::qc::multipleControlledZ,
                                 mlir::qir::multipleControlledZ},
        CompilerPipelineTestCase{"H", qc::h, nullptr, mlir::qc::h,
                                 mlir::qir::h},
        CompilerPipelineTestCase{"SingleControlledH", qc::singleControlledH,
                                 nullptr, mlir::qc::singleControlledH,
                                 mlir::qir::singleControlledH},
        CompilerPipelineTestCase{"MultipleControlledH", qc::multipleControlledH,
                                 nullptr, mlir::qc::multipleControlledH,
                                 mlir::qir::multipleControlledH},
        CompilerPipelineTestCase{"S", qc::s, nullptr, mlir::qc::s,
                                 mlir::qir::s},
        CompilerPipelineTestCase{"SingleControlledS", qc::singleControlledS,
                                 nullptr, mlir::qc::singleControlledS,
                                 mlir::qir::singleControlledS},
        CompilerPipelineTestCase{"MultipleControlledS", qc::multipleControlledS,
                                 nullptr, mlir::qc::multipleControlledS,
                                 mlir::qir::multipleControlledS},
        CompilerPipelineTestCase{"Sdg", qc::sdg, nullptr, mlir::qc::sdg,
                                 mlir::qir::sdg},
        CompilerPipelineTestCase{"SingleControlledSdg", qc::singleControlledSdg,
                                 nullptr, mlir::qc::singleControlledSdg,
                                 mlir::qir::singleControlledSdg},
        CompilerPipelineTestCase{
            "MultipleControlledSdg", qc::multipleControlledSdg, nullptr,
            mlir::qc::multipleControlledSdg, mlir::qir::multipleControlledSdg},
        CompilerPipelineTestCase{"T", qc::t_, nullptr, mlir::qc::t_,
                                 mlir::qir::t_},
        CompilerPipelineTestCase{"SingleControlledT", qc::singleControlledT,
                                 nullptr, mlir::qc::singleControlledT,
                                 mlir::qir::singleControlledT},
        CompilerPipelineTestCase{"MultipleControlledT", qc::multipleControlledT,
                                 nullptr, mlir::qc::multipleControlledT,
                                 mlir::qir::multipleControlledT},
        CompilerPipelineTestCase{"Tdg", qc::tdg, nullptr, mlir::qc::tdg,
                                 mlir::qir::tdg},
        CompilerPipelineTestCase{"SingleControlledTdg", qc::singleControlledTdg,
                                 nullptr, mlir::qc::singleControlledTdg,
                                 mlir::qir::singleControlledTdg},
        CompilerPipelineTestCase{
            "MultipleControlledTdg", qc::multipleControlledTdg, nullptr,
            mlir::qc::multipleControlledTdg, mlir::qir::multipleControlledTdg},
        CompilerPipelineTestCase{"SX", qc::sx, nullptr, mlir::qc::sx,
                                 mlir::qir::sx},
        CompilerPipelineTestCase{"SingleControlledSX", qc::singleControlledSx,
                                 nullptr, mlir::qc::singleControlledSx,
                                 mlir::qir::singleControlledSx},
        CompilerPipelineTestCase{
            "MultipleControlledSX", qc::multipleControlledSx, nullptr,
            mlir::qc::multipleControlledSx, mlir::qir::multipleControlledSx},
        CompilerPipelineTestCase{"SXdg", qc::sxdg, nullptr, mlir::qc::sxdg,
                                 mlir::qir::sxdg},
        CompilerPipelineTestCase{
            "SingleControlledSXdg", qc::singleControlledSxdg, nullptr,
            mlir::qc::singleControlledSxdg, mlir::qir::singleControlledSxdg},
        CompilerPipelineTestCase{"MultipleControlledSXdg",
                                 qc::multipleControlledSxdg, nullptr,
                                 mlir::qc::multipleControlledSxdg,
                                 mlir::qir::multipleControlledSxdg},
        CompilerPipelineTestCase{"RX", qc::rx, nullptr, mlir::qc::rx,
                                 mlir::qir::rx},
        CompilerPipelineTestCase{"SingleControlledRX", qc::singleControlledRx,
                                 nullptr, mlir::qc::singleControlledRx,
                                 mlir::qir::singleControlledRx},
        CompilerPipelineTestCase{
            "MultipleControlledRX", qc::multipleControlledRx, nullptr,
            mlir::qc::multipleControlledRx, mlir::qir::multipleControlledRx},
        CompilerPipelineTestCase{"RY", qc::ry, nullptr, mlir::qc::ry,
                                 mlir::qir::ry},
        CompilerPipelineTestCase{"SingleControlledRY", qc::singleControlledRy,
                                 nullptr, mlir::qc::singleControlledRy,
                                 mlir::qir::singleControlledRy},
        CompilerPipelineTestCase{
            "MultipleControlledRY", qc::multipleControlledRy, nullptr,
            mlir::qc::multipleControlledRy, mlir::qir::multipleControlledRy},
        CompilerPipelineTestCase{"RZ", qc::rz, nullptr, mlir::qc::rz,
                                 mlir::qir::rz},
        CompilerPipelineTestCase{"SingleControlledRZ", qc::singleControlledRz,
                                 nullptr, mlir::qc::singleControlledRz,
                                 mlir::qir::singleControlledRz},
        CompilerPipelineTestCase{
            "MultipleControlledRZ", qc::multipleControlledRz, nullptr,
            mlir::qc::multipleControlledRz, mlir::qir::multipleControlledRz},
        CompilerPipelineTestCase{"P", qc::p, nullptr, mlir::qc::p,
                                 mlir::qir::p},
        CompilerPipelineTestCase{"SingleControlledP", qc::singleControlledP,
                                 nullptr, mlir::qc::singleControlledP,
                                 mlir::qir::singleControlledP},
        CompilerPipelineTestCase{"MultipleControlledP", qc::multipleControlledP,
                                 nullptr, mlir::qc::multipleControlledP,
                                 mlir::qir::multipleControlledP},
        CompilerPipelineTestCase{"R", qc::r, nullptr, mlir::qc::r,
                                 mlir::qir::r},
        CompilerPipelineTestCase{"SingleControlledR", qc::singleControlledR,
                                 nullptr, mlir::qc::singleControlledR,
                                 mlir::qir::singleControlledR},
        CompilerPipelineTestCase{"MultipleControlledR", qc::multipleControlledR,
                                 nullptr, mlir::qc::multipleControlledR,
                                 mlir::qir::multipleControlledR},
        CompilerPipelineTestCase{"U2", qc::u2, nullptr, mlir::qc::u2,
                                 mlir::qir::u2},
        CompilerPipelineTestCase{"SingleControlledU2", qc::singleControlledU2,
                                 nullptr, mlir::qc::singleControlledU2,
                                 mlir::qir::singleControlledU2},
        CompilerPipelineTestCase{
            "MultipleControlledU2", qc::multipleControlledU2, nullptr,
            mlir::qc::multipleControlledU2, mlir::qir::multipleControlledU2},
        CompilerPipelineTestCase{"U", qc::u, nullptr, mlir::qc::u,
                                 mlir::qir::u},
        CompilerPipelineTestCase{"SingleControlledU", qc::singleControlledU,
                                 nullptr, mlir::qc::singleControlledU,
                                 mlir::qir::singleControlledU},
        CompilerPipelineTestCase{"MultipleControlledU", qc::multipleControlledU,
                                 nullptr, mlir::qc::multipleControlledU,
                                 mlir::qir::multipleControlledU},
        CompilerPipelineTestCase{"SWAP", qc::swap, nullptr, mlir::qc::swap,
                                 mlir::qir::swap},
        CompilerPipelineTestCase{
            "SingleControlledSWAP", qc::singleControlledSwap, nullptr,
            mlir::qc::singleControlledSwap, mlir::qir::singleControlledSwap},
        CompilerPipelineTestCase{"MultipleControlledSWAP",
                                 qc::multipleControlledSwap, nullptr,
                                 mlir::qc::multipleControlledSwap,
                                 mlir::qir::multipleControlledSwap},
        CompilerPipelineTestCase{"iSWAP", qc::iswap, nullptr, mlir::qc::iswap,
                                 mlir::qir::iswap},
        CompilerPipelineTestCase{
            "SingleControllediSWAP", qc::singleControlledIswap, nullptr,
            mlir::qc::singleControlledIswap, mlir::qir::singleControlledIswap},
        CompilerPipelineTestCase{"MultipleControllediSWAP",
                                 qc::multipleControlledIswap, nullptr,
                                 mlir::qc::multipleControlledIswap,
                                 mlir::qir::multipleControlledIswap},
        CompilerPipelineTestCase{"InverseISWAP", qc::inverseIswap, nullptr,
                                 mlir::qc::inverseIswap, nullptr, true, false},
        CompilerPipelineTestCase{"InverseMultiControlledISWAP",
                                 qc::inverseMultipleControlledIswap, nullptr,
                                 mlir::qc::inverseMultipleControlledIswap,
                                 nullptr, true, false},
        CompilerPipelineTestCase{"DCX", qc::dcx, nullptr, mlir::qc::dcx,
                                 mlir::qir::dcx},
        CompilerPipelineTestCase{"SingleControlledDCX", qc::singleControlledDcx,
                                 nullptr, mlir::qc::singleControlledDcx,
                                 mlir::qir::singleControlledDcx},
        CompilerPipelineTestCase{
            "MultipleControlledDCX", qc::multipleControlledDcx, nullptr,
            mlir::qc::multipleControlledDcx, mlir::qir::multipleControlledDcx},
        CompilerPipelineTestCase{"ECR", qc::ecr, nullptr, mlir::qc::ecr,
                                 mlir::qir::ecr},
        CompilerPipelineTestCase{"SingleControlledECR", qc::singleControlledEcr,
                                 nullptr, mlir::qc::singleControlledEcr,
                                 mlir::qir::singleControlledEcr},
        CompilerPipelineTestCase{
            "MultipleControlledECR", qc::multipleControlledEcr, nullptr,
            mlir::qc::multipleControlledEcr, mlir::qir::multipleControlledEcr},
        CompilerPipelineTestCase{"RXX", qc::rxx, nullptr, mlir::qc::rxx,
                                 mlir::qir::rxx},
        CompilerPipelineTestCase{"SingleControlledRXX", qc::singleControlledRxx,
                                 nullptr, mlir::qc::singleControlledRxx,
                                 mlir::qir::singleControlledRxx},
        CompilerPipelineTestCase{
            "MultipleControlledRXX", qc::multipleControlledRxx, nullptr,
            mlir::qc::multipleControlledRxx, mlir::qir::multipleControlledRxx},
        CompilerPipelineTestCase{"TripleControlledRXX", qc::tripleControlledRxx,
                                 nullptr, mlir::qc::tripleControlledRxx,
                                 mlir::qir::tripleControlledRxx},
        CompilerPipelineTestCase{"RYY", qc::ryy, nullptr, mlir::qc::ryy,
                                 mlir::qir::ryy},
        CompilerPipelineTestCase{"SingleControlledRYY", qc::singleControlledRyy,
                                 nullptr, mlir::qc::singleControlledRyy,
                                 mlir::qir::singleControlledRyy},
        CompilerPipelineTestCase{
            "MultipleControlledRYY", qc::multipleControlledRyy, nullptr,
            mlir::qc::multipleControlledRyy, mlir::qir::multipleControlledRyy},
        CompilerPipelineTestCase{"RZX", qc::rzx, nullptr, mlir::qc::rzx,
                                 mlir::qir::rzx},
        CompilerPipelineTestCase{"SingleControlledRZX", qc::singleControlledRzx,
                                 nullptr, mlir::qc::singleControlledRzx,
                                 mlir::qir::singleControlledRzx},
        CompilerPipelineTestCase{
            "MultipleControlledRZX", qc::multipleControlledRzx, nullptr,
            mlir::qc::multipleControlledRzx, mlir::qir::multipleControlledRzx},
        CompilerPipelineTestCase{"RZZ", qc::rzz, nullptr, mlir::qc::rzz,
                                 mlir::qir::rzz},
        CompilerPipelineTestCase{"SingleControlledRZZ", qc::singleControlledRzz,
                                 nullptr, mlir::qc::singleControlledRzz,
                                 mlir::qir::singleControlledRzz},
        CompilerPipelineTestCase{
            "MultipleControlledRZZ", qc::multipleControlledRzz, nullptr,
            mlir::qc::multipleControlledRzz, mlir::qir::multipleControlledRzz},
        CompilerPipelineTestCase{"XXPlusYY", qc::xxPlusYY, nullptr,
                                 mlir::qc::xxPlusYY, mlir::qir::xxPlusYY},
        CompilerPipelineTestCase{"SingleControlledXXPlusYY",
                                 qc::singleControlledXxPlusYY, nullptr,
                                 mlir::qc::singleControlledXxPlusYY,
                                 mlir::qir::singleControlledXxPlusYY},
        CompilerPipelineTestCase{"MultipleControlledXXPlusYY",
                                 qc::multipleControlledXxPlusYY, nullptr,
                                 mlir::qc::multipleControlledXxPlusYY,
                                 mlir::qir::multipleControlledXxPlusYY},
        CompilerPipelineTestCase{"XXMinusYY", qc::xxMinusYY, nullptr,
                                 mlir::qc::xxMinusYY, mlir::qir::xxMinusYY},
        CompilerPipelineTestCase{"SingleControlledXXMinusYY",
                                 qc::singleControlledXxMinusYY, nullptr,
                                 mlir::qc::singleControlledXxMinusYY,
                                 mlir::qir::singleControlledXxMinusYY},
        CompilerPipelineTestCase{"MultipleControlledXXMinusYY",
                                 qc::multipleControlledXxMinusYY, nullptr,
                                 mlir::qc::multipleControlledXxMinusYY,
                                 mlir::qir::multipleControlledXxMinusYY}),
    printTestName);
