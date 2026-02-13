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
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"
#include "qc_programs.h"
#include "quantum_computation_programs.h"

#include <gtest/gtest.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Verifier.h>
#include <string>

namespace {

struct QuantumComputationTranslationTestCase {
  std::string name;
  llvm::function_ref<void(::qc::QuantumComputation&)> programBuilder;
  llvm::function_ref<void(mlir::qc::QCProgramBuilder&)> referenceBuilder;
};

class QuantumComputationTranslationTest
    : public testing::TestWithParam<QuantumComputationTranslationTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> translated;
  mlir::OwningOpRef<mlir::ModuleOp> reference;

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    context = std::make_unique<mlir::MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

std::string printTestName(
    const testing::TestParamInfo<QuantumComputationTranslationTestCase>& info) {
  return info.param.name;
}

TEST_P(QuantumComputationTranslationTest, ProgramEquivalence) {
  auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";

  ::qc::QuantumComputation comp;
  programBuilder(comp);

  translated = mlir::translateQuantumComputationToQC(context.get(), comp);
  ASSERT_TRUE(translated);
  mlir::printProgram(translated.get(), "Translated QC IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*translated).succeeded());

  runCanonicalizationPasses(translated.get());
  mlir::printProgram(translated.get(), "Canonicalized Translated QC IR" + name,
                     llvm::errs());
  EXPECT_TRUE(mlir::verify(*translated).succeeded());

  reference =
      mlir::qc::QCProgramBuilder::build(context.get(), referenceBuilder);
  ASSERT_TRUE(reference);
  mlir::printProgram(reference.get(), "Reference QC IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*reference).succeeded());

  runCanonicalizationPasses(reference.get());
  mlir::printProgram(reference.get(), "Canonicalized Reference QC IR" + name,
                     llvm::errs());
  EXPECT_TRUE(mlir::verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(translated.get(), reference.get()));
}

INSTANTIATE_TEST_SUITE_P(
    QuantumComputationTranslationProgramsTest,
    QuantumComputationTranslationTest,
    testing::Values(
        QuantumComputationTranslationTestCase{"AllocQubit", qc::allocQubit,
                                              mlir::qc::allocQubit},
        QuantumComputationTranslationTestCase{"AllocQubitRegister",
                                              qc::allocQubitRegister,
                                              mlir::qc::allocQubitRegister},
        QuantumComputationTranslationTestCase{
            "AllocMultipleQubitRegisters", qc::allocMultipleQubitRegisters,
            mlir::qc::allocMultipleQubitRegisters},
        QuantumComputationTranslationTestCase{"AllocLargeRegister",
                                              qc::allocLargeRegister,
                                              mlir::qc::allocLargeRegister},
        QuantumComputationTranslationTestCase{
            "SingleMeasurementToSingleBit", qc::singleMeasurementToSingleBit,
            mlir::qc::singleMeasurementToSingleBit},
        QuantumComputationTranslationTestCase{
            "RepeatedMeasurementToSameBit", qc::repeatedMeasurementToSameBit,
            mlir::qc::repeatedMeasurementToSameBit},
        QuantumComputationTranslationTestCase{
            "RepeatedMeasurementToDifferentBits",
            qc::repeatedMeasurementToDifferentBits,
            mlir::qc::repeatedMeasurementToDifferentBits},
        QuantumComputationTranslationTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            qc::multipleClassicalRegistersAndMeasurements,
            mlir::qc::multipleClassicalRegistersAndMeasurements},
        QuantumComputationTranslationTestCase{
            "ResetQubitAfterSingleOp", qc::resetQubitAfterSingleOp,
            mlir::qc::resetQubitAfterSingleOp},
        QuantumComputationTranslationTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            qc::resetMultipleQubitsAfterSingleOp,
            mlir::qc::resetMultipleQubitsAfterSingleOp},
        QuantumComputationTranslationTestCase{
            "RepeatedResetAfterSingleOp", qc::repeatedResetAfterSingleOp,
            mlir::qc::repeatedResetAfterSingleOp},
        QuantumComputationTranslationTestCase{"GlobalPhase", qc::globalPhase,
                                              mlir::qc::globalPhase},
        QuantumComputationTranslationTestCase{"Identity", qc::identity,
                                              mlir::qc::identity},
        QuantumComputationTranslationTestCase{
            "SingleControlledIdentity", qc::singleControlledIdentity,
            mlir::qc::singleControlledIdentity},
        QuantumComputationTranslationTestCase{
            "MultipleControlledIdentity", qc::multipleControlledIdentity,
            mlir::qc::multipleControlledIdentity},
        QuantumComputationTranslationTestCase{"X", qc::x, mlir::qc::x},
        QuantumComputationTranslationTestCase{"SingleControlledX",
                                              qc::singleControlledX,
                                              mlir::qc::singleControlledX},
        QuantumComputationTranslationTestCase{"MultipleControlledX",
                                              qc::multipleControlledX,
                                              mlir::qc::multipleControlledX},
        QuantumComputationTranslationTestCase{"Y", qc::y, mlir::qc::y},
        QuantumComputationTranslationTestCase{"SingleControlledY",
                                              qc::singleControlledY,
                                              mlir::qc::singleControlledY},
        QuantumComputationTranslationTestCase{"MultipleControlledY",
                                              qc::multipleControlledY,
                                              mlir::qc::multipleControlledY},
        QuantumComputationTranslationTestCase{"Z", qc::z, mlir::qc::z},
        QuantumComputationTranslationTestCase{"SingleControlledZ",
                                              qc::singleControlledZ,
                                              mlir::qc::singleControlledZ},
        QuantumComputationTranslationTestCase{"MultipleControlledZ",
                                              qc::multipleControlledZ,
                                              mlir::qc::multipleControlledZ},
        QuantumComputationTranslationTestCase{"H", qc::h, mlir::qc::h},
        QuantumComputationTranslationTestCase{"SingleControlledH",
                                              qc::singleControlledH,
                                              mlir::qc::singleControlledH},
        QuantumComputationTranslationTestCase{"MultipleControlledH",
                                              qc::multipleControlledH,
                                              mlir::qc::multipleControlledH},
        QuantumComputationTranslationTestCase{"S", qc::s, mlir::qc::s},
        QuantumComputationTranslationTestCase{"SingleControlledS",
                                              qc::singleControlledS,
                                              mlir::qc::singleControlledS},
        QuantumComputationTranslationTestCase{"MultipleControlledS",
                                              qc::multipleControlledS,
                                              mlir::qc::multipleControlledS},
        QuantumComputationTranslationTestCase{"Sdg", qc::sdg, mlir::qc::sdg},
        QuantumComputationTranslationTestCase{"SingleControlledSdg",
                                              qc::singleControlledSdg,
                                              mlir::qc::singleControlledSdg},
        QuantumComputationTranslationTestCase{"MultipleControlledSdg",
                                              qc::multipleControlledSdg,
                                              mlir::qc::multipleControlledSdg},
        QuantumComputationTranslationTestCase{"T", qc::t_, mlir::qc::t_},
        QuantumComputationTranslationTestCase{"SingleControlledT",
                                              qc::singleControlledT,
                                              mlir::qc::singleControlledT},
        QuantumComputationTranslationTestCase{"MultipleControlledT",
                                              qc::multipleControlledT,
                                              mlir::qc::multipleControlledT},
        QuantumComputationTranslationTestCase{"Tdg", qc::tdg, mlir::qc::tdg},
        QuantumComputationTranslationTestCase{"SingleControlledTdg",
                                              qc::singleControlledTdg,
                                              mlir::qc::singleControlledTdg},
        QuantumComputationTranslationTestCase{"MultipleControlledTdg",
                                              qc::multipleControlledTdg,
                                              mlir::qc::multipleControlledTdg},
        QuantumComputationTranslationTestCase{"SX", qc::sx, mlir::qc::sx},
        QuantumComputationTranslationTestCase{"SingleControlledSX",
                                              qc::singleControlledSx,
                                              mlir::qc::singleControlledSx},
        QuantumComputationTranslationTestCase{"MultipleControlledSX",
                                              qc::multipleControlledSx,
                                              mlir::qc::multipleControlledSx},
        QuantumComputationTranslationTestCase{"SXdg", qc::sxdg, mlir::qc::sxdg},
        QuantumComputationTranslationTestCase{"SingleControlledSXdg",
                                              qc::singleControlledSxdg,
                                              mlir::qc::singleControlledSxdg},
        QuantumComputationTranslationTestCase{"MultipleControlledSXdg",
                                              qc::multipleControlledSxdg,
                                              mlir::qc::multipleControlledSxdg},
        QuantumComputationTranslationTestCase{"RX", qc::rx, mlir::qc::rx},
        QuantumComputationTranslationTestCase{"SingleControlledRX",
                                              qc::singleControlledRx,
                                              mlir::qc::singleControlledRx},
        QuantumComputationTranslationTestCase{"MultipleControlledRX",
                                              qc::multipleControlledRx,
                                              mlir::qc::multipleControlledRx},
        QuantumComputationTranslationTestCase{"RY", qc::ry, mlir::qc::ry},
        QuantumComputationTranslationTestCase{"SingleControlledRY",
                                              qc::singleControlledRy,
                                              mlir::qc::singleControlledRy},
        QuantumComputationTranslationTestCase{"MultipleControlledRY",
                                              qc::multipleControlledRy,
                                              mlir::qc::multipleControlledRy},
        QuantumComputationTranslationTestCase{"RZ", qc::rz, mlir::qc::rz},
        QuantumComputationTranslationTestCase{"SingleControlledRZ",
                                              qc::singleControlledRz,
                                              mlir::qc::singleControlledRz},
        QuantumComputationTranslationTestCase{"MultipleControlledRZ",
                                              qc::multipleControlledRz,
                                              mlir::qc::multipleControlledRz},
        QuantumComputationTranslationTestCase{"P", qc::p, mlir::qc::p},
        QuantumComputationTranslationTestCase{"SingleControlledP",
                                              qc::singleControlledP,
                                              mlir::qc::singleControlledP},
        QuantumComputationTranslationTestCase{"MultipleControlledP",
                                              qc::multipleControlledP,
                                              mlir::qc::multipleControlledP},
        QuantumComputationTranslationTestCase{"R", qc::r, mlir::qc::r},
        QuantumComputationTranslationTestCase{"SingleControlledR",
                                              qc::singleControlledR,
                                              mlir::qc::singleControlledR},
        QuantumComputationTranslationTestCase{"MultipleControlledR",
                                              qc::multipleControlledR,
                                              mlir::qc::multipleControlledR},
        QuantumComputationTranslationTestCase{"U2", qc::u2, mlir::qc::u2},
        QuantumComputationTranslationTestCase{"SingleControlledU2",
                                              qc::singleControlledU2,
                                              mlir::qc::singleControlledU2},
        QuantumComputationTranslationTestCase{"MultipleControlledU2",
                                              qc::multipleControlledU2,
                                              mlir::qc::multipleControlledU2},
        QuantumComputationTranslationTestCase{"U", qc::u, mlir::qc::u},
        QuantumComputationTranslationTestCase{"SingleControlledU",
                                              qc::singleControlledU,
                                              mlir::qc::singleControlledU},
        QuantumComputationTranslationTestCase{"MultipleControlledU",
                                              qc::multipleControlledU,
                                              mlir::qc::multipleControlledU},
        QuantumComputationTranslationTestCase{"SWAP", qc::swap, mlir::qc::swap},
        QuantumComputationTranslationTestCase{"SingleControlledSWAP",
                                              qc::singleControlledSwap,
                                              mlir::qc::singleControlledSwap},
        QuantumComputationTranslationTestCase{"MultipleControlledSWAP",
                                              qc::multipleControlledSwap,
                                              mlir::qc::multipleControlledSwap},
        QuantumComputationTranslationTestCase{"iSWAP", qc::iswap,
                                              mlir::qc::iswap},
        QuantumComputationTranslationTestCase{"SingleControllediSWAP",
                                              qc::singleControlledIswap,
                                              mlir::qc::singleControlledIswap},
        QuantumComputationTranslationTestCase{
            "MultipleControllediSWAP", qc::multipleControlledIswap,
            mlir::qc::multipleControlledIswap},
        QuantumComputationTranslationTestCase{"InverseISWAP", qc::inverseIswap,
                                              mlir::qc::inverseIswap},
        QuantumComputationTranslationTestCase{
            "InverseMultiControlledISWAP", qc::inverseMultipleControlledIswap,
            mlir::qc::inverseMultipleControlledIswap},
        QuantumComputationTranslationTestCase{"DCX", qc::dcx, mlir::qc::dcx},
        QuantumComputationTranslationTestCase{"SingleControlledDCX",
                                              qc::singleControlledDcx,
                                              mlir::qc::singleControlledDcx},
        QuantumComputationTranslationTestCase{"MultipleControlledDCX",
                                              qc::multipleControlledDcx,
                                              mlir::qc::multipleControlledDcx},
        QuantumComputationTranslationTestCase{"ECR", qc::ecr, mlir::qc::ecr},
        QuantumComputationTranslationTestCase{"SingleControlledECR",
                                              qc::singleControlledEcr,
                                              mlir::qc::singleControlledEcr},
        QuantumComputationTranslationTestCase{"MultipleControlledECR",
                                              qc::multipleControlledEcr,
                                              mlir::qc::multipleControlledEcr},
        QuantumComputationTranslationTestCase{"RXX", qc::rxx, mlir::qc::rxx},
        QuantumComputationTranslationTestCase{"SingleControlledRXX",
                                              qc::singleControlledRxx,
                                              mlir::qc::singleControlledRxx},
        QuantumComputationTranslationTestCase{"MultipleControlledRXX",
                                              qc::multipleControlledRxx,
                                              mlir::qc::multipleControlledRxx},
        QuantumComputationTranslationTestCase{"TripleControlledRXX",
                                              qc::tripleControlledRxx,
                                              mlir::qc::tripleControlledRxx},
        QuantumComputationTranslationTestCase{"RYY", qc::ryy, mlir::qc::ryy},
        QuantumComputationTranslationTestCase{"SingleControlledRYY",
                                              qc::singleControlledRyy,
                                              mlir::qc::singleControlledRyy},
        QuantumComputationTranslationTestCase{"MultipleControlledRYY",
                                              qc::multipleControlledRyy,
                                              mlir::qc::multipleControlledRyy},
        QuantumComputationTranslationTestCase{"RZX", qc::rzx, mlir::qc::rzx},
        QuantumComputationTranslationTestCase{"SingleControlledRZX",
                                              qc::singleControlledRzx,
                                              mlir::qc::singleControlledRzx},
        QuantumComputationTranslationTestCase{"MultipleControlledRZX",
                                              qc::multipleControlledRzx,
                                              mlir::qc::multipleControlledRzx},
        QuantumComputationTranslationTestCase{"RZZ", qc::rzz, mlir::qc::rzz},
        QuantumComputationTranslationTestCase{"SingleControlledRZZ",
                                              qc::singleControlledRzz,
                                              mlir::qc::singleControlledRzz},
        QuantumComputationTranslationTestCase{"MultipleControlledRZZ",
                                              qc::multipleControlledRzz,
                                              mlir::qc::multipleControlledRzz},
        QuantumComputationTranslationTestCase{"XXPlusYY", qc::xxPlusYY,
                                              mlir::qc::xxPlusYY},
        QuantumComputationTranslationTestCase{
            "SingleControlledXXPlusYY", qc::singleControlledXxPlusYY,
            mlir::qc::singleControlledXxPlusYY},
        QuantumComputationTranslationTestCase{
            "MultipleControlledXXPlusYY", qc::multipleControlledXxPlusYY,
            mlir::qc::multipleControlledXxPlusYY},
        QuantumComputationTranslationTestCase{"XXMinusYY", qc::xxMinusYY,
                                              mlir::qc::xxMinusYY},
        QuantumComputationTranslationTestCase{
            "SingleControlledXXMinusYY", qc::singleControlledXxMinusYY,
            mlir::qc::singleControlledXxMinusYY},
        QuantumComputationTranslationTestCase{
            "MultipleControlledXXMinusYY", qc::multipleControlledXxMinusYY,
            mlir::qc::multipleControlledXxMinusYY},
        QuantumComputationTranslationTestCase{"Barrier", qc::barrier,
                                              mlir::qc::barrier},
        QuantumComputationTranslationTestCase{"BarrierTwoQubits",
                                              qc::barrierTwoQubits,
                                              mlir::qc::barrierTwoQubits},
        QuantumComputationTranslationTestCase{"BarrierMultipleQubits",
                                              qc::barrierMultipleQubits,
                                              mlir::qc::barrierMultipleQubits}),
    printTestName);

} // namespace
