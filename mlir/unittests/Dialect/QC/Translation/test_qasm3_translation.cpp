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
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <filesystem>
#include <memory>
#include <ostream>
#include <string>

using namespace mlir;

namespace {

struct QASM3TranslationTestCase {
  std::string name;
  std::string path;
  mqt::test::NamedBuilder<qc::QCProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QASM3TranslationTestCase& test);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const QASM3TranslationTestCase& test) {
  return os << "QASM3Translation{" << test.name << ", original=" << test.path
            << ", reference="
            << mqt::test::displayName(test.referenceBuilder.name) << "}";
}

class QASM3TranslationTest
    : public testing::TestWithParam<QASM3TranslationTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qc::QCDialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

TEST_P(QASM3TranslationTest, ProgramEquivalence) {
  const auto name = " (" + GetParam().name + ")";
  const auto path = (std::filesystem::path(__FILE__).parent_path() /
                     "../../../programs/qasm_programs" / GetParam().path)
                        .lexically_normal()
                        .string();
  const auto referenceBuilder = GetParam().referenceBuilder;
  mqt::test::DeferredPrinter printer;

  auto translated = qc::translateQASM3ToQC(context.get(), path);
  ASSERT_TRUE(translated);
  printer.record(translated.get(), "Translated QC IR" + name);
  EXPECT_TRUE(verify(*translated).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(translated.get()).succeeded());
  printer.record(translated.get(), "Canonicalized Translated QC IR" + name);
  EXPECT_TRUE(verify(*translated).succeeded());

  auto reference =
      qc::QCProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(translated.get(), reference.get()));
}

INSTANTIATE_TEST_SUITE_P(
    QASM3TranslationProgramsTest, QASM3TranslationTest,
    testing::Values(

        QASM3TranslationTestCase{"AllocQubit", "alloc_qubit.qasm",
                                 MQT_NAMED_BUILDER(qc::allocQubit)},
        QASM3TranslationTestCase{"AllocQubitRegister",
                                 "alloc_qubit_register.qasm",
                                 MQT_NAMED_BUILDER(qc::allocQubitRegister)},
        QASM3TranslationTestCase{
            "AllocMultipleQubitRegisters",
            "alloc_multiple_qubit_registers.qasm",
            MQT_NAMED_BUILDER(qc::allocMultipleQubitRegisters)},
        QASM3TranslationTestCase{"AllocLargeRegister",
                                 "alloc_large_register.qasm",
                                 MQT_NAMED_BUILDER(qc::allocLargeRegister)},
        QASM3TranslationTestCase{
            "SingleMeasurementToSingleBit",
            "single_measurement_to_single_bit.qasm",
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit)},
        QASM3TranslationTestCase{
            "RepeatedMeasurementToSameBit",
            "repeated_measurement_to_same_bit.qasm",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit)},
        QASM3TranslationTestCase{
            "RepeatedMeasurementToDifferentBits",
            "repeated_measurement_to_different_bits.qasm",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits)},
        QASM3TranslationTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            "multiple_classical_registers_and_measurements.qasm",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements)},
        QASM3TranslationTestCase{
            "ResetQubitAfterSingleOp", "reset_qubit_after_single_op.qasm",
            MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)},
        QASM3TranslationTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            "reset_multiple_qubits_after_single_op.qasm",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp)},
        QASM3TranslationTestCase{
            "RepeatedResetAfterSingleOp", "repeated_reset_after_single_op.qasm",
            MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp)},
        QASM3TranslationTestCase{"GlobalPhase", "global_phase.qasm",
                                 MQT_NAMED_BUILDER(qc::globalPhase)},
        QASM3TranslationTestCase{"Identity", "identity.qasm",
                                 MQT_NAMED_BUILDER(qc::identity)},
        QASM3TranslationTestCase{
            "SingleControlledIdentity", "single_controlled_identity.qasm",
            MQT_NAMED_BUILDER(qc::singleControlledIdentity)},
        QASM3TranslationTestCase{
            "MultipleControlledIdentity", "multiple_controlled_identity.qasm",
            MQT_NAMED_BUILDER(qc::multipleControlledIdentity)},
        QASM3TranslationTestCase{"X", "x.qasm", MQT_NAMED_BUILDER(qc::x)},
        QASM3TranslationTestCase{"SingleControlledX",
                                 "single_controlled_x.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledX)},
        QASM3TranslationTestCase{"MultipleControlledX",
                                 "multiple_controlled_x.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledX)},
        QASM3TranslationTestCase{"Y", "y.qasm", MQT_NAMED_BUILDER(qc::y)},
        QASM3TranslationTestCase{"SingleControlledY",
                                 "single_controlled_y.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledY)},
        QASM3TranslationTestCase{"MultipleControlledY",
                                 "multiple_controlled_y.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledY)},
        QASM3TranslationTestCase{"Z", "z.qasm", MQT_NAMED_BUILDER(qc::z)},
        QASM3TranslationTestCase{"SingleControlledZ",
                                 "single_controlled_z.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledZ)},
        QASM3TranslationTestCase{"MultipleControlledZ",
                                 "multiple_controlled_z.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledZ)},
        QASM3TranslationTestCase{"H", "h.qasm", MQT_NAMED_BUILDER(qc::h)},
        QASM3TranslationTestCase{"SingleControlledH",
                                 "single_controlled_h.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledH)},
        QASM3TranslationTestCase{"MultipleControlledH",
                                 "multiple_controlled_h.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledH)},
        QASM3TranslationTestCase{"S", "s.qasm", MQT_NAMED_BUILDER(qc::s)},
        QASM3TranslationTestCase{"SingleControlledS",
                                 "single_controlled_s.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledS)},
        QASM3TranslationTestCase{"MultipleControlledS",
                                 "multiple_controlled_s.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledS)},
        QASM3TranslationTestCase{"Sdg", "sdg.qasm", MQT_NAMED_BUILDER(qc::sdg)},
        QASM3TranslationTestCase{"SingleControlledSdg",
                                 "single_controlled_sdg.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledSdg)},
        QASM3TranslationTestCase{"MultipleControlledSdg",
                                 "multiple_controlled_sdg.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledSdg)},
        QASM3TranslationTestCase{"T", "t.qasm", MQT_NAMED_BUILDER(qc::t_)},
        QASM3TranslationTestCase{"SingleControlledT",
                                 "single_controlled_t.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledT)},
        QASM3TranslationTestCase{"MultipleControlledT",
                                 "multiple_controlled_t.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledT)},
        QASM3TranslationTestCase{"Tdg", "tdg.qasm", MQT_NAMED_BUILDER(qc::tdg)},
        QASM3TranslationTestCase{"SingleControlledTdg",
                                 "single_controlled_tdg.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledTdg)},
        QASM3TranslationTestCase{"MultipleControlledTdg",
                                 "multiple_controlled_tdg.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledTdg)},
        QASM3TranslationTestCase{"SX", "sx.qasm", MQT_NAMED_BUILDER(qc::sx)},
        QASM3TranslationTestCase{"SingleControlledSX",
                                 "single_controlled_sx.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledSx)},
        QASM3TranslationTestCase{"MultipleControlledSX",
                                 "multiple_controlled_sx.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledSx)},
        QASM3TranslationTestCase{"SXdg", "sxdg.qasm",
                                 MQT_NAMED_BUILDER(qc::sxdg)},
        QASM3TranslationTestCase{"SingleControlledSXdg",
                                 "single_controlled_sxdg.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledSxdg)},
        QASM3TranslationTestCase{"MultipleControlledSXdg",
                                 "multiple_controlled_sxdg.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledSxdg)},
        QASM3TranslationTestCase{"RX", "rx.qasm", MQT_NAMED_BUILDER(qc::rx)},
        QASM3TranslationTestCase{"SingleControlledRX",
                                 "single_controlled_rx.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledRx)},
        QASM3TranslationTestCase{"MultipleControlledRX",
                                 "multiple_controlled_rx.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledRx)},
        QASM3TranslationTestCase{"RY", "ry.qasm", MQT_NAMED_BUILDER(qc::ry)},
        QASM3TranslationTestCase{"SingleControlledRY",
                                 "single_controlled_ry.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledRy)},
        QASM3TranslationTestCase{"MultipleControlledRY",
                                 "multiple_controlled_ry.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledRy)},
        QASM3TranslationTestCase{"RZ", "rz.qasm", MQT_NAMED_BUILDER(qc::rz)},
        QASM3TranslationTestCase{"SingleControlledRZ",
                                 "single_controlled_rz.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledRz)},
        QASM3TranslationTestCase{"MultipleControlledRZ",
                                 "multiple_controlled_rz.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledRz)},
        QASM3TranslationTestCase{"P", "p.qasm", MQT_NAMED_BUILDER(qc::p)},
        QASM3TranslationTestCase{"SingleControlledP",
                                 "single_controlled_p.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledP)},
        QASM3TranslationTestCase{"MultipleControlledP",
                                 "multiple_controlled_p.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledP)},
        QASM3TranslationTestCase{"R", "r.qasm", MQT_NAMED_BUILDER(qc::r)},
        QASM3TranslationTestCase{"SingleControlledR",
                                 "single_controlled_r.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledR)},
        QASM3TranslationTestCase{"MultipleControlledR",
                                 "multiple_controlled_r.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledR)},
        QASM3TranslationTestCase{"U2", "u2.qasm", MQT_NAMED_BUILDER(qc::u2)},
        QASM3TranslationTestCase{"SingleControlledU2",
                                 "single_controlled_u2.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledU2)},
        QASM3TranslationTestCase{"MultipleControlledU2",
                                 "multiple_controlled_u2.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledU2)},
        QASM3TranslationTestCase{"U", "u.qasm", MQT_NAMED_BUILDER(qc::u)},
        QASM3TranslationTestCase{"SingleControlledU",
                                 "single_controlled_u.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledU)},
        QASM3TranslationTestCase{"MultipleControlledU",
                                 "multiple_controlled_u.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledU)},
        QASM3TranslationTestCase{"SWAP", "swap.qasm",
                                 MQT_NAMED_BUILDER(qc::swap)},
        QASM3TranslationTestCase{"SingleControlledSWAP",
                                 "single_controlled_swap.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledSwap)},
        QASM3TranslationTestCase{"MultipleControlledSWAP",
                                 "multiple_controlled_swap.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledSwap)},
        QASM3TranslationTestCase{"iSWAP", "iswap.qasm",
                                 MQT_NAMED_BUILDER(qc::iswap)},
        QASM3TranslationTestCase{"SingleControllediSWAP",
                                 "single_controlled_iswap.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledIswap)},
        QASM3TranslationTestCase{
            "MultipleControllediSWAP", "multiple_controlled_iswap.qasm",
            MQT_NAMED_BUILDER(qc::multipleControlledIswap)},
        QASM3TranslationTestCase{"InverseISWAP", "inverse_iswap.qasm",
                                 MQT_NAMED_BUILDER(qc::inverseIswap)},
        QASM3TranslationTestCase{
            "InverseMultiControlledISWAP",
            "inverse_multiple_controlled_iswap.qasm",
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap)},
        QASM3TranslationTestCase{"DCX", "dcx.qasm", MQT_NAMED_BUILDER(qc::dcx)},
        QASM3TranslationTestCase{"SingleControlledDCX",
                                 "single_controlled_dcx.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledDcx)},
        QASM3TranslationTestCase{"MultipleControlledDCX",
                                 "multiple_controlled_dcx.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledDcx)},
        QASM3TranslationTestCase{"ECR", "ecr.qasm", MQT_NAMED_BUILDER(qc::ecr)},
        QASM3TranslationTestCase{"SingleControlledECR",
                                 "single_controlled_ecr.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledEcr)},
        QASM3TranslationTestCase{"MultipleControlledECR",
                                 "multiple_controlled_ecr.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledEcr)},
        QASM3TranslationTestCase{"RXX", "rxx.qasm", MQT_NAMED_BUILDER(qc::rxx)},
        QASM3TranslationTestCase{"SingleControlledRXX",
                                 "single_controlled_rxx.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledRxx)},
        QASM3TranslationTestCase{"MultipleControlledRXX",
                                 "multiple_controlled_rxx.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledRxx)},
        QASM3TranslationTestCase{"TripleControlledRXX",
                                 "triple_controlled_rxx.qasm",
                                 MQT_NAMED_BUILDER(qc::tripleControlledRxx)},
        QASM3TranslationTestCase{"RYY", "ryy.qasm", MQT_NAMED_BUILDER(qc::ryy)},
        QASM3TranslationTestCase{"SingleControlledRYY",
                                 "single_controlled_ryy.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledRyy)},
        QASM3TranslationTestCase{"MultipleControlledRYY",
                                 "multiple_controlled_ryy.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledRyy)},
        QASM3TranslationTestCase{"RZX", "rzx.qasm", MQT_NAMED_BUILDER(qc::rzx)},
        QASM3TranslationTestCase{"SingleControlledRZX",
                                 "single_controlled_rzx.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledRzx)},
        QASM3TranslationTestCase{"MultipleControlledRZX",
                                 "multiple_controlled_rzx.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledRzx)},
        QASM3TranslationTestCase{"RZZ", "rzz.qasm", MQT_NAMED_BUILDER(qc::rzz)},
        QASM3TranslationTestCase{"SingleControlledRZZ",
                                 "single_controlled_rzz.qasm",
                                 MQT_NAMED_BUILDER(qc::singleControlledRzz)},
        QASM3TranslationTestCase{"MultipleControlledRZZ",
                                 "multiple_controlled_rzz.qasm",
                                 MQT_NAMED_BUILDER(qc::multipleControlledRzz)},
        QASM3TranslationTestCase{"XXPlusYY", "xx_plus_yy.qasm",
                                 MQT_NAMED_BUILDER(qc::xxPlusYY)},
        QASM3TranslationTestCase{
            "SingleControlledXXPlusYY", "single_controlled_xx_plus_yy.qasm",
            MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY)},
        QASM3TranslationTestCase{
            "MultipleControlledXXPlusYY", "multiple_controlled_xx_plus_yy.qasm",
            MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY)},
        QASM3TranslationTestCase{"XXMinusYY", "xx_minus_yy.qasm",
                                 MQT_NAMED_BUILDER(qc::xxMinusYY)},
        QASM3TranslationTestCase{
            "SingleControlledXXMinusYY", "single_controlled_xx_minus_yy.qasm",
            MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY)},
        QASM3TranslationTestCase{
            "MultipleControlledXXMinusYY",
            "multiple_controlled_xx_minus_yy.qasm",
            MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY)},
        QASM3TranslationTestCase{"Barrier", "barrier.qasm",
                                 MQT_NAMED_BUILDER(qc::barrier)},
        QASM3TranslationTestCase{"BarrierTwoQubits", "barrier_two_qubits.qasm",
                                 MQT_NAMED_BUILDER(qc::barrierTwoQubits)},
        QASM3TranslationTestCase{"BarrierMultipleQubits",
                                 "barrier_multiple_qubits.qasm",
                                 MQT_NAMED_BUILDER(qc::barrierMultipleQubits)},
        QASM3TranslationTestCase{"SimpleIf", "simple_if.qasm",
                                 MQT_NAMED_BUILDER(qc::simpleIf)},
        QASM3TranslationTestCase{"IfElse", "if_else.qasm",
                                 MQT_NAMED_BUILDER(qc::ifElse)}));
