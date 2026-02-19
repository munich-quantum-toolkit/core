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
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <iosfwd>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <ostream>
#include <string>

using namespace mlir;
using namespace mlir::qc;

struct QCTestCase {
  std::string name;
  mqt::test::NamedBuilder<QCProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<QCProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os, const QCTestCase& info);
};

class QCTest : public testing::TestWithParam<QCTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override;
};

void QCTest::SetUp() {
  // Register all necessary dialects
  DialectRegistry registry;
  registry.insert<QCDialect, arith::ArithDialect, func::FuncDialect>();
  context = std::make_unique<MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

std::ostream& operator<<(std::ostream& os, const QCTestCase& info) {
  return os << "QC{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

TEST_P(QCTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";

  auto program = QCProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printProgram(program.get(), "Original QC IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printProgram(program.get(), "Canonicalized QC IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = QCProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printProgram(reference.get(), "Reference QC IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*reference).succeeded());

  runCanonicalizationPasses(reference.get());
  printProgram(reference.get(), "Canonicalized Reference QC IR" + name,
               llvm::errs());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// \name QC/Modifiers/CtrlOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCCtrlOpTest, QCTest,
    testing::Values(QCTestCase{"TrivialCtrl", MQT_NAMED_BUILDER(trivialCtrl),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"NestedCtrl", MQT_NAMED_BUILDER(nestedCtrl),
                               MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCTestCase{"TripleNestedCtrl",
                               MQT_NAMED_BUILDER(tripleNestedCtrl),
                               MQT_NAMED_BUILDER(tripleControlledRxx)},
                    QCTestCase{"CtrlInvSandwich",
                               MQT_NAMED_BUILDER(ctrlInvSandwich),
                               MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCTestCase{"DoubleNestedCtrlTwoQubits",
                               MQT_NAMED_BUILDER(doubleNestedCtrlTwoQubits),
                               MQT_NAMED_BUILDER(fourControlledRxx)}));
/// @}

/// \name QC/Modifiers/InvOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCInvOpTest, QCTest,
    testing::Values(QCTestCase{"NestedInv", MQT_NAMED_BUILDER(nestedInv),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"TripleNestedInv",
                               MQT_NAMED_BUILDER(tripleNestedInv),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"InvControlSandwich",
                               MQT_NAMED_BUILDER(invCtrlSandwich),
                               MQT_NAMED_BUILDER(singleControlledRxx)}));
/// @}

/// \name QC/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCMeasureOpTest, QCTest,
    testing::Values(
        QCTestCase{"SingleMeasurementToSingleBit",
                   MQT_NAMED_BUILDER(singleMeasurementToSingleBit),
                   MQT_NAMED_BUILDER(singleMeasurementToSingleBit)},
        QCTestCase{"RepeatedMeasurementToSameBit",
                   MQT_NAMED_BUILDER(repeatedMeasurementToSameBit),
                   MQT_NAMED_BUILDER(repeatedMeasurementToSameBit)},
        QCTestCase{"RepeatedMeasurementToDifferentBits",
                   MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits),
                   MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits)},
        QCTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(multipleClassicalRegistersAndMeasurements)}));
/// @}

/// \name QC/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCResetOpTest, QCTest,
    testing::Values(QCTestCase{"ResetQubitWithoutOp",
                               MQT_NAMED_BUILDER(resetQubitWithoutOp),
                               MQT_NAMED_BUILDER(resetQubitWithoutOp)},
                    QCTestCase{"ResetMultipleQubitsWithoutOp",
                               MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp),
                               MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp)},
                    QCTestCase{"RepeatedResetWithoutOp",
                               MQT_NAMED_BUILDER(repeatedResetWithoutOp),
                               MQT_NAMED_BUILDER(repeatedResetWithoutOp)},
                    QCTestCase{"ResetQubitAfterSingleOp",
                               MQT_NAMED_BUILDER(resetQubitAfterSingleOp),
                               MQT_NAMED_BUILDER(resetQubitAfterSingleOp)},
                    QCTestCase{
                        "ResetMultipleQubitsAfterSingleOp",
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp),
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp)},
                    QCTestCase{"RepeatedResetAfterSingleOp",
                               MQT_NAMED_BUILDER(repeatedResetAfterSingleOp),
                               MQT_NAMED_BUILDER(repeatedResetAfterSingleOp)}));
/// @}

/// \name QC/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCBarrierOpTest, QCTest,
    testing::Values(QCTestCase{"Barrier", MQT_NAMED_BUILDER(barrier),
                               MQT_NAMED_BUILDER(barrier)},
                    QCTestCase{"BarrierTwoQubits",
                               MQT_NAMED_BUILDER(barrierTwoQubits),
                               MQT_NAMED_BUILDER(barrierTwoQubits)},
                    QCTestCase{"BarrierMultipleQubits",
                               MQT_NAMED_BUILDER(barrierMultipleQubits),
                               MQT_NAMED_BUILDER(barrierMultipleQubits)},
                    QCTestCase{"SingleControlledBarrier",
                               MQT_NAMED_BUILDER(singleControlledBarrier),
                               MQT_NAMED_BUILDER(barrier)},
                    QCTestCase{"InverseBarrier",
                               MQT_NAMED_BUILDER(inverseBarrier),
                               MQT_NAMED_BUILDER(barrier)}));
/// @}

/// \name QC/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCDCXOpTest, QCTest,
    testing::Values(QCTestCase{"DCX", MQT_NAMED_BUILDER(dcx),
                               MQT_NAMED_BUILDER(dcx)},
                    QCTestCase{"SingleControlledDCX",
                               MQT_NAMED_BUILDER(singleControlledDcx),
                               MQT_NAMED_BUILDER(singleControlledDcx)},
                    QCTestCase{"MultipleControlledDCX",
                               MQT_NAMED_BUILDER(multipleControlledDcx),
                               MQT_NAMED_BUILDER(multipleControlledDcx)},
                    QCTestCase{"NestedControlledDCX",
                               MQT_NAMED_BUILDER(nestedControlledDcx),
                               MQT_NAMED_BUILDER(multipleControlledDcx)},
                    QCTestCase{"TrivialControlledDCX",
                               MQT_NAMED_BUILDER(trivialControlledDcx),
                               MQT_NAMED_BUILDER(dcx)},
                    QCTestCase{"InverseDCX", MQT_NAMED_BUILDER(inverseDcx),
                               MQT_NAMED_BUILDER(dcx)},
                    QCTestCase{"InverseMultipleControlledDCX",
                               MQT_NAMED_BUILDER(inverseMultipleControlledDcx),
                               MQT_NAMED_BUILDER(multipleControlledDcx)}));
/// @}

/// \name QC/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCECROpTest, QCTest,
    testing::Values(QCTestCase{"ECR", MQT_NAMED_BUILDER(ecr),
                               MQT_NAMED_BUILDER(ecr)},
                    QCTestCase{"SingleControlledECR",
                               MQT_NAMED_BUILDER(singleControlledEcr),
                               MQT_NAMED_BUILDER(singleControlledEcr)},
                    QCTestCase{"MultipleControlledECR",
                               MQT_NAMED_BUILDER(multipleControlledEcr),
                               MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCTestCase{"NestedControlledECR",
                               MQT_NAMED_BUILDER(nestedControlledEcr),
                               MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCTestCase{"TrivialControlledECR",
                               MQT_NAMED_BUILDER(trivialControlledEcr),
                               MQT_NAMED_BUILDER(ecr)},
                    QCTestCase{"InverseECR", MQT_NAMED_BUILDER(inverseEcr),
                               MQT_NAMED_BUILDER(ecr)},
                    QCTestCase{"InverseMultipleControlledECR",
                               MQT_NAMED_BUILDER(inverseMultipleControlledEcr),
                               MQT_NAMED_BUILDER(multipleControlledEcr)}));
/// @}

/// \name QC/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCGPhaseOpTest, QCTest,
    testing::Values(
        QCTestCase{"GlobalPhase", MQT_NAMED_BUILDER(globalPhase),
                   MQT_NAMED_BUILDER(globalPhase)},
        QCTestCase{"SingleControlledGlobalPhase",
                   MQT_NAMED_BUILDER(singleControlledGlobalPhase),
                   MQT_NAMED_BUILDER(p)},
        QCTestCase{"MultipleControlledGlobalPhase",
                   MQT_NAMED_BUILDER(multipleControlledGlobalPhase),
                   MQT_NAMED_BUILDER(multipleControlledP)},
        QCTestCase{"NestedControlledGlobalPhase",
                   MQT_NAMED_BUILDER(nestedControlledGlobalPhase),
                   MQT_NAMED_BUILDER(singleControlledP)},
        QCTestCase{"TrivialControlledGlobalPhase",
                   MQT_NAMED_BUILDER(trivialControlledGlobalPhase),
                   MQT_NAMED_BUILDER(globalPhase)},
        QCTestCase{"InverseGlobalPhase", MQT_NAMED_BUILDER(inverseGlobalPhase),
                   MQT_NAMED_BUILDER(globalPhase)},
        QCTestCase{"InverseMultipleControlledGlobalPhase",
                   MQT_NAMED_BUILDER(inverseMultipleControlledGlobalPhase),
                   MQT_NAMED_BUILDER(multipleControlledP)}));
/// @}

/// \name QC/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCHOpTest, QCTest,
    testing::Values(
        QCTestCase{"H", MQT_NAMED_BUILDER(h), MQT_NAMED_BUILDER(h)},
        QCTestCase{"SingleControlledH", MQT_NAMED_BUILDER(singleControlledH),
                   MQT_NAMED_BUILDER(singleControlledH)},
        QCTestCase{"MultipleControlledH",
                   MQT_NAMED_BUILDER(multipleControlledH),
                   MQT_NAMED_BUILDER(multipleControlledH)},
        QCTestCase{"NestedControlledH", MQT_NAMED_BUILDER(nestedControlledH),
                   MQT_NAMED_BUILDER(multipleControlledH)},
        QCTestCase{"TrivialControlledH", MQT_NAMED_BUILDER(trivialControlledH),
                   MQT_NAMED_BUILDER(h)},
        QCTestCase{"InverseH", MQT_NAMED_BUILDER(inverseH),
                   MQT_NAMED_BUILDER(h)},
        QCTestCase{"InverseMultipleControlledH",
                   MQT_NAMED_BUILDER(inverseMultipleControlledH),
                   MQT_NAMED_BUILDER(multipleControlledH)}));
/// @}

/// \name QC/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCIDOpTest, QCTest,
    testing::Values(
        QCTestCase{"Identity", MQT_NAMED_BUILDER(identity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"SingleControlledIdentity",
                   MQT_NAMED_BUILDER(singleControlledIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"MultipleControlledIdentity",
                   MQT_NAMED_BUILDER(multipleControlledIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"NestedControlledIdentity",
                   MQT_NAMED_BUILDER(nestedControlledIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"TrivialControlledIdentity",
                   MQT_NAMED_BUILDER(trivialControlledIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"InverseIdentity", MQT_NAMED_BUILDER(inverseIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"InverseMultipleControlledIdentity",
                   MQT_NAMED_BUILDER(inverseMultipleControlledIdentity),
                   MQT_NAMED_BUILDER(identity)}));
/// @}

/// \name QC/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCiSWAPOpTest, QCTest,
    testing::Values(
        QCTestCase{"iSWAP", MQT_NAMED_BUILDER(iswap), MQT_NAMED_BUILDER(iswap)},
        QCTestCase{"SingleControllediSWAP",
                   MQT_NAMED_BUILDER(singleControlledIswap),
                   MQT_NAMED_BUILDER(singleControlledIswap)},
        QCTestCase{"MultipleControllediSWAP",
                   MQT_NAMED_BUILDER(multipleControlledIswap),
                   MQT_NAMED_BUILDER(multipleControlledIswap)},
        QCTestCase{"NestedControllediSWAP",
                   MQT_NAMED_BUILDER(nestedControlledIswap),
                   MQT_NAMED_BUILDER(multipleControlledIswap)},
        QCTestCase{"TrivialControllediSWAP",
                   MQT_NAMED_BUILDER(trivialControlledIswap),
                   MQT_NAMED_BUILDER(iswap)},
        QCTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(inverseIswap),
                   MQT_NAMED_BUILDER(inverseIswap)},
        QCTestCase{"InverseMultipleControllediSWAP",
                   MQT_NAMED_BUILDER(inverseMultipleControlledIswap),
                   MQT_NAMED_BUILDER(inverseMultipleControlledIswap)}));
/// @}

/// \name QC/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCPOpTest, QCTest,
    testing::Values(
        QCTestCase{"P", MQT_NAMED_BUILDER(p), MQT_NAMED_BUILDER(p)},
        QCTestCase{"SingleControlledP", MQT_NAMED_BUILDER(singleControlledP),
                   MQT_NAMED_BUILDER(singleControlledP)},
        QCTestCase{"MultipleControlledP",
                   MQT_NAMED_BUILDER(multipleControlledP),
                   MQT_NAMED_BUILDER(multipleControlledP)},
        QCTestCase{"NestedControlledP", MQT_NAMED_BUILDER(nestedControlledP),
                   MQT_NAMED_BUILDER(multipleControlledP)},
        QCTestCase{"TrivialControlledP", MQT_NAMED_BUILDER(trivialControlledP),
                   MQT_NAMED_BUILDER(p)},
        QCTestCase{"InverseP", MQT_NAMED_BUILDER(inverseP),
                   MQT_NAMED_BUILDER(p)},
        QCTestCase{"InverseMultipleControlledP",
                   MQT_NAMED_BUILDER(inverseMultipleControlledP),
                   MQT_NAMED_BUILDER(multipleControlledP)}));
/// @}

/// \name QC/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCROpTest, QCTest,
    testing::Values(
        QCTestCase{"R", MQT_NAMED_BUILDER(r), MQT_NAMED_BUILDER(r)},
        QCTestCase{"SingleControlledR", MQT_NAMED_BUILDER(singleControlledR),
                   MQT_NAMED_BUILDER(singleControlledR)},
        QCTestCase{"MultipleControlledR",
                   MQT_NAMED_BUILDER(multipleControlledR),
                   MQT_NAMED_BUILDER(multipleControlledR)},
        QCTestCase{"NestedControlledR", MQT_NAMED_BUILDER(nestedControlledR),
                   MQT_NAMED_BUILDER(multipleControlledR)},
        QCTestCase{"TrivialControlledR", MQT_NAMED_BUILDER(trivialControlledR),
                   MQT_NAMED_BUILDER(r)},
        QCTestCase{"InverseR", MQT_NAMED_BUILDER(inverseR),
                   MQT_NAMED_BUILDER(r)},
        QCTestCase{"InverseMultipleControlledR",
                   MQT_NAMED_BUILDER(inverseMultipleControlledR),
                   MQT_NAMED_BUILDER(multipleControlledR)}));
/// @}

/// \name QC/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRXOpTest, QCTest,
    testing::Values(
        QCTestCase{"RX", MQT_NAMED_BUILDER(rx), MQT_NAMED_BUILDER(rx)},
        QCTestCase{"SingleControlledRX", MQT_NAMED_BUILDER(singleControlledRx),
                   MQT_NAMED_BUILDER(singleControlledRx)},
        QCTestCase{"MultipleControlledRX",
                   MQT_NAMED_BUILDER(multipleControlledRx),
                   MQT_NAMED_BUILDER(multipleControlledRx)},
        QCTestCase{"NestedControlledRX", MQT_NAMED_BUILDER(nestedControlledRx),
                   MQT_NAMED_BUILDER(multipleControlledRx)},
        QCTestCase{"TrivialControlledRX",
                   MQT_NAMED_BUILDER(trivialControlledRx),
                   MQT_NAMED_BUILDER(rx)},
        QCTestCase{"InverseRX", MQT_NAMED_BUILDER(inverseRx),
                   MQT_NAMED_BUILDER(rx)},
        QCTestCase{"InverseMultipleControlledRX",
                   MQT_NAMED_BUILDER(inverseMultipleControlledRx),
                   MQT_NAMED_BUILDER(multipleControlledRx)}));
/// @}

/// \name QC/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRXXOpTest, QCTest,
    testing::Values(QCTestCase{"RXX", MQT_NAMED_BUILDER(rxx),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"SingleControlledRXX",
                               MQT_NAMED_BUILDER(singleControlledRxx),
                               MQT_NAMED_BUILDER(singleControlledRxx)},
                    QCTestCase{"MultipleControlledRXX",
                               MQT_NAMED_BUILDER(multipleControlledRxx),
                               MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCTestCase{"NestedControlledRXX",
                               MQT_NAMED_BUILDER(nestedControlledRxx),
                               MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCTestCase{"TrivialControlledRXX",
                               MQT_NAMED_BUILDER(trivialControlledRxx),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"InverseRXX", MQT_NAMED_BUILDER(inverseRxx),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"InverseMultipleControlledRXX",
                               MQT_NAMED_BUILDER(inverseMultipleControlledRxx),
                               MQT_NAMED_BUILDER(multipleControlledRxx)}));
/// @}

/// \name QC/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRYOpTest, QCTest,
    testing::Values(
        QCTestCase{"RY", MQT_NAMED_BUILDER(ry), MQT_NAMED_BUILDER(ry)},
        QCTestCase{"SingleControlledRY", MQT_NAMED_BUILDER(singleControlledRy),
                   MQT_NAMED_BUILDER(singleControlledRy)},
        QCTestCase{"MultipleControlledRY",
                   MQT_NAMED_BUILDER(multipleControlledRy),
                   MQT_NAMED_BUILDER(multipleControlledRy)},
        QCTestCase{"NestedControlledRY", MQT_NAMED_BUILDER(nestedControlledRy),
                   MQT_NAMED_BUILDER(multipleControlledRy)},
        QCTestCase{"TrivialControlledRY",
                   MQT_NAMED_BUILDER(trivialControlledRy),
                   MQT_NAMED_BUILDER(ry)},
        QCTestCase{"InverseRY", MQT_NAMED_BUILDER(inverseRy),
                   MQT_NAMED_BUILDER(ry)},
        QCTestCase{"InverseMultipleControlledRY",
                   MQT_NAMED_BUILDER(inverseMultipleControlledRy),
                   MQT_NAMED_BUILDER(multipleControlledRy)}));
/// @}

/// \name QC/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRYYOpTest, QCTest,
    testing::Values(QCTestCase{"RYY", MQT_NAMED_BUILDER(ryy),
                               MQT_NAMED_BUILDER(ryy)},
                    QCTestCase{"SingleControlledRYY",
                               MQT_NAMED_BUILDER(singleControlledRyy),
                               MQT_NAMED_BUILDER(singleControlledRyy)},
                    QCTestCase{"MultipleControlledRYY",
                               MQT_NAMED_BUILDER(multipleControlledRyy),
                               MQT_NAMED_BUILDER(multipleControlledRyy)},
                    QCTestCase{"NestedControlledRYY",
                               MQT_NAMED_BUILDER(nestedControlledRyy),
                               MQT_NAMED_BUILDER(multipleControlledRyy)},
                    QCTestCase{"TrivialControlledRYY",
                               MQT_NAMED_BUILDER(trivialControlledRyy),
                               MQT_NAMED_BUILDER(ryy)},
                    QCTestCase{"InverseRYY", MQT_NAMED_BUILDER(inverseRyy),
                               MQT_NAMED_BUILDER(ryy)},
                    QCTestCase{"InverseMultipleControlledRYY",
                               MQT_NAMED_BUILDER(inverseMultipleControlledRyy),
                               MQT_NAMED_BUILDER(multipleControlledRyy)}));
/// @}

/// \name QC/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZOpTest, QCTest,
    testing::Values(
        QCTestCase{"RZ", MQT_NAMED_BUILDER(rz), MQT_NAMED_BUILDER(rz)},
        QCTestCase{"SingleControlledRZ", MQT_NAMED_BUILDER(singleControlledRz),
                   MQT_NAMED_BUILDER(singleControlledRz)},
        QCTestCase{"MultipleControlledRZ",
                   MQT_NAMED_BUILDER(multipleControlledRz),
                   MQT_NAMED_BUILDER(multipleControlledRz)},
        QCTestCase{"NestedControlledRZ", MQT_NAMED_BUILDER(nestedControlledRz),
                   MQT_NAMED_BUILDER(multipleControlledRz)},
        QCTestCase{"TrivialControlledRZ",
                   MQT_NAMED_BUILDER(trivialControlledRz),
                   MQT_NAMED_BUILDER(rz)},
        QCTestCase{"InverseRZ", MQT_NAMED_BUILDER(inverseRz),
                   MQT_NAMED_BUILDER(rz)},
        QCTestCase{"InverseMultipleControlledRZ",
                   MQT_NAMED_BUILDER(inverseMultipleControlledRz),
                   MQT_NAMED_BUILDER(multipleControlledRz)}));
/// @}

/// \name QC/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZXOpTest, QCTest,
    testing::Values(QCTestCase{"RZX", MQT_NAMED_BUILDER(rzx),
                               MQT_NAMED_BUILDER(rzx)},
                    QCTestCase{"SingleControlledRZX",
                               MQT_NAMED_BUILDER(singleControlledRzx),
                               MQT_NAMED_BUILDER(singleControlledRzx)},
                    QCTestCase{"MultipleControlledRZX",
                               MQT_NAMED_BUILDER(multipleControlledRzx),
                               MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCTestCase{"NestedControlledRZX",
                               MQT_NAMED_BUILDER(nestedControlledRzx),
                               MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCTestCase{"TrivialControlledRZX",
                               MQT_NAMED_BUILDER(trivialControlledRzx),
                               MQT_NAMED_BUILDER(rzx)},
                    QCTestCase{"InverseRZX", MQT_NAMED_BUILDER(inverseRzx),
                               MQT_NAMED_BUILDER(rzx)},
                    QCTestCase{"InverseMultipleControlledRZX",
                               MQT_NAMED_BUILDER(inverseMultipleControlledRzx),
                               MQT_NAMED_BUILDER(multipleControlledRzx)}));
/// @}

/// \name QC/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZZOpTest, QCTest,
    testing::Values(QCTestCase{"RZZ", MQT_NAMED_BUILDER(rzz),
                               MQT_NAMED_BUILDER(rzz)},
                    QCTestCase{"SingleControlledRZZ",
                               MQT_NAMED_BUILDER(singleControlledRzz),
                               MQT_NAMED_BUILDER(singleControlledRzz)},
                    QCTestCase{"MultipleControlledRZZ",
                               MQT_NAMED_BUILDER(multipleControlledRzz),
                               MQT_NAMED_BUILDER(multipleControlledRzz)},
                    QCTestCase{"NestedControlledRZZ",
                               MQT_NAMED_BUILDER(nestedControlledRzz),
                               MQT_NAMED_BUILDER(multipleControlledRzz)},
                    QCTestCase{"TrivialControlledRZZ",
                               MQT_NAMED_BUILDER(trivialControlledRzz),
                               MQT_NAMED_BUILDER(rzz)},
                    QCTestCase{"InverseRZZ", MQT_NAMED_BUILDER(inverseRzz),
                               MQT_NAMED_BUILDER(rzz)},
                    QCTestCase{"InverseMultipleControlledRZZ",
                               MQT_NAMED_BUILDER(inverseMultipleControlledRzz),
                               MQT_NAMED_BUILDER(multipleControlledRzz)}));
/// @}

/// \name QC/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSOpTest, QCTest,
    testing::Values(
        QCTestCase{"S", MQT_NAMED_BUILDER(s), MQT_NAMED_BUILDER(s)},
        QCTestCase{"SingleControlledS", MQT_NAMED_BUILDER(singleControlledS),
                   MQT_NAMED_BUILDER(singleControlledS)},
        QCTestCase{"MultipleControlledS",
                   MQT_NAMED_BUILDER(multipleControlledS),
                   MQT_NAMED_BUILDER(multipleControlledS)},
        QCTestCase{"NestedControlledS", MQT_NAMED_BUILDER(nestedControlledS),
                   MQT_NAMED_BUILDER(multipleControlledS)},
        QCTestCase{"TrivialControlledS", MQT_NAMED_BUILDER(trivialControlledS),
                   MQT_NAMED_BUILDER(s)},
        QCTestCase{"InverseS", MQT_NAMED_BUILDER(inverseS),
                   MQT_NAMED_BUILDER(sdg)},
        QCTestCase{"InverseMultipleControlledS",
                   MQT_NAMED_BUILDER(inverseMultipleControlledS),
                   MQT_NAMED_BUILDER(multipleControlledSdg)}));
/// @}

/// \name QC/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSdgOpTest, QCTest,
    testing::Values(QCTestCase{"Sdg", MQT_NAMED_BUILDER(sdg),
                               MQT_NAMED_BUILDER(sdg)},
                    QCTestCase{"SingleControlledSdg",
                               MQT_NAMED_BUILDER(singleControlledSdg),
                               MQT_NAMED_BUILDER(singleControlledSdg)},
                    QCTestCase{"MultipleControlledSdg",
                               MQT_NAMED_BUILDER(multipleControlledSdg),
                               MQT_NAMED_BUILDER(multipleControlledSdg)},
                    QCTestCase{"NestedControlledSdg",
                               MQT_NAMED_BUILDER(nestedControlledSdg),
                               MQT_NAMED_BUILDER(multipleControlledSdg)},
                    QCTestCase{"TrivialControlledSdg",
                               MQT_NAMED_BUILDER(trivialControlledSdg),
                               MQT_NAMED_BUILDER(sdg)},
                    QCTestCase{"InverseSdg", MQT_NAMED_BUILDER(inverseSdg),
                               MQT_NAMED_BUILDER(s)},
                    QCTestCase{"InverseMultipleControlledSdg",
                               MQT_NAMED_BUILDER(inverseMultipleControlledSdg),
                               MQT_NAMED_BUILDER(multipleControlledS)}));
/// @}

/// \name QC/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSWAPOpTest, QCTest,
    testing::Values(QCTestCase{"SWAP", MQT_NAMED_BUILDER(swap),
                               MQT_NAMED_BUILDER(swap)},
                    QCTestCase{"SingleControlledSWAP",
                               MQT_NAMED_BUILDER(singleControlledSwap),
                               MQT_NAMED_BUILDER(singleControlledSwap)},
                    QCTestCase{"MultipleControlledSWAP",
                               MQT_NAMED_BUILDER(multipleControlledSwap),
                               MQT_NAMED_BUILDER(multipleControlledSwap)},
                    QCTestCase{"NestedControlledSWAP",
                               MQT_NAMED_BUILDER(nestedControlledSwap),
                               MQT_NAMED_BUILDER(multipleControlledSwap)},
                    QCTestCase{"TrivialControlledSWAP",
                               MQT_NAMED_BUILDER(trivialControlledSwap),
                               MQT_NAMED_BUILDER(swap)},
                    QCTestCase{"InverseSWAP", MQT_NAMED_BUILDER(inverseSwap),
                               MQT_NAMED_BUILDER(swap)},
                    QCTestCase{"InverseMultipleControlledSWAP",
                               MQT_NAMED_BUILDER(inverseMultipleControlledSwap),
                               MQT_NAMED_BUILDER(multipleControlledSwap)}));
/// @}

/// \name QC/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSXOpTest, QCTest,
    testing::Values(
        QCTestCase{"SX", MQT_NAMED_BUILDER(sx), MQT_NAMED_BUILDER(sx)},
        QCTestCase{"SingleControlledSX", MQT_NAMED_BUILDER(singleControlledSx),
                   MQT_NAMED_BUILDER(singleControlledSx)},
        QCTestCase{"MultipleControlledSX",
                   MQT_NAMED_BUILDER(multipleControlledSx),
                   MQT_NAMED_BUILDER(multipleControlledSx)},
        QCTestCase{"NestedControlledSX", MQT_NAMED_BUILDER(nestedControlledSx),
                   MQT_NAMED_BUILDER(multipleControlledSx)},
        QCTestCase{"TrivialControlledSX",
                   MQT_NAMED_BUILDER(trivialControlledSx),
                   MQT_NAMED_BUILDER(sx)},
        QCTestCase{"InverseSX", MQT_NAMED_BUILDER(inverseSx),
                   MQT_NAMED_BUILDER(sxdg)},
        QCTestCase{"InverseMultipleControlledSX",
                   MQT_NAMED_BUILDER(inverseMultipleControlledSx),
                   MQT_NAMED_BUILDER(multipleControlledSxdg)}));
/// @}

/// \name QC/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSXdgOpTest, QCTest,
    testing::Values(QCTestCase{"SXdg", MQT_NAMED_BUILDER(sxdg),
                               MQT_NAMED_BUILDER(sxdg)},
                    QCTestCase{"SingleControlledSXdg",
                               MQT_NAMED_BUILDER(singleControlledSxdg),
                               MQT_NAMED_BUILDER(singleControlledSxdg)},
                    QCTestCase{"MultipleControlledSXdg",
                               MQT_NAMED_BUILDER(multipleControlledSxdg),
                               MQT_NAMED_BUILDER(multipleControlledSxdg)},
                    QCTestCase{"NestedControlledSXdg",
                               MQT_NAMED_BUILDER(nestedControlledSxdg),
                               MQT_NAMED_BUILDER(multipleControlledSxdg)},
                    QCTestCase{"TrivialControlledSXdg",
                               MQT_NAMED_BUILDER(trivialControlledSxdg),
                               MQT_NAMED_BUILDER(sxdg)},
                    QCTestCase{"InverseSXdg", MQT_NAMED_BUILDER(inverseSxdg),
                               MQT_NAMED_BUILDER(sx)},
                    QCTestCase{"InverseMultipleControlledSXdg",
                               MQT_NAMED_BUILDER(inverseMultipleControlledSxdg),
                               MQT_NAMED_BUILDER(multipleControlledSx)}));
/// @}

/// \name QC/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCTOpTest, QCTest,
    testing::Values(
        QCTestCase{"T", MQT_NAMED_BUILDER(t_), MQT_NAMED_BUILDER(t_)},
        QCTestCase{"SingleControlledT", MQT_NAMED_BUILDER(singleControlledT),
                   MQT_NAMED_BUILDER(singleControlledT)},
        QCTestCase{"MultipleControlledT",
                   MQT_NAMED_BUILDER(multipleControlledT),
                   MQT_NAMED_BUILDER(multipleControlledT)},
        QCTestCase{"NestedControlledT", MQT_NAMED_BUILDER(nestedControlledT),
                   MQT_NAMED_BUILDER(multipleControlledT)},
        QCTestCase{"TrivialControlledT", MQT_NAMED_BUILDER(trivialControlledT),
                   MQT_NAMED_BUILDER(t_)},
        QCTestCase{"InverseT", MQT_NAMED_BUILDER(inverseT),
                   MQT_NAMED_BUILDER(tdg)},
        QCTestCase{"InverseMultipleControlledT",
                   MQT_NAMED_BUILDER(inverseMultipleControlledT),
                   MQT_NAMED_BUILDER(multipleControlledTdg)}));
/// @}

/// \name QC/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCTdgOpTest, QCTest,
    testing::Values(QCTestCase{"Tdg", MQT_NAMED_BUILDER(tdg),
                               MQT_NAMED_BUILDER(tdg)},
                    QCTestCase{"SingleControlledTdg",
                               MQT_NAMED_BUILDER(singleControlledTdg),
                               MQT_NAMED_BUILDER(singleControlledTdg)},
                    QCTestCase{"MultipleControlledTdg",
                               MQT_NAMED_BUILDER(multipleControlledTdg),
                               MQT_NAMED_BUILDER(multipleControlledTdg)},
                    QCTestCase{"NestedControlledTdg",
                               MQT_NAMED_BUILDER(nestedControlledTdg),
                               MQT_NAMED_BUILDER(multipleControlledTdg)},
                    QCTestCase{"TrivialControlledTdg",
                               MQT_NAMED_BUILDER(trivialControlledTdg),
                               MQT_NAMED_BUILDER(tdg)},
                    QCTestCase{"InverseTdg", MQT_NAMED_BUILDER(inverseTdg),
                               MQT_NAMED_BUILDER(t_)},
                    QCTestCase{"InverseMultipleControlledTdg",
                               MQT_NAMED_BUILDER(inverseMultipleControlledTdg),
                               MQT_NAMED_BUILDER(multipleControlledT)}));
/// @}

/// \name QC/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCU2OpTest, QCTest,
    testing::Values(
        QCTestCase{"U2", MQT_NAMED_BUILDER(u2), MQT_NAMED_BUILDER(u2)},
        QCTestCase{"SingleControlledU2", MQT_NAMED_BUILDER(singleControlledU2),
                   MQT_NAMED_BUILDER(singleControlledU2)},
        QCTestCase{"MultipleControlledU2",
                   MQT_NAMED_BUILDER(multipleControlledU2),
                   MQT_NAMED_BUILDER(multipleControlledU2)},
        QCTestCase{"NestedControlledU2", MQT_NAMED_BUILDER(nestedControlledU2),
                   MQT_NAMED_BUILDER(multipleControlledU2)},
        QCTestCase{"TrivialControlledU2",
                   MQT_NAMED_BUILDER(trivialControlledU2),
                   MQT_NAMED_BUILDER(u2)},
        QCTestCase{"InverseU2", MQT_NAMED_BUILDER(inverseU2),
                   MQT_NAMED_BUILDER(u2)},
        QCTestCase{"InverseMultipleControlledU2",
                   MQT_NAMED_BUILDER(inverseMultipleControlledU2),
                   MQT_NAMED_BUILDER(multipleControlledU2)}));
/// @}

/// \name QC/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCUOpTest, QCTest,
    testing::Values(
        QCTestCase{"U", MQT_NAMED_BUILDER(u), MQT_NAMED_BUILDER(u)},
        QCTestCase{"SingleControlledU", MQT_NAMED_BUILDER(singleControlledU),
                   MQT_NAMED_BUILDER(singleControlledU)},
        QCTestCase{"MultipleControlledU",
                   MQT_NAMED_BUILDER(multipleControlledU),
                   MQT_NAMED_BUILDER(multipleControlledU)},
        QCTestCase{"NestedControlledU", MQT_NAMED_BUILDER(nestedControlledU),
                   MQT_NAMED_BUILDER(multipleControlledU)},
        QCTestCase{"TrivialControlledU", MQT_NAMED_BUILDER(trivialControlledU),
                   MQT_NAMED_BUILDER(u)},
        QCTestCase{"InverseU", MQT_NAMED_BUILDER(inverseU),
                   MQT_NAMED_BUILDER(u)},
        QCTestCase{"InverseMultipleControlledU",
                   MQT_NAMED_BUILDER(inverseMultipleControlledU),
                   MQT_NAMED_BUILDER(multipleControlledU)}));
/// @}

/// \name QC/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXOpTest, QCTest,
    testing::Values(
        QCTestCase{"X", MQT_NAMED_BUILDER(x), MQT_NAMED_BUILDER(x)},
        QCTestCase{"SingleControlledX", MQT_NAMED_BUILDER(singleControlledX),
                   MQT_NAMED_BUILDER(singleControlledX)},
        QCTestCase{"MultipleControlledX",
                   MQT_NAMED_BUILDER(multipleControlledX),
                   MQT_NAMED_BUILDER(multipleControlledX)},
        QCTestCase{"NestedControlledX", MQT_NAMED_BUILDER(nestedControlledX),
                   MQT_NAMED_BUILDER(multipleControlledX)},
        QCTestCase{"TrivialControlledX", MQT_NAMED_BUILDER(trivialControlledX),
                   MQT_NAMED_BUILDER(x)},
        QCTestCase{"InverseX", MQT_NAMED_BUILDER(inverseX),
                   MQT_NAMED_BUILDER(x)},
        QCTestCase{"InverseMultipleControlledX",
                   MQT_NAMED_BUILDER(inverseMultipleControlledX),
                   MQT_NAMED_BUILDER(multipleControlledX)}));
/// @}

/// \name QC/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXXMinusYYOpTest, QCTest,
    testing::Values(
        QCTestCase{"XXMinusYY", MQT_NAMED_BUILDER(xxMinusYY),
                   MQT_NAMED_BUILDER(xxMinusYY)},
        QCTestCase{"SingleControlledXXMinusYY",
                   MQT_NAMED_BUILDER(singleControlledXxMinusYY),
                   MQT_NAMED_BUILDER(singleControlledXxMinusYY)},
        QCTestCase{"MultipleControlledXXMinusYY",
                   MQT_NAMED_BUILDER(multipleControlledXxMinusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCTestCase{"NestedControlledXXMinusYY",
                   MQT_NAMED_BUILDER(nestedControlledXxMinusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCTestCase{"TrivialControlledXXMinusYY",
                   MQT_NAMED_BUILDER(trivialControlledXxMinusYY),
                   MQT_NAMED_BUILDER(xxMinusYY)},
        QCTestCase{"InverseXXMinusYY", MQT_NAMED_BUILDER(inverseXxMinusYY),
                   MQT_NAMED_BUILDER(xxMinusYY)},
        QCTestCase{"InverseMultipleControlledXXMinusYY",
                   MQT_NAMED_BUILDER(inverseMultipleControlledXxMinusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxMinusYY)}));
/// @}

/// \name QC/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXXPlusYYOpTest, QCTest,
    testing::Values(
        QCTestCase{"XXPlusYY", MQT_NAMED_BUILDER(xxPlusYY),
                   MQT_NAMED_BUILDER(xxPlusYY)},
        QCTestCase{"SingleControlledXXPlusYY",
                   MQT_NAMED_BUILDER(singleControlledXxPlusYY),
                   MQT_NAMED_BUILDER(singleControlledXxPlusYY)},
        QCTestCase{"MultipleControlledXXPlusYY",
                   MQT_NAMED_BUILDER(multipleControlledXxPlusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCTestCase{"NestedControlledXXPlusYY",
                   MQT_NAMED_BUILDER(nestedControlledXxPlusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCTestCase{"TrivialControlledXXPlusYY",
                   MQT_NAMED_BUILDER(trivialControlledXxPlusYY),
                   MQT_NAMED_BUILDER(xxPlusYY)},
        QCTestCase{"InverseXXPlusYY", MQT_NAMED_BUILDER(inverseXxPlusYY),
                   MQT_NAMED_BUILDER(xxPlusYY)},
        QCTestCase{"InverseMultipleControlledXXPlusYY",
                   MQT_NAMED_BUILDER(inverseMultipleControlledXxPlusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxPlusYY)}));
/// @}

/// \name QC/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCYOpTest, QCTest,
    testing::Values(
        QCTestCase{"Y", MQT_NAMED_BUILDER(y), MQT_NAMED_BUILDER(y)},
        QCTestCase{"SingleControlledY", MQT_NAMED_BUILDER(singleControlledY),
                   MQT_NAMED_BUILDER(singleControlledY)},
        QCTestCase{"MultipleControlledY",
                   MQT_NAMED_BUILDER(multipleControlledY),
                   MQT_NAMED_BUILDER(multipleControlledY)},
        QCTestCase{"NestedControlledY", MQT_NAMED_BUILDER(nestedControlledY),
                   MQT_NAMED_BUILDER(multipleControlledY)},
        QCTestCase{"TrivialControlledY", MQT_NAMED_BUILDER(trivialControlledY),
                   MQT_NAMED_BUILDER(y)},
        QCTestCase{"InverseY", MQT_NAMED_BUILDER(inverseY),
                   MQT_NAMED_BUILDER(y)},
        QCTestCase{"InverseMultipleControlledY",
                   MQT_NAMED_BUILDER(inverseMultipleControlledY),
                   MQT_NAMED_BUILDER(multipleControlledY)}));
/// @}

/// \name QC/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCZOpTest, QCTest,
    testing::Values(
        QCTestCase{"Z", MQT_NAMED_BUILDER(z), MQT_NAMED_BUILDER(z)},
        QCTestCase{"SingleControlledZ", MQT_NAMED_BUILDER(singleControlledZ),
                   MQT_NAMED_BUILDER(singleControlledZ)},
        QCTestCase{"MultipleControlledZ",
                   MQT_NAMED_BUILDER(multipleControlledZ),
                   MQT_NAMED_BUILDER(multipleControlledZ)},
        QCTestCase{"NestedControlledZ", MQT_NAMED_BUILDER(nestedControlledZ),
                   MQT_NAMED_BUILDER(multipleControlledZ)},
        QCTestCase{"TrivialControlledZ", MQT_NAMED_BUILDER(trivialControlledZ),
                   MQT_NAMED_BUILDER(z)},
        QCTestCase{"InverseZ", MQT_NAMED_BUILDER(inverseZ),
                   MQT_NAMED_BUILDER(z)},
        QCTestCase{"InverseMultipleControlledZ",
                   MQT_NAMED_BUILDER(inverseMultipleControlledZ),
                   MQT_NAMED_BUILDER(multipleControlledZ)}));
/// @}

/// \name QC/QubitManagement/QubitManagement.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCQubitManagementTest, QCTest,
    testing::Values(
        QCTestCase{"AllocQubit", MQT_NAMED_BUILDER(allocQubit),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"AllocQubitRegister", MQT_NAMED_BUILDER(allocQubitRegister),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"AllocMultipleQubitRegisters",
                   MQT_NAMED_BUILDER(allocMultipleQubitRegisters),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"AllocLargeRegister", MQT_NAMED_BUILDER(allocLargeRegister),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"StaticQubits", MQT_NAMED_BUILDER(staticQubits),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"AllocDeallocPair", MQT_NAMED_BUILDER(allocDeallocPair),
                   MQT_NAMED_BUILDER(emptyQC)}));
/// @}
