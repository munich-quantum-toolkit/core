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
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"
#include "qc_programs.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <ostream>
#include <string>

using namespace mlir;

struct QCOToQCTestCase {
  std::string name;
  mqt::test::NamedBuilder<qco::QCOProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<qc::QCProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QCOToQCTestCase& info);
};

class QCOToQCTest : public testing::TestWithParam<QCOToQCTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                    func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

std::ostream& operator<<(std::ostream& os, const QCOToQCTestCase& info) {
  return os << "QCOToQC{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

static LogicalResult runQCOToQCConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCOToQC());
  return pm.run(module);
}

TEST_P(QCOToQCTest, ProgramEquivalence) {
  const auto& [nameStr, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + nameStr + ")";

  auto program =
      qco::QCOProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printProgram(program.get(), "Original QCO IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printProgram(program.get(), "Canonicalized QCO IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCOToQCConversion(program.get())));
  printProgram(program.get(), "Converted QC IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printProgram(program.get(), "Canonicalized Converted QC IR" + name,
               llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      qc::QCProgramBuilder::build(context.get(), referenceBuilder.fn);
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

/// \name QCOToQC/Modifiers/InvOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOInvOpTest, QCOToQCTest,
    testing::Values(
        // iSWAP cannot be inverted with current canonicalization
        QCOToQCTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(qco::inverseIswap),
                        MQT_NAMED_BUILDER(qc::inverseIswap)},
        QCOToQCTestCase{"InverseMultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qco::inverseMultipleControlledIswap),
                        MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap)},
        // Inverse DCX is not canonicalized in QCO
        QCOToQCTestCase{"InverseDCX", MQT_NAMED_BUILDER(qco::inverseDcx),
                        MQT_NAMED_BUILDER(qc::dcx)},
        QCOToQCTestCase{"InverseMultipleControlledDCX",
                        MQT_NAMED_BUILDER(qco::inverseMultipleControlledDcx),
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOBarrierOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Barrier", MQT_NAMED_BUILDER(qco::barrier),
                                    MQT_NAMED_BUILDER(qc::barrier)},
                    QCOToQCTestCase{"BarrierTwoQubits",
                                    MQT_NAMED_BUILDER(qco::barrierTwoQubits),
                                    MQT_NAMED_BUILDER(qc::barrierTwoQubits)},
                    QCOToQCTestCase{
                        "BarrierMultipleQubits",
                        MQT_NAMED_BUILDER(qco::barrierMultipleQubits),
                        MQT_NAMED_BUILDER(qc::barrierMultipleQubits)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCODCXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"DCX", MQT_NAMED_BUILDER(qco::dcx),
                                    MQT_NAMED_BUILDER(qc::dcx)},
                    QCOToQCTestCase{"SingleControlledDCX",
                                    MQT_NAMED_BUILDER(qco::singleControlledDcx),
                                    MQT_NAMED_BUILDER(qc::singleControlledDcx)},
                    QCOToQCTestCase{
                        "MultipleControlledDCX",
                        MQT_NAMED_BUILDER(qco::multipleControlledDcx),
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOECROpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"ECR", MQT_NAMED_BUILDER(qco::ecr),
                                    MQT_NAMED_BUILDER(qc::ecr)},
                    QCOToQCTestCase{"SingleControlledECR",
                                    MQT_NAMED_BUILDER(qco::singleControlledEcr),
                                    MQT_NAMED_BUILDER(qc::singleControlledEcr)},
                    QCOToQCTestCase{
                        "MultipleControlledECR",
                        MQT_NAMED_BUILDER(qco::multipleControlledEcr),
                        MQT_NAMED_BUILDER(qc::multipleControlledEcr)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCOGPhaseOpTest, QCOToQCTest,
                         testing::Values(QCOToQCTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qco::globalPhase),
                             MQT_NAMED_BUILDER(qc::globalPhase)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOHOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"H", MQT_NAMED_BUILDER(qco::h),
                                    MQT_NAMED_BUILDER(qc::h)},
                    QCOToQCTestCase{"SingleControlledH",
                                    MQT_NAMED_BUILDER(qco::singleControlledH),
                                    MQT_NAMED_BUILDER(qc::singleControlledH)},
                    QCOToQCTestCase{
                        "MultipleControlledH",
                        MQT_NAMED_BUILDER(qco::multipleControlledH),
                        MQT_NAMED_BUILDER(qc::multipleControlledH)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOiSWAPOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"iSWAP", MQT_NAMED_BUILDER(qco::iswap),
                        MQT_NAMED_BUILDER(qc::iswap)},
        QCOToQCTestCase{"SingleControllediSWAP",
                        MQT_NAMED_BUILDER(qco::singleControlledIswap),
                        MQT_NAMED_BUILDER(qc::singleControlledIswap)},
        QCOToQCTestCase{"MultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qco::multipleControlledIswap),
                        MQT_NAMED_BUILDER(qc::multipleControlledIswap)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"P", MQT_NAMED_BUILDER(qco::p),
                                    MQT_NAMED_BUILDER(qc::p)},
                    QCOToQCTestCase{"SingleControlledP",
                                    MQT_NAMED_BUILDER(qco::singleControlledP),
                                    MQT_NAMED_BUILDER(qc::singleControlledP)},
                    QCOToQCTestCase{
                        "MultipleControlledP",
                        MQT_NAMED_BUILDER(qco::multipleControlledP),
                        MQT_NAMED_BUILDER(qc::multipleControlledP)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOROpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"R", MQT_NAMED_BUILDER(qco::r),
                                    MQT_NAMED_BUILDER(qc::r)},
                    QCOToQCTestCase{"SingleControlledR",
                                    MQT_NAMED_BUILDER(qco::singleControlledR),
                                    MQT_NAMED_BUILDER(qc::singleControlledR)},
                    QCOToQCTestCase{
                        "MultipleControlledR",
                        MQT_NAMED_BUILDER(qco::multipleControlledR),
                        MQT_NAMED_BUILDER(qc::multipleControlledR)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RX", MQT_NAMED_BUILDER(qco::rx),
                                    MQT_NAMED_BUILDER(qc::rx)},
                    QCOToQCTestCase{"SingleControlledRX",
                                    MQT_NAMED_BUILDER(qco::singleControlledRx),
                                    MQT_NAMED_BUILDER(qc::singleControlledRx)},
                    QCOToQCTestCase{
                        "MultipleControlledRX",
                        MQT_NAMED_BUILDER(qco::multipleControlledRx),
                        MQT_NAMED_BUILDER(qc::multipleControlledRx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RXX", MQT_NAMED_BUILDER(qco::rxx),
                                    MQT_NAMED_BUILDER(qc::rxx)},
                    QCOToQCTestCase{"SingleControlledRXX",
                                    MQT_NAMED_BUILDER(qco::singleControlledRxx),
                                    MQT_NAMED_BUILDER(qc::singleControlledRxx)},
                    QCOToQCTestCase{
                        "MultipleControlledRXX",
                        MQT_NAMED_BUILDER(qco::multipleControlledRxx),
                        MQT_NAMED_BUILDER(qc::multipleControlledRxx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RY", MQT_NAMED_BUILDER(qco::ry),
                                    MQT_NAMED_BUILDER(qc::ry)},
                    QCOToQCTestCase{"SingleControlledRY",
                                    MQT_NAMED_BUILDER(qco::singleControlledRy),
                                    MQT_NAMED_BUILDER(qc::singleControlledRy)},
                    QCOToQCTestCase{
                        "MultipleControlledRY",
                        MQT_NAMED_BUILDER(qco::multipleControlledRy),
                        MQT_NAMED_BUILDER(qc::multipleControlledRy)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RYY", MQT_NAMED_BUILDER(qco::ryy),
                                    MQT_NAMED_BUILDER(qc::ryy)},
                    QCOToQCTestCase{"SingleControlledRYY",
                                    MQT_NAMED_BUILDER(qco::singleControlledRyy),
                                    MQT_NAMED_BUILDER(qc::singleControlledRyy)},
                    QCOToQCTestCase{
                        "MultipleControlledRYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledRyy),
                        MQT_NAMED_BUILDER(qc::multipleControlledRyy)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RZ", MQT_NAMED_BUILDER(qco::rz),
                                    MQT_NAMED_BUILDER(qc::rz)},
                    QCOToQCTestCase{"SingleControlledRZ",
                                    MQT_NAMED_BUILDER(qco::singleControlledRz),
                                    MQT_NAMED_BUILDER(qc::singleControlledRz)},
                    QCOToQCTestCase{
                        "MultipleControlledRZ",
                        MQT_NAMED_BUILDER(qco::multipleControlledRz),
                        MQT_NAMED_BUILDER(qc::multipleControlledRz)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RZX", MQT_NAMED_BUILDER(qco::rzx),
                                    MQT_NAMED_BUILDER(qc::rzx)},
                    QCOToQCTestCase{"SingleControlledRZX",
                                    MQT_NAMED_BUILDER(qco::singleControlledRzx),
                                    MQT_NAMED_BUILDER(qc::singleControlledRzx)},
                    QCOToQCTestCase{
                        "MultipleControlledRZX",
                        MQT_NAMED_BUILDER(qco::multipleControlledRzx),
                        MQT_NAMED_BUILDER(qc::multipleControlledRzx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZZOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RZZ", MQT_NAMED_BUILDER(qco::rzz),
                                    MQT_NAMED_BUILDER(qc::rzz)},
                    QCOToQCTestCase{"SingleControlledRZZ",
                                    MQT_NAMED_BUILDER(qco::singleControlledRzz),
                                    MQT_NAMED_BUILDER(qc::singleControlledRzz)},
                    QCOToQCTestCase{
                        "MultipleControlledRZZ",
                        MQT_NAMED_BUILDER(qco::multipleControlledRzz),
                        MQT_NAMED_BUILDER(qc::multipleControlledRzz)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"S", MQT_NAMED_BUILDER(qco::s),
                                    MQT_NAMED_BUILDER(qc::s)},
                    QCOToQCTestCase{"SingleControlledS",
                                    MQT_NAMED_BUILDER(qco::singleControlledS),
                                    MQT_NAMED_BUILDER(qc::singleControlledS)},
                    QCOToQCTestCase{
                        "MultipleControlledS",
                        MQT_NAMED_BUILDER(qco::multipleControlledS),
                        MQT_NAMED_BUILDER(qc::multipleControlledS)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSdgOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Sdg", MQT_NAMED_BUILDER(qco::sdg),
                                    MQT_NAMED_BUILDER(qc::sdg)},
                    QCOToQCTestCase{"SingleControlledSdg",
                                    MQT_NAMED_BUILDER(qco::singleControlledSdg),
                                    MQT_NAMED_BUILDER(qc::singleControlledSdg)},
                    QCOToQCTestCase{
                        "MultipleControlledSdg",
                        MQT_NAMED_BUILDER(qco::multipleControlledSdg),
                        MQT_NAMED_BUILDER(qc::multipleControlledSdg)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSWAPOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SWAP", MQT_NAMED_BUILDER(qco::swap),
                        MQT_NAMED_BUILDER(qc::swap)},
        QCOToQCTestCase{"SingleControlledSWAP",
                        MQT_NAMED_BUILDER(qco::singleControlledSwap),
                        MQT_NAMED_BUILDER(qc::singleControlledSwap)},
        QCOToQCTestCase{"MultipleControlledSWAP",
                        MQT_NAMED_BUILDER(qco::multipleControlledSwap),
                        MQT_NAMED_BUILDER(qc::multipleControlledSwap)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"SX", MQT_NAMED_BUILDER(qco::sx),
                                    MQT_NAMED_BUILDER(qc::sx)},
                    QCOToQCTestCase{"SingleControlledSX",
                                    MQT_NAMED_BUILDER(qco::singleControlledSx),
                                    MQT_NAMED_BUILDER(qc::singleControlledSx)},
                    QCOToQCTestCase{
                        "MultipleControlledSX",
                        MQT_NAMED_BUILDER(qco::multipleControlledSx),
                        MQT_NAMED_BUILDER(qc::multipleControlledSx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXdgOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SXdg", MQT_NAMED_BUILDER(qco::sxdg),
                        MQT_NAMED_BUILDER(qc::sxdg)},
        QCOToQCTestCase{"SingleControlledSXdg",
                        MQT_NAMED_BUILDER(qco::singleControlledSxdg),
                        MQT_NAMED_BUILDER(qc::singleControlledSxdg)},
        QCOToQCTestCase{"MultipleControlledSXdg",
                        MQT_NAMED_BUILDER(qco::multipleControlledSxdg),
                        MQT_NAMED_BUILDER(qc::multipleControlledSxdg)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"T", MQT_NAMED_BUILDER(qco::t_),
                                    MQT_NAMED_BUILDER(qc::t_)},
                    QCOToQCTestCase{"SingleControlledT",
                                    MQT_NAMED_BUILDER(qco::singleControlledT),
                                    MQT_NAMED_BUILDER(qc::singleControlledT)},
                    QCOToQCTestCase{
                        "MultipleControlledT",
                        MQT_NAMED_BUILDER(qco::multipleControlledT),
                        MQT_NAMED_BUILDER(qc::multipleControlledT)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTdgOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Tdg", MQT_NAMED_BUILDER(qco::tdg),
                                    MQT_NAMED_BUILDER(qc::tdg)},
                    QCOToQCTestCase{"SingleControlledTdg",
                                    MQT_NAMED_BUILDER(qco::singleControlledTdg),
                                    MQT_NAMED_BUILDER(qc::singleControlledTdg)},
                    QCOToQCTestCase{
                        "MultipleControlledTdg",
                        MQT_NAMED_BUILDER(qco::multipleControlledTdg),
                        MQT_NAMED_BUILDER(qc::multipleControlledTdg)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOU2OpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"U2", MQT_NAMED_BUILDER(qco::u2),
                                    MQT_NAMED_BUILDER(qc::u2)},
                    QCOToQCTestCase{"SingleControlledU2",
                                    MQT_NAMED_BUILDER(qco::singleControlledU2),
                                    MQT_NAMED_BUILDER(qc::singleControlledU2)},
                    QCOToQCTestCase{
                        "MultipleControlledU2",
                        MQT_NAMED_BUILDER(qco::multipleControlledU2),
                        MQT_NAMED_BUILDER(qc::multipleControlledU2)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOUOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"U", MQT_NAMED_BUILDER(qco::u),
                                    MQT_NAMED_BUILDER(qc::u)},
                    QCOToQCTestCase{"SingleControlledU",
                                    MQT_NAMED_BUILDER(qco::singleControlledU),
                                    MQT_NAMED_BUILDER(qc::singleControlledU)},
                    QCOToQCTestCase{
                        "MultipleControlledU",
                        MQT_NAMED_BUILDER(qco::multipleControlledU),
                        MQT_NAMED_BUILDER(qc::multipleControlledU)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"X", MQT_NAMED_BUILDER(qco::x),
                                    MQT_NAMED_BUILDER(qc::x)},
                    QCOToQCTestCase{"SingleControlledX",
                                    MQT_NAMED_BUILDER(qco::singleControlledX),
                                    MQT_NAMED_BUILDER(qc::singleControlledX)},
                    QCOToQCTestCase{
                        "MultipleControlledX",
                        MQT_NAMED_BUILDER(qco::multipleControlledX),
                        MQT_NAMED_BUILDER(qc::multipleControlledX)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXMinusYYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qco::xxMinusYY),
                        MQT_NAMED_BUILDER(qc::xxMinusYY)},
        QCOToQCTestCase{"SingleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::singleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY)},
        QCOToQCTestCase{"MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXPlusYYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qco::xxPlusYY),
                        MQT_NAMED_BUILDER(qc::xxPlusYY)},
        QCOToQCTestCase{"SingleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY)},
        QCOToQCTestCase{"MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Y", MQT_NAMED_BUILDER(qco::y),
                                    MQT_NAMED_BUILDER(qc::y)},
                    QCOToQCTestCase{"SingleControlledY",
                                    MQT_NAMED_BUILDER(qco::singleControlledY),
                                    MQT_NAMED_BUILDER(qc::singleControlledY)},
                    QCOToQCTestCase{
                        "MultipleControlledY",
                        MQT_NAMED_BUILDER(qco::multipleControlledY),
                        MQT_NAMED_BUILDER(qc::multipleControlledY)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOZOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Z", MQT_NAMED_BUILDER(qco::z),
                                    MQT_NAMED_BUILDER(qc::z)},
                    QCOToQCTestCase{"SingleControlledZ",
                                    MQT_NAMED_BUILDER(qco::singleControlledZ),
                                    MQT_NAMED_BUILDER(qc::singleControlledZ)},
                    QCOToQCTestCase{
                        "MultipleControlledZ",
                        MQT_NAMED_BUILDER(qco::multipleControlledZ),
                        MQT_NAMED_BUILDER(qc::multipleControlledZ)}));
/// @}

/// \name QCOToQC/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOMeasureOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SingleMeasurementToSingleBit",
                        MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit),
                        MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit)},
        QCOToQCTestCase{"RepeatedMeasurementToSameBit",
                        MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit),
                        MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit)},
        QCOToQCTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits)},
        QCOToQCTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qco::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements)}));
/// @}

/// \name QCOToQC/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"ResetQubitAfterSingleOp",
                        MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp),
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)},
        QCOToQCTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp)},
        QCOToQCTestCase{"RepeatedResetAfterSingleOp",
                        MQT_NAMED_BUILDER(qco::repeatedResetAfterSingleOp),
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)}));
/// @}
