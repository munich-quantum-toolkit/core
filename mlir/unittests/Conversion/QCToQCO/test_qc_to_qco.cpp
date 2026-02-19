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
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
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

struct QCToQCOTestCase {
  std::string name;
  mqt::test::NamedBuilder<qc::QCProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<qco::QCOProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QCToQCOTestCase& info);
};

class QCToQCOTest : public testing::TestWithParam<QCToQCOTestCase> {
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

std::ostream& operator<<(std::ostream& os, const QCToQCOTestCase& info) {
  return os << "QCToQCO{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

static LogicalResult runQCToQCOConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCToQCO());
  return pm.run(module);
}

TEST_P(QCToQCOTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  mqt::test::DeferredPrinter printer;

  auto program = qc::QCProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printer.record(program.get(), "Canonicalized QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCToQCOConversion(program.get())));
  printer.record(program.get(), "Converted QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printer.record(program.get(), "Canonicalized Converted QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      qco::QCOProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  runCanonicalizationPasses(reference.get());
  printer.record(reference.get(), "Canonicalized Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// \name QCToQCO/Modifiers/InvOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCInvOpTest, QCToQCOTest,
    testing::Values(
        // iSWAP cannot be inverted with current canonicalization
        QCToQCOTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(qc::inverseIswap),
                        MQT_NAMED_BUILDER(qco::inverseIswap)},
        QCToQCOTestCase{
            "InverseMultipleControllediSWAP",
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap),
            MQT_NAMED_BUILDER(qco::inverseMultipleControlledIswap)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCBarrierOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"Barrier", MQT_NAMED_BUILDER(qc::barrier),
                                    MQT_NAMED_BUILDER(qco::barrier)},
                    QCToQCOTestCase{"BarrierTwoQubits",
                                    MQT_NAMED_BUILDER(qc::barrierTwoQubits),
                                    MQT_NAMED_BUILDER(qco::barrierTwoQubits)},
                    QCToQCOTestCase{
                        "BarrierMultipleQubits",
                        MQT_NAMED_BUILDER(qc::barrierMultipleQubits),
                        MQT_NAMED_BUILDER(qco::barrierMultipleQubits)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCDCXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"DCX", MQT_NAMED_BUILDER(qc::dcx),
                        MQT_NAMED_BUILDER(qco::dcx)},
        QCToQCOTestCase{"SingleControlledDCX",
                        MQT_NAMED_BUILDER(qc::singleControlledDcx),
                        MQT_NAMED_BUILDER(qco::singleControlledDcx)},
        QCToQCOTestCase{"MultipleControlledDCX",
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx),
                        MQT_NAMED_BUILDER(qco::multipleControlledDcx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCECROpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"ECR", MQT_NAMED_BUILDER(qc::ecr),
                        MQT_NAMED_BUILDER(qco::ecr)},
        QCToQCOTestCase{"SingleControlledECR",
                        MQT_NAMED_BUILDER(qc::singleControlledEcr),
                        MQT_NAMED_BUILDER(qco::singleControlledEcr)},
        QCToQCOTestCase{"MultipleControlledECR",
                        MQT_NAMED_BUILDER(qc::multipleControlledEcr),
                        MQT_NAMED_BUILDER(qco::multipleControlledEcr)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCGPhaseOpTest, QCToQCOTest,
                         testing::Values(QCToQCOTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qc::globalPhase),
                             MQT_NAMED_BUILDER(qco::globalPhase)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCHOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                                    MQT_NAMED_BUILDER(qco::h)},
                    QCToQCOTestCase{"SingleControlledH",
                                    MQT_NAMED_BUILDER(qc::singleControlledH),
                                    MQT_NAMED_BUILDER(qco::singleControlledH)},
                    QCToQCOTestCase{
                        "MultipleControlledH",
                        MQT_NAMED_BUILDER(qc::multipleControlledH),
                        MQT_NAMED_BUILDER(qco::multipleControlledH)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCIDOpTest, QCToQCOTest,
                         testing::Values(QCToQCOTestCase{
                             "Identity", MQT_NAMED_BUILDER(qc::identity),
                             MQT_NAMED_BUILDER(qco::emptyQCO)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCiSWAPOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"iSWAP", MQT_NAMED_BUILDER(qc::iswap),
                        MQT_NAMED_BUILDER(qco::iswap)},
        QCToQCOTestCase{"SingleControllediSWAP",
                        MQT_NAMED_BUILDER(qc::singleControlledIswap),
                        MQT_NAMED_BUILDER(qco::singleControlledIswap)},
        QCToQCOTestCase{"MultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qc::multipleControlledIswap),
                        MQT_NAMED_BUILDER(qco::multipleControlledIswap)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCPOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"P", MQT_NAMED_BUILDER(qc::p),
                                    MQT_NAMED_BUILDER(qco::p)},
                    QCToQCOTestCase{"SingleControlledP",
                                    MQT_NAMED_BUILDER(qc::singleControlledP),
                                    MQT_NAMED_BUILDER(qco::singleControlledP)},
                    QCToQCOTestCase{
                        "MultipleControlledP",
                        MQT_NAMED_BUILDER(qc::multipleControlledP),
                        MQT_NAMED_BUILDER(qco::multipleControlledP)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCROpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"R", MQT_NAMED_BUILDER(qc::r),
                                    MQT_NAMED_BUILDER(qco::r)},
                    QCToQCOTestCase{"SingleControlledR",
                                    MQT_NAMED_BUILDER(qc::singleControlledR),
                                    MQT_NAMED_BUILDER(qco::singleControlledR)},
                    QCToQCOTestCase{
                        "MultipleControlledR",
                        MQT_NAMED_BUILDER(qc::multipleControlledR),
                        MQT_NAMED_BUILDER(qco::multipleControlledR)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRXOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"RX", MQT_NAMED_BUILDER(qc::rx),
                                    MQT_NAMED_BUILDER(qco::rx)},
                    QCToQCOTestCase{"SingleControlledRX",
                                    MQT_NAMED_BUILDER(qc::singleControlledRx),
                                    MQT_NAMED_BUILDER(qco::singleControlledRx)},
                    QCToQCOTestCase{
                        "MultipleControlledRX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRx),
                        MQT_NAMED_BUILDER(qco::multipleControlledRx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRXXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RXX", MQT_NAMED_BUILDER(qc::rxx),
                        MQT_NAMED_BUILDER(qco::rxx)},
        QCToQCOTestCase{"SingleControlledRXX",
                        MQT_NAMED_BUILDER(qc::singleControlledRxx),
                        MQT_NAMED_BUILDER(qco::singleControlledRxx)},
        QCToQCOTestCase{"MultipleControlledRXX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRxx),
                        MQT_NAMED_BUILDER(qco::multipleControlledRxx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRYOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"RY", MQT_NAMED_BUILDER(qc::ry),
                                    MQT_NAMED_BUILDER(qco::ry)},
                    QCToQCOTestCase{"SingleControlledRY",
                                    MQT_NAMED_BUILDER(qc::singleControlledRy),
                                    MQT_NAMED_BUILDER(qco::singleControlledRy)},
                    QCToQCOTestCase{
                        "MultipleControlledRY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRy),
                        MQT_NAMED_BUILDER(qco::multipleControlledRy)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRYYOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RYY", MQT_NAMED_BUILDER(qc::ryy),
                        MQT_NAMED_BUILDER(qco::ryy)},
        QCToQCOTestCase{"SingleControlledRYY",
                        MQT_NAMED_BUILDER(qc::singleControlledRyy),
                        MQT_NAMED_BUILDER(qco::singleControlledRyy)},
        QCToQCOTestCase{"MultipleControlledRYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRyy),
                        MQT_NAMED_BUILDER(qco::multipleControlledRyy)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz),
                                    MQT_NAMED_BUILDER(qco::rz)},
                    QCToQCOTestCase{"SingleControlledRZ",
                                    MQT_NAMED_BUILDER(qc::singleControlledRz),
                                    MQT_NAMED_BUILDER(qco::singleControlledRz)},
                    QCToQCOTestCase{
                        "MultipleControlledRZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRz),
                        MQT_NAMED_BUILDER(qco::multipleControlledRz)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RZX", MQT_NAMED_BUILDER(qc::rzx),
                        MQT_NAMED_BUILDER(qco::rzx)},
        QCToQCOTestCase{"SingleControlledRZX",
                        MQT_NAMED_BUILDER(qc::singleControlledRzx),
                        MQT_NAMED_BUILDER(qco::singleControlledRzx)},
        QCToQCOTestCase{"MultipleControlledRZX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzx),
                        MQT_NAMED_BUILDER(qco::multipleControlledRzx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZZOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RZZ", MQT_NAMED_BUILDER(qc::rzz),
                        MQT_NAMED_BUILDER(qco::rzz)},
        QCToQCOTestCase{"SingleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::singleControlledRzz),
                        MQT_NAMED_BUILDER(qco::singleControlledRzz)},
        QCToQCOTestCase{"MultipleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzz),
                        MQT_NAMED_BUILDER(qco::multipleControlledRzz)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"S", MQT_NAMED_BUILDER(qc::s),
                                    MQT_NAMED_BUILDER(qco::s)},
                    QCToQCOTestCase{"SingleControlledS",
                                    MQT_NAMED_BUILDER(qc::singleControlledS),
                                    MQT_NAMED_BUILDER(qco::singleControlledS)},
                    QCToQCOTestCase{
                        "MultipleControlledS",
                        MQT_NAMED_BUILDER(qc::multipleControlledS),
                        MQT_NAMED_BUILDER(qco::multipleControlledS)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSdgOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"Sdg", MQT_NAMED_BUILDER(qc::sdg),
                        MQT_NAMED_BUILDER(qco::sdg)},
        QCToQCOTestCase{"SingleControlledSdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSdg),
                        MQT_NAMED_BUILDER(qco::singleControlledSdg)},
        QCToQCOTestCase{"MultipleControlledSdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSdg),
                        MQT_NAMED_BUILDER(qco::multipleControlledSdg)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSWAPOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SWAP", MQT_NAMED_BUILDER(qc::swap),
                        MQT_NAMED_BUILDER(qco::swap)},
        QCToQCOTestCase{"SingleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::singleControlledSwap),
                        MQT_NAMED_BUILDER(qco::singleControlledSwap)},
        QCToQCOTestCase{"MultipleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::multipleControlledSwap),
                        MQT_NAMED_BUILDER(qco::multipleControlledSwap)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSXOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"SX", MQT_NAMED_BUILDER(qc::sx),
                                    MQT_NAMED_BUILDER(qco::sx)},
                    QCToQCOTestCase{"SingleControlledSX",
                                    MQT_NAMED_BUILDER(qc::singleControlledSx),
                                    MQT_NAMED_BUILDER(qco::singleControlledSx)},
                    QCToQCOTestCase{
                        "MultipleControlledSX",
                        MQT_NAMED_BUILDER(qc::multipleControlledSx),
                        MQT_NAMED_BUILDER(qco::multipleControlledSx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSXdgOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SXdg", MQT_NAMED_BUILDER(qc::sxdg),
                        MQT_NAMED_BUILDER(qco::sxdg)},
        QCToQCOTestCase{"SingleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSxdg),
                        MQT_NAMED_BUILDER(qco::singleControlledSxdg)},
        QCToQCOTestCase{"MultipleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSxdg),
                        MQT_NAMED_BUILDER(qco::multipleControlledSxdg)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCTOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"T", MQT_NAMED_BUILDER(qc::t_),
                                    MQT_NAMED_BUILDER(qco::t_)},
                    QCToQCOTestCase{"SingleControlledT",
                                    MQT_NAMED_BUILDER(qc::singleControlledT),
                                    MQT_NAMED_BUILDER(qco::singleControlledT)},
                    QCToQCOTestCase{
                        "MultipleControlledT",
                        MQT_NAMED_BUILDER(qc::multipleControlledT),
                        MQT_NAMED_BUILDER(qco::multipleControlledT)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCTdgOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg),
                        MQT_NAMED_BUILDER(qco::tdg)},
        QCToQCOTestCase{"SingleControlledTdg",
                        MQT_NAMED_BUILDER(qc::singleControlledTdg),
                        MQT_NAMED_BUILDER(qco::singleControlledTdg)},
        QCToQCOTestCase{"MultipleControlledTdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledTdg),
                        MQT_NAMED_BUILDER(qco::multipleControlledTdg)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCU2OpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"U2", MQT_NAMED_BUILDER(qc::u2),
                                    MQT_NAMED_BUILDER(qco::u2)},
                    QCToQCOTestCase{"SingleControlledU2",
                                    MQT_NAMED_BUILDER(qc::singleControlledU2),
                                    MQT_NAMED_BUILDER(qco::singleControlledU2)},
                    QCToQCOTestCase{
                        "MultipleControlledU2",
                        MQT_NAMED_BUILDER(qc::multipleControlledU2),
                        MQT_NAMED_BUILDER(qco::multipleControlledU2)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCUOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                                    MQT_NAMED_BUILDER(qco::u)},
                    QCToQCOTestCase{"SingleControlledU",
                                    MQT_NAMED_BUILDER(qc::singleControlledU),
                                    MQT_NAMED_BUILDER(qco::singleControlledU)},
                    QCToQCOTestCase{
                        "MultipleControlledU",
                        MQT_NAMED_BUILDER(qc::multipleControlledU),
                        MQT_NAMED_BUILDER(qco::multipleControlledU)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"X", MQT_NAMED_BUILDER(qc::x),
                                    MQT_NAMED_BUILDER(qco::x)},
                    QCToQCOTestCase{"SingleControlledX",
                                    MQT_NAMED_BUILDER(qc::singleControlledX),
                                    MQT_NAMED_BUILDER(qco::singleControlledX)},
                    QCToQCOTestCase{
                        "MultipleControlledX",
                        MQT_NAMED_BUILDER(qc::multipleControlledX),
                        MQT_NAMED_BUILDER(qco::multipleControlledX)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXXMinusYYOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qc::xxMinusYY),
                        MQT_NAMED_BUILDER(qco::xxMinusYY)},
        QCToQCOTestCase{"SingleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qco::singleControlledXxMinusYY)},
        QCToQCOTestCase{"MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qco::multipleControlledXxMinusYY)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXXPlusYYOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
                        MQT_NAMED_BUILDER(qco::xxPlusYY)},
        QCToQCOTestCase{"SingleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY)},
        QCToQCOTestCase{"MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCYOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                                    MQT_NAMED_BUILDER(qco::y)},
                    QCToQCOTestCase{"SingleControlledY",
                                    MQT_NAMED_BUILDER(qc::singleControlledY),
                                    MQT_NAMED_BUILDER(qco::singleControlledY)},
                    QCToQCOTestCase{
                        "MultipleControlledY",
                        MQT_NAMED_BUILDER(qc::multipleControlledY),
                        MQT_NAMED_BUILDER(qco::multipleControlledY)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCZOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"Z", MQT_NAMED_BUILDER(qc::z),
                                    MQT_NAMED_BUILDER(qco::z)},
                    QCToQCOTestCase{"SingleControlledZ",
                                    MQT_NAMED_BUILDER(qc::singleControlledZ),
                                    MQT_NAMED_BUILDER(qco::singleControlledZ)},
                    QCToQCOTestCase{
                        "MultipleControlledZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledZ),
                        MQT_NAMED_BUILDER(qco::multipleControlledZ)}));
/// @}

/// \name QCToQCO/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCMeasureOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SingleMeasurementToSingleBit",
                        MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit),
                        MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit)},
        QCToQCOTestCase{"RepeatedMeasurementToSameBit",
                        MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit),
                        MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit)},
        QCToQCOTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits)},
        QCToQCOTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(
                qco::multipleClassicalRegistersAndMeasurements)}));
/// @}

/// \name QCToQCO/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCResetOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"ResetQubitAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp),
                        MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp)},
        QCToQCOTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp)},
        QCToQCOTestCase{"RepeatedResetAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp),
                        MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp)}));
/// @}
