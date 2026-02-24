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
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <jeff/IR/JeffDialect.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <ostream>
#include <string>

using namespace mlir;

struct JeffRoundTripTestCase {
  std::string name;
  mqt::test::NamedBuilder<qco::QCOProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<qco::QCOProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const JeffRoundTripTestCase& info);
};

class JeffRoundTripTest : public testing::TestWithParam<JeffRoundTripTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, jeff::JeffDialect,
                    mlir::qco::QCODialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

std::ostream& operator<<(std::ostream& os, const JeffRoundTripTestCase& info) {
  return os << "JeffRoundTrip{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

static LogicalResult runJeffRoundTrip(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCOToJeff());
  pm.addPass(createJeffToQCO());
  return pm.run(module);
}

TEST_P(JeffRoundTripTest, ProgramEquivalence) {
  const auto& [nameStr, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + nameStr + ")";
  mqt::test::DeferredPrinter printer;

  auto program =
      qco::QCOProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printer.record(program.get(), "Canonicalized QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runJeffRoundTrip(program.get())));
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

/// \name JeffRoundTrip/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOBarrierOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Barrier", MQT_NAMED_BUILDER(qco::barrier),
                              MQT_NAMED_BUILDER(qco::barrier)},
        JeffRoundTripTestCase{"BarrierTwoQubits",
                              MQT_NAMED_BUILDER(qco::barrierTwoQubits),
                              MQT_NAMED_BUILDER(qco::barrierTwoQubits)},
        JeffRoundTripTestCase{"BarrierMultipleQubits",
                              MQT_NAMED_BUILDER(qco::barrierMultipleQubits),
                              MQT_NAMED_BUILDER(qco::barrierMultipleQubits)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCODCXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"DCX", MQT_NAMED_BUILDER(qco::dcx),
                              MQT_NAMED_BUILDER(qco::dcx)},
        JeffRoundTripTestCase{"SingleControlledDCX",
                              MQT_NAMED_BUILDER(qco::singleControlledDcx),
                              MQT_NAMED_BUILDER(qco::singleControlledDcx)},
        JeffRoundTripTestCase{"MultipleControlledDCX",
                              MQT_NAMED_BUILDER(qco::multipleControlledDcx),
                              MQT_NAMED_BUILDER(qco::multipleControlledDcx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOECROpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"ECR", MQT_NAMED_BUILDER(qco::ecr),
                              MQT_NAMED_BUILDER(qco::ecr)},
        JeffRoundTripTestCase{"SingleControlledECR",
                              MQT_NAMED_BUILDER(qco::singleControlledEcr),
                              MQT_NAMED_BUILDER(qco::singleControlledEcr)},
        JeffRoundTripTestCase{"MultipleControlledECR",
                              MQT_NAMED_BUILDER(qco::multipleControlledEcr),
                              MQT_NAMED_BUILDER(qco::multipleControlledEcr)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCOGPhaseOpTest, JeffRoundTripTest,
                         testing::Values(JeffRoundTripTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qco::globalPhase),
                             MQT_NAMED_BUILDER(qco::globalPhase)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOHOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"H", MQT_NAMED_BUILDER(qco::h),
                              MQT_NAMED_BUILDER(qco::h)},
        JeffRoundTripTestCase{"SingleControlledH",
                              MQT_NAMED_BUILDER(qco::singleControlledH),
                              MQT_NAMED_BUILDER(qco::singleControlledH)},
        JeffRoundTripTestCase{"MultipleControlledH",
                              MQT_NAMED_BUILDER(qco::multipleControlledH),
                              MQT_NAMED_BUILDER(qco::multipleControlledH)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOiSWAPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"iSWAP", MQT_NAMED_BUILDER(qco::iswap),
                              MQT_NAMED_BUILDER(qco::iswap)},
        JeffRoundTripTestCase{"SingleControllediSWAP",
                              MQT_NAMED_BUILDER(qco::singleControlledIswap),
                              MQT_NAMED_BUILDER(qco::singleControlledIswap)},
        JeffRoundTripTestCase{
            "MultipleControllediSWAP",
            MQT_NAMED_BUILDER(qco::multipleControlledIswap),
            MQT_NAMED_BUILDER(qco::multipleControlledIswap)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCOIdOpTest, JeffRoundTripTest,
                         testing::Values(JeffRoundTripTestCase{
                             "Identity", MQT_NAMED_BUILDER(qco::identity),
                             MQT_NAMED_BUILDER(qco::identity)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"P", MQT_NAMED_BUILDER(qco::p),
                              MQT_NAMED_BUILDER(qco::p)},
        JeffRoundTripTestCase{"SingleControlledP",
                              MQT_NAMED_BUILDER(qco::singleControlledP),
                              MQT_NAMED_BUILDER(qco::singleControlledP)},
        JeffRoundTripTestCase{"MultipleControlledP",
                              MQT_NAMED_BUILDER(qco::multipleControlledP),
                              MQT_NAMED_BUILDER(qco::multipleControlledP)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    DISABLED_QCOROpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"R", MQT_NAMED_BUILDER(qco::r),
                              MQT_NAMED_BUILDER(qco::r)},
        JeffRoundTripTestCase{"SingleControlledR",
                              MQT_NAMED_BUILDER(qco::singleControlledR),
                              MQT_NAMED_BUILDER(qco::singleControlledR)},
        JeffRoundTripTestCase{"MultipleControlledR",
                              MQT_NAMED_BUILDER(qco::multipleControlledR),
                              MQT_NAMED_BUILDER(qco::multipleControlledR)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RX", MQT_NAMED_BUILDER(qco::rx),
                              MQT_NAMED_BUILDER(qco::rx)},
        JeffRoundTripTestCase{"SingleControlledRX",
                              MQT_NAMED_BUILDER(qco::singleControlledRx),
                              MQT_NAMED_BUILDER(qco::singleControlledRx)},
        JeffRoundTripTestCase{"MultipleControlledRX",
                              MQT_NAMED_BUILDER(qco::multipleControlledRx),
                              MQT_NAMED_BUILDER(qco::multipleControlledRx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RY", MQT_NAMED_BUILDER(qco::ry),
                              MQT_NAMED_BUILDER(qco::ry)},
        JeffRoundTripTestCase{"SingleControlledRY",
                              MQT_NAMED_BUILDER(qco::singleControlledRy),
                              MQT_NAMED_BUILDER(qco::singleControlledRy)},
        JeffRoundTripTestCase{"MultipleControlledRY",
                              MQT_NAMED_BUILDER(qco::multipleControlledRy),
                              MQT_NAMED_BUILDER(qco::multipleControlledRy)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RZ", MQT_NAMED_BUILDER(qco::rz),
                              MQT_NAMED_BUILDER(qco::rz)},
        JeffRoundTripTestCase{"SingleControlledRZ",
                              MQT_NAMED_BUILDER(qco::singleControlledRz),
                              MQT_NAMED_BUILDER(qco::singleControlledRz)},
        JeffRoundTripTestCase{"MultipleControlledRZ",
                              MQT_NAMED_BUILDER(qco::multipleControlledRz),
                              MQT_NAMED_BUILDER(qco::multipleControlledRz)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"S", MQT_NAMED_BUILDER(qco::s),
                              MQT_NAMED_BUILDER(qco::s)},
        JeffRoundTripTestCase{"SingleControlledS",
                              MQT_NAMED_BUILDER(qco::singleControlledS),
                              MQT_NAMED_BUILDER(qco::singleControlledS)},
        JeffRoundTripTestCase{"MultipleControlledS",
                              MQT_NAMED_BUILDER(qco::multipleControlledS),
                              MQT_NAMED_BUILDER(qco::multipleControlledS)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Sdg", MQT_NAMED_BUILDER(qco::sdg),
                              MQT_NAMED_BUILDER(qco::sdg)},
        JeffRoundTripTestCase{"SingleControlledSdg",
                              MQT_NAMED_BUILDER(qco::singleControlledSdg),
                              MQT_NAMED_BUILDER(qco::singleControlledSdg)},
        JeffRoundTripTestCase{"MultipleControlledSdg",
                              MQT_NAMED_BUILDER(qco::multipleControlledSdg),
                              MQT_NAMED_BUILDER(qco::multipleControlledSdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSWAPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SWAP", MQT_NAMED_BUILDER(qco::swap),
                              MQT_NAMED_BUILDER(qco::swap)},
        JeffRoundTripTestCase{"SingleControlledSWAP",
                              MQT_NAMED_BUILDER(qco::singleControlledSwap),
                              MQT_NAMED_BUILDER(qco::singleControlledSwap)},
        JeffRoundTripTestCase{"MultipleControlledSWAP",
                              MQT_NAMED_BUILDER(qco::multipleControlledSwap),
                              MQT_NAMED_BUILDER(qco::multipleControlledSwap)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SX", MQT_NAMED_BUILDER(qco::sx),
                              MQT_NAMED_BUILDER(qco::sx)},
        JeffRoundTripTestCase{"SingleControlledSX",
                              MQT_NAMED_BUILDER(qco::singleControlledSx),
                              MQT_NAMED_BUILDER(qco::singleControlledSx)},
        JeffRoundTripTestCase{"MultipleControlledSX",
                              MQT_NAMED_BUILDER(qco::multipleControlledSx),
                              MQT_NAMED_BUILDER(qco::multipleControlledSx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SXdg", MQT_NAMED_BUILDER(qco::sxdg),
                              MQT_NAMED_BUILDER(qco::sxdg)},
        JeffRoundTripTestCase{"SingleControlledSXdg",
                              MQT_NAMED_BUILDER(qco::singleControlledSxdg),
                              MQT_NAMED_BUILDER(qco::singleControlledSxdg)},
        JeffRoundTripTestCase{"MultipleControlledSXdg",
                              MQT_NAMED_BUILDER(qco::multipleControlledSxdg),
                              MQT_NAMED_BUILDER(qco::multipleControlledSxdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"T", MQT_NAMED_BUILDER(qco::t_),
                              MQT_NAMED_BUILDER(qco::t_)},
        JeffRoundTripTestCase{"SingleControlledT",
                              MQT_NAMED_BUILDER(qco::singleControlledT),
                              MQT_NAMED_BUILDER(qco::singleControlledT)},
        JeffRoundTripTestCase{"MultipleControlledT",
                              MQT_NAMED_BUILDER(qco::multipleControlledT),
                              MQT_NAMED_BUILDER(qco::multipleControlledT)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Tdg", MQT_NAMED_BUILDER(qco::tdg),
                              MQT_NAMED_BUILDER(qco::tdg)},
        JeffRoundTripTestCase{"SingleControlledTdg",
                              MQT_NAMED_BUILDER(qco::singleControlledTdg),
                              MQT_NAMED_BUILDER(qco::singleControlledTdg)},
        JeffRoundTripTestCase{"MultipleControlledTdg",
                              MQT_NAMED_BUILDER(qco::multipleControlledTdg),
                              MQT_NAMED_BUILDER(qco::multipleControlledTdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    DISABLED_QCOU2OpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"U2", MQT_NAMED_BUILDER(qco::u2),
                              MQT_NAMED_BUILDER(qco::u2)},
        JeffRoundTripTestCase{"SingleControlledU2",
                              MQT_NAMED_BUILDER(qco::singleControlledU2),
                              MQT_NAMED_BUILDER(qco::singleControlledU2)},
        JeffRoundTripTestCase{"MultipleControlledU2",
                              MQT_NAMED_BUILDER(qco::multipleControlledU2),
                              MQT_NAMED_BUILDER(qco::multipleControlledU2)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOUOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"U", MQT_NAMED_BUILDER(qco::u),
                              MQT_NAMED_BUILDER(qco::u)},
        JeffRoundTripTestCase{"SingleControlledU",
                              MQT_NAMED_BUILDER(qco::singleControlledU),
                              MQT_NAMED_BUILDER(qco::singleControlledU)},
        JeffRoundTripTestCase{"MultipleControlledU",
                              MQT_NAMED_BUILDER(qco::multipleControlledU),
                              MQT_NAMED_BUILDER(qco::multipleControlledU)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"X", MQT_NAMED_BUILDER(qco::x),
                              MQT_NAMED_BUILDER(qco::x)},
        JeffRoundTripTestCase{"SingleControlledX",
                              MQT_NAMED_BUILDER(qco::singleControlledX),
                              MQT_NAMED_BUILDER(qco::singleControlledX)},
        JeffRoundTripTestCase{"MultipleControlledX",
                              MQT_NAMED_BUILDER(qco::multipleControlledX),
                              MQT_NAMED_BUILDER(qco::multipleControlledX)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    DISABLED_QCOXXMinusYYOpTest, JeffRoundTripTest,
    testing::Values(JeffRoundTripTestCase{"XXMinusYY",
                                          MQT_NAMED_BUILDER(qco::xxMinusYY),
                                          MQT_NAMED_BUILDER(qco::xxMinusYY)},
                    JeffRoundTripTestCase{
                        "SingleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::singleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qco::singleControlledXxMinusYY)},
                    JeffRoundTripTestCase{
                        "MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qco::multipleControlledXxMinusYY)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    DISABLED_QCOXXPlusYYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qco::xxPlusYY),
                              MQT_NAMED_BUILDER(qco::xxPlusYY)},
        JeffRoundTripTestCase{"SingleControlledXXPlusYY",
                              MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY),
                              MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY)},
        JeffRoundTripTestCase{
            "MultipleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY),
            MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Y", MQT_NAMED_BUILDER(qco::y),
                              MQT_NAMED_BUILDER(qco::y)},
        JeffRoundTripTestCase{"SingleControlledY",
                              MQT_NAMED_BUILDER(qco::singleControlledY),
                              MQT_NAMED_BUILDER(qco::singleControlledY)},
        JeffRoundTripTestCase{"MultipleControlledY",
                              MQT_NAMED_BUILDER(qco::multipleControlledY),
                              MQT_NAMED_BUILDER(qco::multipleControlledY)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOZOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Z", MQT_NAMED_BUILDER(qco::z),
                              MQT_NAMED_BUILDER(qco::z)},
        JeffRoundTripTestCase{"SingleControlledZ",
                              MQT_NAMED_BUILDER(qco::singleControlledZ),
                              MQT_NAMED_BUILDER(qco::singleControlledZ)},
        JeffRoundTripTestCase{"MultipleControlledZ",
                              MQT_NAMED_BUILDER(qco::multipleControlledZ),
                              MQT_NAMED_BUILDER(qco::multipleControlledZ)}));
/// @}

/// \name JeffRoundTrip/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOMeasureOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit)},
        JeffRoundTripTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit)},
        JeffRoundTripTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits)},
        JeffRoundTripTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qco::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(
                qco::multipleClassicalRegistersAndMeasurements)}));
/// @}

/// \name JeffRoundTrip/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"ResetQubitAfterSingleOp",
                              MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp),
                              MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp)},
        JeffRoundTripTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp)}));
/// @}
