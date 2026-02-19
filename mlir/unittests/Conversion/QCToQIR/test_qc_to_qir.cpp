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
#include "mlir/Conversion/QCToQIR/QCToQIR.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"
#include "qc_programs.h"
#include "qir_programs.h"

#include <gtest/gtest.h>
#include <iosfwd>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <ostream>
#include <string>

using namespace mlir;

struct QCToQIRTestCase {
  std::string name;
  mqt::test::NamedBuilder<qc::QCProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<qir::QIRProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QCToQIRTestCase& info);
};

class QCToQIRTest : public testing::TestWithParam<QCToQIRTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qc::QCDialect, LLVM::LLVMDialect, arith::ArithDialect,
                    func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

std::ostream& operator<<(std::ostream& os, const QCToQIRTestCase& info) {
  return os << "QCToQIR{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

static LogicalResult runQCToQIRConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCToQIR());
  return pm.run(module);
}

TEST_P(QCToQIRTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";

  auto program = qc::QCProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printProgram(program.get(), "Original QC IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printProgram(program.get(), "Canonicalized QC IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCToQIRConversion(program.get())));
  printProgram(program.get(), "Converted QIR IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printProgram(program.get(), "Canonicalized Converted QIR IR" + name,
               llvm::errs());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      qir::QIRProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printProgram(reference.get(), "Reference QIR IR" + name, llvm::errs());
  EXPECT_TRUE(verify(*reference).succeeded());

  runCanonicalizationPasses(reference.get());
  printProgram(reference.get(), "Canonicalized Reference QIR IR" + name,
               llvm::errs());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// \name QCToQIR/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBarrierOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"Barrier", MQT_NAMED_BUILDER(qc::barrier),
                        MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRTestCase{"BarrierTwoQubits",
                        MQT_NAMED_BUILDER(qc::barrierTwoQubits),
                        MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRTestCase{"BarrierMultipleQubits",
                        MQT_NAMED_BUILDER(qc::barrierMultipleQubits),
                        MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRTestCase{"SingleControlledBarrier",
                        MQT_NAMED_BUILDER(qc::singleControlledBarrier),
                        MQT_NAMED_BUILDER(qir::emptyQIR)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRDCXOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"DCX", MQT_NAMED_BUILDER(qc::dcx),
                        MQT_NAMED_BUILDER(qir::dcx)},
        QCToQIRTestCase{"SingleControlledDCX",
                        MQT_NAMED_BUILDER(qc::singleControlledDcx),
                        MQT_NAMED_BUILDER(qir::singleControlledDcx)},
        QCToQIRTestCase{"MultipleControlledDCX",
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx),
                        MQT_NAMED_BUILDER(qir::multipleControlledDcx)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRECROpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"ECR", MQT_NAMED_BUILDER(qc::ecr),
                        MQT_NAMED_BUILDER(qir::ecr)},
        QCToQIRTestCase{"SingleControlledECR",
                        MQT_NAMED_BUILDER(qc::singleControlledEcr),
                        MQT_NAMED_BUILDER(qir::singleControlledEcr)},
        QCToQIRTestCase{"MultipleControlledECR",
                        MQT_NAMED_BUILDER(qc::multipleControlledEcr),
                        MQT_NAMED_BUILDER(qir::multipleControlledEcr)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCToQIRGPhaseOpTest, QCToQIRTest,
                         testing::Values(QCToQIRTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qc::globalPhase),
                             MQT_NAMED_BUILDER(qir::globalPhase)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRHOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                                    MQT_NAMED_BUILDER(qir::h)},
                    QCToQIRTestCase{"SingleControlledH",
                                    MQT_NAMED_BUILDER(qc::singleControlledH),
                                    MQT_NAMED_BUILDER(qir::singleControlledH)},
                    QCToQIRTestCase{
                        "MultipleControlledH",
                        MQT_NAMED_BUILDER(qc::multipleControlledH),
                        MQT_NAMED_BUILDER(qir::multipleControlledH)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRIDOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"Identity", MQT_NAMED_BUILDER(qc::identity),
                        MQT_NAMED_BUILDER(qir::identity)},
        QCToQIRTestCase{"SingleControlledIdentity",
                        MQT_NAMED_BUILDER(qc::singleControlledIdentity),
                        MQT_NAMED_BUILDER(qir::identity)},
        QCToQIRTestCase{"MultipleControlledIdentity",
                        MQT_NAMED_BUILDER(qc::multipleControlledIdentity),
                        MQT_NAMED_BUILDER(qir::identity)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRiSWAPOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"iSWAP", MQT_NAMED_BUILDER(qc::iswap),
                        MQT_NAMED_BUILDER(qir::iswap)},
        QCToQIRTestCase{"SingleControllediSWAP",
                        MQT_NAMED_BUILDER(qc::singleControlledIswap),
                        MQT_NAMED_BUILDER(qir::singleControlledIswap)},
        QCToQIRTestCase{"MultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qc::multipleControlledIswap),
                        MQT_NAMED_BUILDER(qir::multipleControlledIswap)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRPOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"P", MQT_NAMED_BUILDER(qc::p),
                                    MQT_NAMED_BUILDER(qir::p)},
                    QCToQIRTestCase{"SingleControlledP",
                                    MQT_NAMED_BUILDER(qc::singleControlledP),
                                    MQT_NAMED_BUILDER(qir::singleControlledP)},
                    QCToQIRTestCase{
                        "MultipleControlledP",
                        MQT_NAMED_BUILDER(qc::multipleControlledP),
                        MQT_NAMED_BUILDER(qir::multipleControlledP)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRROpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"R", MQT_NAMED_BUILDER(qc::r),
                                    MQT_NAMED_BUILDER(qir::r)},
                    QCToQIRTestCase{"SingleControlledR",
                                    MQT_NAMED_BUILDER(qc::singleControlledR),
                                    MQT_NAMED_BUILDER(qir::singleControlledR)},
                    QCToQIRTestCase{
                        "MultipleControlledR",
                        MQT_NAMED_BUILDER(qc::multipleControlledR),
                        MQT_NAMED_BUILDER(qir::multipleControlledR)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRRXOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"RX", MQT_NAMED_BUILDER(qc::rx),
                                    MQT_NAMED_BUILDER(qir::rx)},
                    QCToQIRTestCase{"SingleControlledRX",
                                    MQT_NAMED_BUILDER(qc::singleControlledRx),
                                    MQT_NAMED_BUILDER(qir::singleControlledRx)},
                    QCToQIRTestCase{
                        "MultipleControlledRX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRx)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRRXXOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"RXX", MQT_NAMED_BUILDER(qc::rxx),
                        MQT_NAMED_BUILDER(qir::rxx)},
        QCToQIRTestCase{"SingleControlledRXX",
                        MQT_NAMED_BUILDER(qc::singleControlledRxx),
                        MQT_NAMED_BUILDER(qir::singleControlledRxx)},
        QCToQIRTestCase{"MultipleControlledRXX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRxx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRxx)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRRYOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"RY", MQT_NAMED_BUILDER(qc::ry),
                                    MQT_NAMED_BUILDER(qir::ry)},
                    QCToQIRTestCase{"SingleControlledRY",
                                    MQT_NAMED_BUILDER(qc::singleControlledRy),
                                    MQT_NAMED_BUILDER(qir::singleControlledRy)},
                    QCToQIRTestCase{
                        "MultipleControlledRY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRy),
                        MQT_NAMED_BUILDER(qir::multipleControlledRy)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRRYYOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"RYY", MQT_NAMED_BUILDER(qc::ryy),
                        MQT_NAMED_BUILDER(qir::ryy)},
        QCToQIRTestCase{"SingleControlledRYY",
                        MQT_NAMED_BUILDER(qc::singleControlledRyy),
                        MQT_NAMED_BUILDER(qir::singleControlledRyy)},
        QCToQIRTestCase{"MultipleControlledRYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRyy),
                        MQT_NAMED_BUILDER(qir::multipleControlledRyy)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRRZOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz),
                                    MQT_NAMED_BUILDER(qir::rz)},
                    QCToQIRTestCase{"SingleControlledRZ",
                                    MQT_NAMED_BUILDER(qc::singleControlledRz),
                                    MQT_NAMED_BUILDER(qir::singleControlledRz)},
                    QCToQIRTestCase{
                        "MultipleControlledRZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRz),
                        MQT_NAMED_BUILDER(qir::multipleControlledRz)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRRZXOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"RZX", MQT_NAMED_BUILDER(qc::rzx),
                        MQT_NAMED_BUILDER(qir::rzx)},
        QCToQIRTestCase{"SingleControlledRZX",
                        MQT_NAMED_BUILDER(qc::singleControlledRzx),
                        MQT_NAMED_BUILDER(qir::singleControlledRzx)},
        QCToQIRTestCase{"MultipleControlledRZX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRzx)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRRZZOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"RZZ", MQT_NAMED_BUILDER(qc::rzz),
                        MQT_NAMED_BUILDER(qir::rzz)},
        QCToQIRTestCase{"SingleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::singleControlledRzz),
                        MQT_NAMED_BUILDER(qir::singleControlledRzz)},
        QCToQIRTestCase{"MultipleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzz),
                        MQT_NAMED_BUILDER(qir::multipleControlledRzz)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRSOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"S", MQT_NAMED_BUILDER(qc::s),
                                    MQT_NAMED_BUILDER(qir::s)},
                    QCToQIRTestCase{"SingleControlledS",
                                    MQT_NAMED_BUILDER(qc::singleControlledS),
                                    MQT_NAMED_BUILDER(qir::singleControlledS)},
                    QCToQIRTestCase{
                        "MultipleControlledS",
                        MQT_NAMED_BUILDER(qc::multipleControlledS),
                        MQT_NAMED_BUILDER(qir::multipleControlledS)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRSdgOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"Sdg", MQT_NAMED_BUILDER(qc::sdg),
                        MQT_NAMED_BUILDER(qir::sdg)},
        QCToQIRTestCase{"SingleControlledSdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSdg),
                        MQT_NAMED_BUILDER(qir::singleControlledSdg)},
        QCToQIRTestCase{"MultipleControlledSdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSdg),
                        MQT_NAMED_BUILDER(qir::multipleControlledSdg)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRSWAPOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"SWAP", MQT_NAMED_BUILDER(qc::swap),
                        MQT_NAMED_BUILDER(qir::swap)},
        QCToQIRTestCase{"SingleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::singleControlledSwap),
                        MQT_NAMED_BUILDER(qir::singleControlledSwap)},
        QCToQIRTestCase{"MultipleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::multipleControlledSwap),
                        MQT_NAMED_BUILDER(qir::multipleControlledSwap)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRSXOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"SX", MQT_NAMED_BUILDER(qc::sx),
                                    MQT_NAMED_BUILDER(qir::sx)},
                    QCToQIRTestCase{"SingleControlledSX",
                                    MQT_NAMED_BUILDER(qc::singleControlledSx),
                                    MQT_NAMED_BUILDER(qir::singleControlledSx)},
                    QCToQIRTestCase{
                        "MultipleControlledSX",
                        MQT_NAMED_BUILDER(qc::multipleControlledSx),
                        MQT_NAMED_BUILDER(qir::multipleControlledSx)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRSXdgOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"SXdg", MQT_NAMED_BUILDER(qc::sxdg),
                        MQT_NAMED_BUILDER(qir::sxdg)},
        QCToQIRTestCase{"SingleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSxdg),
                        MQT_NAMED_BUILDER(qir::singleControlledSxdg)},
        QCToQIRTestCase{"MultipleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSxdg),
                        MQT_NAMED_BUILDER(qir::multipleControlledSxdg)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRTOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"T", MQT_NAMED_BUILDER(qc::t_),
                                    MQT_NAMED_BUILDER(qir::t_)},
                    QCToQIRTestCase{"SingleControlledT",
                                    MQT_NAMED_BUILDER(qc::singleControlledT),
                                    MQT_NAMED_BUILDER(qir::singleControlledT)},
                    QCToQIRTestCase{
                        "MultipleControlledT",
                        MQT_NAMED_BUILDER(qc::multipleControlledT),
                        MQT_NAMED_BUILDER(qir::multipleControlledT)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRTdgOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg),
                        MQT_NAMED_BUILDER(qir::tdg)},
        QCToQIRTestCase{"SingleControlledTdg",
                        MQT_NAMED_BUILDER(qc::singleControlledTdg),
                        MQT_NAMED_BUILDER(qir::singleControlledTdg)},
        QCToQIRTestCase{"MultipleControlledTdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledTdg),
                        MQT_NAMED_BUILDER(qir::multipleControlledTdg)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRU2OpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"U2", MQT_NAMED_BUILDER(qc::u2),
                                    MQT_NAMED_BUILDER(qir::u2)},
                    QCToQIRTestCase{"SingleControlledU2",
                                    MQT_NAMED_BUILDER(qc::singleControlledU2),
                                    MQT_NAMED_BUILDER(qir::singleControlledU2)},
                    QCToQIRTestCase{
                        "MultipleControlledU2",
                        MQT_NAMED_BUILDER(qc::multipleControlledU2),
                        MQT_NAMED_BUILDER(qir::multipleControlledU2)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRUOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                                    MQT_NAMED_BUILDER(qir::u)},
                    QCToQIRTestCase{"SingleControlledU",
                                    MQT_NAMED_BUILDER(qc::singleControlledU),
                                    MQT_NAMED_BUILDER(qir::singleControlledU)},
                    QCToQIRTestCase{
                        "MultipleControlledU",
                        MQT_NAMED_BUILDER(qc::multipleControlledU),
                        MQT_NAMED_BUILDER(qir::multipleControlledU)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRXOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"X", MQT_NAMED_BUILDER(qc::x),
                                    MQT_NAMED_BUILDER(qir::x)},
                    QCToQIRTestCase{"SingleControlledX",
                                    MQT_NAMED_BUILDER(qc::singleControlledX),
                                    MQT_NAMED_BUILDER(qir::singleControlledX)},
                    QCToQIRTestCase{
                        "MultipleControlledX",
                        MQT_NAMED_BUILDER(qc::multipleControlledX),
                        MQT_NAMED_BUILDER(qir::multipleControlledX)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRXXMinusYYOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qc::xxMinusYY),
                        MQT_NAMED_BUILDER(qir::xxMinusYY)},
        QCToQIRTestCase{"SingleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qir::singleControlledXxMinusYY)},
        QCToQIRTestCase{"MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qir::multipleControlledXxMinusYY)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRXXPlusYYOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
                        MQT_NAMED_BUILDER(qir::xxPlusYY)},
        QCToQIRTestCase{"SingleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qir::singleControlledXxPlusYY)},
        QCToQIRTestCase{"MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qir::multipleControlledXxPlusYY)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRYOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                                    MQT_NAMED_BUILDER(qir::y)},
                    QCToQIRTestCase{"SingleControlledY",
                                    MQT_NAMED_BUILDER(qc::singleControlledY),
                                    MQT_NAMED_BUILDER(qir::singleControlledY)},
                    QCToQIRTestCase{
                        "MultipleControlledY",
                        MQT_NAMED_BUILDER(qc::multipleControlledY),
                        MQT_NAMED_BUILDER(qir::multipleControlledY)}));
/// @}

/// \name QCToQIR/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRZOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"Z", MQT_NAMED_BUILDER(qc::z),
                                    MQT_NAMED_BUILDER(qir::z)},
                    QCToQIRTestCase{"SingleControlledZ",
                                    MQT_NAMED_BUILDER(qc::singleControlledZ),
                                    MQT_NAMED_BUILDER(qir::singleControlledZ)},
                    QCToQIRTestCase{
                        "MultipleControlledZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledZ),
                        MQT_NAMED_BUILDER(qir::multipleControlledZ)}));
/// @}

/// \name QCToQIR/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRMeasureOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"SingleMeasurementToSingleBit",
                        MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit),
                        MQT_NAMED_BUILDER(qir::singleMeasurementToSingleBit)},
        QCToQIRTestCase{"RepeatedMeasurementToSameBit",
                        MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit),
                        MQT_NAMED_BUILDER(qir::repeatedMeasurementToSameBit)},
        QCToQIRTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qir::repeatedMeasurementToDifferentBits)},
        QCToQIRTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(
                qir::multipleClassicalRegistersAndMeasurements)}));
/// @}

/// \name QCToQIR/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRResetOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"ResetQubitWithoutOp",
                        MQT_NAMED_BUILDER(qc::resetQubitWithoutOp),
                        MQT_NAMED_BUILDER(qir::resetQubitWithoutOp)},
        QCToQIRTestCase{"ResetMultipleQubitsWithoutOp",
                        MQT_NAMED_BUILDER(qc::resetMultipleQubitsWithoutOp),
                        MQT_NAMED_BUILDER(qir::resetMultipleQubitsWithoutOp)},
        QCToQIRTestCase{"RepeatedResetWithoutOp",
                        MQT_NAMED_BUILDER(qc::repeatedResetWithoutOp),
                        MQT_NAMED_BUILDER(qir::repeatedResetWithoutOp)},
        QCToQIRTestCase{"ResetQubitAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp),
                        MQT_NAMED_BUILDER(qir::resetQubitAfterSingleOp)},
        QCToQIRTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qir::resetMultipleQubitsAfterSingleOp)},
        QCToQIRTestCase{"RepeatedResetAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp),
                        MQT_NAMED_BUILDER(qir::repeatedResetAfterSingleOp)}));
/// @}

/// \name QCToQIR/QubitManagement/QubitManagement.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRQubitManagementTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"AllocQubit", MQT_NAMED_BUILDER(qc::allocQubit),
                        MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRTestCase{"AllocQubitRegister",
                        MQT_NAMED_BUILDER(qc::allocQubitRegister),
                        MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRTestCase{"AllocMultipleQubitRegisters",
                        MQT_NAMED_BUILDER(qc::allocMultipleQubitRegisters),
                        MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRTestCase{"AllocLargeRegister",
                        MQT_NAMED_BUILDER(qc::allocLargeRegister),
                        MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRTestCase{"StaticQubits", MQT_NAMED_BUILDER(qc::staticQubits),
                        MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRTestCase{"AllocDeallocPair",
                        MQT_NAMED_BUILDER(qc::allocDeallocPair),
                        MQT_NAMED_BUILDER(qir::emptyQIR)}));
/// @}
