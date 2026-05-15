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
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"
#include "qir_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>

using namespace mlir;

namespace {

struct QCToQIRBaseTestCase {
  std::string name;
  mqt::test::NamedBuilder<qc::QCProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<qir::QIRProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QCToQIRBaseTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os, const QCToQIRBaseTestCase& info) {
  return os << "QCToQIRBase{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

class QCToQIRBaseTest : public testing::TestWithParam<QCToQIRBaseTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qc::QCDialect, LLVM::LLVMDialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                    cf::ControlFlowDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

static LogicalResult runQCToQIRBaseConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCToQIRBase());
  return pm.run(module);
}

TEST_P(QCToQIRBaseTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  mqt::test::DeferredPrinter printer;

  auto program = qc::QCProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCToQIRBaseConversion(program.get())));
  printer.record(program.get(), "Converted QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized Converted QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      qir::QIRProgramBuilder::build(context.get(), referenceBuilder.fn,
                                    qir::QIRProgramBuilder::Profile::Base);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QIR IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QIR IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// \name QCToQIRBase/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseBarrierOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"Barrier", MQT_NAMED_BUILDER(qc::barrier),
                            MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRBaseTestCase{"BarrierTwoQubits",
                            MQT_NAMED_BUILDER(qc::barrierTwoQubits),
                            MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRBaseTestCase{"BarrierMultipleQubits",
                            MQT_NAMED_BUILDER(qc::barrierMultipleQubits),
                            MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRBaseTestCase{"SingleControlledBarrier",
                            MQT_NAMED_BUILDER(qc::singleControlledBarrier),
                            MQT_NAMED_BUILDER(qir::emptyQIR)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseDCXOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"DCX", MQT_NAMED_BUILDER(qc::dcx),
                            MQT_NAMED_BUILDER(qir::dcx)},
        QCToQIRBaseTestCase{"SingleControlledDCX",
                            MQT_NAMED_BUILDER(qc::singleControlledDcx),
                            MQT_NAMED_BUILDER(qir::singleControlledDcx)},
        QCToQIRBaseTestCase{"MultipleControlledDCX",
                            MQT_NAMED_BUILDER(qc::multipleControlledDcx),
                            MQT_NAMED_BUILDER(qir::multipleControlledDcx)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseECROpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"ECR", MQT_NAMED_BUILDER(qc::ecr),
                            MQT_NAMED_BUILDER(qir::ecr)},
        QCToQIRBaseTestCase{"SingleControlledECR",
                            MQT_NAMED_BUILDER(qc::singleControlledEcr),
                            MQT_NAMED_BUILDER(qir::singleControlledEcr)},
        QCToQIRBaseTestCase{"MultipleControlledECR",
                            MQT_NAMED_BUILDER(qc::multipleControlledEcr),
                            MQT_NAMED_BUILDER(qir::multipleControlledEcr)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCToQIRBaseGPhaseOpTest, QCToQIRBaseTest,
                         testing::Values(QCToQIRBaseTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qc::globalPhase),
                             MQT_NAMED_BUILDER(qir::globalPhase)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseHOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                            MQT_NAMED_BUILDER(qir::h)},
        QCToQIRBaseTestCase{"SingleControlledH",
                            MQT_NAMED_BUILDER(qc::singleControlledH),
                            MQT_NAMED_BUILDER(qir::singleControlledH)},
        QCToQIRBaseTestCase{"MultipleControlledH",
                            MQT_NAMED_BUILDER(qc::multipleControlledH),
                            MQT_NAMED_BUILDER(qir::multipleControlledH)},
        QCToQIRBaseTestCase{"HWithoutRegister",
                            MQT_NAMED_BUILDER(qc::hWithoutRegister),
                            MQT_NAMED_BUILDER(qir::hWithoutRegister)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseIDOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"Identity", MQT_NAMED_BUILDER(qc::identity),
                            MQT_NAMED_BUILDER(qir::identity)},
        QCToQIRBaseTestCase{"SingleControlledIdentity",
                            MQT_NAMED_BUILDER(qc::singleControlledIdentity),
                            MQT_NAMED_BUILDER(qir::identity)},
        QCToQIRBaseTestCase{"MultipleControlledIdentity",
                            MQT_NAMED_BUILDER(qc::multipleControlledIdentity),
                            MQT_NAMED_BUILDER(qir::identity)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseiSWAPOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"iSWAP", MQT_NAMED_BUILDER(qc::iswap),
                            MQT_NAMED_BUILDER(qir::iswap)},
        QCToQIRBaseTestCase{"SingleControllediSWAP",
                            MQT_NAMED_BUILDER(qc::singleControlledIswap),
                            MQT_NAMED_BUILDER(qir::singleControlledIswap)},
        QCToQIRBaseTestCase{"MultipleControllediSWAP",
                            MQT_NAMED_BUILDER(qc::multipleControlledIswap),
                            MQT_NAMED_BUILDER(qir::multipleControlledIswap)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBasePOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"P", MQT_NAMED_BUILDER(qc::p),
                            MQT_NAMED_BUILDER(qir::p)},
        QCToQIRBaseTestCase{"SingleControlledP",
                            MQT_NAMED_BUILDER(qc::singleControlledP),
                            MQT_NAMED_BUILDER(qir::singleControlledP)},
        QCToQIRBaseTestCase{"MultipleControlledP",
                            MQT_NAMED_BUILDER(qc::multipleControlledP),
                            MQT_NAMED_BUILDER(qir::multipleControlledP)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseROpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"R", MQT_NAMED_BUILDER(qc::r),
                            MQT_NAMED_BUILDER(qir::r)},
        QCToQIRBaseTestCase{"SingleControlledR",
                            MQT_NAMED_BUILDER(qc::singleControlledR),
                            MQT_NAMED_BUILDER(qir::singleControlledR)},
        QCToQIRBaseTestCase{"MultipleControlledR",
                            MQT_NAMED_BUILDER(qc::multipleControlledR),
                            MQT_NAMED_BUILDER(qir::multipleControlledR)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRXOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RX", MQT_NAMED_BUILDER(qc::rx),
                            MQT_NAMED_BUILDER(qir::rx)},
        QCToQIRBaseTestCase{"SingleControlledRX",
                            MQT_NAMED_BUILDER(qc::singleControlledRx),
                            MQT_NAMED_BUILDER(qir::singleControlledRx)},
        QCToQIRBaseTestCase{"MultipleControlledRX",
                            MQT_NAMED_BUILDER(qc::multipleControlledRx),
                            MQT_NAMED_BUILDER(qir::multipleControlledRx)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRXXOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RXX", MQT_NAMED_BUILDER(qc::rxx),
                            MQT_NAMED_BUILDER(qir::rxx)},
        QCToQIRBaseTestCase{"SingleControlledRXX",
                            MQT_NAMED_BUILDER(qc::singleControlledRxx),
                            MQT_NAMED_BUILDER(qir::singleControlledRxx)},
        QCToQIRBaseTestCase{"MultipleControlledRXX",
                            MQT_NAMED_BUILDER(qc::multipleControlledRxx),
                            MQT_NAMED_BUILDER(qir::multipleControlledRxx)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRYOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RY", MQT_NAMED_BUILDER(qc::ry),
                            MQT_NAMED_BUILDER(qir::ry)},
        QCToQIRBaseTestCase{"SingleControlledRY",
                            MQT_NAMED_BUILDER(qc::singleControlledRy),
                            MQT_NAMED_BUILDER(qir::singleControlledRy)},
        QCToQIRBaseTestCase{"MultipleControlledRY",
                            MQT_NAMED_BUILDER(qc::multipleControlledRy),
                            MQT_NAMED_BUILDER(qir::multipleControlledRy)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRYYOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RYY", MQT_NAMED_BUILDER(qc::ryy),
                            MQT_NAMED_BUILDER(qir::ryy)},
        QCToQIRBaseTestCase{"SingleControlledRYY",
                            MQT_NAMED_BUILDER(qc::singleControlledRyy),
                            MQT_NAMED_BUILDER(qir::singleControlledRyy)},
        QCToQIRBaseTestCase{"MultipleControlledRYY",
                            MQT_NAMED_BUILDER(qc::multipleControlledRyy),
                            MQT_NAMED_BUILDER(qir::multipleControlledRyy)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRZOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz),
                            MQT_NAMED_BUILDER(qir::rz)},
        QCToQIRBaseTestCase{"SingleControlledRZ",
                            MQT_NAMED_BUILDER(qc::singleControlledRz),
                            MQT_NAMED_BUILDER(qir::singleControlledRz)},
        QCToQIRBaseTestCase{"MultipleControlledRZ",
                            MQT_NAMED_BUILDER(qc::multipleControlledRz),
                            MQT_NAMED_BUILDER(qir::multipleControlledRz)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRZXOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RZX", MQT_NAMED_BUILDER(qc::rzx),
                            MQT_NAMED_BUILDER(qir::rzx)},
        QCToQIRBaseTestCase{"SingleControlledRZX",
                            MQT_NAMED_BUILDER(qc::singleControlledRzx),
                            MQT_NAMED_BUILDER(qir::singleControlledRzx)},
        QCToQIRBaseTestCase{"MultipleControlledRZX",
                            MQT_NAMED_BUILDER(qc::multipleControlledRzx),
                            MQT_NAMED_BUILDER(qir::multipleControlledRzx)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRZZOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RZZ", MQT_NAMED_BUILDER(qc::rzz),
                            MQT_NAMED_BUILDER(qir::rzz)},
        QCToQIRBaseTestCase{"SingleControlledRZZ",
                            MQT_NAMED_BUILDER(qc::singleControlledRzz),
                            MQT_NAMED_BUILDER(qir::singleControlledRzz)},
        QCToQIRBaseTestCase{"MultipleControlledRZZ",
                            MQT_NAMED_BUILDER(qc::multipleControlledRzz),
                            MQT_NAMED_BUILDER(qir::multipleControlledRzz)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"S", MQT_NAMED_BUILDER(qc::s),
                            MQT_NAMED_BUILDER(qir::s)},
        QCToQIRBaseTestCase{"SingleControlledS",
                            MQT_NAMED_BUILDER(qc::singleControlledS),
                            MQT_NAMED_BUILDER(qir::singleControlledS)},
        QCToQIRBaseTestCase{"MultipleControlledS",
                            MQT_NAMED_BUILDER(qc::multipleControlledS),
                            MQT_NAMED_BUILDER(qir::multipleControlledS)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSdgOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"Sdg", MQT_NAMED_BUILDER(qc::sdg),
                            MQT_NAMED_BUILDER(qir::sdg)},
        QCToQIRBaseTestCase{"SingleControlledSdg",
                            MQT_NAMED_BUILDER(qc::singleControlledSdg),
                            MQT_NAMED_BUILDER(qir::singleControlledSdg)},
        QCToQIRBaseTestCase{"MultipleControlledSdg",
                            MQT_NAMED_BUILDER(qc::multipleControlledSdg),
                            MQT_NAMED_BUILDER(qir::multipleControlledSdg)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSWAPOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"SWAP", MQT_NAMED_BUILDER(qc::swap),
                            MQT_NAMED_BUILDER(qir::swap)},
        QCToQIRBaseTestCase{"SingleControlledSWAP",
                            MQT_NAMED_BUILDER(qc::singleControlledSwap),
                            MQT_NAMED_BUILDER(qir::singleControlledSwap)},
        QCToQIRBaseTestCase{"MultipleControlledSWAP",
                            MQT_NAMED_BUILDER(qc::multipleControlledSwap),
                            MQT_NAMED_BUILDER(qir::multipleControlledSwap)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSXOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"SX", MQT_NAMED_BUILDER(qc::sx),
                            MQT_NAMED_BUILDER(qir::sx)},
        QCToQIRBaseTestCase{"SingleControlledSX",
                            MQT_NAMED_BUILDER(qc::singleControlledSx),
                            MQT_NAMED_BUILDER(qir::singleControlledSx)},
        QCToQIRBaseTestCase{"MultipleControlledSX",
                            MQT_NAMED_BUILDER(qc::multipleControlledSx),
                            MQT_NAMED_BUILDER(qir::multipleControlledSx)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSXdgOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"SXdg", MQT_NAMED_BUILDER(qc::sxdg),
                            MQT_NAMED_BUILDER(qir::sxdg)},
        QCToQIRBaseTestCase{"SingleControlledSXdg",
                            MQT_NAMED_BUILDER(qc::singleControlledSxdg),
                            MQT_NAMED_BUILDER(qir::singleControlledSxdg)},
        QCToQIRBaseTestCase{"MultipleControlledSXdg",
                            MQT_NAMED_BUILDER(qc::multipleControlledSxdg),
                            MQT_NAMED_BUILDER(qir::multipleControlledSxdg)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseTOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"T", MQT_NAMED_BUILDER(qc::t_),
                            MQT_NAMED_BUILDER(qir::t_)},
        QCToQIRBaseTestCase{"SingleControlledT",
                            MQT_NAMED_BUILDER(qc::singleControlledT),
                            MQT_NAMED_BUILDER(qir::singleControlledT)},
        QCToQIRBaseTestCase{"MultipleControlledT",
                            MQT_NAMED_BUILDER(qc::multipleControlledT),
                            MQT_NAMED_BUILDER(qir::multipleControlledT)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseTdgOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg),
                            MQT_NAMED_BUILDER(qir::tdg)},
        QCToQIRBaseTestCase{"SingleControlledTdg",
                            MQT_NAMED_BUILDER(qc::singleControlledTdg),
                            MQT_NAMED_BUILDER(qir::singleControlledTdg)},
        QCToQIRBaseTestCase{"MultipleControlledTdg",
                            MQT_NAMED_BUILDER(qc::multipleControlledTdg),
                            MQT_NAMED_BUILDER(qir::multipleControlledTdg)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseU2OpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"U2", MQT_NAMED_BUILDER(qc::u2),
                            MQT_NAMED_BUILDER(qir::u2)},
        QCToQIRBaseTestCase{"SingleControlledU2",
                            MQT_NAMED_BUILDER(qc::singleControlledU2),
                            MQT_NAMED_BUILDER(qir::singleControlledU2)},
        QCToQIRBaseTestCase{"MultipleControlledU2",
                            MQT_NAMED_BUILDER(qc::multipleControlledU2),
                            MQT_NAMED_BUILDER(qir::multipleControlledU2)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseUOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                            MQT_NAMED_BUILDER(qir::u)},
        QCToQIRBaseTestCase{"SingleControlledU",
                            MQT_NAMED_BUILDER(qc::singleControlledU),
                            MQT_NAMED_BUILDER(qir::singleControlledU)},
        QCToQIRBaseTestCase{"MultipleControlledU",
                            MQT_NAMED_BUILDER(qc::multipleControlledU),
                            MQT_NAMED_BUILDER(qir::multipleControlledU)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseXOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"X", MQT_NAMED_BUILDER(qc::x),
                            MQT_NAMED_BUILDER(qir::x)},
        QCToQIRBaseTestCase{"SingleControlledX",
                            MQT_NAMED_BUILDER(qc::singleControlledX),
                            MQT_NAMED_BUILDER(qir::singleControlledX)},
        QCToQIRBaseTestCase{"MultipleControlledX",
                            MQT_NAMED_BUILDER(qc::multipleControlledX),
                            MQT_NAMED_BUILDER(qir::multipleControlledX)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseXXMinusYYOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qc::xxMinusYY),
                            MQT_NAMED_BUILDER(qir::xxMinusYY)},
        QCToQIRBaseTestCase{"SingleControlledXXMinusYY",
                            MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY),
                            MQT_NAMED_BUILDER(qir::singleControlledXxMinusYY)},
        QCToQIRBaseTestCase{
            "MultipleControlledXXMinusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY),
            MQT_NAMED_BUILDER(qir::multipleControlledXxMinusYY)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseXXPlusYYOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
                            MQT_NAMED_BUILDER(qir::xxPlusYY)},
        QCToQIRBaseTestCase{"SingleControlledXXPlusYY",
                            MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY),
                            MQT_NAMED_BUILDER(qir::singleControlledXxPlusYY)},
        QCToQIRBaseTestCase{
            "MultipleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY),
            MQT_NAMED_BUILDER(qir::multipleControlledXxPlusYY)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseYOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                            MQT_NAMED_BUILDER(qir::y)},
        QCToQIRBaseTestCase{"SingleControlledY",
                            MQT_NAMED_BUILDER(qc::singleControlledY),
                            MQT_NAMED_BUILDER(qir::singleControlledY)},
        QCToQIRBaseTestCase{"MultipleControlledY",
                            MQT_NAMED_BUILDER(qc::multipleControlledY),
                            MQT_NAMED_BUILDER(qir::multipleControlledY)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseZOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"Z", MQT_NAMED_BUILDER(qc::z),
                            MQT_NAMED_BUILDER(qir::z)},
        QCToQIRBaseTestCase{"SingleControlledZ",
                            MQT_NAMED_BUILDER(qc::singleControlledZ),
                            MQT_NAMED_BUILDER(qir::singleControlledZ)},
        QCToQIRBaseTestCase{"MultipleControlledZ",
                            MQT_NAMED_BUILDER(qc::multipleControlledZ),
                            MQT_NAMED_BUILDER(qir::multipleControlledZ)}));
/// @}

/// \name QCToQIRBase/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseMeasureOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(qir::singleMeasurementToSingleBit)},
        QCToQIRBaseTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(qir::repeatedMeasurementToSameBit)},
        QCToQIRBaseTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qir::repeatedMeasurementToDifferentBits)},
        QCToQIRBaseTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(qir::multipleClassicalRegistersAndMeasurements)},
        QCToQIRBaseTestCase{
            "MeasurementWithoutRegisters",
            MQT_NAMED_BUILDER(qc::measurementWithoutRegisters),
            MQT_NAMED_BUILDER(qir::measurementWithoutRegisters)}));
/// @}

/// \name QCToQIRBase/QubitManagement/QubitManagement.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseQubitManagementTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"AllocQubit", MQT_NAMED_BUILDER(qc::allocQubit),
                            MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRBaseTestCase{"AllocQubitRegister",
                            MQT_NAMED_BUILDER(qc::allocQubitRegister),
                            MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRBaseTestCase{"AllocMultipleQubitRegisters",
                            MQT_NAMED_BUILDER(qc::allocMultipleQubitRegisters),
                            MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRBaseTestCase{"AllocLargeRegister",
                            MQT_NAMED_BUILDER(qc::allocLargeRegister),
                            MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRBaseTestCase{"StaticQubits", MQT_NAMED_BUILDER(qc::staticQubits),
                            MQT_NAMED_BUILDER(qir::emptyQIR)},
        QCToQIRBaseTestCase{"StaticQubitsWithOps",
                            MQT_NAMED_BUILDER(qc::staticQubitsWithOps),
                            MQT_NAMED_BUILDER(qir::staticQubitsWithOps)},
        QCToQIRBaseTestCase{
            "StaticQubitsWithParametricOps",
            MQT_NAMED_BUILDER(qc::staticQubitsWithParametricOps),
            MQT_NAMED_BUILDER(qir::staticQubitsWithParametricOps)},
        QCToQIRBaseTestCase{
            "StaticQubitsWithTwoTargetOps",
            MQT_NAMED_BUILDER(qc::staticQubitsWithTwoTargetOps),
            MQT_NAMED_BUILDER(qir::staticQubitsWithTwoTargetOps)},
        QCToQIRBaseTestCase{"StaticQubitsWithCtrl",
                            MQT_NAMED_BUILDER(qc::staticQubitsWithCtrl),
                            MQT_NAMED_BUILDER(qir::staticQubitsWithCtrl)},
        QCToQIRBaseTestCase{"StaticQubitsWithInv",
                            MQT_NAMED_BUILDER(qc::staticQubitsWithInv),
                            MQT_NAMED_BUILDER(qir::staticQubitsWithInv)},
        QCToQIRBaseTestCase{"AllocDeallocPair",
                            MQT_NAMED_BUILDER(qc::allocDeallocPair),
                            MQT_NAMED_BUILDER(qir::emptyQIR)}));
/// @}
