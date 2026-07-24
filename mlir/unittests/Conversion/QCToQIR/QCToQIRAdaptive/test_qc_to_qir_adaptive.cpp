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
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
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
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>

using namespace mlir;

namespace {

struct QCToQIRAdaptiveTestCase {
  std::string name;
  mqt::test::NamedMLIRBuilder<qc::QCProgramBuilder> programBuilder;
  mqt::test::NamedMLIRBuilder<qir::QIRProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QCToQIRAdaptiveTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const QCToQIRAdaptiveTestCase& info) {
  return os << "QCToQIRAdaptive{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

class QCToQIRAdaptiveTest
    : public testing::TestWithParam<QCToQIRAdaptiveTestCase> {
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

static LogicalResult runQCToQIRAdaptiveConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCToQIRAdaptive());
  return pm.run(module);
}

TEST(QCToQIRAdaptiveNativeTest, LowersControlFlowAssertions) {
  MLIRContext context;
  context
      .loadDialect<qc::QCDialect, arith::ArithDialect, cf::ControlFlowDialect,
                   func::FuncDialect, LLVM::LLVMDialect>();
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto condition = LLVM::UndefOp::create(builder, builder.getI1Type());
  cf::AssertOp::create(builder, condition, "runtime precondition");
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversion(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>("abort"));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>("puts"));
  EXPECT_TRUE(module->lookupSymbol<LLVM::GlobalOp>("assert_msg"));
  bool retainsAssertion = false;
  bool hasConditionalBranch = false;
  bool hasUnreachableFailure = false;
  module->walk([&](Operation* operation) {
    retainsAssertion |= isa<cf::AssertOp>(operation);
    hasConditionalBranch |= isa<LLVM::CondBrOp>(operation);
    hasUnreachableFailure |= isa<LLVM::UnreachableOp>(operation);
  });
  EXPECT_FALSE(retainsAssertion);
  EXPECT_TRUE(hasConditionalBranch);
  EXPECT_TRUE(hasUnreachableFailure);
}

TEST_P(QCToQIRAdaptiveTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  mqt::test::DeferredPrinter printer;

  auto program = mqt::test::buildMLIRProgram(context.get(), programBuilder);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCToQIRAdaptiveConversion(program.get())));
  printer.record(program.get(), "Converted QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(program.get(), true).succeeded());
  printer.record(program.get(), "Canonicalized Converted QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      mqt::test::buildMLIRProgram(context.get(), referenceBuilder,
                                  qir::QIRProgramBuilder::Profile::Adaptive);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QIR IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(reference.get(), true).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QIR IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// \name QCToQIRAdaptive/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveBarrierOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{
            "Barrier", MQT_NAMED_BUILDER(qc::barrier),
            MQT_NAMED_BUILDER(qir::alloc1QubitRegister<true>)},
        QCToQIRAdaptiveTestCase{
            "BarrierTwoQubits", MQT_NAMED_BUILDER(qc::barrierTwoQubits),
            MQT_NAMED_BUILDER(qir::allocQubitRegister<true>)},
        QCToQIRAdaptiveTestCase{
            "BarrierMultipleQubits",
            MQT_NAMED_BUILDER(qc::barrierMultipleQubits),
            MQT_NAMED_BUILDER(qir::alloc3QubitRegister<true>)},
        QCToQIRAdaptiveTestCase{
            "SingleControlledBarrier",
            MQT_NAMED_BUILDER(qc::singleControlledBarrier),
            MQT_NAMED_BUILDER(qir::allocQubitRegister<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveDCXOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"DCX", MQT_NAMED_BUILDER(qc::dcx),
                                            MQT_NAMED_BUILDER(qir::dcx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledDCX",
                        MQT_NAMED_BUILDER(qc::singleControlledDcx),
                        MQT_NAMED_BUILDER(qir::singleControlledDcx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledDCX",
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx),
                        MQT_NAMED_BUILDER(qir::multipleControlledDcx<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveECROpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"ECR", MQT_NAMED_BUILDER(qc::ecr),
                                            MQT_NAMED_BUILDER(qir::ecr<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledECR",
                        MQT_NAMED_BUILDER(qc::singleControlledEcr),
                        MQT_NAMED_BUILDER(qir::singleControlledEcr<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledECR",
                        MQT_NAMED_BUILDER(qc::multipleControlledEcr),
                        MQT_NAMED_BUILDER(qir::multipleControlledEcr<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCToQIRAdaptiveGPhaseOpTest, QCToQIRAdaptiveTest,
                         testing::Values(QCToQIRAdaptiveTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qc::globalPhase),
                             MQT_NAMED_BUILDER(qir::globalPhase<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveHOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                                MQT_NAMED_BUILDER(qir::h<true>)},
        QCToQIRAdaptiveTestCase{
            "SingleControlledH", MQT_NAMED_BUILDER(qc::singleControlledH),
            MQT_NAMED_BUILDER(qir::singleControlledH<true>)},
        QCToQIRAdaptiveTestCase{
            "MultipleControlledH", MQT_NAMED_BUILDER(qc::multipleControlledH),
            MQT_NAMED_BUILDER(qir::multipleControlledH<true>)},
        QCToQIRAdaptiveTestCase{"HWithoutRegister",
                                MQT_NAMED_BUILDER(qc::hWithoutRegister),
                                MQT_NAMED_BUILDER(qir::hWithoutRegister)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveIDOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{"Identity", MQT_NAMED_BUILDER(qc::identity),
                                MQT_NAMED_BUILDER(qir::identity<true>)},
        QCToQIRAdaptiveTestCase{
            "SingleControlledIdentity",
            MQT_NAMED_BUILDER(qc::singleControlledIdentity),
            MQT_NAMED_BUILDER(qir::twoQubitsOneIdentity<true>)},
        QCToQIRAdaptiveTestCase{
            "MultipleControlledIdentity",
            MQT_NAMED_BUILDER(qc::multipleControlledIdentity),
            MQT_NAMED_BUILDER(qir::threeQubitsOneIdentity<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveiSWAPOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{"iSWAP", MQT_NAMED_BUILDER(qc::iswap),
                                MQT_NAMED_BUILDER(qir::iswap<true>)},
        QCToQIRAdaptiveTestCase{
            "SingleControllediSWAP",
            MQT_NAMED_BUILDER(qc::singleControlledIswap),
            MQT_NAMED_BUILDER(qir::singleControlledIswap<true>)},
        QCToQIRAdaptiveTestCase{
            "MultipleControllediSWAP",
            MQT_NAMED_BUILDER(qc::multipleControlledIswap),
            MQT_NAMED_BUILDER(qir::multipleControlledIswap<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptivePOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"P", MQT_NAMED_BUILDER(qc::p),
                                            MQT_NAMED_BUILDER(qir::p<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledP",
                        MQT_NAMED_BUILDER(qc::singleControlledP),
                        MQT_NAMED_BUILDER(qir::singleControlledP<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledP",
                        MQT_NAMED_BUILDER(qc::multipleControlledP),
                        MQT_NAMED_BUILDER(qir::multipleControlledP<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/RCCXOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveRCCXOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"RCCX", MQT_NAMED_BUILDER(qc::rccx),
                                            MQT_NAMED_BUILDER(qir::rccx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledRCCX",
                        MQT_NAMED_BUILDER(qc::singleControlledRccx),
                        MQT_NAMED_BUILDER(qir::singleControlledRccx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledRCCX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRccx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRccx<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveROpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"R", MQT_NAMED_BUILDER(qc::r),
                                            MQT_NAMED_BUILDER(qir::r<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledR",
                        MQT_NAMED_BUILDER(qc::singleControlledR),
                        MQT_NAMED_BUILDER(qir::singleControlledR<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledR",
                        MQT_NAMED_BUILDER(qc::multipleControlledR),
                        MQT_NAMED_BUILDER(qir::multipleControlledR<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveRXOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"RX", MQT_NAMED_BUILDER(qc::rx),
                                            MQT_NAMED_BUILDER(qir::rx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledRX",
                        MQT_NAMED_BUILDER(qc::singleControlledRx),
                        MQT_NAMED_BUILDER(qir::singleControlledRx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledRX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRx<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveRXXOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"RXX", MQT_NAMED_BUILDER(qc::rxx),
                                            MQT_NAMED_BUILDER(qir::rxx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledRXX",
                        MQT_NAMED_BUILDER(qc::singleControlledRxx),
                        MQT_NAMED_BUILDER(qir::singleControlledRxx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledRXX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRxx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRxx<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveRYOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"RY", MQT_NAMED_BUILDER(qc::ry),
                                            MQT_NAMED_BUILDER(qir::ry<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledRY",
                        MQT_NAMED_BUILDER(qc::singleControlledRy),
                        MQT_NAMED_BUILDER(qir::singleControlledRy<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledRY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRy),
                        MQT_NAMED_BUILDER(qir::multipleControlledRy<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveRYYOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"RYY", MQT_NAMED_BUILDER(qc::ryy),
                                            MQT_NAMED_BUILDER(qir::ryy<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledRYY",
                        MQT_NAMED_BUILDER(qc::singleControlledRyy),
                        MQT_NAMED_BUILDER(qir::singleControlledRyy<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledRYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRyy),
                        MQT_NAMED_BUILDER(qir::multipleControlledRyy<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveRZOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz),
                                            MQT_NAMED_BUILDER(qir::rz<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledRZ",
                        MQT_NAMED_BUILDER(qc::singleControlledRz),
                        MQT_NAMED_BUILDER(qir::singleControlledRz<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledRZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRz),
                        MQT_NAMED_BUILDER(qir::multipleControlledRz<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveRZXOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"RZX", MQT_NAMED_BUILDER(qc::rzx),
                                            MQT_NAMED_BUILDER(qir::rzx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledRZX",
                        MQT_NAMED_BUILDER(qc::singleControlledRzx),
                        MQT_NAMED_BUILDER(qir::singleControlledRzx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledRZX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRzx<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveRZZOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"RZZ", MQT_NAMED_BUILDER(qc::rzz),
                                            MQT_NAMED_BUILDER(qir::rzz<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::singleControlledRzz),
                        MQT_NAMED_BUILDER(qir::singleControlledRzz<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzz),
                        MQT_NAMED_BUILDER(qir::multipleControlledRzz<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveSOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"S", MQT_NAMED_BUILDER(qc::s),
                                            MQT_NAMED_BUILDER(qir::s<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledS",
                        MQT_NAMED_BUILDER(qc::singleControlledS),
                        MQT_NAMED_BUILDER(qir::singleControlledS<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledS",
                        MQT_NAMED_BUILDER(qc::multipleControlledS),
                        MQT_NAMED_BUILDER(qir::multipleControlledS<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveSdgOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"Sdg", MQT_NAMED_BUILDER(qc::sdg),
                                            MQT_NAMED_BUILDER(qir::sdg<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledSdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSdg),
                        MQT_NAMED_BUILDER(qir::singleControlledSdg<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledSdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSdg),
                        MQT_NAMED_BUILDER(qir::multipleControlledSdg<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveSWAPOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"SWAP", MQT_NAMED_BUILDER(qc::swap),
                                            MQT_NAMED_BUILDER(qir::swap<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::singleControlledSwap),
                        MQT_NAMED_BUILDER(qir::singleControlledSwap<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::multipleControlledSwap),
                        MQT_NAMED_BUILDER(qir::multipleControlledSwap<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveSXOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"SX", MQT_NAMED_BUILDER(qc::sx),
                                            MQT_NAMED_BUILDER(qir::sx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledSX",
                        MQT_NAMED_BUILDER(qc::singleControlledSx),
                        MQT_NAMED_BUILDER(qir::singleControlledSx<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledSX",
                        MQT_NAMED_BUILDER(qc::multipleControlledSx),
                        MQT_NAMED_BUILDER(qir::multipleControlledSx<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveSXdgOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"SXdg", MQT_NAMED_BUILDER(qc::sxdg),
                                            MQT_NAMED_BUILDER(qir::sxdg<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSxdg),
                        MQT_NAMED_BUILDER(qir::singleControlledSxdg<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSxdg),
                        MQT_NAMED_BUILDER(qir::multipleControlledSxdg<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveTOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"T", MQT_NAMED_BUILDER(qc::t_),
                                            MQT_NAMED_BUILDER(qir::t_<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledT",
                        MQT_NAMED_BUILDER(qc::singleControlledT),
                        MQT_NAMED_BUILDER(qir::singleControlledT<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledT",
                        MQT_NAMED_BUILDER(qc::multipleControlledT),
                        MQT_NAMED_BUILDER(qir::multipleControlledT<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveTdgOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg),
                                            MQT_NAMED_BUILDER(qir::tdg<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledTdg",
                        MQT_NAMED_BUILDER(qc::singleControlledTdg),
                        MQT_NAMED_BUILDER(qir::singleControlledTdg<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledTdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledTdg),
                        MQT_NAMED_BUILDER(qir::multipleControlledTdg<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveU2OpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"U2", MQT_NAMED_BUILDER(qc::u2),
                                            MQT_NAMED_BUILDER(qir::u2<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledU2",
                        MQT_NAMED_BUILDER(qc::singleControlledU2),
                        MQT_NAMED_BUILDER(qir::singleControlledU2<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledU2",
                        MQT_NAMED_BUILDER(qc::multipleControlledU2),
                        MQT_NAMED_BUILDER(qir::multipleControlledU2<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveUOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                                            MQT_NAMED_BUILDER(qir::u<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledU",
                        MQT_NAMED_BUILDER(qc::singleControlledU),
                        MQT_NAMED_BUILDER(qir::singleControlledU<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledU",
                        MQT_NAMED_BUILDER(qc::multipleControlledU),
                        MQT_NAMED_BUILDER(qir::multipleControlledU<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveXOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"X", MQT_NAMED_BUILDER(qc::x),
                                            MQT_NAMED_BUILDER(qir::x<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledX",
                        MQT_NAMED_BUILDER(qc::singleControlledX),
                        MQT_NAMED_BUILDER(qir::singleControlledX<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledX",
                        MQT_NAMED_BUILDER(qc::multipleControlledX),
                        MQT_NAMED_BUILDER(qir::multipleControlledX<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveXXMinusYYOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qc::xxMinusYY),
                                MQT_NAMED_BUILDER(qir::xxMinusYY<true>)},
        QCToQIRAdaptiveTestCase{
            "SingleControlledXXMinusYY",
            MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY),
            MQT_NAMED_BUILDER(qir::singleControlledXxMinusYY<true>)},
        QCToQIRAdaptiveTestCase{
            "MultipleControlledXXMinusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY),
            MQT_NAMED_BUILDER(qir::multipleControlledXxMinusYY<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveXXPlusYYOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
                                MQT_NAMED_BUILDER(qir::xxPlusYY<true>)},
        QCToQIRAdaptiveTestCase{
            "SingleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY),
            MQT_NAMED_BUILDER(qir::singleControlledXxPlusYY<true>)},
        QCToQIRAdaptiveTestCase{
            "MultipleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY),
            MQT_NAMED_BUILDER(qir::multipleControlledXxPlusYY<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveYOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                                            MQT_NAMED_BUILDER(qir::y<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledY",
                        MQT_NAMED_BUILDER(qc::singleControlledY),
                        MQT_NAMED_BUILDER(qir::singleControlledY<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledY",
                        MQT_NAMED_BUILDER(qc::multipleControlledY),
                        MQT_NAMED_BUILDER(qir::multipleControlledY<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveZOpTest, QCToQIRAdaptiveTest,
    testing::Values(QCToQIRAdaptiveTestCase{"Z", MQT_NAMED_BUILDER(qc::z),
                                            MQT_NAMED_BUILDER(qir::z<true>)},
                    QCToQIRAdaptiveTestCase{
                        "SingleControlledZ",
                        MQT_NAMED_BUILDER(qc::singleControlledZ),
                        MQT_NAMED_BUILDER(qir::singleControlledZ<true>)},
                    QCToQIRAdaptiveTestCase{
                        "MultipleControlledZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledZ),
                        MQT_NAMED_BUILDER(qir::multipleControlledZ<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveMeasureOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(qir::singleMeasurementToSingleBit)},
        QCToQIRAdaptiveTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(qir::repeatedMeasurementToSameBit)},
        QCToQIRAdaptiveTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qir::repeatedMeasurementToDifferentBits)},
        QCToQIRAdaptiveTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(qir::multipleClassicalRegistersAndMeasurements)},
        QCToQIRAdaptiveTestCase{
            "PartialMeasurementToRegister",
            MQT_NAMED_BUILDER(qc::partialMeasurementToRegister),
            MQT_NAMED_BUILDER(qir::partialMeasurementToRegister)},
        QCToQIRAdaptiveTestCase{
            "DynamicallyIndexedMeasurement",
            MQT_NAMED_BUILDER(qc::dynamicallyIndexedMeasurement),
            MQT_NAMED_BUILDER(qir::dynamicallyIndexedMeasurement)},
        QCToQIRAdaptiveTestCase{
            "MeasurementWithoutRegisters",
            MQT_NAMED_BUILDER(qc::measurementWithoutRegisters),
            MQT_NAMED_BUILDER(qir::measurementWithoutRegisters)}));
/// @}

/// \name QCToQIRAdaptive/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveResetOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{
            "ResetQubitWithoutOp", MQT_NAMED_BUILDER(qc::resetQubitWithoutOp),
            MQT_NAMED_BUILDER(qir::resetQubitWithoutOp<true>)},
        QCToQIRAdaptiveTestCase{
            "ResetMultipleQubitsWithoutOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsWithoutOp),
            MQT_NAMED_BUILDER(qir::resetMultipleQubitsWithoutOp<true>)},
        QCToQIRAdaptiveTestCase{
            "RepeatedResetWithoutOp",
            MQT_NAMED_BUILDER(qc::repeatedResetWithoutOp),
            MQT_NAMED_BUILDER(qir::repeatedResetWithoutOp<true>)},
        QCToQIRAdaptiveTestCase{
            "ResetQubitAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp),
            MQT_NAMED_BUILDER(qir::resetQubitAfterSingleOp<true>)},
        QCToQIRAdaptiveTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qir::resetMultipleQubitsAfterSingleOp<true>)},
        QCToQIRAdaptiveTestCase{
            "RepeatedResetAfterSingleOp",
            MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp),
            MQT_NAMED_BUILDER(qir::repeatedResetAfterSingleOp<true>)}));
/// @}

/// \name QCToQIRAdaptive/QubitManagement/QubitManagement.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRAdaptiveQubitManagementTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{"AllocQubit", MQT_NAMED_BUILDER(qc::allocQubit),
                                MQT_NAMED_BUILDER(qir::allocQubit<true>)},
        QCToQIRAdaptiveTestCase{
            "AllocQubitRegister", MQT_NAMED_BUILDER(qc::allocQubitRegister),
            MQT_NAMED_BUILDER(qir::allocQubitRegister<true>)},
        QCToQIRAdaptiveTestCase{
            "AllocMultipleQubitRegisters",
            MQT_NAMED_BUILDER(qc::allocMultipleQubitRegisters),
            MQT_NAMED_BUILDER(qir::allocMultipleQubitRegisters<true>)},
        QCToQIRAdaptiveTestCase{
            "AllocMultipleQubitRegistersWithOps",
            MQT_NAMED_BUILDER(qc::allocMultipleQubitRegistersWithOps),
            MQT_NAMED_BUILDER(qir::allocMultipleQubitRegistersWithOps<true>)},
        QCToQIRAdaptiveTestCase{
            "AllocLargeRegister", MQT_NAMED_BUILDER(qc::allocLargeRegister),
            MQT_NAMED_BUILDER(qir::allocQubitRegister<true>)},
        QCToQIRAdaptiveTestCase{"StaticQubits",
                                MQT_NAMED_BUILDER(qc::staticQubits),
                                MQT_NAMED_BUILDER(qir::staticQubits)},
        QCToQIRAdaptiveTestCase{"StaticQubitsWithOps",
                                MQT_NAMED_BUILDER(qc::staticQubitsWithOps),
                                MQT_NAMED_BUILDER(qir::staticQubitsWithOps)},
        QCToQIRAdaptiveTestCase{
            "StaticQubitsWithParametricOps",
            MQT_NAMED_BUILDER(qc::staticQubitsWithParametricOps),
            MQT_NAMED_BUILDER(qir::staticQubitsWithParametricOps)},
        QCToQIRAdaptiveTestCase{
            "StaticQubitsWithTwoTargetOps",
            MQT_NAMED_BUILDER(qc::staticQubitsWithTwoTargetOps),
            MQT_NAMED_BUILDER(qir::staticQubitsWithTwoTargetOps)},
        QCToQIRAdaptiveTestCase{"StaticQubitsWithCtrl",
                                MQT_NAMED_BUILDER(qc::staticQubitsWithCtrl),
                                MQT_NAMED_BUILDER(qir::staticQubitsWithCtrl)},
        QCToQIRAdaptiveTestCase{"StaticQubitsWithInv",
                                MQT_NAMED_BUILDER(qc::staticQubitsWithInv),
                                MQT_NAMED_BUILDER(qir::staticQubitsWithInv)},
        QCToQIRAdaptiveTestCase{"AllocDeallocPair",
                                MQT_NAMED_BUILDER(qc::allocDeallocPair),
                                MQT_NAMED_BUILDER(qir::emptyQIR<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/IfOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFIfOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{"SimpleIfOp", MQT_NAMED_BUILDER(qc::simpleIf),
                                MQT_NAMED_BUILDER(qir::simpleIf)},
        QCToQIRAdaptiveTestCase{"IfTwoQubits",
                                MQT_NAMED_BUILDER(qc::ifTwoQubits),
                                MQT_NAMED_BUILDER(qir::ifTwoQubits)},
        QCToQIRAdaptiveTestCase{"IfElse", MQT_NAMED_BUILDER(qc::ifElse),
                                MQT_NAMED_BUILDER(qir::ifElse)},
        QCToQIRAdaptiveTestCase{
            "NestedIfOpForLoop", MQT_NAMED_BUILDER(qc::nestedIfOpForLoop),
            MQT_NAMED_BUILDER(qir::nestedIfOpForLoop<true>)}));
/// @}

/// \name QCToQIRAdaptive/Operations/WhileOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFWhileOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{"SimpleWhile",
                                MQT_NAMED_BUILDER(qc::simpleWhileReset),
                                MQT_NAMED_BUILDER(qir::simpleWhileReset<true>)},
        QCToQIRAdaptiveTestCase{
            "SimpleDoWhile", MQT_NAMED_BUILDER(qc::simpleDoWhileReset),
            MQT_NAMED_BUILDER(qir::simpleDoWhileReset<true>)}));

/// \name QCToQIRAdaptive/Operations/ForOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFForOpTest, QCToQIRAdaptiveTest,
    testing::Values(
        QCToQIRAdaptiveTestCase{"SimpleForLoop",
                                MQT_NAMED_BUILDER(qc::simpleForLoop),
                                MQT_NAMED_BUILDER(qir::simpleForLoop<true>)},
        QCToQIRAdaptiveTestCase{
            "NestedForLoopIfOp", MQT_NAMED_BUILDER(qc::nestedForLoopIfOp),
            MQT_NAMED_BUILDER(qir::nestedForLoopIfOp<true>)},
        QCToQIRAdaptiveTestCase{
            "NestedForLoopWhileOp", MQT_NAMED_BUILDER(qc::nestedForLoopWhileOp),
            MQT_NAMED_BUILDER(qir::nestedForLoopWhileOp<true>)},
        QCToQIRAdaptiveTestCase{
            "NestedForLoopCtrlOpWithSeparateQubit",
            MQT_NAMED_BUILDER(qc::nestedForLoopCtrlOpWithSeparateQubit),
            MQT_NAMED_BUILDER(qir::nestedForLoopCtrlOpWithSeparateQubit<true>)},
        QCToQIRAdaptiveTestCase{
            "NestedForLoopCtrlOpWithExtractedQubit",
            MQT_NAMED_BUILDER(qc::nestedForLoopCtrlOpWithExtractedQubit),
            MQT_NAMED_BUILDER(
                qir::nestedForLoopCtrlOpWithExtractedQubit<true>)}));

/// \name QCToQIRAdaptive/Modifiers/CtrlOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCToQIRCtrlOpTest, QCToQIRAdaptiveTest,
                         testing::Values(QCToQIRAdaptiveTestCase{
                             "NestedCtrlTwo", MQT_NAMED_BUILDER(qc::ctrlTwo),
                             MQT_NAMED_BUILDER(qir::ctrlTwo<true>)}));
/// @}
