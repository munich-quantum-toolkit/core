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
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>

using namespace mlir;
using namespace mlir::qco;

namespace {

struct QCOTestCase {
  std::string name;
  mqt::test::NamedMLIRBuilder<QCOProgramBuilder> programBuilder;
  mqt::test::NamedMLIRBuilder<QCOProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os, const QCOTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os, const QCOTestCase& info) {
  return os << "QCO{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

class QCOTest : public testing::TestWithParam<QCOTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    qtensor::QTensorDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};
} // namespace

TEST_P(QCOTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  mqt::test::DeferredPrinter printer;

  auto program = mqt::test::buildMLIRProgram(context.get(), programBuilder);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = mqt::test::buildMLIRProgram(context.get(), referenceBuilder);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(QCOTest, BuilderRejectsMixedStaticAndDynamicQubitAllocationModes) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        mixedStaticThenDynamicQubit(builder);
      },
      "Cannot mix static and dynamic qubit allocation modes");

  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        mixedDynamicRegisterThenStaticQubit(builder);
      },
      "Cannot mix dynamic and static qubit allocation modes");
}

TEST_F(QCOTest, DirectIfBuilder) {
  // Test If construction directly
  QCOProgramBuilder builder(context.get());
  builder.initialize({builder.getI1Type(), builder.getI1Type()});
  auto c0 = arith::ConstantIndexOp::create(builder, 0);
  auto c1 = arith::ConstantIndexOp::create(builder, 1);
  auto r0 = qtensor::AllocOp::create(builder, c1);
  auto extractOp = qtensor::ExtractOp::create(builder, r0, c0);
  auto q1 = HOp::create(builder, extractOp.getResult());
  auto measureOp = MeasureOp::create(builder, q1);
  auto ifOp =
      IfOp::create(builder, measureOp.getResult(), measureOp.getQubitOut(),
                   [&](ValueRange qubits) -> SmallVector<Value> {
                     auto innerQubit = XOp::create(builder, qubits[0]);
                     return SmallVector<Value>{innerQubit};
                   });
  auto finalMeasureOp = MeasureOp::create(builder, ifOp.getResult(0));
  auto r2 = qtensor::InsertOp::create(builder, finalMeasureOp.getQubitOut(),
                                      extractOp.getOutTensor(), c0);
  qtensor::DeallocOp::create(builder, r2);

  auto directBuilder =
      builder.finalize({measureOp.getResult(), finalMeasureOp.getResult()});
  ASSERT_TRUE(directBuilder);
  EXPECT_TRUE(verify(*directBuilder).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(directBuilder.get()).succeeded());
  EXPECT_TRUE(verify(*directBuilder).succeeded());

  auto refBuilder =
      mqt::test::buildMLIRProgram(context.get(), MQT_NAMED_BUILDER(simpleIf));
  ASSERT_TRUE(refBuilder);
  EXPECT_TRUE(verify(*refBuilder).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(refBuilder.get()).succeeded());
  EXPECT_TRUE(verify(*refBuilder).succeeded());

  EXPECT_TRUE(areModulesEquivalentWithPermutations(directBuilder.get(),
                                                   refBuilder.get()));
}

TEST_F(QCOTest, IfOpParser) {
  // Test IfOp parser
  const char* mlirCode = R"(
      module {
        func.func @main() -> i1 attributes {passthrough = ["entry_point"]} {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %q0_0 = qco.alloc : !qco.qubit
            %t0 = qtensor.alloc(%c1) : tensor<1x!qco.qubit>
            %q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit
            %q0_2, %cond = qco.measure %q0_1 : !qco.qubit
            %q0_4, %t3 = qco.if %cond args(%arg0 = %q0_2, %arg1 = %t0) -> (!qco.qubit, tensor<1x!qco.qubit>) {
                %q0_3 = qco.x %arg0 : !qco.qubit -> !qco.qubit
                %t1, %q1_0 = qtensor.extract %arg1[%c0] : tensor<1x!qco.qubit>
                %q1_1 = qco.x %q1_0 : !qco.qubit -> !qco.qubit
                %t2 = qtensor.insert %q1_1 into %t1[%c0] : tensor<1x!qco.qubit>
                qco.yield %q0_3, %t2 : !qco.qubit, tensor<1x!qco.qubit>
            } else args(%arg0 = %q0_2, %arg1 = %t0) {
                qco.yield %arg0, %arg1 : !qco.qubit, tensor<1x!qco.qubit>
            }
            %q0_5, %c = qco.measure %q0_4 : !qco.qubit
            qco.sink %q0_5 : !qco.qubit
            qtensor.dealloc %t3 : tensor<1x!qco.qubit>
            return %c : i1
        }
    })";

  auto parsedSourceModule =
      parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(parsedSourceModule);
  EXPECT_TRUE(verify(*parsedSourceModule).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(parsedSourceModule.get()).succeeded());
  EXPECT_TRUE(verify(*parsedSourceModule).succeeded());

  auto refBuilder = mqt::test::buildMLIRProgram(
      context.get(), MQT_NAMED_BUILDER(ifOneQubitOneTensor));
  ASSERT_TRUE(refBuilder);
  EXPECT_TRUE(verify(*refBuilder).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(refBuilder.get()).succeeded());
  EXPECT_TRUE(verify(*refBuilder).succeeded());

  EXPECT_TRUE(areModulesEquivalentWithPermutations(parsedSourceModule.get(),
                                                   refBuilder.get()));
}

/// \name QCO/SCF/IfOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOIfOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SimpleIf", MQT_NAMED_BUILDER(simpleIf),
                    MQT_NAMED_BUILDER(simpleIf)},
        QCOTestCase{"TwoQubitIf", MQT_NAMED_BUILDER(ifTwoQubits),
                    MQT_NAMED_BUILDER(ifTwoQubits)},
        QCOTestCase{"IfElse", MQT_NAMED_BUILDER(ifElse),
                    MQT_NAMED_BUILDER(ifElse)},
        QCOTestCase{"ConstantTrueIf", MQT_NAMED_BUILDER(constantTrueIf),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"ConstantFalseIf", MQT_NAMED_BUILDER(constantFalseIf),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"NestedTrueIf", MQT_NAMED_BUILDER(nestedTrueIf),
                    MQT_NAMED_BUILDER(simpleIf)},
        QCOTestCase{"NestedFalseIf", MQT_NAMED_BUILDER(nestedFalseIf),
                    MQT_NAMED_BUILDER(ifElse)}));
/// @}

/// \name QCO/Modifiers/CtrlOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOCtrlOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"TrivialCtrl", MQT_NAMED_BUILDER(trivialCtrl),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"EmptyCtrl", MQT_NAMED_BUILDER(emptyCtrl),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"NestedCtrl", MQT_NAMED_BUILDER(nestedCtrl),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"TripleNestedCtrl", MQT_NAMED_BUILDER(tripleNestedCtrl),
                    MQT_NAMED_BUILDER(tripleControlledRxx)},
        QCOTestCase{"CtrlInvSandwich", MQT_NAMED_BUILDER(ctrlInvSandwich),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"DoubleNestedCtrlTwoQubits",
                    MQT_NAMED_BUILDER(doubleNestedCtrlTwoQubits),
                    MQT_NAMED_BUILDER(fourControlledRxx)},
        QCOTestCase{"NestedCtrlTwo", MQT_NAMED_BUILDER(nestedCtrlTwo),
                    MQT_NAMED_BUILDER(ctrlTwo)}));
/// @}

/// \name QCO/Modifiers/InvOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOInvOpTest, QCOTest,
    testing::Values(QCOTestCase{"EmptyInv", MQT_NAMED_BUILDER(emptyInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"NestedInv", MQT_NAMED_BUILDER(nestedInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"TripleNestedInv",
                                MQT_NAMED_BUILDER(tripleNestedInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"InvControlSandwich",
                                MQT_NAMED_BUILDER(invCtrlSandwich),
                                MQT_NAMED_BUILDER(singleControlledRxx)},
                    QCOTestCase{"InvCtrlTwo", MQT_NAMED_BUILDER(invCtrlTwo),
                                MQT_NAMED_BUILDER(ctrlInvTwo)},
                    QCOTestCase{"InverseT", MQT_NAMED_BUILDER(inverseT),
                                MQT_NAMED_BUILDER(tdg)}));
/// @}

/// \name QCO/Modifiers/PowOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPowOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Pow1Inline", MQT_NAMED_BUILDER(pow1Inline),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"Pow0Erase", MQT_NAMED_BUILDER(pow0Erase),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"Pow0Two", MQT_NAMED_BUILDER(pow0Two),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"EmptyPow", MQT_NAMED_BUILDER(emptyPow),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"NestedPow", MQT_NAMED_BUILDER(nestedPow),
                    MQT_NAMED_BUILDER(powSingleExponent)},
        QCOTestCase{"NegPowRx", MQT_NAMED_BUILDER(negPowRx),
                    MQT_NAMED_BUILDER(powRxNeg)},
        QCOTestCase{"InvPowRx", MQT_NAMED_BUILDER(invPowRx),
                    MQT_NAMED_BUILDER(powRxNeg)},
        QCOTestCase{"InvPowReordered", MQT_NAMED_BUILDER(invPowReordered),
                    MQT_NAMED_BUILDER(invPowReorderedRef)},
        QCOTestCase{"MergeNestedPowReordered",
                    MQT_NAMED_BUILDER(mergeNestedPowReordered),
                    MQT_NAMED_BUILDER(mergeNestedPowReorderedRef)},
        QCOTestCase{"PowCtrlRx", MQT_NAMED_BUILDER(powCtrlRx),
                    MQT_NAMED_BUILDER(ctrlPowRx)},
        QCOTestCase{"NegPowInvIswap", MQT_NAMED_BUILDER(negPowInvIswap),
                    MQT_NAMED_BUILDER(negPowInvIswapRef)},
        QCOTestCase{"InvPowHFrac", MQT_NAMED_BUILDER(invPowHFrac),
                    MQT_NAMED_BUILDER(powHFracNeg)},
        QCOTestCase{"InvPowEvenH", MQT_NAMED_BUILDER(invPowEvenH),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"InvPowEvenSwap", MQT_NAMED_BUILDER(invPowEvenSwap),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"InvPowSquaredZ", MQT_NAMED_BUILDER(invPowSquaredZ),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)}));
/// @}

TEST_F(QCOTest, PowExponentIsUnitaryParameter) {
  auto program =
      mqt::test::buildMLIRProgram(context.get(), MQT_NAMED_BUILDER(powRxx));
  ASSERT_TRUE(program);

  auto funcOp = cast<func::FuncOp>(program->getBody()->front());
  auto powOp = *funcOp.getBody().getOps<PowOp>().begin();
  auto unitary = cast<UnitaryOpInterface>(powOp.getOperation());
  EXPECT_EQ(unitary.getNumParams(), 1);
  EXPECT_EQ(unitary.getParameter(0), powOp.getExponent());
  ASSERT_EQ(unitary.getParameters().size(), 1);
  EXPECT_EQ(unitary.getParameters().front(), powOp.getExponent());
}

/// pow(rxx) folds the exponent into the rotation angle: pow(2){rxx(θ)} =>
/// rxx(2θ). Verify that PowOp is folded away by the cleanup pipeline.
TEST_F(QCOTest, PowRxxFold) {
  auto program =
      mqt::test::buildMLIRProgram(context.get(), MQT_NAMED_BUILDER(powRxx));
  ASSERT_TRUE(program);
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  int powCount = 0;
  program->walk([&](PowOp) { ++powCount; });
  EXPECT_EQ(powCount, 0) << "PowOp around rxx should be folded away";
}

/// pow(-0.5) { h } cannot fold a negative fractional exponent
/// into H (no angle to scale). Verify that PowOp survives.
TEST_F(QCOTest, NegPowHNoFold) {
  auto program =
      mqt::test::buildMLIRProgram(context.get(), MQT_NAMED_BUILDER(negPowH));
  ASSERT_TRUE(program);
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  int powCount = 0;
  program->walk([&](PowOp) { ++powCount; });
  EXPECT_EQ(powCount, 1) << "PowOp around h must survive the pipeline";
}

/// pow(sx) inside a ctrl modifier expands into a GPhase + RX kept within the
/// ctrl body, so the controlled global phase is preserved. Verify the CtrlOp
/// survives and the nested PowOp is expanded into a GPhase + RX.
TEST_F(QCOTest, CtrlPowSxExpands) {
  auto program =
      mqt::test::buildMLIRProgram(context.get(), MQT_NAMED_BUILDER(ctrlPowSx));
  ASSERT_TRUE(program);
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  int ctrlCount = 0;
  int powCount = 0;
  int gphaseCount = 0;
  int rxCount = 0;
  program->walk([&](CtrlOp) { ++ctrlCount; });
  program->walk([&](PowOp) { ++powCount; });
  program->walk([&](GPhaseOp) { ++gphaseCount; });
  program->walk([&](RXOp) { ++rxCount; });
  EXPECT_EQ(ctrlCount, 1) << "CtrlOp must survive the pipeline";
  EXPECT_EQ(powCount, 0) << "PowOp inside ctrl must be expanded";
  EXPECT_EQ(gphaseCount, 1) << "SX fold must emit a GPhase";
  EXPECT_EQ(rxCount, 1) << "SX fold must emit an RX";
}

/// \name QCO/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOBarrierOpTest, QCOTest,
    testing::Values(QCOTestCase{"Barrier", MQT_NAMED_BUILDER(barrier),
                                MQT_NAMED_BUILDER(barrier)},
                    QCOTestCase{"BarrierTwoQubits",
                                MQT_NAMED_BUILDER(barrierTwoQubits),
                                MQT_NAMED_BUILDER(barrierTwoQubits)},
                    QCOTestCase{"BarrierMultipleQubits",
                                MQT_NAMED_BUILDER(barrierMultipleQubits),
                                MQT_NAMED_BUILDER(barrierMultipleQubits)},
                    QCOTestCase{"SingleControlledBarrier",
                                MQT_NAMED_BUILDER(singleControlledBarrier),
                                MQT_NAMED_BUILDER(barrier)},
                    QCOTestCase{"InverseBarrier",
                                MQT_NAMED_BUILDER(inverseBarrier),
                                MQT_NAMED_BUILDER(barrier)},
                    QCOTestCase{"TwoBarrier", MQT_NAMED_BUILDER(twoBarrier),
                                MQT_NAMED_BUILDER(barrierTwoQubits)},
                    QCOTestCase{"PowBarrier", MQT_NAMED_BUILDER(powBarrier),
                                MQT_NAMED_BUILDER(barrier)}));
/// @}

/// \name QCO/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCODCXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"DCX", MQT_NAMED_BUILDER(dcx), MQT_NAMED_BUILDER(dcx)},
        QCOTestCase{"SingleControlledDCX",
                    MQT_NAMED_BUILDER(singleControlledDcx),
                    MQT_NAMED_BUILDER(singleControlledDcx)},
        QCOTestCase{"MultipleControlledDCX",
                    MQT_NAMED_BUILDER(multipleControlledDcx),
                    MQT_NAMED_BUILDER(multipleControlledDcx)},
        QCOTestCase{"NestedControlledDCX",
                    MQT_NAMED_BUILDER(nestedControlledDcx),
                    MQT_NAMED_BUILDER(multipleControlledDcx)},
        QCOTestCase{"TrivialControlledDCX",
                    MQT_NAMED_BUILDER(trivialControlledDcx),
                    MQT_NAMED_BUILDER(dcx)},
        QCOTestCase{"InverseDCX", MQT_NAMED_BUILDER(inverseDcx),
                    MQT_NAMED_BUILDER(inverseDcx)},
        QCOTestCase{"InverseMultipleControlledDCX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledDcx),
                    MQT_NAMED_BUILDER(inverseMultipleControlledDcx)},
        QCOTestCase{"TwoDCX", MQT_NAMED_BUILDER(twoDcx),
                    MQT_NAMED_BUILDER(twoDcx)},
        QCOTestCase{"TwoDCXSwappedTargets",
                    MQT_NAMED_BUILDER(twoDcxSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOECROpTest, QCOTest,
    testing::Values(QCOTestCase{"ECR", MQT_NAMED_BUILDER(ecr),
                                MQT_NAMED_BUILDER(ecr)},
                    QCOTestCase{"SingleControlledECR",
                                MQT_NAMED_BUILDER(singleControlledEcr),
                                MQT_NAMED_BUILDER(singleControlledEcr)},
                    QCOTestCase{"MultipleControlledECR",
                                MQT_NAMED_BUILDER(multipleControlledEcr),
                                MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCOTestCase{"NestedControlledECR",
                                MQT_NAMED_BUILDER(nestedControlledEcr),
                                MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCOTestCase{"TrivialControlledECR",
                                MQT_NAMED_BUILDER(trivialControlledEcr),
                                MQT_NAMED_BUILDER(ecr)},
                    QCOTestCase{"InverseECR", MQT_NAMED_BUILDER(inverseEcr),
                                MQT_NAMED_BUILDER(ecr)},
                    QCOTestCase{"InverseMultipleControlledECR",
                                MQT_NAMED_BUILDER(inverseMultipleControlledEcr),
                                MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCOTestCase{"TwoECR", MQT_NAMED_BUILDER(twoEcr),
                                MQT_NAMED_BUILDER(alloc2QubitRegister)},
                    QCOTestCase{"PowEvenECR", MQT_NAMED_BUILDER(powEvenEcr),
                                MQT_NAMED_BUILDER(alloc2QubitRegister)},
                    QCOTestCase{"PowOddECR", MQT_NAMED_BUILDER(powOddEcr),
                                MQT_NAMED_BUILDER(ecr)}));
/// @}

/// \name QCO/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOGPhaseOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"GlobalPhase", MQT_NAMED_BUILDER(globalPhase),
                    MQT_NAMED_BUILDER(globalPhase)},
        QCOTestCase{"SingleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(singleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"MultipleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(multipleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"InverseGlobalPhase", MQT_NAMED_BUILDER(inverseGlobalPhase),
                    MQT_NAMED_BUILDER(globalPhase)},
        QCOTestCase{"InverseMultipleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(inverseMultipleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(multipleControlledGlobalPhase)},
        QCOTestCase{"PowGphaseScaled", MQT_NAMED_BUILDER(powGphaseScaled),
                    MQT_NAMED_BUILDER(powGphaseScaledRef)},
        QCOTestCase{"NegPowGphase", MQT_NAMED_BUILDER(negPowGphase),
                    MQT_NAMED_BUILDER(negPowGphaseRef)}));
/// @}

/// \name QCO/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOHOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"H", MQT_NAMED_BUILDER(h), MQT_NAMED_BUILDER(h)},
        QCOTestCase{"SingleControlledH", MQT_NAMED_BUILDER(singleControlledH),
                    MQT_NAMED_BUILDER(singleControlledH)},
        QCOTestCase{"MultipleControlledH",
                    MQT_NAMED_BUILDER(multipleControlledH),
                    MQT_NAMED_BUILDER(multipleControlledH)},
        QCOTestCase{"NestedControlledH", MQT_NAMED_BUILDER(nestedControlledH),
                    MQT_NAMED_BUILDER(multipleControlledH)},
        QCOTestCase{"TrivialControlledH", MQT_NAMED_BUILDER(trivialControlledH),
                    MQT_NAMED_BUILDER(h)},
        QCOTestCase{"InverseH", MQT_NAMED_BUILDER(inverseH),
                    MQT_NAMED_BUILDER(h)},
        QCOTestCase{"InverseMultipleControlledH",
                    MQT_NAMED_BUILDER(inverseMultipleControlledH),
                    MQT_NAMED_BUILDER(multipleControlledH)},
        QCOTestCase{"TwoH", MQT_NAMED_BUILDER(twoH),
                    MQT_NAMED_BUILDER(allocQubit)},
        QCOTestCase{"PowEvenH", MQT_NAMED_BUILDER(powEvenH),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowOddH", MQT_NAMED_BUILDER(powOddH),
                    MQT_NAMED_BUILDER(h)}));
/// @}

/// \name QCO/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOIDOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Identity", MQT_NAMED_BUILDER(identity),
                    MQT_NAMED_BUILDER(allocQubit)},
        QCOTestCase{"SingleControlledIdentity",
                    MQT_NAMED_BUILDER(singleControlledIdentity),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"MultipleControlledIdentity",
                    MQT_NAMED_BUILDER(multipleControlledIdentity),
                    MQT_NAMED_BUILDER(alloc3QubitRegister)},
        QCOTestCase{"NestedControlledIdentity",
                    MQT_NAMED_BUILDER(nestedControlledIdentity),
                    MQT_NAMED_BUILDER(alloc3QubitRegister)},
        QCOTestCase{"TrivialControlledIdentity",
                    MQT_NAMED_BUILDER(trivialControlledIdentity),
                    MQT_NAMED_BUILDER(allocQubit)},
        QCOTestCase{"InverseIdentity", MQT_NAMED_BUILDER(inverseIdentity),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"InverseMultipleControlledIdentity",
                    MQT_NAMED_BUILDER(inverseMultipleControlledIdentity),
                    MQT_NAMED_BUILDER(alloc3QubitRegister)},
        QCOTestCase{"PowId", MQT_NAMED_BUILDER(powId),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOiSWAPOpTest, QCOTest,
    testing::Values(QCOTestCase{"iSWAP", MQT_NAMED_BUILDER(iswap),
                                MQT_NAMED_BUILDER(iswap)},
                    QCOTestCase{"SingleControllediSWAP",
                                MQT_NAMED_BUILDER(singleControlledIswap),
                                MQT_NAMED_BUILDER(singleControlledIswap)},
                    QCOTestCase{"MultipleControllediSWAP",
                                MQT_NAMED_BUILDER(multipleControlledIswap),
                                MQT_NAMED_BUILDER(multipleControlledIswap)},
                    QCOTestCase{"NestedControllediSWAP",
                                MQT_NAMED_BUILDER(nestedControlledIswap),
                                MQT_NAMED_BUILDER(multipleControlledIswap)},
                    QCOTestCase{"TrivialControllediSWAP",
                                MQT_NAMED_BUILDER(trivialControlledIswap),
                                MQT_NAMED_BUILDER(iswap)},
                    QCOTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(inverseIswap),
                                MQT_NAMED_BUILDER(inverseIswap)},
                    QCOTestCase{
                        "InverseMultipleControllediSWAP",
                        MQT_NAMED_BUILDER(inverseMultipleControlledIswap),
                        MQT_NAMED_BUILDER(inverseMultipleControlledIswap)},
                    QCOTestCase{"PowHalfiSWAP", MQT_NAMED_BUILDER(powHalfIswap),
                                MQT_NAMED_BUILDER(powHalfIswapRef)}));
/// @}

/// \name QCO/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"P", MQT_NAMED_BUILDER(p), MQT_NAMED_BUILDER(p)},
        QCOTestCase{"SingleControlledP", MQT_NAMED_BUILDER(singleControlledP),
                    MQT_NAMED_BUILDER(singleControlledP)},
        QCOTestCase{"MultipleControlledP",
                    MQT_NAMED_BUILDER(multipleControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"NestedControlledP", MQT_NAMED_BUILDER(nestedControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"TrivialControlledP", MQT_NAMED_BUILDER(trivialControlledP),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"InverseP", MQT_NAMED_BUILDER(inverseP),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"InverseMultipleControlledP",
                    MQT_NAMED_BUILDER(inverseMultipleControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"TwoPOppositePhase", MQT_NAMED_BUILDER(twoPOppositePhase),
                    MQT_NAMED_BUILDER(allocQubit)}));
/// @}

/// \name QCO/Operations/StandardGates/RCCXOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORCCXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RCCX", MQT_NAMED_BUILDER(rccx), MQT_NAMED_BUILDER(rccx)},
        QCOTestCase{"SingleControlledRCCX",
                    MQT_NAMED_BUILDER(singleControlledRccx),
                    MQT_NAMED_BUILDER(singleControlledRccx)},
        QCOTestCase{"MultipleControlledRCCX",
                    MQT_NAMED_BUILDER(multipleControlledRccx),
                    MQT_NAMED_BUILDER(multipleControlledRccx)},
        QCOTestCase{"NestedControlledRCCX",
                    MQT_NAMED_BUILDER(nestedControlledRccx),
                    MQT_NAMED_BUILDER(multipleControlledRccx)},
        QCOTestCase{"TrivialControlledRCCX",
                    MQT_NAMED_BUILDER(trivialControlledRccx),
                    MQT_NAMED_BUILDER(rccx)},
        QCOTestCase{"InverseRCCX", MQT_NAMED_BUILDER(inverseRccx),
                    MQT_NAMED_BUILDER(rccx)},
        QCOTestCase{"InverseMultipleControlledRCCX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRccx),
                    MQT_NAMED_BUILDER(multipleControlledRccx)},
        QCOTestCase{"TwoRCCX", MQT_NAMED_BUILDER(twoRccx),
                    MQT_NAMED_BUILDER(alloc3QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOROpTest, QCOTest,
    testing::Values(
        QCOTestCase{"R", MQT_NAMED_BUILDER(r), MQT_NAMED_BUILDER(r)},
        QCOTestCase{"SingleControlledR", MQT_NAMED_BUILDER(singleControlledR),
                    MQT_NAMED_BUILDER(singleControlledR)},
        QCOTestCase{"MultipleControlledR",
                    MQT_NAMED_BUILDER(multipleControlledR),
                    MQT_NAMED_BUILDER(multipleControlledR)},
        QCOTestCase{"NestedControlledR", MQT_NAMED_BUILDER(nestedControlledR),
                    MQT_NAMED_BUILDER(multipleControlledR)},
        QCOTestCase{"TrivialControlledR", MQT_NAMED_BUILDER(trivialControlledR),
                    MQT_NAMED_BUILDER(r)},
        QCOTestCase{"InverseR", MQT_NAMED_BUILDER(inverseR),
                    MQT_NAMED_BUILDER(r)},
        QCOTestCase{"InverseMultipleControlledR",
                    MQT_NAMED_BUILDER(inverseMultipleControlledR),
                    MQT_NAMED_BUILDER(multipleControlledR)},
        QCOTestCase{"CanonicalizeRToRx", MQT_NAMED_BUILDER(canonicalizeRToRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"CanonicalizeRToRy", MQT_NAMED_BUILDER(canonicalizeRToRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"TwoR", MQT_NAMED_BUILDER(twoR), MQT_NAMED_BUILDER(r)},
        QCOTestCase{"PowRScaled", MQT_NAMED_BUILDER(powRScaled),
                    MQT_NAMED_BUILDER(powRScaledRef)}));
/// @}

/// \name QCO/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RX", MQT_NAMED_BUILDER(rx), MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"SingleControlledRX", MQT_NAMED_BUILDER(singleControlledRx),
                    MQT_NAMED_BUILDER(singleControlledRx)},
        QCOTestCase{"MultipleControlledRX",
                    MQT_NAMED_BUILDER(multipleControlledRx),
                    MQT_NAMED_BUILDER(multipleControlledRx)},
        QCOTestCase{"NestedControlledRX", MQT_NAMED_BUILDER(nestedControlledRx),
                    MQT_NAMED_BUILDER(multipleControlledRx)},
        QCOTestCase{"TrivialControlledRX",
                    MQT_NAMED_BUILDER(trivialControlledRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"InverseRX", MQT_NAMED_BUILDER(inverseRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"InverseMultipleControlledRX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRx),
                    MQT_NAMED_BUILDER(multipleControlledRx)},
        QCOTestCase{"TwoRXOppositePhase", MQT_NAMED_BUILDER(twoRxOppositePhase),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowRxScaled", MQT_NAMED_BUILDER(powRxScaled),
                    MQT_NAMED_BUILDER(rxScaled)}));
/// @}

/// \name QCO/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RXX", MQT_NAMED_BUILDER(rxx), MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"SingleControlledRXX",
                    MQT_NAMED_BUILDER(singleControlledRxx),
                    MQT_NAMED_BUILDER(singleControlledRxx)},
        QCOTestCase{"MultipleControlledRXX",
                    MQT_NAMED_BUILDER(multipleControlledRxx),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"NestedControlledRXX",
                    MQT_NAMED_BUILDER(nestedControlledRxx),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"TrivialControlledRXX",
                    MQT_NAMED_BUILDER(trivialControlledRxx),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"InverseRXX", MQT_NAMED_BUILDER(inverseRxx),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"InverseMultipleControlledRXX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRxx),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"TwoRXX", MQT_NAMED_BUILDER(twoRxx),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"TwoRXXSwappedTargets",
                    MQT_NAMED_BUILDER(twoRxxSwappedTargets),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"TwoRXXOppositePhase",
                    MQT_NAMED_BUILDER(twoRxxOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoRXXOppositePhaseSwappedTargets",
                    MQT_NAMED_BUILDER(twoRxxOppositePhaseSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RY", MQT_NAMED_BUILDER(ry), MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"SingleControlledRY", MQT_NAMED_BUILDER(singleControlledRy),
                    MQT_NAMED_BUILDER(singleControlledRy)},
        QCOTestCase{"MultipleControlledRY",
                    MQT_NAMED_BUILDER(multipleControlledRy),
                    MQT_NAMED_BUILDER(multipleControlledRy)},
        QCOTestCase{"NestedControlledRY", MQT_NAMED_BUILDER(nestedControlledRy),
                    MQT_NAMED_BUILDER(multipleControlledRy)},
        QCOTestCase{"TrivialControlledRY",
                    MQT_NAMED_BUILDER(trivialControlledRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"InverseRY", MQT_NAMED_BUILDER(inverseRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"InverseMultipleControlledRY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRy),
                    MQT_NAMED_BUILDER(multipleControlledRy)},
        QCOTestCase{"TwoRYOppositePhase", MQT_NAMED_BUILDER(twoRyOppositePhase),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RYY", MQT_NAMED_BUILDER(ryy), MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"SingleControlledRYY",
                    MQT_NAMED_BUILDER(singleControlledRyy),
                    MQT_NAMED_BUILDER(singleControlledRyy)},
        QCOTestCase{"MultipleControlledRYY",
                    MQT_NAMED_BUILDER(multipleControlledRyy),
                    MQT_NAMED_BUILDER(multipleControlledRyy)},
        QCOTestCase{"NestedControlledRYY",
                    MQT_NAMED_BUILDER(nestedControlledRyy),
                    MQT_NAMED_BUILDER(multipleControlledRyy)},
        QCOTestCase{"TrivialControlledRYY",
                    MQT_NAMED_BUILDER(trivialControlledRyy),
                    MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"InverseRYY", MQT_NAMED_BUILDER(inverseRyy),
                    MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"InverseMultipleControlledRYY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRyy),
                    MQT_NAMED_BUILDER(multipleControlledRyy)},
        QCOTestCase{"TwoRYY", MQT_NAMED_BUILDER(twoRyy),
                    MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"TwoRYYSwappedTargets",
                    MQT_NAMED_BUILDER(twoRyySwappedTargets),
                    MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"TwoRYYOppositePhaseSwappedTargets",
                    MQT_NAMED_BUILDER(twoRyyOppositePhaseSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoRYYOppositePhase",
                    MQT_NAMED_BUILDER(twoRyyOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RZ", MQT_NAMED_BUILDER(rz), MQT_NAMED_BUILDER(rz)},
        QCOTestCase{"SingleControlledRZ", MQT_NAMED_BUILDER(singleControlledRz),
                    MQT_NAMED_BUILDER(singleControlledRz)},
        QCOTestCase{"MultipleControlledRZ",
                    MQT_NAMED_BUILDER(multipleControlledRz),
                    MQT_NAMED_BUILDER(multipleControlledRz)},
        QCOTestCase{"NestedControlledRZ", MQT_NAMED_BUILDER(nestedControlledRz),
                    MQT_NAMED_BUILDER(multipleControlledRz)},
        QCOTestCase{"TrivialControlledRZ",
                    MQT_NAMED_BUILDER(trivialControlledRz),
                    MQT_NAMED_BUILDER(rz)},
        QCOTestCase{"InverseRZ", MQT_NAMED_BUILDER(inverseRz),
                    MQT_NAMED_BUILDER(rz)},
        QCOTestCase{"InverseMultipleControlledRZ",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRz),
                    MQT_NAMED_BUILDER(multipleControlledRz)},
        QCOTestCase{"TwoRZOppositePhase", MQT_NAMED_BUILDER(twoRzOppositePhase),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZXOpTest, QCOTest,
    testing::Values(QCOTestCase{"RZX", MQT_NAMED_BUILDER(rzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QCOTestCase{"SingleControlledRZX",
                                MQT_NAMED_BUILDER(singleControlledRzx),
                                MQT_NAMED_BUILDER(singleControlledRzx)},
                    QCOTestCase{"MultipleControlledRZX",
                                MQT_NAMED_BUILDER(multipleControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCOTestCase{"NestedControlledRZX",
                                MQT_NAMED_BUILDER(nestedControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCOTestCase{"TrivialControlledRZX",
                                MQT_NAMED_BUILDER(trivialControlledRzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QCOTestCase{"InverseRZX", MQT_NAMED_BUILDER(inverseRzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QCOTestCase{"InverseMultipleControlledRZX",
                                MQT_NAMED_BUILDER(inverseMultipleControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCOTestCase{"TwoRZXOppositePhase",
                                MQT_NAMED_BUILDER(twoRzxOppositePhase),
                                MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RZZ", MQT_NAMED_BUILDER(rzz), MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"SingleControlledRZZ",
                    MQT_NAMED_BUILDER(singleControlledRzz),
                    MQT_NAMED_BUILDER(singleControlledRzz)},
        QCOTestCase{"MultipleControlledRZZ",
                    MQT_NAMED_BUILDER(multipleControlledRzz),
                    MQT_NAMED_BUILDER(multipleControlledRzz)},
        QCOTestCase{"NestedControlledRZZ",
                    MQT_NAMED_BUILDER(nestedControlledRzz),
                    MQT_NAMED_BUILDER(multipleControlledRzz)},
        QCOTestCase{"TrivialControlledRZZ",
                    MQT_NAMED_BUILDER(trivialControlledRzz),
                    MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"InverseRZZ", MQT_NAMED_BUILDER(inverseRzz),
                    MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"InverseMultipleControlledRZZ",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRzz),
                    MQT_NAMED_BUILDER(multipleControlledRzz)},
        QCOTestCase{"TwoRZZ", MQT_NAMED_BUILDER(twoRzz),
                    MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"TwoRZZSwappedTargets",
                    MQT_NAMED_BUILDER(twoRzzSwappedTargets),
                    MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"TwoRZZOppositePhaseSwappedTargets",
                    MQT_NAMED_BUILDER(twoRzzOppositePhaseSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoRZZOppositePhase",
                    MQT_NAMED_BUILDER(twoRzzOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"S", MQT_NAMED_BUILDER(s), MQT_NAMED_BUILDER(s)},
        QCOTestCase{"SingleControlledS", MQT_NAMED_BUILDER(singleControlledS),
                    MQT_NAMED_BUILDER(singleControlledS)},
        QCOTestCase{"MultipleControlledS",
                    MQT_NAMED_BUILDER(multipleControlledS),
                    MQT_NAMED_BUILDER(multipleControlledS)},
        QCOTestCase{"NestedControlledS", MQT_NAMED_BUILDER(nestedControlledS),
                    MQT_NAMED_BUILDER(multipleControlledS)},
        QCOTestCase{"TrivialControlledS", MQT_NAMED_BUILDER(trivialControlledS),
                    MQT_NAMED_BUILDER(s)},
        QCOTestCase{"InverseS", MQT_NAMED_BUILDER(inverseS),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"InverseMultipleControlledS",
                    MQT_NAMED_BUILDER(inverseMultipleControlledS),
                    MQT_NAMED_BUILDER(multipleControlledSdg)},
        QCOTestCase{"SThenSdg", MQT_NAMED_BUILDER(sThenSdg),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoS", MQT_NAMED_BUILDER(twoS), MQT_NAMED_BUILDER(z)},
        QCOTestCase{"PowTwoS", MQT_NAMED_BUILDER(powTwoS),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"PowFourSErase", MQT_NAMED_BUILDER(powFourS),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowHalfSToT", MQT_NAMED_BUILDER(powHalfS),
                    MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"PowThirdSToP", MQT_NAMED_BUILDER(powThirdS),
                    MQT_NAMED_BUILDER(powThirdSRef)}));
/// @}

/// \name QCO/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSdgOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Sdg", MQT_NAMED_BUILDER(sdg), MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"SingleControlledSdg",
                    MQT_NAMED_BUILDER(singleControlledSdg),
                    MQT_NAMED_BUILDER(singleControlledSdg)},
        QCOTestCase{"MultipleControlledSdg",
                    MQT_NAMED_BUILDER(multipleControlledSdg),
                    MQT_NAMED_BUILDER(multipleControlledSdg)},
        QCOTestCase{"NestedControlledSdg",
                    MQT_NAMED_BUILDER(nestedControlledSdg),
                    MQT_NAMED_BUILDER(multipleControlledSdg)},
        QCOTestCase{"TrivialControlledSdg",
                    MQT_NAMED_BUILDER(trivialControlledSdg),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"InverseSdg", MQT_NAMED_BUILDER(inverseSdg),
                    MQT_NAMED_BUILDER(s)},
        QCOTestCase{"InverseMultipleControlledSdg",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSdg),
                    MQT_NAMED_BUILDER(multipleControlledS)},
        QCOTestCase{"SdgThenS", MQT_NAMED_BUILDER(sdgThenS),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoSdg", MQT_NAMED_BUILDER(twoSdg), MQT_NAMED_BUILDER(z)},
        QCOTestCase{"PowTwoSdg", MQT_NAMED_BUILDER(powTwoSdg),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"PowHalfSdgToTdg", MQT_NAMED_BUILDER(powHalfSdg),
                    MQT_NAMED_BUILDER(tdg)},
        QCOTestCase{"PowThirdSdgToP", MQT_NAMED_BUILDER(powThirdSdg),
                    MQT_NAMED_BUILDER(powThirdSdgRef)}));
/// @}

/// \name QCO/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSWAPOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SWAP", MQT_NAMED_BUILDER(swap), MQT_NAMED_BUILDER(swap)},
        QCOTestCase{"SingleControlledSWAP",
                    MQT_NAMED_BUILDER(singleControlledSwap),
                    MQT_NAMED_BUILDER(singleControlledSwap)},
        QCOTestCase{"MultipleControlledSWAP",
                    MQT_NAMED_BUILDER(multipleControlledSwap),
                    MQT_NAMED_BUILDER(multipleControlledSwap)},
        QCOTestCase{"NestedControlledSWAP",
                    MQT_NAMED_BUILDER(nestedControlledSwap),
                    MQT_NAMED_BUILDER(multipleControlledSwap)},
        QCOTestCase{"TrivialControlledSWAP",
                    MQT_NAMED_BUILDER(trivialControlledSwap),
                    MQT_NAMED_BUILDER(swap)},
        QCOTestCase{"InverseSWAP", MQT_NAMED_BUILDER(inverseSwap),
                    MQT_NAMED_BUILDER(swap)},
        QCOTestCase{"InverseMultipleControlledSWAP",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSwap),
                    MQT_NAMED_BUILDER(multipleControlledSwap)},
        QCOTestCase{"TwoSWAP", MQT_NAMED_BUILDER(twoSwap),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoSWAPSwappedTargets",
                    MQT_NAMED_BUILDER(twoSwapSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"PowEvenSWAP", MQT_NAMED_BUILDER(powEvenSwap),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"PowOddSWAP", MQT_NAMED_BUILDER(powOddSwap),
                    MQT_NAMED_BUILDER(swap)}));
/// @}

/// \name QCO/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SX", MQT_NAMED_BUILDER(sx), MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"SingleControlledSX", MQT_NAMED_BUILDER(singleControlledSx),
                    MQT_NAMED_BUILDER(singleControlledSx)},
        QCOTestCase{"MultipleControlledSX",
                    MQT_NAMED_BUILDER(multipleControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"NestedControlledSX", MQT_NAMED_BUILDER(nestedControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"TrivialControlledSX",
                    MQT_NAMED_BUILDER(trivialControlledSx),
                    MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"InverseSX", MQT_NAMED_BUILDER(inverseSx),
                    MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"InverseMultipleControlledSX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)},
        QCOTestCase{"SXThenSXdg", MQT_NAMED_BUILDER(sxThenSxdg),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoSX", MQT_NAMED_BUILDER(twoSx), MQT_NAMED_BUILDER(x)},
        QCOTestCase{"PowTwoSX", MQT_NAMED_BUILDER(powTwoSx),
                    MQT_NAMED_BUILDER(powTwoSxRef)},
        QCOTestCase{"PowThirdSxGeneral", MQT_NAMED_BUILDER(powThirdSx),
                    MQT_NAMED_BUILDER(powThirdSxRef)}));
/// @}

/// \name QCO/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXdgOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SXdg", MQT_NAMED_BUILDER(sxdg), MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"SingleControlledSXdg",
                    MQT_NAMED_BUILDER(singleControlledSxdg),
                    MQT_NAMED_BUILDER(singleControlledSxdg)},
        QCOTestCase{"MultipleControlledSXdg",
                    MQT_NAMED_BUILDER(multipleControlledSxdg),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)},
        QCOTestCase{"NestedControlledSXdg",
                    MQT_NAMED_BUILDER(nestedControlledSxdg),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)},
        QCOTestCase{"TrivialControlledSXdg",
                    MQT_NAMED_BUILDER(trivialControlledSxdg),
                    MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"InverseSXdg", MQT_NAMED_BUILDER(inverseSxdg),
                    MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"InverseMultipleControlledSXdg",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSxdg),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"SXdgThenSX", MQT_NAMED_BUILDER(sxdgThenSx),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoSXdg", MQT_NAMED_BUILDER(twoSxdg),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"PowTwoSXdg", MQT_NAMED_BUILDER(powTwoSxdg),
                    MQT_NAMED_BUILDER(powTwoSxdgRef)},
        QCOTestCase{"PowThirdSxdgGeneral", MQT_NAMED_BUILDER(powThirdSxdg),
                    MQT_NAMED_BUILDER(powThirdSxdgRef)}));
/// @}

/// \name QCO/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"T", MQT_NAMED_BUILDER(t_), MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"SingleControlledT", MQT_NAMED_BUILDER(singleControlledT),
                    MQT_NAMED_BUILDER(singleControlledT)},
        QCOTestCase{"MultipleControlledT",
                    MQT_NAMED_BUILDER(multipleControlledT),
                    MQT_NAMED_BUILDER(multipleControlledT)},
        QCOTestCase{"NestedControlledT", MQT_NAMED_BUILDER(nestedControlledT),
                    MQT_NAMED_BUILDER(multipleControlledT)},
        QCOTestCase{"TrivialControlledT", MQT_NAMED_BUILDER(trivialControlledT),
                    MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"InverseT", MQT_NAMED_BUILDER(inverseT),
                    MQT_NAMED_BUILDER(tdg)},
        QCOTestCase{"InverseMultipleControlledT",
                    MQT_NAMED_BUILDER(inverseMultipleControlledT),
                    MQT_NAMED_BUILDER(multipleControlledTdg)},
        QCOTestCase{"TThenTdg", MQT_NAMED_BUILDER(tThenTdg),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoT", MQT_NAMED_BUILDER(twoT), MQT_NAMED_BUILDER(s)},
        QCOTestCase{"PowTwoT", MQT_NAMED_BUILDER(powTwoT),
                    MQT_NAMED_BUILDER(s)},
        QCOTestCase{"PowThirdTToP", MQT_NAMED_BUILDER(powThirdT),
                    MQT_NAMED_BUILDER(powThirdTRef)}));
/// @}

/// \name QCO/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTdgOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Tdg", MQT_NAMED_BUILDER(tdg), MQT_NAMED_BUILDER(tdg)},
        QCOTestCase{"SingleControlledTdg",
                    MQT_NAMED_BUILDER(singleControlledTdg),
                    MQT_NAMED_BUILDER(singleControlledTdg)},
        QCOTestCase{"MultipleControlledTdg",
                    MQT_NAMED_BUILDER(multipleControlledTdg),
                    MQT_NAMED_BUILDER(multipleControlledTdg)},
        QCOTestCase{"NestedControlledTdg",
                    MQT_NAMED_BUILDER(nestedControlledTdg),
                    MQT_NAMED_BUILDER(multipleControlledTdg)},
        QCOTestCase{"TrivialControlledTdg",
                    MQT_NAMED_BUILDER(trivialControlledTdg),
                    MQT_NAMED_BUILDER(tdg)},
        QCOTestCase{"InverseTdg", MQT_NAMED_BUILDER(inverseTdg),
                    MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"InverseMultipleControlledTdg",
                    MQT_NAMED_BUILDER(inverseMultipleControlledTdg),
                    MQT_NAMED_BUILDER(multipleControlledT)},
        QCOTestCase{"TdgThenT", MQT_NAMED_BUILDER(tdgThenT),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoTdg", MQT_NAMED_BUILDER(twoTdg),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"PowTwoTdg", MQT_NAMED_BUILDER(powTwoTdg),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"PowThirdTdgToP", MQT_NAMED_BUILDER(powThirdTdg),
                    MQT_NAMED_BUILDER(powThirdTdgRef)}));
/// @}

/// \name QCO/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOU2OpTest, QCOTest,
    testing::Values(
        QCOTestCase{"U2", MQT_NAMED_BUILDER(u2), MQT_NAMED_BUILDER(u2)},
        QCOTestCase{"SingleControlledU2", MQT_NAMED_BUILDER(singleControlledU2),
                    MQT_NAMED_BUILDER(singleControlledU2)},
        QCOTestCase{"MultipleControlledU2",
                    MQT_NAMED_BUILDER(multipleControlledU2),
                    MQT_NAMED_BUILDER(multipleControlledU2)},
        QCOTestCase{"NestedControlledU2", MQT_NAMED_BUILDER(nestedControlledU2),
                    MQT_NAMED_BUILDER(multipleControlledU2)},
        QCOTestCase{"TrivialControlledU2",
                    MQT_NAMED_BUILDER(trivialControlledU2),
                    MQT_NAMED_BUILDER(u2)},
        QCOTestCase{"InverseU2", MQT_NAMED_BUILDER(inverseU2),
                    MQT_NAMED_BUILDER(u2)},
        QCOTestCase{"InverseMultipleControlledU2",
                    MQT_NAMED_BUILDER(inverseMultipleControlledU2),
                    MQT_NAMED_BUILDER(multipleControlledU2)},
        QCOTestCase{"CanonicalizeU2ToH", MQT_NAMED_BUILDER(canonicalizeU2ToH),
                    MQT_NAMED_BUILDER(h)},
        QCOTestCase{"CanonicalizeU2ToRx", MQT_NAMED_BUILDER(canonicalizeU2ToRx),
                    MQT_NAMED_BUILDER(rxPiOver2)},
        QCOTestCase{"CanonicalizeU2ToRy", MQT_NAMED_BUILDER(canonicalizeU2ToRy),
                    MQT_NAMED_BUILDER(ryPiOver2)}));
/// @}

/// \name QCO/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOUOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"U", MQT_NAMED_BUILDER(u), MQT_NAMED_BUILDER(u)},
        QCOTestCase{"SingleControlledU", MQT_NAMED_BUILDER(singleControlledU),
                    MQT_NAMED_BUILDER(singleControlledU)},
        QCOTestCase{"MultipleControlledU",
                    MQT_NAMED_BUILDER(multipleControlledU),
                    MQT_NAMED_BUILDER(multipleControlledU)},
        QCOTestCase{"NestedControlledU", MQT_NAMED_BUILDER(nestedControlledU),
                    MQT_NAMED_BUILDER(multipleControlledU)},
        QCOTestCase{"TrivialControlledU", MQT_NAMED_BUILDER(trivialControlledU),
                    MQT_NAMED_BUILDER(u)},
        QCOTestCase{"InverseU", MQT_NAMED_BUILDER(inverseU),
                    MQT_NAMED_BUILDER(u)},
        QCOTestCase{"InverseMultipleControlledU",
                    MQT_NAMED_BUILDER(inverseMultipleControlledU),
                    MQT_NAMED_BUILDER(multipleControlledU)},
        QCOTestCase{"CanonicalizeUToP", MQT_NAMED_BUILDER(canonicalizeUToP),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"CanonicalizeUToRx", MQT_NAMED_BUILDER(canonicalizeUToRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"CanonicalizeUToRy", MQT_NAMED_BUILDER(canonicalizeUToRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"CanonicalizeUToU2", MQT_NAMED_BUILDER(canonicalizeUToU2),
                    MQT_NAMED_BUILDER(u2)}));
/// @}

/// \name QCO/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"X", MQT_NAMED_BUILDER(x), MQT_NAMED_BUILDER(x)},
        QCOTestCase{"SingleControlledX", MQT_NAMED_BUILDER(singleControlledX),
                    MQT_NAMED_BUILDER(singleControlledX)},
        QCOTestCase{"MultipleControlledX",
                    MQT_NAMED_BUILDER(multipleControlledX),
                    MQT_NAMED_BUILDER(multipleControlledX)},
        QCOTestCase{"NestedControlledX", MQT_NAMED_BUILDER(nestedControlledX),
                    MQT_NAMED_BUILDER(multipleControlledX)},
        QCOTestCase{"TrivialControlledX", MQT_NAMED_BUILDER(trivialControlledX),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"InverseX", MQT_NAMED_BUILDER(inverseX),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"InverseMultipleControlledX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledX),
                    MQT_NAMED_BUILDER(multipleControlledX)},
        QCOTestCase{"TwoX", MQT_NAMED_BUILDER(twoX),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"ControlledTwoX", MQT_NAMED_BUILDER(controlledTwoX),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"InverseTwoX", MQT_NAMED_BUILDER(inverseTwoX),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowHalfX", MQT_NAMED_BUILDER(powHalfX),
                    MQT_NAMED_BUILDER(powHalfXRef)},
        QCOTestCase{"PowNegHalfXToSXdg", MQT_NAMED_BUILDER(powNegHalfX),
                    MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"PowThirdXGeneral", MQT_NAMED_BUILDER(powThirdX),
                    MQT_NAMED_BUILDER(powThirdXRef)}));
/// @}

/// \name QCO/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXMinusYYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"XXMinusYY", MQT_NAMED_BUILDER(xxMinusYY),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"SingleControlledXXMinusYY",
                    MQT_NAMED_BUILDER(singleControlledXxMinusYY),
                    MQT_NAMED_BUILDER(singleControlledXxMinusYY)},
        QCOTestCase{"MultipleControlledXXMinusYY",
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCOTestCase{"NestedControlledXXMinusYY",
                    MQT_NAMED_BUILDER(nestedControlledXxMinusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCOTestCase{"TrivialControlledXXMinusYY",
                    MQT_NAMED_BUILDER(trivialControlledXxMinusYY),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"InverseXXMinusYY", MQT_NAMED_BUILDER(inverseXxMinusYY),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"InverseMultipleControlledXXMinusYY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledXxMinusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCOTestCase{"TwoXXMinusYYOppositePhase",
                    MQT_NAMED_BUILDER(twoXxMinusYYOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoXXMinusYYSwappedTargets",
                    MQT_NAMED_BUILDER(twoXxMinusYYSwappedTargets),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"PowXxMinusYYScaled", MQT_NAMED_BUILDER(powXxMinusYYScaled),
                    MQT_NAMED_BUILDER(powXxMinusYYScaledRef)}));
/// @}

/// \name QCO/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXPlusYYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"XXPlusYY", MQT_NAMED_BUILDER(xxPlusYY),
                    MQT_NAMED_BUILDER(xxPlusYY)},
        QCOTestCase{"SingleControlledXXPlusYY",
                    MQT_NAMED_BUILDER(singleControlledXxPlusYY),
                    MQT_NAMED_BUILDER(singleControlledXxPlusYY)},
        QCOTestCase{"MultipleControlledXXPlusYY",
                    MQT_NAMED_BUILDER(multipleControlledXxPlusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCOTestCase{"NestedControlledXXPlusYY",
                    MQT_NAMED_BUILDER(nestedControlledXxPlusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCOTestCase{"TrivialControlledXXPlusYY",
                    MQT_NAMED_BUILDER(trivialControlledXxPlusYY),
                    MQT_NAMED_BUILDER(xxPlusYY)},
        QCOTestCase{"InverseXXPlusYY", MQT_NAMED_BUILDER(inverseXxPlusYY),
                    MQT_NAMED_BUILDER(xxPlusYY)},
        QCOTestCase{"InverseMultipleControlledXXPlusYY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledXxPlusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCOTestCase{"TwoXXPlusYYOppositePhase",
                    MQT_NAMED_BUILDER(twoXxPlusYYOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoXXPlusYYSwappedTargets",
                    MQT_NAMED_BUILDER(twoXxPlusYYSwappedTargets),
                    MQT_NAMED_BUILDER(xxPlusYY)},
        QCOTestCase{"PowXxPlusYYScaled", MQT_NAMED_BUILDER(powXxPlusYYScaled),
                    MQT_NAMED_BUILDER(powXxPlusYYScaledRef)}));
/// @}

/// \name QCO/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Y", MQT_NAMED_BUILDER(y), MQT_NAMED_BUILDER(y)},
        QCOTestCase{"SingleControlledY", MQT_NAMED_BUILDER(singleControlledY),
                    MQT_NAMED_BUILDER(singleControlledY)},
        QCOTestCase{"MultipleControlledY",
                    MQT_NAMED_BUILDER(multipleControlledY),
                    MQT_NAMED_BUILDER(multipleControlledY)},
        QCOTestCase{"NestedControlledY", MQT_NAMED_BUILDER(nestedControlledY),
                    MQT_NAMED_BUILDER(multipleControlledY)},
        QCOTestCase{"TrivialControlledY", MQT_NAMED_BUILDER(trivialControlledY),
                    MQT_NAMED_BUILDER(y)},
        QCOTestCase{"InverseY", MQT_NAMED_BUILDER(inverseY),
                    MQT_NAMED_BUILDER(y)},
        QCOTestCase{"InverseMultipleControlledY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledY),
                    MQT_NAMED_BUILDER(multipleControlledY)},
        QCOTestCase{"TwoY", MQT_NAMED_BUILDER(twoY),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowHalfY", MQT_NAMED_BUILDER(powHalfY),
                    MQT_NAMED_BUILDER(powHalfYRef)}));
/// @}

/// \name QCO/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Z", MQT_NAMED_BUILDER(z), MQT_NAMED_BUILDER(z)},
        QCOTestCase{"SingleControlledZ", MQT_NAMED_BUILDER(singleControlledZ),
                    MQT_NAMED_BUILDER(singleControlledZ)},
        QCOTestCase{"MultipleControlledZ",
                    MQT_NAMED_BUILDER(multipleControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"NestedControlledZ", MQT_NAMED_BUILDER(nestedControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"TrivialControlledZ", MQT_NAMED_BUILDER(trivialControlledZ),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"InverseZ", MQT_NAMED_BUILDER(inverseZ),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"InverseMultipleControlledZ",
                    MQT_NAMED_BUILDER(inverseMultipleControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"TwoZ", MQT_NAMED_BUILDER(twoZ),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowHalfZ", MQT_NAMED_BUILDER(powHalfZ),
                    MQT_NAMED_BUILDER(s)},
        QCOTestCase{"NormalizeAngleWrapZ", MQT_NAMED_BUILDER(powThreeHalvesZ),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"PowThirdZToP", MQT_NAMED_BUILDER(powThirdZ),
                    MQT_NAMED_BUILDER(powThirdZRef)}));
/// @}

/// \name QCO/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOMeasureOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SingleMeasurementToSingleBit",
                    MQT_NAMED_BUILDER(singleMeasurementToSingleBit),
                    MQT_NAMED_BUILDER(singleMeasurementToSingleBit)},
        QCOTestCase{"RepeatedMeasurementToSameBit",
                    MQT_NAMED_BUILDER(repeatedMeasurementToSameBit),
                    MQT_NAMED_BUILDER(repeatedMeasurementToSameBit)},
        QCOTestCase{"RepeatedMeasurementToDifferentBits",
                    MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits),
                    MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits)},
        QCOTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(multipleClassicalRegistersAndMeasurements)}));
/// @}

/// \name QCO/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, QCOTest,
    testing::Values(QCOTestCase{"ResetQubitWithoutOp",
                                MQT_NAMED_BUILDER(resetQubitWithoutOp),
                                MQT_NAMED_BUILDER(allocQubit)},
                    QCOTestCase{"ResetMultipleQubitsWithoutOp",
                                MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp),
                                MQT_NAMED_BUILDER(alloc2QubitRegister)},
                    QCOTestCase{"RepeatedResetWithoutOp",
                                MQT_NAMED_BUILDER(repeatedResetWithoutOp),
                                MQT_NAMED_BUILDER(allocQubit)},
                    QCOTestCase{"ResetQubitAfterSingleOp",
                                MQT_NAMED_BUILDER(resetQubitAfterSingleOp),
                                MQT_NAMED_BUILDER(resetQubitAfterSingleOp)},
                    QCOTestCase{
                        "ResetMultipleQubitsAfterSingleOp",
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp),
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp)},
                    QCOTestCase{"RepeatedResetAfterSingleOp",
                                MQT_NAMED_BUILDER(repeatedResetAfterSingleOp),
                                MQT_NAMED_BUILDER(resetQubitAfterSingleOp)}));
/// @}

/// \name QCO/QubitManagement/QubitManagement.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOQubitManagementTest, QCOTest,
    testing::Values(
        QCOTestCase{"AllocQubit", MQT_NAMED_BUILDER(allocQubitNoMeasure),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"StaticQubitsNoMeasure",
                    MQT_NAMED_BUILDER(staticQubitsNoMeasure),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"StaticQubitsWithOps",
                    MQT_NAMED_BUILDER(staticQubitsWithOps),
                    MQT_NAMED_BUILDER(staticQubitsWithOps)},
        QCOTestCase{"StaticQubitsWithParametricOps",
                    MQT_NAMED_BUILDER(staticQubitsWithParametricOps),
                    MQT_NAMED_BUILDER(staticQubitsWithParametricOps)},
        QCOTestCase{"StaticQubitsWithTwoTargetOps",
                    MQT_NAMED_BUILDER(staticQubitsWithTwoTargetOps),
                    MQT_NAMED_BUILDER(staticQubitsWithTwoTargetOps)},
        QCOTestCase{"StaticQubitsWithCtrl",
                    MQT_NAMED_BUILDER(staticQubitsWithCtrl),
                    MQT_NAMED_BUILDER(staticQubitsWithCtrl)},
        QCOTestCase{"StaticQubitsWithInv",
                    MQT_NAMED_BUILDER(staticQubitsWithInv),
                    MQT_NAMED_BUILDER(staticQubitsWithInv)},
        QCOTestCase{"DeadGateElimination", MQT_NAMED_BUILDER(deadGatesProgram),
                    MQT_NAMED_BUILDER(alloc2Qubits)},
        QCOTestCase{"DeadGateEliminationIfOp",
                    MQT_NAMED_BUILDER(deadGatesWithIfOpProgram),
                    MQT_NAMED_BUILDER(deadGatesWithIfOpSimplified)},
        QCOTestCase{"AllocSinkPair", MQT_NAMED_BUILDER(allocSinkPair),
                    MQT_NAMED_BUILDER(allocQubitNoMeasure)}));
/// @}
