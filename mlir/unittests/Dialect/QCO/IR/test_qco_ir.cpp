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
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <iosfwd>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <ostream>
#include <string>

using namespace mlir;
using namespace mlir::qco;

struct QCOTestCase {
  std::string name;
  mqt::test::NamedBuilder<QCOProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<QCOProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os, const QCOTestCase& info);
};

class QCOTest : public testing::TestWithParam<QCOTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

std::ostream& operator<<(std::ostream& os, const QCOTestCase& info) {
  return os << "QCO{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

TEST_P(QCOTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  mqt::test::DeferredPrinter printer;

  auto program = QCOProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());
  runCanonicalizationPasses(program.get());
  printer.record(program.get(), "Canonicalized QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = QCOProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  runCanonicalizationPasses(reference.get());
  printer.record(reference.get(), "Canonicalized Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(QCOTest, DirectIfBuilder) {
  // Test If construction directly
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto q0 = AllocOp::create(builder);
  auto q1 = HOp::create(builder, q0);
  auto measureOp = MeasureOp::create(builder, q1);
  auto ifOp =
      IfOp::create(builder, measureOp.getResult(), measureOp.getQubitOut(),
                   [&](ValueRange qubits) -> llvm::SmallVector<Value> {
                     auto innerQubit = XOp::create(builder, qubits[0]);
                     return llvm::SmallVector<mlir::Value>{innerQubit};
                   });
  DeallocOp::create(builder, ifOp.getResult(0));

  auto directBuilder = builder.finalize();
  ASSERT_TRUE(directBuilder);
  EXPECT_TRUE(verify(*directBuilder).succeeded());
  runCanonicalizationPasses(directBuilder.get());
  EXPECT_TRUE(verify(*directBuilder).succeeded());

  auto refBuilder =
      QCOProgramBuilder::build(context.get(), MQT_NAMED_BUILDER(simpleIf).fn);
  ASSERT_TRUE(refBuilder);
  EXPECT_TRUE(verify(*refBuilder).succeeded());
  runCanonicalizationPasses(refBuilder.get());
  EXPECT_TRUE(verify(*refBuilder).succeeded());

  EXPECT_TRUE(areModulesEquivalentWithPermutations(directBuilder.get(),
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
    testing::Values(QCOTestCase{"TrivialCtrl", MQT_NAMED_BUILDER(trivialCtrl),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"NestedCtrl", MQT_NAMED_BUILDER(nestedCtrl),
                                MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCOTestCase{"TripleNestedCtrl",
                                MQT_NAMED_BUILDER(tripleNestedCtrl),
                                MQT_NAMED_BUILDER(tripleControlledRxx)},
                    QCOTestCase{"CtrlInvSandwich",
                                MQT_NAMED_BUILDER(ctrlInvSandwich),
                                MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCOTestCase{"DoubleNestedCtrlTwoQubits",
                                MQT_NAMED_BUILDER(doubleNestedCtrlTwoQubits),
                                MQT_NAMED_BUILDER(fourControlledRxx)}));
/// @}

/// \name QCO/Modifiers/InvOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOInvOpTest, QCOTest,
    testing::Values(QCOTestCase{"NestedInv", MQT_NAMED_BUILDER(nestedInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"TripleNestedInv",
                                MQT_NAMED_BUILDER(tripleNestedInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"InvControlSandwich",
                                MQT_NAMED_BUILDER(invCtrlSandwich),
                                MQT_NAMED_BUILDER(singleControlledRxx)}));
/// @}

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
                                MQT_NAMED_BUILDER(barrierTwoQubits)}));
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
                    MQT_NAMED_BUILDER(inverseMultipleControlledDcx)}));
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
                                MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(multipleControlledGlobalPhase)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
/// @}

/// \name QCO/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOIDOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Identity", MQT_NAMED_BUILDER(identity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"SingleControlledIdentity",
                    MQT_NAMED_BUILDER(singleControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"MultipleControlledIdentity",
                    MQT_NAMED_BUILDER(multipleControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"NestedControlledIdentity",
                    MQT_NAMED_BUILDER(nestedControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"TrivialControlledIdentity",
                    MQT_NAMED_BUILDER(trivialControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"InverseIdentity", MQT_NAMED_BUILDER(inverseIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"InverseMultipleControlledIdentity",
                    MQT_NAMED_BUILDER(inverseMultipleControlledIdentity),
                    MQT_NAMED_BUILDER(emptyQCO)}));
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
                        MQT_NAMED_BUILDER(inverseMultipleControlledIswap)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(ry)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
/// @}

/// \name QCO/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXXOpTest, QCOTest,
    testing::Values(QCOTestCase{"RXX", MQT_NAMED_BUILDER(rxx),
                                MQT_NAMED_BUILDER(rxx)},
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
                    QCOTestCase{"TwoRXXOppositePhase",
                                MQT_NAMED_BUILDER(twoRxxOppositePhase),
                                MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
/// @}

/// \name QCO/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYYOpTest, QCOTest,
    testing::Values(QCOTestCase{"RYY", MQT_NAMED_BUILDER(ryy),
                                MQT_NAMED_BUILDER(ryy)},
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
                    QCOTestCase{"TwoRYYOppositePhase",
                                MQT_NAMED_BUILDER(twoRyyOppositePhase),
                                MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
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
                                MQT_NAMED_BUILDER(emptyQCO)}));
/// @}

/// \name QCO/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZZOpTest, QCOTest,
    testing::Values(QCOTestCase{"RZZ", MQT_NAMED_BUILDER(rzz),
                                MQT_NAMED_BUILDER(rzz)},
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
                    QCOTestCase{"TwoRZZOppositePhase",
                                MQT_NAMED_BUILDER(twoRzzOppositePhase),
                                MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"TwoS", MQT_NAMED_BUILDER(twoS), MQT_NAMED_BUILDER(z)}));
/// @}

/// \name QCO/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSdgOpTest, QCOTest,
    testing::Values(QCOTestCase{"Sdg", MQT_NAMED_BUILDER(sdg),
                                MQT_NAMED_BUILDER(sdg)},
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
                                MQT_NAMED_BUILDER(emptyQCO)},
                    QCOTestCase{"TwoSdg", MQT_NAMED_BUILDER(twoSdg),
                                MQT_NAMED_BUILDER(z)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"TwoSX", MQT_NAMED_BUILDER(twoSx), MQT_NAMED_BUILDER(x)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"TwoSXdg", MQT_NAMED_BUILDER(twoSxdg),
                    MQT_NAMED_BUILDER(x)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"TwoT", MQT_NAMED_BUILDER(twoT), MQT_NAMED_BUILDER(s)}));
/// @}

/// \name QCO/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTdgOpTest, QCOTest,
    testing::Values(QCOTestCase{"Tdg", MQT_NAMED_BUILDER(tdg),
                                MQT_NAMED_BUILDER(tdg)},
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
                    QCOTestCase{"TdgThenS", MQT_NAMED_BUILDER(tdgThenT),
                                MQT_NAMED_BUILDER(emptyQCO)},
                    QCOTestCase{"TwoTdg", MQT_NAMED_BUILDER(twoTdg),
                                MQT_NAMED_BUILDER(sdg)}));
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
                    MQT_NAMED_BUILDER(ry)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
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
                    MQT_NAMED_BUILDER(emptyQCO)}));
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
                                MQT_NAMED_BUILDER(emptyQCO)},
                    QCOTestCase{"ResetMultipleQubitsWithoutOp",
                                MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp),
                                MQT_NAMED_BUILDER(emptyQCO)},
                    QCOTestCase{"RepeatedResetWithoutOp",
                                MQT_NAMED_BUILDER(repeatedResetWithoutOp),
                                MQT_NAMED_BUILDER(emptyQCO)},
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
        QCOTestCase{"AllocQubit", MQT_NAMED_BUILDER(allocQubit),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"AllocQubitRegister", MQT_NAMED_BUILDER(allocQubitRegister),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"AllocMultipleQubitRegisters",
                    MQT_NAMED_BUILDER(allocMultipleQubitRegisters),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"AllocLargeRegister", MQT_NAMED_BUILDER(allocLargeRegister),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"StaticQubits", MQT_NAMED_BUILDER(staticQubits),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"AllocDeallocPair", MQT_NAMED_BUILDER(allocDeallocPair),
                    MQT_NAMED_BUILDER(emptyQCO)}));
/// @}
