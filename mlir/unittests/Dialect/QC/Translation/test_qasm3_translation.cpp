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
#include "qasm_programs.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <ostream>
#include <string>

using namespace mlir;

namespace {

struct QASM3TranslationTestCase {
  std::string name;
  std::string source;
  mqt::test::NamedMLIRBuilder<qc::QCProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QASM3TranslationTestCase& test);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const QASM3TranslationTestCase& test) {
  return os << "QASM3Translation{" << test.name << ", reference="
            << mqt::test::displayName(test.referenceBuilder.name) << "}";
}

class QASM3TranslationTest
    : public testing::TestWithParam<QASM3TranslationTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                    memref::MemRefDialect, qc::QCDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

static SmallVector<Value> twoX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.x(q[0]);
  b.x(q[1]);
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  return {c0, c1};
}

static SmallVector<Value> singleNegControlledX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.x(q[0]);
  b.cx(q[0], q[1]);
  b.x(q[0]);
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  return {c0, c1};
}

static SmallVector<Value> tripleControlledX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcx({q[0], q[1], q[2]}, q[3]);
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  auto c2 = b.measure(q[2]);
  auto c3 = b.measure(q[3]);
  return {c0, c1, c2, c3};
}

static SmallVector<Value> mixedControlledX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.x(q[1]);
  b.mcx({q[0], q[1]}, q[2]);
  b.x(q[1]);
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  auto c2 = b.measure(q[2]);
  return {c0, c1, c2};
}

static SmallVector<Value> twoMixedControlledX(qc::QCProgramBuilder& b) {
  auto q1 = b.allocQubitRegister(2);
  auto q2 = b.allocQubitRegister(2);
  auto q3 = b.allocQubitRegister(2);
  b.x(q2[0]);
  b.mcx({q1[0], q2[0]}, q3[0]);
  b.x(q2[0]);
  b.x(q2[1]);
  b.mcx({q1[1], q2[1]}, q3[1]);
  b.x(q2[1]);
  auto c0 = b.measure(q1[0]);
  auto c1 = b.measure(q1[1]);
  auto c2 = b.measure(q2[0]);
  auto c3 = b.measure(q2[1]);
  auto c4 = b.measure(q3[0]);
  auto c5 = b.measure(q3[1]);
  return {c0, c1, c2, c3, c4, c5};
}

static Value ifNot(qc::QCProgramBuilder& b) {
  auto trueValue = b.boolConstant(true);
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto c = b.measure(q[0]);
  auto cond = arith::XOrIOp::create(b, c, trueValue).getResult();
  b.scfIf(cond, [&] { b.x(q[0]); });
  auto out = b.measure(q[0]);
  return out;
}

static SmallVector<Value> broadcastRegisterAndQubit(qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto scalar = b.allocQubitRegister(1);
  auto q = scalar[0];
  b.cx(reg[0], q);
  b.cx(reg[1], q);
  b.cx(reg[2], q);
  return {b.measure(reg[0]), b.measure(reg[1]), b.measure(reg[2]),
          b.measure(q)};
}

static SmallVector<Value> broadcastCompoundGate(qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto scalar = b.allocQubitRegister(1);
  auto q = scalar[0];
  for (auto qubit : reg.qubits) {
    b.x(qubit);
    b.cx(qubit, q);
  }
  return {b.measure(reg[0]), b.measure(reg[1]), b.measure(reg[2]),
          b.measure(q)};
}

static Value expressionArithmetic(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.rx((((1.0 + 2.0) * 3.0) / 2.0) - 0.5, q[0]);
  return b.measure(q[0]);
}

static Value expressionUnaryMinus(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.rx(-0.5, q[0]);
  b.ry(-(1.0 + 2.0), q[0]);
  b.rz(-(-0.25), q[0]);
  return b.measure(q[0]);
}

static Value expressionBuiltinConstants(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.rx(std::numbers::pi / 2.0, q[0]);
  b.ry((2.0 * std::numbers::pi) / 4.0, q[0]);
  b.rz(std::numbers::e, q[0]);
  return b.measure(q[0]);
}

static Value expressionMathFunctions(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.rx(std::acos(0.5), q[0]);
  b.rx(std::asin(0.5), q[0]);
  b.rx(std::atan(0.5), q[0]);
  b.rx(std::cos(0.5), q[0]);
  b.rx(std::exp(0.5), q[0]);
  b.rx(std::numbers::ln2, q[0]);
  b.rx(std::fmod(5.5, 2.0), q[0]);
  b.rx(std::pow(2.0, 3.0), q[0]);
  b.rx(std::sin(0.5), q[0]);
  b.rx(std::numbers::sqrt2, q[0]);
  b.rx(std::tan(0.5), q[0]);
  return b.measure(q[0]);
}

static Value expressionNestedMathFunctions(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.rx(std::sqrt(std::pow(std::sin(0.5), 2.0) + std::pow(std::cos(0.5), 2.0)),
       q[0]);
  return b.measure(q[0]);
}

static Value expressionConstFloat(qc::QCProgramBuilder& b) {
  constexpr double theta = std::numbers::pi / 4.0;
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.rx(theta, q[0]);
  b.ry(theta * 2.0, q[0]);
  return b.measure(q[0]);
}

static Value expressionMutableFloat(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.rx(0.5, q[0]);
  b.ry(0.75, q[0]);
  return b.measure(q[0]);
}

static SmallVector<Value>
expressionConstIntArithmetic(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(8);
  b.h(q[3]);
  b.h(q[5]);
  b.rx(8.0, q[3]);
  return {b.measure(q[3]), b.measure(q[5])};
}

static SmallVector<Value> conditionLiteral(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.scfIf(true, [&] { b.x(q[0]); });
  b.scfIf(false, [&] { b.x(q[1]); });
  return {b.measure(q[0]), b.measure(q[1])};
}

static Value conditionMeasurement(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  auto condition = b.measure(q[0]);
  b.scfIf(condition, [&] { b.x(q[1]); });
  return b.measure(q[1]);
}

static SmallVector<Value> conditionAnd(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.h(q[0]);
  b.h(q[1]);
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  auto condition = arith::AndIOp::create(b, c0, c1);
  b.scfIf(condition, [&] { b.x(q[2]); });
  return {c0, c1, b.measure(q[2])};
}

static SmallVector<Value> conditionOr(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.h(q[0]);
  b.h(q[1]);
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  auto condition = arith::OrIOp::create(b, c0, c1);
  b.scfIf(condition, [&] { b.x(q[2]); }, [&] { b.h(q[2]); });
  return {c0, c1, b.measure(q[2])};
}

static SmallVector<Value> conditionNotAndOr(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.h(q[0]);
  b.h(q[1]);
  b.h(q[2]);
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  auto c2 = b.measure(q[2]);
  auto both = arith::AndIOp::create(b, c0, c1);
  auto notBoth = arith::XOrIOp::create(b, both, b.boolConstant(true));
  auto condition = arith::OrIOp::create(b, notBoth, c2);
  b.scfIf(condition, [&] { b.x(q[3]); });
  return {c0, c1, c2, b.measure(q[3])};
}

static SmallVector<Value> conditionBoolVariable(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.h(q[0]);
  b.h(q[1]);
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  auto both = arith::AndIOp::create(b, c0, c1);
  auto neither = arith::XOrIOp::create(b, both, b.boolConstant(true));
  b.scfIf(neither, [&] { b.x(q[2]); });
  return {c0, c1, b.measure(q[2])};
}

static SmallVector<Value> conditionIndexedBit(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.h(q[0]);
  b.h(q[1]);
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  b.scfIf(c1, [&] { b.x(q[2]); });
  return {c0, c1, b.measure(q[2])};
}

TEST_P(QASM3TranslationTest, ProgramEquivalence) {
  const auto name = " (" + GetParam().name + ")";
  const auto& source = GetParam().source;
  const auto referenceBuilder = GetParam().referenceBuilder;
  mqt::test::DeferredPrinter printer;

  auto translated = qc::translateQASM3ToQC(source, context.get());
  ASSERT_TRUE(translated);
  printer.record(translated.get(), "Translated QC IR" + name);
  EXPECT_TRUE(verify(*translated).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(translated.get()).succeeded());
  printer.record(translated.get(), "Canonicalized Translated QC IR" + name);
  EXPECT_TRUE(verify(*translated).succeeded());

  auto reference = mqt::test::buildMLIRProgram(context.get(), referenceBuilder);
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

        QASM3TranslationTestCase{"AllocQubit", qasm::allocQubit,
                                 MQT_NAMED_BUILDER(qc::alloc1QubitRegister)},
        QASM3TranslationTestCase{"AllocQubitRegister", qasm::allocQubitRegister,
                                 MQT_NAMED_BUILDER(qc::allocQubitRegister)},
        QASM3TranslationTestCase{
            "AllocMultipleQubitRegisters", qasm::allocMultipleQubitRegisters,
            MQT_NAMED_BUILDER(qc::allocMultipleQubitRegisters)},
        QASM3TranslationTestCase{"AllocLargeRegister", qasm::allocLargeRegister,
                                 MQT_NAMED_BUILDER(qc::allocLargeRegister)},
        QASM3TranslationTestCase{
            "SingleMeasurementToSingleBit", qasm::singleMeasurementToSingleBit,
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit)},
        QASM3TranslationTestCase{
            "RepeatedMeasurementToSameBit", qasm::repeatedMeasurementToSameBit,
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit)},
        QASM3TranslationTestCase{
            "RepeatedMeasurementToDifferentBits",
            qasm::repeatedMeasurementToDifferentBits,
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits)},
        QASM3TranslationTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            qasm::multipleClassicalRegistersAndMeasurements,
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements)},
        QASM3TranslationTestCase{
            "ResetQubitAfterSingleOp", qasm::resetQubitAfterSingleOp,
            MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)},
        QASM3TranslationTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            qasm::resetMultipleQubitsAfterSingleOp,
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp)},
        QASM3TranslationTestCase{
            "RepeatedResetAfterSingleOp", qasm::repeatedResetAfterSingleOp,
            MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp)},
        QASM3TranslationTestCase{"GlobalPhase", qasm::globalPhase,
                                 MQT_NAMED_BUILDER(qc::globalPhase)},
        QASM3TranslationTestCase{"InverseGlobalPhase", qasm::inverseGlobalPhase,
                                 MQT_NAMED_BUILDER(qc::inverseGlobalPhase)},
        QASM3TranslationTestCase{"Identity", qasm::identity,
                                 MQT_NAMED_BUILDER(qc::identity)},
        QASM3TranslationTestCase{"SingleControlledIdentity",
                                 qasm::singleControlledIdentity,
                                 MQT_NAMED_BUILDER(qc::twoQubitsOneIdentity)},
        QASM3TranslationTestCase{"MultipleControlledIdentity",
                                 qasm::multipleControlledIdentity,
                                 MQT_NAMED_BUILDER(qc::threeQubitsOneIdentity)},
        QASM3TranslationTestCase{"X", qasm::x, MQT_NAMED_BUILDER(qc::x)},
        QASM3TranslationTestCase{"TwoX", qasm::twoX, MQT_NAMED_BUILDER(twoX)},
        QASM3TranslationTestCase{"SingleControlledX", qasm::singleControlledX,
                                 MQT_NAMED_BUILDER(qc::singleControlledX)},
        QASM3TranslationTestCase{"SingleNegControlledX",
                                 qasm::singleNegControlledX,
                                 MQT_NAMED_BUILDER(singleNegControlledX)},
        QASM3TranslationTestCase{"MultipleControlledX",
                                 qasm::multipleControlledX,
                                 MQT_NAMED_BUILDER(qc::multipleControlledX)},
        QASM3TranslationTestCase{"TripleControlledXOpenQASM2",
                                 qasm::tripleControlledXOpenQASM2,
                                 MQT_NAMED_BUILDER(tripleControlledX)},
        QASM3TranslationTestCase{"MixedControlledX", qasm::mixedControlledX,
                                 MQT_NAMED_BUILDER(mixedControlledX)},
        QASM3TranslationTestCase{"TwoMixedControlledX",
                                 qasm::twoMixedControlledX,
                                 MQT_NAMED_BUILDER(twoMixedControlledX)},
        QASM3TranslationTestCase{"InverseX", qasm::inverseX,
                                 MQT_NAMED_BUILDER(qc::inverseX)},
        QASM3TranslationTestCase{
            "InverseMultipleControlledX", qasm::inverseMultipleControlledX,
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledX)},
        QASM3TranslationTestCase{"Y", qasm::y, MQT_NAMED_BUILDER(qc::y)},
        QASM3TranslationTestCase{"SingleControlledY", qasm::singleControlledY,
                                 MQT_NAMED_BUILDER(qc::singleControlledY)},
        QASM3TranslationTestCase{"MultipleControlledY",
                                 qasm::multipleControlledY,
                                 MQT_NAMED_BUILDER(qc::multipleControlledY)},
        QASM3TranslationTestCase{"Z", qasm::z, MQT_NAMED_BUILDER(qc::z)},
        QASM3TranslationTestCase{"SingleControlledZ", qasm::singleControlledZ,
                                 MQT_NAMED_BUILDER(qc::singleControlledZ)},
        QASM3TranslationTestCase{"MultipleControlledZ",
                                 qasm::multipleControlledZ,
                                 MQT_NAMED_BUILDER(qc::multipleControlledZ)},
        QASM3TranslationTestCase{"H", qasm::h, MQT_NAMED_BUILDER(qc::h)},
        QASM3TranslationTestCase{"SingleControlledH", qasm::singleControlledH,
                                 MQT_NAMED_BUILDER(qc::singleControlledH)},
        QASM3TranslationTestCase{"MultipleControlledH",
                                 qasm::multipleControlledH,
                                 MQT_NAMED_BUILDER(qc::multipleControlledH)},
        QASM3TranslationTestCase{"S", qasm::s, MQT_NAMED_BUILDER(qc::s)},
        QASM3TranslationTestCase{"SingleControlledS", qasm::singleControlledS,
                                 MQT_NAMED_BUILDER(qc::singleControlledS)},
        QASM3TranslationTestCase{"MultipleControlledS",
                                 qasm::multipleControlledS,
                                 MQT_NAMED_BUILDER(qc::multipleControlledS)},
        QASM3TranslationTestCase{"Sdg", qasm::sdg, MQT_NAMED_BUILDER(qc::sdg)},
        QASM3TranslationTestCase{"SingleControlledSdg",
                                 qasm::singleControlledSdg,
                                 MQT_NAMED_BUILDER(qc::singleControlledSdg)},
        QASM3TranslationTestCase{"MultipleControlledSdg",
                                 qasm::multipleControlledSdg,
                                 MQT_NAMED_BUILDER(qc::multipleControlledSdg)},
        QASM3TranslationTestCase{"T", qasm::t_, MQT_NAMED_BUILDER(qc::t_)},
        QASM3TranslationTestCase{"SingleControlledT", qasm::singleControlledT,
                                 MQT_NAMED_BUILDER(qc::singleControlledT)},
        QASM3TranslationTestCase{"MultipleControlledT",
                                 qasm::multipleControlledT,
                                 MQT_NAMED_BUILDER(qc::multipleControlledT)},
        QASM3TranslationTestCase{"Tdg", qasm::tdg, MQT_NAMED_BUILDER(qc::tdg)},
        QASM3TranslationTestCase{"SingleControlledTdg",
                                 qasm::singleControlledTdg,
                                 MQT_NAMED_BUILDER(qc::singleControlledTdg)},
        QASM3TranslationTestCase{"MultipleControlledTdg",
                                 qasm::multipleControlledTdg,
                                 MQT_NAMED_BUILDER(qc::multipleControlledTdg)},
        QASM3TranslationTestCase{"SX", qasm::sx, MQT_NAMED_BUILDER(qc::sx)},
        QASM3TranslationTestCase{"SingleControlledSX", qasm::singleControlledSx,
                                 MQT_NAMED_BUILDER(qc::singleControlledSx)},
        QASM3TranslationTestCase{"MultipleControlledSX",
                                 qasm::multipleControlledSx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledSx)},
        QASM3TranslationTestCase{"SXdg", qasm::sxdg,
                                 MQT_NAMED_BUILDER(qc::sxdg)},
        QASM3TranslationTestCase{"SingleControlledSXdg",
                                 qasm::singleControlledSxdg,
                                 MQT_NAMED_BUILDER(qc::singleControlledSxdg)},
        QASM3TranslationTestCase{"MultipleControlledSXdg",
                                 qasm::multipleControlledSxdg,
                                 MQT_NAMED_BUILDER(qc::multipleControlledSxdg)},
        QASM3TranslationTestCase{"RX", qasm::rx, MQT_NAMED_BUILDER(qc::rx)},
        QASM3TranslationTestCase{"SingleControlledRX", qasm::singleControlledRx,
                                 MQT_NAMED_BUILDER(qc::singleControlledRx)},
        QASM3TranslationTestCase{"MultipleControlledRX",
                                 qasm::multipleControlledRx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRx)},
        QASM3TranslationTestCase{"RY", qasm::ry, MQT_NAMED_BUILDER(qc::ry)},
        QASM3TranslationTestCase{"SingleControlledRY", qasm::singleControlledRy,
                                 MQT_NAMED_BUILDER(qc::singleControlledRy)},
        QASM3TranslationTestCase{"MultipleControlledRY",
                                 qasm::multipleControlledRy,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRy)},
        QASM3TranslationTestCase{"RZ", qasm::rz, MQT_NAMED_BUILDER(qc::rz)},
        QASM3TranslationTestCase{"SingleControlledRZ", qasm::singleControlledRz,
                                 MQT_NAMED_BUILDER(qc::singleControlledRz)},
        QASM3TranslationTestCase{"MultipleControlledRZ",
                                 qasm::multipleControlledRz,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRz)},
        QASM3TranslationTestCase{"P", qasm::p, MQT_NAMED_BUILDER(qc::p)},
        QASM3TranslationTestCase{"SingleControlledP", qasm::singleControlledP,
                                 MQT_NAMED_BUILDER(qc::singleControlledP)},
        QASM3TranslationTestCase{"MultipleControlledP",
                                 qasm::multipleControlledP,
                                 MQT_NAMED_BUILDER(qc::multipleControlledP)},
        QASM3TranslationTestCase{"R", qasm::r, MQT_NAMED_BUILDER(qc::r)},
        QASM3TranslationTestCase{"SingleControlledR", qasm::singleControlledR,
                                 MQT_NAMED_BUILDER(qc::singleControlledR)},
        QASM3TranslationTestCase{"MultipleControlledR",
                                 qasm::multipleControlledR,
                                 MQT_NAMED_BUILDER(qc::multipleControlledR)},
        QASM3TranslationTestCase{"U2", qasm::u2, MQT_NAMED_BUILDER(qc::u2)},
        QASM3TranslationTestCase{"SingleControlledU2", qasm::singleControlledU2,
                                 MQT_NAMED_BUILDER(qc::singleControlledU2)},
        QASM3TranslationTestCase{"MultipleControlledU2",
                                 qasm::multipleControlledU2,
                                 MQT_NAMED_BUILDER(qc::multipleControlledU2)},
        QASM3TranslationTestCase{"U", qasm::u, MQT_NAMED_BUILDER(qc::u)},
        QASM3TranslationTestCase{"SingleControlledU", qasm::singleControlledU,
                                 MQT_NAMED_BUILDER(qc::singleControlledU)},
        QASM3TranslationTestCase{"MultipleControlledU",
                                 qasm::multipleControlledU,
                                 MQT_NAMED_BUILDER(qc::multipleControlledU)},
        QASM3TranslationTestCase{"SWAP", qasm::swap,
                                 MQT_NAMED_BUILDER(qc::swap)},
        QASM3TranslationTestCase{"SingleControlledSWAP",
                                 qasm::singleControlledSwap,
                                 MQT_NAMED_BUILDER(qc::singleControlledSwap)},
        QASM3TranslationTestCase{"MultipleControlledSWAP",
                                 qasm::multipleControlledSwap,
                                 MQT_NAMED_BUILDER(qc::multipleControlledSwap)},
        QASM3TranslationTestCase{"iSWAP", qasm::iswap,
                                 MQT_NAMED_BUILDER(qc::iswap)},
        QASM3TranslationTestCase{"SingleControllediSWAP",
                                 qasm::singleControlledIswap,
                                 MQT_NAMED_BUILDER(qc::singleControlledIswap)},
        QASM3TranslationTestCase{
            "MultipleControllediSWAP", qasm::multipleControlledIswap,
            MQT_NAMED_BUILDER(qc::multipleControlledIswap)},
        QASM3TranslationTestCase{"InverseISWAP", qasm::inverseIswap,
                                 MQT_NAMED_BUILDER(qc::inverseIswap)},
        QASM3TranslationTestCase{
            "InverseMultiControlledISWAP", qasm::inverseMultipleControlledIswap,
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap)},
        QASM3TranslationTestCase{"DCX", qasm::dcx, MQT_NAMED_BUILDER(qc::dcx)},
        QASM3TranslationTestCase{"SingleControlledDCX",
                                 qasm::singleControlledDcx,
                                 MQT_NAMED_BUILDER(qc::singleControlledDcx)},
        QASM3TranslationTestCase{"MultipleControlledDCX",
                                 qasm::multipleControlledDcx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledDcx)},
        QASM3TranslationTestCase{"ECR", qasm::ecr, MQT_NAMED_BUILDER(qc::ecr)},
        QASM3TranslationTestCase{"SingleControlledECR",
                                 qasm::singleControlledEcr,
                                 MQT_NAMED_BUILDER(qc::singleControlledEcr)},
        QASM3TranslationTestCase{"MultipleControlledECR",
                                 qasm::multipleControlledEcr,
                                 MQT_NAMED_BUILDER(qc::multipleControlledEcr)},
        QASM3TranslationTestCase{"RXX", qasm::rxx, MQT_NAMED_BUILDER(qc::rxx)},
        QASM3TranslationTestCase{"SingleControlledRXX",
                                 qasm::singleControlledRxx,
                                 MQT_NAMED_BUILDER(qc::singleControlledRxx)},
        QASM3TranslationTestCase{"MultipleControlledRXX",
                                 qasm::multipleControlledRxx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRxx)},
        QASM3TranslationTestCase{"TripleControlledRXX",
                                 qasm::tripleControlledRxx,
                                 MQT_NAMED_BUILDER(qc::tripleControlledRxx)},
        QASM3TranslationTestCase{"RYY", qasm::ryy, MQT_NAMED_BUILDER(qc::ryy)},
        QASM3TranslationTestCase{"SingleControlledRYY",
                                 qasm::singleControlledRyy,
                                 MQT_NAMED_BUILDER(qc::singleControlledRyy)},
        QASM3TranslationTestCase{"MultipleControlledRYY",
                                 qasm::multipleControlledRyy,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRyy)},
        QASM3TranslationTestCase{"RZX", qasm::rzx, MQT_NAMED_BUILDER(qc::rzx)},
        QASM3TranslationTestCase{"SingleControlledRZX",
                                 qasm::singleControlledRzx,
                                 MQT_NAMED_BUILDER(qc::singleControlledRzx)},
        QASM3TranslationTestCase{"MultipleControlledRZX",
                                 qasm::multipleControlledRzx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRzx)},
        QASM3TranslationTestCase{"RZZ", qasm::rzz, MQT_NAMED_BUILDER(qc::rzz)},
        QASM3TranslationTestCase{"SingleControlledRZZ",
                                 qasm::singleControlledRzz,
                                 MQT_NAMED_BUILDER(qc::singleControlledRzz)},
        QASM3TranslationTestCase{"MultipleControlledRZZ",
                                 qasm::multipleControlledRzz,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRzz)},
        QASM3TranslationTestCase{"XXPlusYY", qasm::xxPlusYY,
                                 MQT_NAMED_BUILDER(qc::xxPlusYY)},
        QASM3TranslationTestCase{
            "SingleControlledXXPlusYY", qasm::singleControlledXxPlusYY,
            MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY)},
        QASM3TranslationTestCase{
            "MultipleControlledXXPlusYY", qasm::multipleControlledXxPlusYY,
            MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY)},
        QASM3TranslationTestCase{"XXMinusYY", qasm::xxMinusYY,
                                 MQT_NAMED_BUILDER(qc::xxMinusYY)},
        QASM3TranslationTestCase{
            "SingleControlledXXMinusYY", qasm::singleControlledXxMinusYY,
            MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY)},
        QASM3TranslationTestCase{
            "MultipleControlledXXMinusYY", qasm::multipleControlledXxMinusYY,
            MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY)},
        QASM3TranslationTestCase{"Barrier", qasm::barrier,
                                 MQT_NAMED_BUILDER(qc::barrier)},
        QASM3TranslationTestCase{"BarrierTwoQubits", qasm::barrierTwoQubits,
                                 MQT_NAMED_BUILDER(qc::barrierTwoQubits)},
        QASM3TranslationTestCase{"BarrierMultipleQubits",
                                 qasm::barrierMultipleQubits,
                                 MQT_NAMED_BUILDER(qc::barrierMultipleQubits)},
        QASM3TranslationTestCase{"CtrlTwo", qasm::ctrlTwo,
                                 MQT_NAMED_BUILDER(qc::ctrlTwo)},
        QASM3TranslationTestCase{"CtrlTwoMixed", qasm::ctrlTwoMixed,
                                 MQT_NAMED_BUILDER(qc::ctrlTwoMixed)},
        QASM3TranslationTestCase{"SimpleIf", qasm::simpleIf,
                                 MQT_NAMED_BUILDER(qc::simpleIf)},
        QASM3TranslationTestCase{"IfNot", qasm::ifNot,
                                 MQT_NAMED_BUILDER(ifNot)},
        QASM3TranslationTestCase{"IfTwoQubits", qasm::ifTwoQubits,
                                 MQT_NAMED_BUILDER(qc::ifTwoQubits)},
        QASM3TranslationTestCase{"IfEmptyThen", qasm::ifEmptyThen,
                                 MQT_NAMED_BUILDER(ifNot)},
        QASM3TranslationTestCase{"IfElse", qasm::ifElse,
                                 MQT_NAMED_BUILDER(qc::ifElse)},
        QASM3TranslationTestCase{"BroadcastRegisterAndQubit",
                                 qasm::broadcastRegisterAndQubit,
                                 MQT_NAMED_BUILDER(broadcastRegisterAndQubit)},
        QASM3TranslationTestCase{"BroadcastCompoundGate",
                                 qasm::broadcastCompoundGate,
                                 MQT_NAMED_BUILDER(broadcastCompoundGate)},
        QASM3TranslationTestCase{"ExpressionArithmetic",
                                 qasm::expressionArithmetic,
                                 MQT_NAMED_BUILDER(expressionArithmetic)},
        QASM3TranslationTestCase{"ExpressionUnaryMinus",
                                 qasm::expressionUnaryMinus,
                                 MQT_NAMED_BUILDER(expressionUnaryMinus)},
        QASM3TranslationTestCase{"ExpressionBuiltinConstants",
                                 qasm::expressionBuiltinConstants,
                                 MQT_NAMED_BUILDER(expressionBuiltinConstants)},
        QASM3TranslationTestCase{"ExpressionMathFunctions",
                                 qasm::expressionMathFunctions,
                                 MQT_NAMED_BUILDER(expressionMathFunctions)},
        QASM3TranslationTestCase{
            "ExpressionNestedMathFunctions",
            qasm::expressionNestedMathFunctions,
            MQT_NAMED_BUILDER(expressionNestedMathFunctions)},
        QASM3TranslationTestCase{"ExpressionConstFloat",
                                 qasm::expressionConstFloat,
                                 MQT_NAMED_BUILDER(expressionConstFloat)},
        QASM3TranslationTestCase{"ExpressionMutableFloat",
                                 qasm::expressionMutableFloat,
                                 MQT_NAMED_BUILDER(expressionMutableFloat)},
        QASM3TranslationTestCase{
            "ExpressionConstIntArithmetic", qasm::expressionConstIntArithmetic,
            MQT_NAMED_BUILDER(expressionConstIntArithmetic)},
        QASM3TranslationTestCase{"ConditionLiteral", qasm::conditionLiteral,
                                 MQT_NAMED_BUILDER(conditionLiteral)},
        QASM3TranslationTestCase{"ConditionMeasurement",
                                 qasm::conditionMeasurement,
                                 MQT_NAMED_BUILDER(conditionMeasurement)},
        QASM3TranslationTestCase{"ConditionAnd", qasm::conditionAnd,
                                 MQT_NAMED_BUILDER(conditionAnd)},
        QASM3TranslationTestCase{"ConditionOr", qasm::conditionOr,
                                 MQT_NAMED_BUILDER(conditionOr)},
        QASM3TranslationTestCase{"ConditionNotAndOr", qasm::conditionNotAndOr,
                                 MQT_NAMED_BUILDER(conditionNotAndOr)},
        QASM3TranslationTestCase{"ConditionBoolVariable",
                                 qasm::conditionBoolVariable,
                                 MQT_NAMED_BUILDER(conditionBoolVariable)},
        QASM3TranslationTestCase{"ConditionIndexedBit",
                                 qasm::conditionIndexedBit,
                                 MQT_NAMED_BUILDER(conditionIndexedBit)}));
