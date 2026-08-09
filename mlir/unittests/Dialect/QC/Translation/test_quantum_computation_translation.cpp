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
#include "ir/QuantumComputation.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/Utils/Utils.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"
#include "quantum_computation_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace {

struct QuantumComputationTranslationTestCase {
  std::string name;
  mqt::test::NamedBuilder<::qc::QuantumComputation> programBuilder;
  mqt::test::NamedMLIRBuilder<mlir::qc::QCProgramBuilder> referenceBuilder;

  friend std::ostream&
  operator<<(std::ostream& os,
             const QuantumComputationTranslationTestCase& test);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const QuantumComputationTranslationTestCase& test) {
  return os << "QuantumComputationTranslation{" << test.name
            << ", original=" << mqt::test::displayName(test.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(test.referenceBuilder.name) << "}";
}

class QuantumComputationTranslationTest
    : public testing::TestWithParam<QuantumComputationTranslationTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
    context = std::make_unique<mlir::MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

mlir::SmallVector<mlir::Value>
// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
partiallyMeasuredRegisterControlReference(mlir::qc::QCProgramBuilder& builder) {
  auto q = builder.allocQubitRegister(1);
  auto prefix = builder.allocClassicalBitRegister(2, "prefix");
  auto condition = builder.allocClassicalBitRegister(3, "condition");
  builder.measure(q[0], condition, 0);

  mlir::Value packed;
  for (int64_t bit = 0; bit < 3; ++bit) {
    auto index = mlir::arith::ConstantIndexOp::create(builder, bit);
    auto value =
        mlir::memref::LoadOp::create(builder, condition, index.getResult());
    mlir::Value extended = mlir::arith::ExtUIOp::create(
        builder, builder.getI64Type(), value.getResult());
    if (bit != 0) {
      auto shift = mlir::arith::ConstantIntOp::create(builder, bit, 64);
      extended = mlir::arith::ShLIOp::create(builder, extended, shift);
    }
    packed = packed ? mlir::arith::OrIOp::create(builder, packed, extended)
                    : extended;
  }
  auto expected = builder.intConstant(5);
  auto comparison = mlir::arith::CmpIOp::create(
      builder, mlir::arith::CmpIPredicate::eq, packed, expected);
  builder.scfIf(comparison.getResult(), [&] { builder.x(q[0]); });
  return {prefix, condition};
}

} // namespace

TEST_P(QuantumComputationTranslationTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  mqt::test::DeferredPrinter printer;

  ::qc::QuantumComputation comp;
  programBuilder.fn(comp);

  auto translated = mlir::translateQuantumComputationToQC(context.get(), comp);
  ASSERT_TRUE(translated);
  printer.record(translated.get(), "Translated QC IR" + name);
  EXPECT_TRUE(mlir::verify(*translated).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(translated.get()).succeeded());
  printer.record(translated.get(), "Canonicalized Translated QC IR" + name);
  EXPECT_TRUE(mlir::verify(*translated).succeeded());

  auto reference = mqt::test::buildMLIRProgram(
      context.get(), referenceBuilder,
      mlir::qc::QCProgramBuilder::ClassicalRegisterInitialization::Zero);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QC IR" + name);
  EXPECT_TRUE(mlir::verify(*reference).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QC IR" + name);
  EXPECT_TRUE(mlir::verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(translated.get(), reference.get()));
}

TEST_F(QuantumComputationTranslationTest,
       ReloadsConditionAfterBranchMeasurement) {
  ::qc::QuantumComputation comp;
  const auto& q = comp.addQubitRegister(1, "q");
  const auto& c = comp.addClassicalRegister(1, "c");
  comp.measure(q[0], c[0]);
  comp.emplace_back<::qc::IfElseOperation>(
      std::make_unique<::qc::NonUnitaryOperation>(q[0], c[0]), nullptr, c[0]);
  comp.if_(::qc::X, q[0], c[0]);

  auto translated = mlir::translateQuantumComputationToQC(context.get(), comp);
  ASSERT_TRUE(translated);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*translated)));
}

TEST_F(QuantumComputationTranslationTest, RetainsClassicalRegisterName) {
  ::qc::QuantumComputation comp;
  const auto& q = comp.addQubitRegister(1, "q");
  const auto& c = comp.addClassicalRegister(1, "named_result");
  comp.measure(q[0], c[0]);

  auto translated = mlir::translateQuantumComputationToQC(context.get(), comp);
  ASSERT_TRUE(translated);

  mlir::memref::AllocOp classicalRegister;
  translated->walk([&](mlir::memref::AllocOp op) {
    if (op.getType().getElementType().isInteger(1)) {
      classicalRegister = op;
    }
  });
  ASSERT_TRUE(classicalRegister);
  const auto name = classicalRegister->getAttrOfType<mlir::StringAttr>(
      mlir::utils::CLASSICAL_REGISTER_NAME_ATTR);
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_result");
}

TEST_F(QuantumComputationTranslationTest,
       AllowsSingleBitControlBeforeMeasurement) {
  ::qc::QuantumComputation comp;
  const auto& q = comp.addQubitRegister(1, "q");
  const auto& c = comp.addClassicalRegister(1, "c");
  comp.if_(::qc::X, q[0], c[0], false);

  auto translated = mlir::translateQuantumComputationToQC(context.get(), comp);
  ASSERT_TRUE(translated);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*translated)));

  size_t zeroStores = 0;
  translated->walk([&](mlir::memref::StoreOp store) {
    auto constant = store.getValue().getDefiningOp<mlir::arith::ConstantOp>();
    if (constant == nullptr) {
      return;
    }
    const auto value = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (value && value.getValue().isZero()) {
      ++zeroStores;
    }
  });
  EXPECT_EQ(zeroStores, 1);

  size_t conditionals = 0;
  translated->walk([&](mlir::scf::IfOp) { ++conditionals; });
  EXPECT_EQ(conditionals, 1);
}

TEST_F(QuantumComputationTranslationTest, AllowsUnmeasuredRegisterControl) {
  ::qc::QuantumComputation comp;
  const auto& q = comp.addQubitRegister(1, "q");
  const auto& c = comp.addClassicalRegister(2, "c");
  comp.if_(::qc::X, q[0], c, 0U);

  auto translated = mlir::translateQuantumComputationToQC(context.get(), comp);
  ASSERT_TRUE(translated);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*translated)));

  size_t loads = 0;
  translated->walk([&](mlir::memref::LoadOp load) {
    if (load.getType().isInteger(1)) {
      ++loads;
    }
  });
  EXPECT_EQ(loads, 2);
}

TEST_F(QuantumComputationTranslationTest,
       TranslatesRegisterComparisonsInLittleEndianOrder) {
  constexpr std::array comparisons{
      std::pair{::qc::ComparisonKind::Eq, mlir::arith::CmpIPredicate::eq},
      std::pair{::qc::ComparisonKind::Neq, mlir::arith::CmpIPredicate::ne},
      std::pair{::qc::ComparisonKind::Lt, mlir::arith::CmpIPredicate::ult},
      std::pair{::qc::ComparisonKind::Leq, mlir::arith::CmpIPredicate::ule},
      std::pair{::qc::ComparisonKind::Gt, mlir::arith::CmpIPredicate::ugt},
      std::pair{::qc::ComparisonKind::Geq, mlir::arith::CmpIPredicate::uge},
  };

  for (const auto& [kind, expectedPredicate] : comparisons) {
    SCOPED_TRACE(::qc::toString(kind));
    ::qc::QuantumComputation comp;
    const auto& q = comp.addQubitRegister(1, "q");
    comp.addClassicalRegister(2, "prefix");
    const auto& condition = comp.addClassicalRegister(3, "condition");
    ASSERT_EQ(condition.getStartIndex(), 2U);
    comp.measure(q[0], condition[0]);
    comp.if_(::qc::X, q[0], condition, 5U, kind);

    auto translated =
        mlir::translateQuantumComputationToQC(context.get(), comp);
    ASSERT_TRUE(translated);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*translated)));

    mlir::scf::IfOp conditional;
    translated->walk([&](mlir::scf::IfOp op) { conditional = op; });
    ASSERT_TRUE(conditional);
    auto comparison =
        conditional.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
    ASSERT_TRUE(comparison);
    EXPECT_EQ(comparison.getPredicate(), expectedPredicate);
    const auto comparisonType =
        mlir::dyn_cast<mlir::IntegerType>(comparison.getLhs().getType());
    ASSERT_TRUE(comparisonType);
    EXPECT_EQ(comparisonType.getWidth(), 64U);
    auto expected =
        comparison.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
    ASSERT_TRUE(expected);
    const auto expectedValue =
        mlir::dyn_cast<mlir::IntegerAttr>(expected.getValue());
    ASSERT_TRUE(expectedValue);
    EXPECT_EQ(expectedValue.getValue().getZExtValue(), 5U);

    mlir::SmallVector<int64_t> loadedIndices;
    translated->walk([&](mlir::memref::LoadOp load) {
      auto alloc = load.getMemref().getDefiningOp<mlir::memref::AllocOp>();
      if (alloc == nullptr) {
        return;
      }
      const auto name = alloc->getAttrOfType<mlir::StringAttr>(
          mlir::utils::CLASSICAL_REGISTER_NAME_ATTR);
      if (!name || name.getValue() != "condition") {
        return;
      }
      auto index =
          load.getIndices().front().getDefiningOp<mlir::arith::ConstantOp>();
      if (index == nullptr) {
        return;
      }
      const auto value = mlir::dyn_cast<mlir::IntegerAttr>(index.getValue());
      if (!value) {
        return;
      }
      loadedIndices.push_back(value.getInt());
    });
    llvm::sort(loadedIndices);
    EXPECT_EQ(loadedIndices, (mlir::SmallVector<int64_t>{0, 1, 2}));

    mlir::SmallVector<int64_t> shiftAmounts;
    translated->walk([&](mlir::arith::ShLIOp shift) {
      auto amount = shift.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
      if (amount == nullptr) {
        return;
      }
      const auto value = mlir::dyn_cast<mlir::IntegerAttr>(amount.getValue());
      if (!value) {
        return;
      }
      shiftAmounts.push_back(value.getInt());
    });
    llvm::sort(shiftAmounts);
    EXPECT_EQ(shiftAmounts, (mlir::SmallVector<int64_t>{1, 2}));
  }
}

TEST_F(QuantumComputationTranslationTest,
       MatchesPartiallyMeasuredRegisterControlReference) {
  ::qc::QuantumComputation comp;
  const auto& q = comp.addQubitRegister(1, "q");
  comp.addClassicalRegister(2, "prefix");
  const auto& condition = comp.addClassicalRegister(3, "condition");
  comp.measure(q[0], condition[0]);
  comp.if_(::qc::X, q[0], condition, 5U);

  auto translated = mlir::translateQuantumComputationToQC(context.get(), comp);
  ASSERT_TRUE(translated);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*translated)));
  ASSERT_TRUE(mlir::succeeded(runQCCleanupPipeline(*translated)));

  const auto referenceBuilder =
      mqt::test::namedBuilder("partiallyMeasuredRegisterControlReference",
                              partiallyMeasuredRegisterControlReference);
  auto reference = mqt::test::buildMLIRProgram(
      context.get(), referenceBuilder,
      mlir::qc::QCProgramBuilder::ClassicalRegisterInitialization::Zero);
  ASSERT_TRUE(reference);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*reference)));
  ASSERT_TRUE(mlir::succeeded(runQCCleanupPipeline(*reference)));

  EXPECT_TRUE(areModulesEquivalentWithPermutations(*translated, *reference));
}

TEST_F(QuantumComputationTranslationTest, JoinsMeasurementsFromBothBranches) {
  ::qc::QuantumComputation comp;
  const auto& q = comp.addQubitRegister(2, "q");
  const auto& c = comp.addClassicalRegister(2, "c");
  comp.measure(q[0], c[0]);
  comp.emplace_back<::qc::IfElseOperation>(
      std::make_unique<::qc::NonUnitaryOperation>(q[1], c[1]),
      std::make_unique<::qc::NonUnitaryOperation>(q[1], c[1]), c[0]);
  comp.if_(::qc::X, q[1], c[1]);

  auto translated = mlir::translateQuantumComputationToQC(context.get(), comp);
  ASSERT_TRUE(translated);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*translated)));
}

INSTANTIATE_TEST_SUITE_P(
    QuantumComputationTranslationProgramsTest,
    QuantumComputationTranslationTest,
    testing::Values(
        QuantumComputationTranslationTestCase{
            "AllocQubit", MQT_NAMED_BUILDER(qc::allocQubit),
            MQT_NAMED_BUILDER(mlir::qc::alloc1QubitRegister)},
        QuantumComputationTranslationTestCase{
            "AllocQubitRegister", MQT_NAMED_BUILDER(qc::allocQubitRegister),
            MQT_NAMED_BUILDER(mlir::qc::allocQubitRegister)},
        QuantumComputationTranslationTestCase{
            "AllocMultipleQubitRegisters",
            MQT_NAMED_BUILDER(qc::allocMultipleQubitRegisters),
            MQT_NAMED_BUILDER(mlir::qc::allocMultipleQubitRegisters)},
        QuantumComputationTranslationTestCase{
            "AllocLargeRegister", MQT_NAMED_BUILDER(qc::allocLargeRegister),
            MQT_NAMED_BUILDER(mlir::qc::allocLargeRegister)},
        QuantumComputationTranslationTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(mlir::qc::singleMeasurementToSingleBit)},
        QuantumComputationTranslationTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(mlir::qc::repeatedMeasurementToSameBit)},
        QuantumComputationTranslationTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(mlir::qc::repeatedMeasurementToDifferentBits)},
        QuantumComputationTranslationTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(
                mlir::qc::multipleClassicalRegistersAndMeasurements)},
        QuantumComputationTranslationTestCase{
            "ResetQubitAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qc::resetQubitAfterSingleOp)},
        QuantumComputationTranslationTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qc::resetMultipleQubitsAfterSingleOp)},
        QuantumComputationTranslationTestCase{
            "RepeatedResetAfterSingleOp",
            MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qc::repeatedResetAfterSingleOp)},
        QuantumComputationTranslationTestCase{
            "GlobalPhase", MQT_NAMED_BUILDER(qc::globalPhase),
            MQT_NAMED_BUILDER(mlir::qc::globalPhase)},
        QuantumComputationTranslationTestCase{
            "Identity", MQT_NAMED_BUILDER(qc::identity),
            MQT_NAMED_BUILDER(mlir::qc::identity)},
        QuantumComputationTranslationTestCase{
            "SingleControlledIdentity",
            MQT_NAMED_BUILDER(qc::singleControlledIdentity),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledIdentity)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledIdentity",
            MQT_NAMED_BUILDER(qc::multipleControlledIdentity),
            MQT_NAMED_BUILDER(mlir::qc::threeQubitsOneIdentity)},
        QuantumComputationTranslationTestCase{"X", MQT_NAMED_BUILDER(qc::x),
                                              MQT_NAMED_BUILDER(mlir::qc::x)},
        QuantumComputationTranslationTestCase{
            "SingleControlledX", MQT_NAMED_BUILDER(qc::singleControlledX),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledX)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledX", MQT_NAMED_BUILDER(qc::multipleControlledX),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledX)},
        QuantumComputationTranslationTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                                              MQT_NAMED_BUILDER(mlir::qc::y)},
        QuantumComputationTranslationTestCase{
            "SingleControlledY", MQT_NAMED_BUILDER(qc::singleControlledY),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledY)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledY", MQT_NAMED_BUILDER(qc::multipleControlledY),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledY)},
        QuantumComputationTranslationTestCase{"Z", MQT_NAMED_BUILDER(qc::z),
                                              MQT_NAMED_BUILDER(mlir::qc::z)},
        QuantumComputationTranslationTestCase{
            "SingleControlledZ", MQT_NAMED_BUILDER(qc::singleControlledZ),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledZ)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledZ", MQT_NAMED_BUILDER(qc::multipleControlledZ),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledZ)},
        QuantumComputationTranslationTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                                              MQT_NAMED_BUILDER(mlir::qc::h)},
        QuantumComputationTranslationTestCase{
            "SingleControlledH", MQT_NAMED_BUILDER(qc::singleControlledH),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledH)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledH", MQT_NAMED_BUILDER(qc::multipleControlledH),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledH)},
        QuantumComputationTranslationTestCase{"S", MQT_NAMED_BUILDER(qc::s),
                                              MQT_NAMED_BUILDER(mlir::qc::s)},
        QuantumComputationTranslationTestCase{
            "SingleControlledS", MQT_NAMED_BUILDER(qc::singleControlledS),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledS)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledS", MQT_NAMED_BUILDER(qc::multipleControlledS),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledS)},
        QuantumComputationTranslationTestCase{"Sdg", MQT_NAMED_BUILDER(qc::sdg),
                                              MQT_NAMED_BUILDER(mlir::qc::sdg)},
        QuantumComputationTranslationTestCase{
            "SingleControlledSdg", MQT_NAMED_BUILDER(qc::singleControlledSdg),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledSdg)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledSdg",
            MQT_NAMED_BUILDER(qc::multipleControlledSdg),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSdg)},
        QuantumComputationTranslationTestCase{"T", MQT_NAMED_BUILDER(qc::t_),
                                              MQT_NAMED_BUILDER(mlir::qc::t_)},
        QuantumComputationTranslationTestCase{
            "SingleControlledT", MQT_NAMED_BUILDER(qc::singleControlledT),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledT)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledT", MQT_NAMED_BUILDER(qc::multipleControlledT),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledT)},
        QuantumComputationTranslationTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg),
                                              MQT_NAMED_BUILDER(mlir::qc::tdg)},
        QuantumComputationTranslationTestCase{
            "SingleControlledTdg", MQT_NAMED_BUILDER(qc::singleControlledTdg),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledTdg)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledTdg",
            MQT_NAMED_BUILDER(qc::multipleControlledTdg),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledTdg)},
        QuantumComputationTranslationTestCase{"SX", MQT_NAMED_BUILDER(qc::sx),
                                              MQT_NAMED_BUILDER(mlir::qc::sx)},
        QuantumComputationTranslationTestCase{
            "SingleControlledSX", MQT_NAMED_BUILDER(qc::singleControlledSx),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledSx)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledSX", MQT_NAMED_BUILDER(qc::multipleControlledSx),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSx)},
        QuantumComputationTranslationTestCase{
            "SXdg", MQT_NAMED_BUILDER(qc::sxdg),
            MQT_NAMED_BUILDER(mlir::qc::sxdg)},
        QuantumComputationTranslationTestCase{
            "SingleControlledSXdg", MQT_NAMED_BUILDER(qc::singleControlledSxdg),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledSxdg)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledSXdg",
            MQT_NAMED_BUILDER(qc::multipleControlledSxdg),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSxdg)},
        QuantumComputationTranslationTestCase{"RX", MQT_NAMED_BUILDER(qc::rx),
                                              MQT_NAMED_BUILDER(mlir::qc::rx)},
        QuantumComputationTranslationTestCase{
            "SingleControlledRX", MQT_NAMED_BUILDER(qc::singleControlledRx),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledRx)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledRX", MQT_NAMED_BUILDER(qc::multipleControlledRx),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRx)},
        QuantumComputationTranslationTestCase{"RY", MQT_NAMED_BUILDER(qc::ry),
                                              MQT_NAMED_BUILDER(mlir::qc::ry)},
        QuantumComputationTranslationTestCase{
            "SingleControlledRY", MQT_NAMED_BUILDER(qc::singleControlledRy),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledRy)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledRY", MQT_NAMED_BUILDER(qc::multipleControlledRy),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRy)},
        QuantumComputationTranslationTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz),
                                              MQT_NAMED_BUILDER(mlir::qc::rz)},
        QuantumComputationTranslationTestCase{
            "SingleControlledRZ", MQT_NAMED_BUILDER(qc::singleControlledRz),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledRz)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledRZ", MQT_NAMED_BUILDER(qc::multipleControlledRz),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRz)},
        QuantumComputationTranslationTestCase{"P", MQT_NAMED_BUILDER(qc::p),
                                              MQT_NAMED_BUILDER(mlir::qc::p)},
        QuantumComputationTranslationTestCase{
            "SingleControlledP", MQT_NAMED_BUILDER(qc::singleControlledP),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledP)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledP", MQT_NAMED_BUILDER(qc::multipleControlledP),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledP)},
        QuantumComputationTranslationTestCase{"R", MQT_NAMED_BUILDER(qc::r),
                                              MQT_NAMED_BUILDER(mlir::qc::r)},
        QuantumComputationTranslationTestCase{
            "SingleControlledR", MQT_NAMED_BUILDER(qc::singleControlledR),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledR)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledR", MQT_NAMED_BUILDER(qc::multipleControlledR),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledR)},
        QuantumComputationTranslationTestCase{"U2", MQT_NAMED_BUILDER(qc::u2),
                                              MQT_NAMED_BUILDER(mlir::qc::u2)},
        QuantumComputationTranslationTestCase{
            "SingleControlledU2", MQT_NAMED_BUILDER(qc::singleControlledU2),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledU2)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledU2", MQT_NAMED_BUILDER(qc::multipleControlledU2),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledU2)},
        QuantumComputationTranslationTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                                              MQT_NAMED_BUILDER(mlir::qc::u)},
        QuantumComputationTranslationTestCase{
            "SingleControlledU", MQT_NAMED_BUILDER(qc::singleControlledU),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledU)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledU", MQT_NAMED_BUILDER(qc::multipleControlledU),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledU)},
        QuantumComputationTranslationTestCase{
            "SWAP", MQT_NAMED_BUILDER(qc::swap),
            MQT_NAMED_BUILDER(mlir::qc::swap)},
        QuantumComputationTranslationTestCase{
            "SingleControlledSWAP", MQT_NAMED_BUILDER(qc::singleControlledSwap),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledSwap)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledSWAP",
            MQT_NAMED_BUILDER(qc::multipleControlledSwap),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSwap)},
        QuantumComputationTranslationTestCase{
            "iSWAP", MQT_NAMED_BUILDER(qc::iswap),
            MQT_NAMED_BUILDER(mlir::qc::iswap)},
        QuantumComputationTranslationTestCase{
            "SingleControllediSWAP",
            MQT_NAMED_BUILDER(qc::singleControlledIswap),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledIswap)},
        QuantumComputationTranslationTestCase{
            "MultipleControllediSWAP",
            MQT_NAMED_BUILDER(qc::multipleControlledIswap),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledIswap)},
        QuantumComputationTranslationTestCase{
            "InverseISWAP", MQT_NAMED_BUILDER(qc::inverseIswap),
            MQT_NAMED_BUILDER(mlir::qc::inverseIswap)},
        QuantumComputationTranslationTestCase{
            "InverseMultiControlledISWAP",
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap),
            MQT_NAMED_BUILDER(mlir::qc::inverseMultipleControlledIswap)},
        QuantumComputationTranslationTestCase{"DCX", MQT_NAMED_BUILDER(qc::dcx),
                                              MQT_NAMED_BUILDER(mlir::qc::dcx)},
        QuantumComputationTranslationTestCase{
            "SingleControlledDCX", MQT_NAMED_BUILDER(qc::singleControlledDcx),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledDcx)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledDCX",
            MQT_NAMED_BUILDER(qc::multipleControlledDcx),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledDcx)},
        QuantumComputationTranslationTestCase{"ECR", MQT_NAMED_BUILDER(qc::ecr),
                                              MQT_NAMED_BUILDER(mlir::qc::ecr)},
        QuantumComputationTranslationTestCase{
            "SingleControlledECR", MQT_NAMED_BUILDER(qc::singleControlledEcr),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledEcr)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledECR",
            MQT_NAMED_BUILDER(qc::multipleControlledEcr),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledEcr)},
        QuantumComputationTranslationTestCase{"RXX", MQT_NAMED_BUILDER(qc::rxx),
                                              MQT_NAMED_BUILDER(mlir::qc::rxx)},
        QuantumComputationTranslationTestCase{
            "SingleControlledRXX", MQT_NAMED_BUILDER(qc::singleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledRxx)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledRXX",
            MQT_NAMED_BUILDER(qc::multipleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRxx)},
        QuantumComputationTranslationTestCase{
            "TripleControlledRXX", MQT_NAMED_BUILDER(qc::tripleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qc::tripleControlledRxx)},
        QuantumComputationTranslationTestCase{"RYY", MQT_NAMED_BUILDER(qc::ryy),
                                              MQT_NAMED_BUILDER(mlir::qc::ryy)},
        QuantumComputationTranslationTestCase{
            "SingleControlledRYY", MQT_NAMED_BUILDER(qc::singleControlledRyy),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledRyy)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledRYY",
            MQT_NAMED_BUILDER(qc::multipleControlledRyy),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRyy)},
        QuantumComputationTranslationTestCase{"RZX", MQT_NAMED_BUILDER(qc::rzx),
                                              MQT_NAMED_BUILDER(mlir::qc::rzx)},
        QuantumComputationTranslationTestCase{
            "SingleControlledRZX", MQT_NAMED_BUILDER(qc::singleControlledRzx),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledRzx)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledRZX",
            MQT_NAMED_BUILDER(qc::multipleControlledRzx),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRzx)},
        QuantumComputationTranslationTestCase{"RZZ", MQT_NAMED_BUILDER(qc::rzz),
                                              MQT_NAMED_BUILDER(mlir::qc::rzz)},
        QuantumComputationTranslationTestCase{
            "SingleControlledRZZ", MQT_NAMED_BUILDER(qc::singleControlledRzz),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledRzz)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledRZZ",
            MQT_NAMED_BUILDER(qc::multipleControlledRzz),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRzz)},
        QuantumComputationTranslationTestCase{
            "XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
            MQT_NAMED_BUILDER(mlir::qc::xxPlusYY)},
        QuantumComputationTranslationTestCase{
            "SingleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXxPlusYY)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledXxPlusYY)},
        QuantumComputationTranslationTestCase{
            "XXMinusYY", MQT_NAMED_BUILDER(qc::xxMinusYY),
            MQT_NAMED_BUILDER(mlir::qc::xxMinusYY)},
        QuantumComputationTranslationTestCase{
            "SingleControlledXXMinusYY",
            MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXxMinusYY)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledXXMinusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledXxMinusYY)},
        QuantumComputationTranslationTestCase{
            "RCCX", MQT_NAMED_BUILDER(qc::rccx),
            MQT_NAMED_BUILDER(mlir::qc::rccx)},
        QuantumComputationTranslationTestCase{
            "SingleControlledRCCX", MQT_NAMED_BUILDER(qc::singleControlledRccx),
            MQT_NAMED_BUILDER(mlir::qc::singleControlledRccx)},
        QuantumComputationTranslationTestCase{
            "MultipleControlledRCCX",
            MQT_NAMED_BUILDER(qc::multipleControlledRccx),
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRccx)},
        QuantumComputationTranslationTestCase{
            "Barrier", MQT_NAMED_BUILDER(qc::barrier),
            MQT_NAMED_BUILDER(mlir::qc::barrier)},
        QuantumComputationTranslationTestCase{
            "BarrierTwoQubits", MQT_NAMED_BUILDER(qc::barrierTwoQubits),
            MQT_NAMED_BUILDER(mlir::qc::barrierTwoQubits)},
        QuantumComputationTranslationTestCase{
            "BarrierMultipleQubits",
            MQT_NAMED_BUILDER(qc::barrierMultipleQubits),
            MQT_NAMED_BUILDER(mlir::qc::barrierMultipleQubits)},
        QuantumComputationTranslationTestCase{
            "CtrlTwo", MQT_NAMED_BUILDER(qc::ctrlTwo),
            MQT_NAMED_BUILDER(mlir::qc::ctrlTwo)},
        QuantumComputationTranslationTestCase{
            "CtrlTwoMixed", MQT_NAMED_BUILDER(qc::ctrlTwoMixed),
            MQT_NAMED_BUILDER(mlir::qc::ctrlTwoMixed)},
        QuantumComputationTranslationTestCase{
            "SimpleIf", MQT_NAMED_BUILDER(qc::simpleIf),
            MQT_NAMED_BUILDER(mlir::qc::simpleIf)},
        QuantumComputationTranslationTestCase{
            "IfElse", MQT_NAMED_BUILDER(qc::ifElse),
            MQT_NAMED_BUILDER(mlir::qc::ifElse)},
        QuantumComputationTranslationTestCase{
            "IfTwoQubits", MQT_NAMED_BUILDER(qc::ifTwoQubits),
            MQT_NAMED_BUILDER(mlir::qc::ifTwoQubits)},
        QuantumComputationTranslationTestCase{
            "IfWithMeasurement", MQT_NAMED_BUILDER(qc::ifWithMeasurement),
            MQT_NAMED_BUILDER(mlir::qc::ifWithMeasurement)}));
