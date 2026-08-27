/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Support/IRVerification.h"
#include "TestCaseUtils.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Dialect/QIR/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"
#include "mlir/Support/Passes.h"
#include "qir_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>

using namespace mlir;
using namespace qir;

static LLVM::ModuleFlagAttr findModuleFlag(ModuleOp moduleOp,
                                           const StringRef name) {
  LLVM::ModuleFlagAttr result;
  moduleOp->walk([&](LLVM::ModuleFlagsOp flagsOp) {
    for (const auto flag :
         flagsOp.getFlags().getAsRange<LLVM::ModuleFlagAttr>()) {
      if (flag.getKey().getValue() == name) {
        result = flag;
      }
    }
  });
  return result;
}

namespace {

struct QIRTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<QIRProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<QIRProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os, const QIRTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os, const QIRTestCase& info) {
  return os << "QIR{" << info.name << ", original="
            << ::mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << ::mqt::test::displayName(info.referenceBuilder.name) << "}";
}

class QIRTest : public testing::TestWithParam<QIRTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<LLVM::LLVMDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

TEST_P(QIRTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  ::mqt::test::DeferredPrinter printer;

  auto program = ::mqt::test::buildMLIRProgram(
      context.get(), programBuilder, QIRProgramBuilder::Profile::Adaptive);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(program.get(), true).succeeded());
  printer.record(program.get(), "Canonicalized QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = ::mqt::test::buildMLIRProgram(
      context.get(), referenceBuilder, QIRProgramBuilder::Profile::Adaptive);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QIR IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(reference.get(), true).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QIR IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(QIRTest, BuilderRejectsMixedStaticAndDynamicQubitAllocationModes) {
  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        mixedStaticThenDynamicQubit(builder);
      },
      "Cannot mix static and dynamic qubit allocation modes");

  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        mixedDynamicRegisterThenStaticQubit(builder);
      },
      "Cannot mix dynamic and static qubit allocation modes");
}

TEST_F(QIRTest, BuilderRejectsOutOfBoundsClassicalRegisterIndices) {
  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        auto q = builder.allocQubit();
        const auto c = builder.allocClassicalBitRegister(1);
        builder.measure(q, c, -1);
      },
      "Register index must be non-negative");

  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        auto q = builder.allocQubit();
        const auto c = builder.allocClassicalBitRegister(1);
        builder.measure(q, c, 1);
      },
      "Register index is out of bounds");
}

TEST_F(QIRTest, BuilderReturnsCompleteClassicalRegister) {
  ClassicalRegister returnedRegister;
  auto module = QIRProgramBuilder::build(
      context.get(),
      [&](QIRProgramBuilder& builder) {
        returnedRegister = builder.allocClassicalBitRegister(2, false);
        return builder.intConstant(0);
      },
      QIRProgramBuilder::Profile::Base);

  ASSERT_TRUE(module);
  EXPECT_FALSE(returnedRegister.record);
  EXPECT_EQ(returnedRegister.results.size(), 2U);
  EXPECT_FALSE(returnedRegister.array);
}

TEST_F(QIRTest, AdaptiveBuilderSelectsControlledSpecializationsByArity) {
  auto module = QIRProgramBuilder::build(
      context.get(),
      [](QIRProgramBuilder& builder) {
        auto control0 = builder.allocQubit();
        auto control1 = builder.allocQubit();
        auto control2 = builder.allocQubit();
        auto target = builder.allocQubit();
        builder.crx(0.25, control0, target);
        builder.mcrx(0.5, {control0, control1}, target);
        builder.mcrx(0.75, {control0, control1, control2}, target);
        builder.r(0.5, 0.75, target);
        return builder.intConstant(0);
      },
      QIRProgramBuilder::Profile::Adaptive);

  ASSERT_TRUE(module);
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_CRX));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_CCRX));
  auto controlledRx = module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_RX_CTL);
  ASSERT_TRUE(controlledRx);
  const auto argumentTypes = controlledRx.getFunctionType().getParams();
  ASSERT_EQ(argumentTypes.size(), 2U);
  EXPECT_EQ(argumentTypes[0], LLVM::LLVMPointerType::get(context.get()));
  EXPECT_EQ(argumentTypes[1], LLVM::LLVMPointerType::get(context.get()));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_ARRAY_CREATE));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_TUPLE_CREATE));
  auto main = getMainFunction(module.get());
  ASSERT_TRUE(main);
  const auto passthrough = main->getAttrOfType<ArrayAttr>("passthrough");
  ASSERT_TRUE(passthrough);
  EXPECT_TRUE(llvm::is_contained(
      passthrough,
      ArrayAttr::get(context.get(),
                     {StringAttr::get(context.get(), "qir_profiles"),
                      StringAttr::get(context.get(), "adaptive_profile")})));

  EXPECT_STREQ(QIR_R, "__quantum__qis__prx__body");
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_R));
  EXPECT_FALSE(
      module->lookupSymbol<LLVM::LLVMFuncOp>("__quantum__qis__r__body"));
}

TEST_F(QIRTest, BaseBuilderUsesGenericSpecializationForThreeControls) {
  auto module = QIRProgramBuilder::build(
      context.get(),
      [](QIRProgramBuilder& builder) {
        auto control0 = builder.staticQubit(0);
        auto control1 = builder.staticQubit(1);
        auto control2 = builder.staticQubit(2);
        auto target = builder.staticQubit(3);
        builder.mcrx(0.25, {control0, control1, control2}, target);
        return builder.intConstant(0);
      },
      QIRProgramBuilder::Profile::Base);

  ASSERT_TRUE(module);
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_RX_CTL));
  auto main = getMainFunction(module.get());
  ASSERT_TRUE(main);
  const auto passthrough = main->getAttrOfType<ArrayAttr>("passthrough");
  ASSERT_TRUE(passthrough);
  EXPECT_TRUE(llvm::is_contained(
      passthrough,
      ArrayAttr::get(context.get(),
                     {StringAttr::get(context.get(), "qir_profiles"),
                      StringAttr::get(context.get(), "base_profile")})));
}

TEST_F(QIRTest, UsesQIR21ModuleFlagWidths) {
  const auto build = [&](const QIRProgramBuilder::Profile profile) {
    return QIRProgramBuilder::build(
        context.get(),
        [](QIRProgramBuilder& builder) { return builder.intConstant(0); },
        profile);
  };
  auto base = build(QIRProgramBuilder::Profile::Base);
  ASSERT_TRUE(base);
  const auto baseDynamicQubits =
      findModuleFlag(base.get(), "dynamic_qubit_management");
  ASSERT_TRUE(baseDynamicQubits);
  EXPECT_TRUE(isa<BoolAttr>(baseDynamicQubits.getValue()));
  EXPECT_FALSE(findModuleFlag(base.get(), "backwards_branching"));

  auto adaptive = build(QIRProgramBuilder::Profile::Adaptive);
  ASSERT_TRUE(adaptive);
  const auto backwardsBranching =
      findModuleFlag(adaptive.get(), "backwards_branching");
  ASSERT_TRUE(backwardsBranching);
  const auto backwardsBranchingValue =
      dyn_cast<IntegerAttr>(backwardsBranching.getValue());
  ASSERT_TRUE(backwardsBranchingValue);
  EXPECT_EQ(backwardsBranchingValue.getType().getIntOrFloatBitWidth(), 2U);
  const auto arrays = findModuleFlag(adaptive.get(), "arrays");
  ASSERT_TRUE(arrays);
  EXPECT_TRUE(isa<BoolAttr>(arrays.getValue()));
}

TEST_F(QIRTest, DerivesAdaptiveClassicalCapabilities) {
  OpBuilder builder(context.get());
  const auto location = builder.getUnknownLoc();
  auto moduleOp = ModuleOp::create(location);
  builder.setInsertionPointToStart(moduleOp.getBody());

  const auto mainType = LLVM::LLVMFunctionType::get(builder.getI64Type(), {});
  auto main = LLVM::LLVMFuncOp::create(builder, location, "main", mainType);
  main->setAttr("passthrough", builder.getStrArrayAttr({"entry_point"}));
  auto* mainBlock = main.addEntryBlock(builder);
  builder.setInsertionPointToEnd(mainBlock);
  Value exitCode =
      LLVM::ConstantOp::create(builder, location, builder.getI64IntegerAttr(0))
          .getResult();
  LLVM::ReturnOp::create(builder, location, exitCode);

  builder.setInsertionPointToEnd(moduleOp.getBody());
  const auto helperType = LLVM::LLVMFunctionType::get(
      builder.getF64Type(), {builder.getI8Type(), builder.getF32Type()});
  auto helper =
      LLVM::LLVMFuncOp::create(builder, location, "helper", helperType);
  auto* helperEntry = helper.addEntryBlock(builder);
  auto* defaultBlock = helper.addBlock();
  auto* caseBlock = helper.addBlock();
  builder.setInsertionPointToEnd(helperEntry);
  Value left =
      LLVM::ConstantOp::create(builder, location, builder.getI32IntegerAttr(1))
          .getResult();
  Value right =
      LLVM::ConstantOp::create(builder, location, builder.getI32IntegerAttr(2))
          .getResult();
  std::ignore = LLVM::ICmpOp::create(builder, location,
                                     LLVM::ICmpPredicate::slt, left, right);
  Value floating =
      LLVM::FPExtOp::create(builder, location, builder.getF64Type(),
                            helperEntry->getArgument(1))
          .getResult();
  Value doubled =
      LLVM::FAddOp::create(builder, location, floating, floating).getResult();
  const std::array<APInt, 1> caseValues{APInt(8, 0)};
  const std::array<Block*, 1> caseDestinations{caseBlock};
  const std::array<ValueRange, 1> caseOperands{ValueRange{}};
  LLVM::SwitchOp::create(builder, location, helperEntry->getArgument(0),
                         defaultBlock, ValueRange{}, caseValues,
                         BlockRange(caseDestinations), caseOperands);
  builder.setInsertionPointToEnd(defaultBlock);
  LLVM::ReturnOp::create(builder, location, doubled);
  builder.setInsertionPointToEnd(caseBlock);
  LLVM::ReturnOp::create(builder, location, doubled);

  const auto attachAttributes = [&](const bool useAdaptive) {
    return runWithPassManager(
        moduleOp,
        [&](OpPassManager& manager) {
          manager.addPass(
              qir::createQIRSetAttributesAndMetadata({useAdaptive}));
        },
        "Failed to attach QIR attributes.");
  };
  ASSERT_TRUE(attachAttributes(true).succeeded());
  const auto integerTypes =
      moduleOp->getAttrOfType<ArrayAttr>("qir.int_computations");
  ASSERT_TRUE(integerTypes);
  ASSERT_EQ(integerTypes.size(), 2U);
  EXPECT_EQ(cast<StringAttr>(integerTypes[0]).getValue(), "i32");
  EXPECT_EQ(cast<StringAttr>(integerTypes[1]).getValue(), "i8");
  const auto floatingTypes =
      moduleOp->getAttrOfType<ArrayAttr>("qir.float_computations");
  ASSERT_TRUE(floatingTypes);
  ASSERT_EQ(floatingTypes.size(), 2U);
  EXPECT_EQ(cast<StringAttr>(floatingTypes[0]).getValue(), "double");
  EXPECT_EQ(cast<StringAttr>(floatingTypes[1]).getValue(), "float");

  for (const auto* const flag : {"ir_functions", "multiple_target_branching",
                                 "multiple_return_points"}) {
    const auto moduleFlag = findModuleFlag(moduleOp, flag);
    ASSERT_TRUE(moduleFlag) << flag;
    EXPECT_EQ(moduleFlag.getValue(), builder.getBoolAttr(true)) << flag;
  }

  ASSERT_TRUE(attachAttributes(false).succeeded());
  EXPECT_FALSE(moduleOp->hasAttr("qir.int_computations"));
  EXPECT_FALSE(moduleOp->hasAttr("qir.float_computations"));
  EXPECT_FALSE(findModuleFlag(moduleOp, "ir_functions"));
  EXPECT_FALSE(findModuleFlag(moduleOp, "multiple_target_branching"));
  EXPECT_FALSE(findModuleFlag(moduleOp, "multiple_return_points"));
}

TEST(QIRModuleFlagsTest, RecordsAdaptiveClassicalCapabilities) {
  MLIRContext mlirContext;
  OpBuilder builder(&mlirContext);
  auto sourceModule = ModuleOp::create(builder.getUnknownLoc());
  sourceModule->setAttr("qir.int_computations",
                        builder.getStrArrayAttr({"i8"}));
  sourceModule->setAttr("qir.float_computations",
                        builder.getStrArrayAttr({"double"}));

  llvm::LLVMContext llvmContext;
  llvm::Module moduleOp("adaptive", llvmContext);
  moduleOp.addModuleFlag(llvm::Module::Error, "ir_functions", 1U);
  moduleOp.addModuleFlag(llvm::Module::Error, "multiple_target_branching", 1U);
  moduleOp.addModuleFlag(llvm::Module::Error, "multiple_return_points", 1U);
  normalizeQIRModuleFlags(moduleOp, sourceModule);

  const auto* integerTypes =
      llvm::dyn_cast<llvm::MDNode>(moduleOp.getModuleFlag("int_computations"));
  ASSERT_NE(integerTypes, nullptr);
  ASSERT_EQ(integerTypes->getNumOperands(), 1U);
  EXPECT_EQ(
      llvm::cast<llvm::MDString>(integerTypes->getOperand(0))->getString(),
      "i8");
  const auto* floatingTypes = llvm::dyn_cast<llvm::MDNode>(
      moduleOp.getModuleFlag("float_computations"));
  ASSERT_NE(floatingTypes, nullptr);
  ASSERT_EQ(floatingTypes->getNumOperands(), 1U);
  EXPECT_EQ(
      llvm::cast<llvm::MDString>(floatingTypes->getOperand(0))->getString(),
      "double");
  for (const auto* const flag : {"ir_functions", "multiple_target_branching",
                                 "multiple_return_points"}) {
    const auto* metadata =
        llvm::dyn_cast<llvm::ConstantAsMetadata>(moduleOp.getModuleFlag(flag));
    ASSERT_NE(metadata, nullptr) << flag;
    const auto* value = llvm::dyn_cast<llvm::ConstantInt>(metadata->getValue());
    ASSERT_NE(value, nullptr) << flag;
    EXPECT_EQ(value->getBitWidth(), 1U) << flag;
    EXPECT_TRUE(value->isOne()) << flag;
  }
}

/// \name QIR/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRDCXOpTest, QIRTest,
    testing::Values(QIRTestCase{"DCX", MQT_NAMED_BUILDER(dcx<>),
                                MQT_NAMED_BUILDER(dcx<>)},
                    QIRTestCase{"SingleControlledDCX",
                                MQT_NAMED_BUILDER(singleControlledDcx<>),
                                MQT_NAMED_BUILDER(singleControlledDcx<>)},
                    QIRTestCase{"MultipleControlledDCX",
                                MQT_NAMED_BUILDER(multipleControlledDcx<>),
                                MQT_NAMED_BUILDER(multipleControlledDcx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRECROpTest, QIRTest,
    testing::Values(QIRTestCase{"ECR", MQT_NAMED_BUILDER(ecr<>),
                                MQT_NAMED_BUILDER(ecr<>)},
                    QIRTestCase{"SingleControlledECR",
                                MQT_NAMED_BUILDER(singleControlledEcr<>),
                                MQT_NAMED_BUILDER(singleControlledEcr<>)},
                    QIRTestCase{"MultipleControlledECR",
                                MQT_NAMED_BUILDER(multipleControlledEcr<>),
                                MQT_NAMED_BUILDER(multipleControlledEcr<>)}));
/// @}

/// \name QIR/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QIRGPhaseOpTest, QIRTest,
                         testing::Values(QIRTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(globalPhase<>),
                             MQT_NAMED_BUILDER(globalPhase<>)}));
/// @}

/// \name QIR/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRHOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"H", MQT_NAMED_BUILDER(h<>), MQT_NAMED_BUILDER(h<>)},
        QIRTestCase{"SingleControlledH", MQT_NAMED_BUILDER(singleControlledH<>),
                    MQT_NAMED_BUILDER(singleControlledH<>)},
        QIRTestCase{"MultipleControlledH",
                    MQT_NAMED_BUILDER(multipleControlledH<>),
                    MQT_NAMED_BUILDER(multipleControlledH<>)}));
/// @}

/// \name QIR/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRIDOpTest, QIRTest,
    testing::Values(QIRTestCase{"Identity", MQT_NAMED_BUILDER(identity<>),
                                MQT_NAMED_BUILDER(identity<>)},
                    QIRTestCase{"SingleControlledIdentity",
                                MQT_NAMED_BUILDER(singleControlledIdentity<>),
                                MQT_NAMED_BUILDER(singleControlledIdentity<>)},
                    QIRTestCase{
                        "MultipleControlledIdentity",
                        MQT_NAMED_BUILDER(multipleControlledIdentity<>),
                        MQT_NAMED_BUILDER(multipleControlledIdentity<>)}));
/// @}

/// \name QIR/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRiSWAPOpTest, QIRTest,
    testing::Values(QIRTestCase{"iSWAP", MQT_NAMED_BUILDER(iswap<>),
                                MQT_NAMED_BUILDER(iswap<>)},
                    QIRTestCase{"SingleControllediSWAP",
                                MQT_NAMED_BUILDER(singleControlledIswap<>),
                                MQT_NAMED_BUILDER(singleControlledIswap<>)},
                    QIRTestCase{"MultipleControllediSWAP",
                                MQT_NAMED_BUILDER(multipleControlledIswap<>),
                                MQT_NAMED_BUILDER(multipleControlledIswap<>)}));
/// @}

/// \name QIR/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRPOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"P", MQT_NAMED_BUILDER(p<>), MQT_NAMED_BUILDER(p<>)},
        QIRTestCase{"SingleControlledP", MQT_NAMED_BUILDER(singleControlledP<>),
                    MQT_NAMED_BUILDER(singleControlledP<>)},
        QIRTestCase{"MultipleControlledP",
                    MQT_NAMED_BUILDER(multipleControlledP<>),
                    MQT_NAMED_BUILDER(multipleControlledP<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RCCXOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRCCXOpTest, QIRTest,
    testing::Values(QIRTestCase{"RCCX", MQT_NAMED_BUILDER(rccx<>),
                                MQT_NAMED_BUILDER(rccx<>)},
                    QIRTestCase{"SingleControlledRCCX",
                                MQT_NAMED_BUILDER(singleControlledRccx<>),
                                MQT_NAMED_BUILDER(singleControlledRccx<>)},
                    QIRTestCase{"MultipleControlledRCCX",
                                MQT_NAMED_BUILDER(multipleControlledRccx<>),
                                MQT_NAMED_BUILDER(multipleControlledRccx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRROpTest, QIRTest,
    testing::Values(
        QIRTestCase{"R", MQT_NAMED_BUILDER(r<>), MQT_NAMED_BUILDER(r<>)},
        QIRTestCase{"SingleControlledR", MQT_NAMED_BUILDER(singleControlledR<>),
                    MQT_NAMED_BUILDER(singleControlledR<>)},
        QIRTestCase{"MultipleControlledR",
                    MQT_NAMED_BUILDER(multipleControlledR<>),
                    MQT_NAMED_BUILDER(multipleControlledR<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRXOpTest, QIRTest,
    testing::Values(QIRTestCase{"RX", MQT_NAMED_BUILDER(rx<>),
                                MQT_NAMED_BUILDER(rx<>)},
                    QIRTestCase{"SingleControlledRX",
                                MQT_NAMED_BUILDER(singleControlledRx<>),
                                MQT_NAMED_BUILDER(singleControlledRx<>)},
                    QIRTestCase{"MultipleControlledRX",
                                MQT_NAMED_BUILDER(multipleControlledRx<>),
                                MQT_NAMED_BUILDER(multipleControlledRx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRXXOpTest, QIRTest,
    testing::Values(QIRTestCase{"RXX", MQT_NAMED_BUILDER(rxx<>),
                                MQT_NAMED_BUILDER(rxx<>)},
                    QIRTestCase{"SingleControlledRXX",
                                MQT_NAMED_BUILDER(singleControlledRxx<>),
                                MQT_NAMED_BUILDER(singleControlledRxx<>)},
                    QIRTestCase{"MultipleControlledRXX",
                                MQT_NAMED_BUILDER(multipleControlledRxx<>),
                                MQT_NAMED_BUILDER(multipleControlledRxx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRYOpTest, QIRTest,
    testing::Values(QIRTestCase{"RY", MQT_NAMED_BUILDER(ry<>),
                                MQT_NAMED_BUILDER(ry<>)},
                    QIRTestCase{"SingleControlledRY",
                                MQT_NAMED_BUILDER(singleControlledRy<>),
                                MQT_NAMED_BUILDER(singleControlledRy<>)},
                    QIRTestCase{"MultipleControlledRY",
                                MQT_NAMED_BUILDER(multipleControlledRy<>),
                                MQT_NAMED_BUILDER(multipleControlledRy<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRYYOpTest, QIRTest,
    testing::Values(QIRTestCase{"RYY", MQT_NAMED_BUILDER(ryy<>),
                                MQT_NAMED_BUILDER(ryy<>)},
                    QIRTestCase{"SingleControlledRYY",
                                MQT_NAMED_BUILDER(singleControlledRyy<>),
                                MQT_NAMED_BUILDER(singleControlledRyy<>)},
                    QIRTestCase{"MultipleControlledRYY",
                                MQT_NAMED_BUILDER(multipleControlledRyy<>),
                                MQT_NAMED_BUILDER(multipleControlledRyy<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRZOpTest, QIRTest,
    testing::Values(QIRTestCase{"RZ", MQT_NAMED_BUILDER(rz<>),
                                MQT_NAMED_BUILDER(rz<>)},
                    QIRTestCase{"SingleControlledRZ",
                                MQT_NAMED_BUILDER(singleControlledRz<>),
                                MQT_NAMED_BUILDER(singleControlledRz<>)},
                    QIRTestCase{"MultipleControlledRZ",
                                MQT_NAMED_BUILDER(multipleControlledRz<>),
                                MQT_NAMED_BUILDER(multipleControlledRz<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRZXOpTest, QIRTest,
    testing::Values(QIRTestCase{"RZX", MQT_NAMED_BUILDER(rzx<>),
                                MQT_NAMED_BUILDER(rzx<>)},
                    QIRTestCase{"SingleControlledRZX",
                                MQT_NAMED_BUILDER(singleControlledRzx<>),
                                MQT_NAMED_BUILDER(singleControlledRzx<>)},
                    QIRTestCase{"MultipleControlledRZX",
                                MQT_NAMED_BUILDER(multipleControlledRzx<>),
                                MQT_NAMED_BUILDER(multipleControlledRzx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRZZOpTest, QIRTest,
    testing::Values(QIRTestCase{"RZZ", MQT_NAMED_BUILDER(rzz<>),
                                MQT_NAMED_BUILDER(rzz<>)},
                    QIRTestCase{"SingleControlledRZZ",
                                MQT_NAMED_BUILDER(singleControlledRzz<>),
                                MQT_NAMED_BUILDER(singleControlledRzz<>)},
                    QIRTestCase{"MultipleControlledRZZ",
                                MQT_NAMED_BUILDER(multipleControlledRzz<>),
                                MQT_NAMED_BUILDER(multipleControlledRzz<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"S", MQT_NAMED_BUILDER(s<>), MQT_NAMED_BUILDER(s<>)},
        QIRTestCase{"SingleControlledS", MQT_NAMED_BUILDER(singleControlledS<>),
                    MQT_NAMED_BUILDER(singleControlledS<>)},
        QIRTestCase{"MultipleControlledS",
                    MQT_NAMED_BUILDER(multipleControlledS<>),
                    MQT_NAMED_BUILDER(multipleControlledS<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSdgOpTest, QIRTest,
    testing::Values(QIRTestCase{"Sdg", MQT_NAMED_BUILDER(sdg<>),
                                MQT_NAMED_BUILDER(sdg<>)},
                    QIRTestCase{"SingleControlledSdg",
                                MQT_NAMED_BUILDER(singleControlledSdg<>),
                                MQT_NAMED_BUILDER(singleControlledSdg<>)},
                    QIRTestCase{"MultipleControlledSdg",
                                MQT_NAMED_BUILDER(multipleControlledSdg<>),
                                MQT_NAMED_BUILDER(multipleControlledSdg<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSWAPOpTest, QIRTest,
    testing::Values(QIRTestCase{"SWAP", MQT_NAMED_BUILDER(swap<>),
                                MQT_NAMED_BUILDER(swap<>)},
                    QIRTestCase{"SingleControlledSWAP",
                                MQT_NAMED_BUILDER(singleControlledSwap<>),
                                MQT_NAMED_BUILDER(singleControlledSwap<>)},
                    QIRTestCase{"MultipleControlledSWAP",
                                MQT_NAMED_BUILDER(multipleControlledSwap<>),
                                MQT_NAMED_BUILDER(multipleControlledSwap<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSXOpTest, QIRTest,
    testing::Values(QIRTestCase{"SX", MQT_NAMED_BUILDER(sx<>),
                                MQT_NAMED_BUILDER(sx<>)},
                    QIRTestCase{"SingleControlledSX",
                                MQT_NAMED_BUILDER(singleControlledSx<>),
                                MQT_NAMED_BUILDER(singleControlledSx<>)},
                    QIRTestCase{"MultipleControlledSX",
                                MQT_NAMED_BUILDER(multipleControlledSx<>),
                                MQT_NAMED_BUILDER(multipleControlledSx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSXdgOpTest, QIRTest,
    testing::Values(QIRTestCase{"SXdg", MQT_NAMED_BUILDER(sxdg<>),
                                MQT_NAMED_BUILDER(sxdg<>)},
                    QIRTestCase{"SingleControlledSXdg",
                                MQT_NAMED_BUILDER(singleControlledSxdg<>),
                                MQT_NAMED_BUILDER(singleControlledSxdg<>)},
                    QIRTestCase{"MultipleControlledSXdg",
                                MQT_NAMED_BUILDER(multipleControlledSxdg<>),
                                MQT_NAMED_BUILDER(multipleControlledSxdg<>)}));
/// @}

/// \name QIR/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRTOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"T", MQT_NAMED_BUILDER(t_<>), MQT_NAMED_BUILDER(t_<>)},
        QIRTestCase{"SingleControlledT", MQT_NAMED_BUILDER(singleControlledT<>),
                    MQT_NAMED_BUILDER(singleControlledT<>)},
        QIRTestCase{"MultipleControlledT",
                    MQT_NAMED_BUILDER(multipleControlledT<>),
                    MQT_NAMED_BUILDER(multipleControlledT<>)}));
/// @}

/// \name QIR/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRTdgOpTest, QIRTest,
    testing::Values(QIRTestCase{"Tdg", MQT_NAMED_BUILDER(tdg<>),
                                MQT_NAMED_BUILDER(tdg<>)},
                    QIRTestCase{"SingleControlledTdg",
                                MQT_NAMED_BUILDER(singleControlledTdg<>),
                                MQT_NAMED_BUILDER(singleControlledTdg<>)},
                    QIRTestCase{"MultipleControlledTdg",
                                MQT_NAMED_BUILDER(multipleControlledTdg<>),
                                MQT_NAMED_BUILDER(multipleControlledTdg<>)}));
/// @}

/// \name QIR/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRU2OpTest, QIRTest,
    testing::Values(QIRTestCase{"U2", MQT_NAMED_BUILDER(u2<>),
                                MQT_NAMED_BUILDER(u2<>)},
                    QIRTestCase{"SingleControlledU2",
                                MQT_NAMED_BUILDER(singleControlledU2<>),
                                MQT_NAMED_BUILDER(singleControlledU2<>)},
                    QIRTestCase{"MultipleControlledU2",
                                MQT_NAMED_BUILDER(multipleControlledU2<>),
                                MQT_NAMED_BUILDER(multipleControlledU2<>)}));
/// @}

/// \name QIR/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRUOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"U", MQT_NAMED_BUILDER(u<>), MQT_NAMED_BUILDER(u<>)},
        QIRTestCase{"SingleControlledU", MQT_NAMED_BUILDER(singleControlledU<>),
                    MQT_NAMED_BUILDER(singleControlledU<>)},
        QIRTestCase{"MultipleControlledU",
                    MQT_NAMED_BUILDER(multipleControlledU<>),
                    MQT_NAMED_BUILDER(multipleControlledU<>)}));
/// @}

/// \name QIR/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRXOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"X", MQT_NAMED_BUILDER(x<>), MQT_NAMED_BUILDER(x<>)},
        QIRTestCase{"SingleControlledX", MQT_NAMED_BUILDER(singleControlledX<>),
                    MQT_NAMED_BUILDER(singleControlledX<>)},
        QIRTestCase{"MultipleControlledX",
                    MQT_NAMED_BUILDER(multipleControlledX<>),
                    MQT_NAMED_BUILDER(multipleControlledX<>)}));
/// @}

/// \name QIR/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRXXMinusYYOpTest, QIRTest,
    testing::Values(QIRTestCase{"XXMinusYY", MQT_NAMED_BUILDER(xxMinusYY<>),
                                MQT_NAMED_BUILDER(xxMinusYY<>)},
                    QIRTestCase{"SingleControlledXXMinusYY",
                                MQT_NAMED_BUILDER(singleControlledXxMinusYY<>),
                                MQT_NAMED_BUILDER(singleControlledXxMinusYY<>)},
                    QIRTestCase{
                        "MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(multipleControlledXxMinusYY<>),
                        MQT_NAMED_BUILDER(multipleControlledXxMinusYY<>)}));
/// @}

/// \name QIR/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRXXPlusYYOpTest, QIRTest,
    testing::Values(QIRTestCase{"XXPlusYY", MQT_NAMED_BUILDER(xxPlusYY<>),
                                MQT_NAMED_BUILDER(xxPlusYY<>)},
                    QIRTestCase{"SingleControlledXXPlusYY",
                                MQT_NAMED_BUILDER(singleControlledXxPlusYY<>),
                                MQT_NAMED_BUILDER(singleControlledXxPlusYY<>)},
                    QIRTestCase{
                        "MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(multipleControlledXxPlusYY<>),
                        MQT_NAMED_BUILDER(multipleControlledXxPlusYY<>)}));
/// @}

/// \name QIR/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRYOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"Y", MQT_NAMED_BUILDER(y<>), MQT_NAMED_BUILDER(y<>)},
        QIRTestCase{"SingleControlledY", MQT_NAMED_BUILDER(singleControlledY<>),
                    MQT_NAMED_BUILDER(singleControlledY<>)},
        QIRTestCase{"MultipleControlledY",
                    MQT_NAMED_BUILDER(multipleControlledY<>),
                    MQT_NAMED_BUILDER(multipleControlledY<>)}));
/// @}

/// \name QIR/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRZOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"Z", MQT_NAMED_BUILDER(z<>), MQT_NAMED_BUILDER(z<>)},
        QIRTestCase{"SingleControlledZ", MQT_NAMED_BUILDER(singleControlledZ<>),
                    MQT_NAMED_BUILDER(singleControlledZ<>)},
        QIRTestCase{"MultipleControlledZ",
                    MQT_NAMED_BUILDER(multipleControlledZ<>),
                    MQT_NAMED_BUILDER(multipleControlledZ<>)}));
/// @}

/// \name QIR/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRMeasureOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"SingleMeasurementToSingleBit",
                    MQT_NAMED_BUILDER(singleMeasurementToSingleBit),
                    MQT_NAMED_BUILDER(singleMeasurementToSingleBit)},
        QIRTestCase{"RepeatedMeasurementToSameBit",
                    MQT_NAMED_BUILDER(repeatedMeasurementToSameBit),
                    MQT_NAMED_BUILDER(repeatedMeasurementToSameBit)},
        QIRTestCase{"RepeatedMeasurementToDifferentBits",
                    MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits),
                    MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits)}));
/// @}

/// \name QIR/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRResetOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"ResetQubitWithoutOp",
                    MQT_NAMED_BUILDER(resetQubitWithoutOp<>),
                    MQT_NAMED_BUILDER(resetQubitWithoutOp<>)},
        QIRTestCase{"ResetMultipleQubitsWithoutOp",
                    MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp<>),
                    MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp<>)},
        QIRTestCase{"RepeatedResetWithoutOp",
                    MQT_NAMED_BUILDER(repeatedResetWithoutOp<>),
                    MQT_NAMED_BUILDER(repeatedResetWithoutOp<>)},
        QIRTestCase{"ResetQubitAfterSingleOp",
                    MQT_NAMED_BUILDER(resetQubitAfterSingleOp<>),
                    MQT_NAMED_BUILDER(resetQubitAfterSingleOp<>)},
        QIRTestCase{"ResetMultipleQubitsAfterSingleOp",
                    MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp<>),
                    MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp<>)},
        QIRTestCase{"RepeatedResetAfterSingleOp",
                    MQT_NAMED_BUILDER(repeatedResetAfterSingleOp<>),
                    MQT_NAMED_BUILDER(repeatedResetAfterSingleOp<>)}));
/// @}

/// \name QIR/QubitManagement/QubitManagement.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRQubitManagementTest, QIRTest,
    testing::Values(
        QIRTestCase{"AllocQubit", MQT_NAMED_BUILDER(allocQubit<>),
                    MQT_NAMED_BUILDER(allocQubit<>)},
        QIRTestCase{"AllocQubitRegister",
                    MQT_NAMED_BUILDER(allocQubitRegister<>),
                    MQT_NAMED_BUILDER(allocQubitRegister<>)},
        QIRTestCase{"AllocMultipleQubitRegisters",
                    MQT_NAMED_BUILDER(allocMultipleQubitRegisters<>),
                    MQT_NAMED_BUILDER(allocMultipleQubitRegisters<>)},
        QIRTestCase{"AllocLargeRegister",
                    MQT_NAMED_BUILDER(allocLargeRegister<>),
                    MQT_NAMED_BUILDER(allocLargeRegister<>)},
        QIRTestCase{"StaticQubits", MQT_NAMED_BUILDER(staticQubits),
                    MQT_NAMED_BUILDER(staticQubits)},
        QIRTestCase{"StaticQubitsWithOps",
                    MQT_NAMED_BUILDER(staticQubitsWithOps),
                    MQT_NAMED_BUILDER(staticQubitsWithOps)},
        QIRTestCase{"StaticQubitsWithParametricOps",
                    MQT_NAMED_BUILDER(staticQubitsWithParametricOps),
                    MQT_NAMED_BUILDER(staticQubitsWithParametricOps)},
        QIRTestCase{"StaticQubitsWithTwoTargetOps",
                    MQT_NAMED_BUILDER(staticQubitsWithTwoTargetOps),
                    MQT_NAMED_BUILDER(staticQubitsWithTwoTargetOps)},
        QIRTestCase{"StaticQubitsWithCtrl",
                    MQT_NAMED_BUILDER(staticQubitsWithCtrl),
                    MQT_NAMED_BUILDER(staticQubitsWithCtrl)},
        QIRTestCase{"StaticQubitsWithInv",
                    MQT_NAMED_BUILDER(staticQubitsWithInv),
                    MQT_NAMED_BUILDER(staticQubitsWithInv)},
        QIRTestCase{"StaticQubitsWithDuplicates",
                    MQT_NAMED_BUILDER(staticQubitsWithDuplicates),
                    MQT_NAMED_BUILDER(staticQubitsCanonical)}));
/// @}
