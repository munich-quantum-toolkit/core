/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/OQ3ToQC/OQ3ToQC.h"
#include "mlir/Dialect/OQ3/IR/GateCatalog.h"
#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>

using namespace mlir;

namespace {

class OQ3Test : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<oq3::OQ3Dialect, qc::QCDialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  static oq3::ApplyGateOp
  createApplication(OpBuilder& builder, Location loc, StringRef name,
                    ValueRange parameters, ValueRange qubits,
                    ValueRange modifierOperands = {},
                    ArrayRef<int32_t> modifierKinds = {},
                    ArrayRef<int32_t> modifierIndices = {}) {
    OperationState state(loc, oq3::ApplyGateOp::getOperationName());
    state.addOperands(parameters);
    state.addOperands(qubits);
    state.addOperands(modifierOperands);
    state.addAttribute("callee",
                       FlatSymbolRefAttr::get(builder.getContext(), name));
    state.addAttribute(
        "modifier_kinds",
        DenseI32ArrayAttr::get(builder.getContext(), modifierKinds));
    state.addAttribute(
        "modifier_operand_indices",
        DenseI32ArrayAttr::get(builder.getContext(), modifierIndices));
    state.addAttribute("operandSegmentSizes",
                       DenseI32ArrayAttr::get(
                           builder.getContext(),
                           {static_cast<int32_t>(parameters.size()),
                            static_cast<int32_t>(qubits.size()),
                            static_cast<int32_t>(modifierOperands.size())}));
    return cast<oq3::ApplyGateOp>(builder.create(state));
  }

  OwningOpRef<ModuleOp> buildGateApplication(
      StringRef name, const size_t parameterCount,
      const size_t gateQubitCount = 2, const size_t applicationQubitCount = 2,
      const std::optional<size_t> applicationParameterCount = std::nullopt,
      const ArrayRef<int32_t> modifierKinds = {},
      const ArrayRef<int32_t> modifierIndices = {},
      const ArrayRef<Type> modifierTypes = {}) {
    OpBuilder builder(context.get());
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    SmallVector<Type> gateInputs(parameterCount, builder.getF64Type());
    gateInputs.append(gateQubitCount, qc::QubitType::get(context.get()));
    oq3::GateDeclOp::create(builder, loc, name,
                            builder.getFunctionType(gateInputs, {}));

    auto qubitType = qc::QubitType::get(context.get());
    SmallVector<Type> functionInputs(applicationQubitCount, qubitType);
    auto function = func::FuncOp::create(
        builder, loc, "main", builder.getFunctionType(functionInputs, {}));
    Block* entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    SmallVector<Value> parameters;
    const size_t emittedParameterCount =
        applicationParameterCount.value_or(parameterCount);
    parameters.reserve(emittedParameterCount);
    for (size_t i = 0; i < emittedParameterCount; ++i) {
      parameters.push_back(
          arith::ConstantFloatOp::create(builder, loc, builder.getF64Type(),
                                         APFloat(static_cast<double>(i + 1))));
    }

    SmallVector<Value> modifierOperands;
    modifierOperands.reserve(modifierTypes.size());
    for (auto type : modifierTypes) {
      if (auto integerType = dyn_cast<IntegerType>(type)) {
        modifierOperands.push_back(
            arith::ConstantIntOp::create(builder, loc, integerType, 1));
      } else {
        modifierOperands.push_back(arith::ConstantFloatOp::create(
            builder, loc, cast<FloatType>(type), APFloat(1.0)));
      }
    }

    createApplication(builder, loc, name, parameters, entry->getArguments(),
                      modifierOperands, modifierKinds, modifierIndices);
    func::ReturnOp::create(builder, loc);
    return OwningOpRef<ModuleOp>(module);
  }

  std::unique_ptr<MLIRContext> context;
};

TEST_F(OQ3Test, LowersControlledUGateFamiliesNatively) {
  struct GateCase {
    StringRef name;
    size_t parameterCount;
    size_t expectedPhases;
    size_t expectedUniversals;
  };
  constexpr GateCase cases[] = {
      {"cu", 4, 1, 1}, {"cu1", 1, 1, 0}, {"cu3", 3, 0, 1}};

  for (const auto& gate : cases) {
    auto module = buildGateApplication(gate.name, gate.parameterCount);
    ASSERT_TRUE(succeeded(verify(module.get()))) << gate.name.str();

    PassManager manager(context.get());
    manager.addPass(oq3::createOQ3ToQCPass());
    ASSERT_TRUE(succeeded(manager.run(module.get()))) << gate.name.str();
    EXPECT_TRUE(succeeded(verify(module.get()))) << gate.name.str();

    size_t controls = 0;
    size_t phases = 0;
    size_t universals = 0;
    module->walk([&](Operation* operation) {
      controls += isa<qc::CtrlOp>(operation);
      phases += isa<qc::POp>(operation);
      universals += isa<qc::UOp>(operation);
    });
    EXPECT_EQ(controls, 1) << gate.name.str();
    EXPECT_EQ(phases, gate.expectedPhases) << gate.name.str();
    EXPECT_EQ(universals, gate.expectedUniversals) << gate.name.str();
  }
}

TEST_F(OQ3Test, PreservesInverseNativeGateAliases) {
  auto module = buildGateApplication("iswapdg", 0);
  ASSERT_TRUE(succeeded(verify(module.get())));

  PassManager manager(context.get());
  manager.addPass(oq3::createOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(module.get())));
  EXPECT_TRUE(succeeded(verify(module.get())));

  size_t inverses = 0;
  size_t swaps = 0;
  module->walk([&](Operation* operation) {
    inverses += isa<qc::InvOp>(operation);
    swaps += isa<qc::iSWAPOp>(operation);
  });
  EXPECT_EQ(inverses, 1);
  EXPECT_EQ(swaps, 1);
}

TEST_F(OQ3Test, LowersEveryCanonicalGateCatalogEntry) {
  for (const auto& gate : oq3::getGateCatalog()) {
    auto module = buildGateApplication(gate.name, gate.parameterCount,
                                       gate.qubitCount(), gate.qubitCount());
    ASSERT_TRUE(succeeded(verify(module.get()))) << gate.name.str();

    PassManager manager(context.get());
    manager.addPass(oq3::createOQ3ToQCPass());
    ASSERT_TRUE(succeeded(manager.run(module.get()))) << gate.name.str();
    EXPECT_TRUE(succeeded(verify(module.get()))) << gate.name.str();
  }
}

TEST_F(OQ3Test, LowersControlledCustomGateBodies) {
  OpBuilder builder(context.get());
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  auto qubitType = qc::QubitType::get(context.get());
  oq3::GateDeclOp::create(builder, loc, "x",
                          builder.getFunctionType({qubitType}, {}));
  auto custom = oq3::GateOp::create(builder, loc, "custom",
                                    builder.getFunctionType({qubitType}, {}));
  auto* gateBody = new Block();
  custom.getBody().push_back(gateBody);
  gateBody->addArgument(qubitType, loc);
  builder.setInsertionPointToStart(gateBody);
  createApplication(builder, loc, "x", {}, gateBody->getArguments());
  oq3::YieldOp::create(builder, loc);

  builder.setInsertionPointToEnd(module.getBody());
  auto function =
      func::FuncOp::create(builder, loc, "main",
                           builder.getFunctionType({qubitType, qubitType}, {}));
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  constexpr auto ctrl = static_cast<int32_t>(oq3::GateModifierKind::ctrl);
  createApplication(builder, loc, "custom", {}, entry->getArguments(), {},
                    {ctrl}, {-1});
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(succeeded(verify(module)));
  PassManager manager(context.get());
  manager.addPass(oq3::createOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(module)));
  EXPECT_TRUE(succeeded(verify(module)));

  size_t controls = 0;
  size_t gates = 0;
  module.walk([&](Operation* operation) {
    controls += isa<qc::CtrlOp>(operation);
    gates += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(controls, 1);
  EXPECT_EQ(gates, 1);
}

TEST_F(OQ3Test, RejectsSurplusQubitsWithoutControlModifiers) {
  auto module = buildGateApplication("x", 0, 1);
  EXPECT_TRUE(failed(verify(module.get())));
}

TEST_F(OQ3Test, RejectsMalformedGateContracts) {
  OpBuilder builder(context.get());
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  auto returningGate = oq3::GateDeclOp::create(
      builder, loc, "returning",
      builder.getFunctionType({}, {builder.getI1Type()}));
  EXPECT_TRUE(failed(returningGate.verify()));

  auto untypedDeclaration = oq3::GateDeclOp::create(
      builder, loc, "untyped_declaration", builder.getI64Type());
  EXPECT_TRUE(failed(untypedDeclaration.verify()));

  auto misorderedGate = oq3::GateDeclOp::create(
      builder, loc, "misordered",
      builder.getFunctionType(
          {qc::QubitType::get(context.get()), builder.getF64Type()}, {}));
  EXPECT_TRUE(failed(misorderedGate.verify()));

  auto indexParameter = oq3::GateDeclOp::create(
      builder, loc, "index_parameter",
      builder.getFunctionType({builder.getIndexType()}, {}));
  EXPECT_TRUE(failed(indexParameter.verify()));

  auto memrefParameter = oq3::GateDeclOp::create(
      builder, loc, "memref_parameter",
      builder.getFunctionType({MemRefType::get({2}, builder.getI1Type())}, {}));
  EXPECT_TRUE(failed(memrefParameter.verify()));

  auto emptyGate = oq3::GateOp::create(builder, loc, "empty",
                                       builder.getFunctionType({}, {}));
  EXPECT_TRUE(failed(emptyGate.verify()));

  auto untypedGate =
      oq3::GateOp::create(builder, loc, "untyped", builder.getI64Type());
  EXPECT_TRUE(failed(untypedGate.verify()));

  auto gateType = builder.getFunctionType({builder.getF64Type()}, {});
  auto gate = oq3::GateOp::create(builder, loc, "mismatched", gateType);
  auto* body = new Block();
  gate.getBody().push_back(body);
  body->addArgument(builder.getI64Type(), loc);
  EXPECT_TRUE(failed(gate.verify()));

  auto returningDefinition =
      oq3::GateOp::create(builder, loc, "returning_definition",
                          builder.getFunctionType({}, {builder.getI1Type()}));
  returningDefinition.getBody().emplaceBlock();
  EXPECT_TRUE(failed(returningDefinition.verify()));

  const auto qubitType = qc::QubitType::get(context.get());
  auto nonUnitary = oq3::GateOp::create(
      builder, loc, "non_unitary", builder.getFunctionType({qubitType}, {}));
  auto* nonUnitaryBody = new Block();
  nonUnitary.getBody().push_back(nonUnitaryBody);
  nonUnitaryBody->addArgument(qubitType, loc);
  builder.setInsertionPointToStart(nonUnitaryBody);
  qc::MeasureOp::create(builder, loc, nonUnitaryBody->getArgument(0));
  qc::ResetOp::create(builder, loc, nonUnitaryBody->getArgument(0));
  oq3::YieldOp::create(builder, loc);
  EXPECT_TRUE(failed(nonUnitary.verify()));

  builder.setInsertionPointToEnd(module.getBody());
  auto manufactured = oq3::GateOp::create(builder, loc, "manufactured",
                                          builder.getFunctionType({}, {}));
  auto* manufacturedBody = new Block();
  manufactured.getBody().push_back(manufacturedBody);
  builder.setInsertionPointToStart(manufacturedBody);
  qc::StaticOp::create(builder, loc, 0);
  oq3::YieldOp::create(builder, loc);
  EXPECT_TRUE(failed(manufactured.verify()));
}

TEST_F(OQ3Test, RejectsMalformedGateApplications) {
  OpBuilder builder(context.get());
  constexpr auto inv = static_cast<int32_t>(oq3::GateModifierKind::inv);
  constexpr auto ctrl = static_cast<int32_t>(oq3::GateModifierKind::ctrl);
  constexpr auto pow = static_cast<int32_t>(oq3::GateModifierKind::pow);

  auto noncanonicalCatalogDeclaration = buildGateApplication("x", 0, 2, 2);
  EXPECT_TRUE(failed(verify(noncanonicalCatalogDeclaration.get())));

  auto unknown = buildGateApplication("x", 0, 1, 1);
  oq3::ApplyGateOp unknownApplication;
  unknown->walk(
      [&](oq3::ApplyGateOp application) { unknownApplication = application; });
  ASSERT_TRUE(unknownApplication);
  unknownApplication.setCalleeAttr(
      FlatSymbolRefAttr::get(context.get(), "missing"));
  EXPECT_TRUE(failed(unknownApplication.verify()));

  auto invalidReferencedGate = buildGateApplication("x", 0, 1, 1);
  auto referencedDeclaration =
      *invalidReferencedGate->getOps<oq3::GateDeclOp>().begin();
  referencedDeclaration->setAttr("function_type",
                                 TypeAttr::get(builder.getI64Type()));
  oq3::ApplyGateOp applicationWithInvalidDeclaration;
  invalidReferencedGate->walk([&](oq3::ApplyGateOp application) {
    applicationWithInvalidDeclaration = application;
  });
  EXPECT_TRUE(failed(applicationWithInvalidDeclaration.verify()));

  auto badParameters = buildGateApplication("rx", 1, 1, 1, 0);
  EXPECT_TRUE(failed(verify(badParameters.get())));

  auto mismatchedModifierArrays =
      buildGateApplication("x", 0, 1, 1, std::nullopt, {inv}, {});
  EXPECT_TRUE(failed(verify(mismatchedModifierArrays.get())));

  auto unknownModifier =
      buildGateApplication("x", 0, 1, 1, std::nullopt, {99}, {-1});
  EXPECT_TRUE(failed(verify(unknownModifier.get())));

  auto invWithOperand = buildGateApplication("x", 0, 1, 1, std::nullopt, {inv},
                                             {0}, {builder.getI64Type()});
  EXPECT_TRUE(failed(verify(invWithOperand.get())));

  auto powWithoutOperand =
      buildGateApplication("x", 0, 1, 1, std::nullopt, {pow}, {-1});
  EXPECT_TRUE(failed(verify(powWithoutOperand.get())));

  auto outOfBoundsOperand = buildGateApplication(
      "x", 0, 1, 2, std::nullopt, {ctrl}, {1}, {builder.getI64Type()});
  EXPECT_TRUE(failed(verify(outOfBoundsOperand.get())));

  auto reusedOperand = buildGateApplication(
      "x", 0, 1, 3, std::nullopt, {ctrl, ctrl}, {0, 0}, {builder.getI64Type()});
  EXPECT_TRUE(failed(verify(reusedOperand.get())));

  auto unreferencedOperand = buildGateApplication(
      "x", 0, 1, 2, std::nullopt, {ctrl}, {-1}, {builder.getI64Type()});
  EXPECT_TRUE(failed(verify(unreferencedOperand.get())));

  auto nonIntegerControl = buildGateApplication(
      "x", 0, 1, 2, std::nullopt, {ctrl}, {0}, {builder.getF64Type()});
  EXPECT_TRUE(failed(verify(nonIntegerControl.get())));

  auto nonPositiveControl = buildGateApplication(
      "x", 0, 1, 2, std::nullopt, {ctrl}, {0}, {builder.getI64Type()});
  nonPositiveControl->walk([&](arith::ConstantIntOp constant) {
    constant->setAttr("value", builder.getI64IntegerAttr(0));
  });
  EXPECT_TRUE(failed(verify(nonPositiveControl.get())));

  auto twoControls = buildGateApplication("x", 0, 1, 3, std::nullopt, {ctrl},
                                          {0}, {builder.getI64Type()});
  twoControls->walk([&](arith::ConstantIntOp constant) {
    constant->setAttr("value", builder.getI64IntegerAttr(2));
  });
  EXPECT_TRUE(succeeded(verify(twoControls.get())));

  auto overflowingControls = buildGateApplication(
      "x", 0, 1, 1, std::nullopt, {ctrl, ctrl, ctrl}, {0, 1, 2},
      {builder.getI64Type(), builder.getI64Type(), builder.getI64Type()});
  constexpr std::array CONTROL_COUNTS{std::numeric_limits<int64_t>::max(),
                                      std::numeric_limits<int64_t>::max(),
                                      int64_t{2}};
  std::size_t controlIndex = 0;
  overflowingControls->walk([&](arith::ConstantIntOp constant) {
    constant->setAttr(
        "value", builder.getI64IntegerAttr(CONTROL_COUNTS[controlIndex++]));
  });
  EXPECT_TRUE(failed(verify(overflowingControls.get())));

  auto mixedControlCounts = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(mixedControlCounts.getBody());
  auto qubitType = qc::QubitType::get(context.get());
  oq3::GateDeclOp::create(builder, builder.getUnknownLoc(), "x",
                          builder.getFunctionType({qubitType}, {}));
  auto function = func::FuncOp::create(
      builder, builder.getUnknownLoc(), "mixed_control_counts",
      builder.getFunctionType(
          {builder.getI64Type(), qubitType, qubitType, qubitType}, {}));
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto constantTwo = arith::ConstantIntOp::create(
      builder, builder.getUnknownLoc(), builder.getI64Type(), 2);
  createApplication(builder, builder.getUnknownLoc(), "x", {},
                    entry->getArguments().drop_front(),
                    ValueRange{constantTwo.getResult(), entry->getArgument(0)},
                    {ctrl, ctrl}, {0, 1});
  func::ReturnOp::create(builder, builder.getUnknownLoc());
  EXPECT_TRUE(failed(verify(mixedControlCounts)));

  auto tooFewQubits =
      buildGateApplication("x", 0, 1, 1, std::nullopt, {ctrl}, {-1});
  EXPECT_TRUE(failed(verify(tooFewQubits.get())));

  auto surplusQubits =
      buildGateApplication("x", 0, 1, 3, std::nullopt, {ctrl}, {-1});
  EXPECT_TRUE(failed(verify(surplusQubits.get())));

  auto duplicateQubits = buildGateApplication("cx", 0, 2, 2);
  oq3::ApplyGateOp duplicateApplication;
  duplicateQubits->walk([&](oq3::ApplyGateOp application) {
    duplicateApplication = application;
  });
  ASSERT_TRUE(duplicateApplication);
  duplicateApplication->setOperand(1, duplicateApplication.getQubits().front());
  EXPECT_TRUE(failed(verify(duplicateQubits.get())));
}

TEST_F(OQ3Test, RejectsKnownPhysicalQubitAliases) {
  OpBuilder builder(context.get());
  auto loc = builder.getUnknownLoc();
  auto qubitType = qc::QubitType::get(context.get());
  auto registerType = MemRefType::get({2}, qubitType);

  const auto buildApplication =
      [&](StringRef functionName, TypeRange functionInputs,
          llvm::function_ref<SmallVector<Value>(OpBuilder&, Block&)>
              createQubits) {
        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToStart(module.getBody());
        oq3::GateDeclOp::create(
            builder, loc, "cx",
            builder.getFunctionType({qubitType, qubitType}, {}));
        auto function =
            func::FuncOp::create(builder, loc, functionName,
                                 builder.getFunctionType(functionInputs, {}));
        auto* entry = function.addEntryBlock();
        builder.setInsertionPointToStart(entry);
        auto qubits = createQubits(builder, *entry);
        createApplication(builder, loc, "cx", {}, qubits);
        func::ReturnOp::create(builder, loc);
        return OwningOpRef<ModuleOp>(module);
      };

  auto duplicateStatic =
      buildApplication("duplicate_static", {}, [&](OpBuilder& nested, Block&) {
        return SmallVector<Value>{
            qc::StaticOp::create(nested, loc, 0).getQubit(),
            qc::StaticOp::create(nested, loc, 0).getQubit()};
      });
  EXPECT_TRUE(failed(verify(duplicateStatic.get())));

  const SmallVector<Type> registerInputs{registerType};
  auto duplicateLoad = buildApplication(
      "duplicate_load", registerInputs, [&](OpBuilder& nested, Block& entry) {
        auto firstIndex = arith::ConstantIndexOp::create(nested, loc, 0);
        auto secondIndex = arith::ConstantIndexOp::create(nested, loc, 0);
        return SmallVector<Value>{
            memref::LoadOp::create(nested, loc, entry.getArgument(0),
                                   ValueRange{firstIndex})
                .getResult(),
            memref::LoadOp::create(nested, loc, entry.getArgument(0),
                                   ValueRange{secondIndex})
                .getResult()};
      });
  EXPECT_TRUE(failed(verify(duplicateLoad.get())));

  const SmallVector<Type> dynamicRegisterInputs{
      registerType, builder.getIndexType(), builder.getIndexType()};
  auto unknownDynamicRelation = buildApplication(
      "unknown_dynamic_relation", dynamicRegisterInputs,
      [&](OpBuilder& nested, Block& entry) {
        return SmallVector<Value>{
            memref::LoadOp::create(nested, loc, entry.getArgument(0),
                                   ValueRange{entry.getArgument(1)})
                .getResult(),
            memref::LoadOp::create(nested, loc, entry.getArgument(0),
                                   ValueRange{entry.getArgument(2)})
                .getResult()};
      });
  EXPECT_TRUE(succeeded(verify(unknownDynamicRelation.get())));
}

TEST_F(OQ3Test, IgnoresUnusedRecursiveAndExponentiallyGrowingGates) {
  OpBuilder builder(context.get());
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto qubitType = qc::QubitType::get(context.get());
  auto gateType = builder.getFunctionType({qubitType}, {});
  oq3::GateDeclOp::create(builder, loc, "x", gateType);

  const auto createGate = [&](StringRef name, StringRef callee,
                              const unsigned applications) {
    builder.setInsertionPointToEnd(module.getBody());
    auto gate = oq3::GateOp::create(builder, loc, name, gateType);
    auto* body = new Block();
    gate.getBody().push_back(body);
    body->addArgument(qubitType, loc);
    builder.setInsertionPointToStart(body);
    for (unsigned i = 0; i < applications; ++i) {
      createApplication(builder, loc, callee, {}, body->getArguments());
    }
    oq3::YieldOp::create(builder, loc);
  };

  createGate("recursive", "recursive", 1);
  createGate("doubling_0", "x", 1);
  for (unsigned i = 1; i < 20; ++i) {
    const auto name = "doubling_" + std::to_string(i);
    const auto callee = "doubling_" + std::to_string(i - 1);
    createGate(name, callee, 2);
  }

  builder.setInsertionPointToEnd(module.getBody());
  auto function = func::FuncOp::create(builder, loc, "main", gateType);
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  createApplication(builder, loc, "x", {}, entry->getArguments());
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(succeeded(verify(module)));
  PassManager manager(context.get());
  manager.addPass(oq3::createOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(module)));
  ASSERT_TRUE(succeeded(verify(module)));
  EXPECT_TRUE(module.getOps<oq3::GateOp>().empty());
}

TEST_F(OQ3Test, RejectsReachableRecursiveCustomGates) {
  OpBuilder builder(context.get());
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto qubitType = qc::QubitType::get(context.get());
  auto gateType = builder.getFunctionType({qubitType}, {});
  auto recursive = oq3::GateOp::create(builder, loc, "recursive", gateType);
  auto* body = new Block();
  recursive.getBody().push_back(body);
  body->addArgument(qubitType, loc);
  builder.setInsertionPointToStart(body);
  createApplication(builder, loc, "recursive", {}, body->getArguments());
  oq3::YieldOp::create(builder, loc);

  builder.setInsertionPointToEnd(module.getBody());
  auto function = func::FuncOp::create(builder, loc, "main", gateType);
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  createApplication(builder, loc, "recursive", {}, entry->getArguments());
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(succeeded(verify(module)));
  PassManager manager(context.get());
  manager.addPass(oq3::createOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(module)));
}

TEST_F(OQ3Test, RejectsReachableIndirectlyRecursiveCustomGates) {
  OpBuilder builder(context.get());
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto qubitType = qc::QubitType::get(context.get());
  auto gateType = builder.getFunctionType({qubitType}, {});

  const auto createGate = [&](StringRef name, StringRef callee) {
    builder.setInsertionPointToEnd(module.getBody());
    auto gate = oq3::GateOp::create(builder, loc, name, gateType);
    auto* body = new Block();
    gate.getBody().push_back(body);
    body->addArgument(qubitType, loc);
    builder.setInsertionPointToStart(body);
    createApplication(builder, loc, callee, {}, body->getArguments());
    oq3::YieldOp::create(builder, loc);
  };

  createGate("first", "second");
  createGate("second", "first");

  builder.setInsertionPointToEnd(module.getBody());
  auto function = func::FuncOp::create(builder, loc, "main", gateType);
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  createApplication(builder, loc, "first", {}, entry->getArguments());
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(succeeded(verify(module)));
  std::string diagnostic;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& value) {
    llvm::raw_string_ostream(diagnostic) << value;
    return success();
  });
  PassManager manager(context.get());
  manager.addPass(oq3::createOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(module)));
  EXPECT_NE(diagnostic.find("recursive custom gates cannot be lowered"),
            std::string::npos);
}

TEST_F(OQ3Test, RejectsReachableExcessiveCustomGateExpansion) {
  OpBuilder builder(context.get());
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto qubitType = qc::QubitType::get(context.get());
  auto gateType = builder.getFunctionType({qubitType}, {});
  oq3::GateDeclOp::create(builder, loc, "x", gateType);

  const auto createGate = [&](StringRef name, StringRef callee,
                              const unsigned applications) {
    builder.setInsertionPointToEnd(module.getBody());
    auto gate = oq3::GateOp::create(builder, loc, name, gateType);
    auto* body = new Block();
    gate.getBody().push_back(body);
    body->addArgument(qubitType, loc);
    builder.setInsertionPointToStart(body);
    for (unsigned i = 0; i < applications; ++i) {
      createApplication(builder, loc, callee, {}, body->getArguments());
    }
    oq3::YieldOp::create(builder, loc);
  };

  createGate("doubling_0", "x", 1);
  for (unsigned i = 1; i < 20; ++i) {
    const auto name = "doubling_" + std::to_string(i);
    const auto callee = "doubling_" + std::to_string(i - 1);
    createGate(name, callee, 2);
  }

  builder.setInsertionPointToEnd(module.getBody());
  auto function = func::FuncOp::create(builder, loc, "main", gateType);
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  createApplication(builder, loc, "doubling_19", {}, entry->getArguments());
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(succeeded(verify(module)));
  std::string diagnostic;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& value) {
    llvm::raw_string_ostream(diagnostic) << value;
    return success();
  });
  PassManager manager(context.get());
  manager.addPass(oq3::createOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(module)));
  EXPECT_NE(
      diagnostic.find("custom-gate expansion exceeds the safe lowering limit"),
      std::string::npos);
}

TEST_F(OQ3Test, RejectsExcessiveModuleWideCustomGateExpansion) {
  OpBuilder builder(context.get());
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto qubitType = qc::QubitType::get(context.get());
  auto gateType = builder.getFunctionType({qubitType}, {});
  oq3::GateDeclOp::create(builder, loc, "x", gateType);

  const auto createGate = [&](StringRef name, StringRef callee,
                              const unsigned applications) {
    builder.setInsertionPointToEnd(module.getBody());
    auto gate = oq3::GateOp::create(builder, loc, name, gateType);
    auto* body = new Block();
    gate.getBody().push_back(body);
    body->addArgument(qubitType, loc);
    builder.setInsertionPointToStart(body);
    for (unsigned i = 0; i < applications; ++i) {
      createApplication(builder, loc, callee, {}, body->getArguments());
    }
    oq3::YieldOp::create(builder, loc);
  };

  createGate("doubling_0", "x", 1);
  for (unsigned i = 1; i <= 14; ++i) {
    const auto name = "doubling_" + std::to_string(i);
    const auto callee = "doubling_" + std::to_string(i - 1);
    createGate(name, callee, 2);
  }

  builder.setInsertionPointToEnd(module.getBody());
  auto function = func::FuncOp::create(builder, loc, "main", gateType);
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  for (unsigned i = 0; i < 3; ++i) {
    createApplication(builder, loc, "doubling_14", {}, entry->getArguments());
  }
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(succeeded(verify(module)));
  std::string diagnostic;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& value) {
    llvm::raw_string_ostream(diagnostic) << value;
    return success();
  });
  PassManager manager(context.get());
  manager.addPass(oq3::createOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(module)));
  EXPECT_NE(diagnostic.find("module custom-gate expansion exceeds"),
            std::string::npos);
}

} // namespace
