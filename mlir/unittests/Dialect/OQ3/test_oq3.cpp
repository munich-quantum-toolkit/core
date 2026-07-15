/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"
#include "mlir/Dialect/OQ3/Transforms/Passes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>
#include <string>

using namespace mlir;

namespace {

class OQ3Test : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<oq3::OQ3Dialect, qc::QCDialect, arith::ArithDialect,
                    func::FuncDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  OwningOpRef<ModuleOp> buildGateApplication(const StringRef name,
                                             const size_t parameterCount) {
    OpBuilder builder(context.get());
    const Location loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    SmallVector<Type> gateInputs(parameterCount, builder.getF64Type());
    gateInputs.append(2, qc::QubitType::get(context.get()));
    oq3::GateDeclOp::create(builder, loc, name,
                            builder.getFunctionType(gateInputs, {}));

    const auto qubitType = qc::QubitType::get(context.get());
    auto function = func::FuncOp::create(
        builder, loc, "main",
        builder.getFunctionType({qubitType, qubitType}, {}));
    Block* entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    SmallVector<Value> parameters;
    parameters.reserve(parameterCount);
    for (size_t i = 0; i < parameterCount; ++i) {
      parameters.push_back(
          arith::ConstantFloatOp::create(builder, loc, builder.getF64Type(),
                                         APFloat(static_cast<double>(i + 1))));
    }

    OperationState state(loc, oq3::ApplyGateOp::getOperationName());
    state.addOperands(parameters);
    state.addOperands(entry->getArguments());
    state.addAttribute("callee", FlatSymbolRefAttr::get(context.get(), name));
    state.addAttribute("modifier_kinds",
                       DenseI32ArrayAttr::get(context.get(), {}));
    state.addAttribute("modifier_operand_indices",
                       DenseI32ArrayAttr::get(context.get(), {}));
    state.addAttribute(
        "operandSegmentSizes",
        DenseI32ArrayAttr::get(context.get(),
                               {static_cast<int32_t>(parameterCount), 2, 0}));
    builder.create(state);
    func::ReturnOp::create(builder, loc);
    return OwningOpRef<ModuleOp>(module);
  }

  std::unique_ptr<MLIRContext> context;
};

TEST_F(OQ3Test, RejectsZeroWidthSourceTypes) {
  EXPECT_FALSE(oq3::BitType::getChecked(
      [&]() { return emitError(UnknownLoc::get(context.get())); },
      context.get(), 0U));
  EXPECT_FALSE(oq3::AngleType::getChecked(
      [&]() { return emitError(UnknownLoc::get(context.get())); },
      context.get(), 0U));
}

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
    manager.addPass(oq3::createLowerOQ3ToQCPass());
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

TEST_F(OQ3Test, DiagnosesUnprovenDynamicRangeStepsAtLowering) {
  OpBuilder builder(context.get());
  const Location loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  auto function =
      func::FuncOp::create(builder, loc, "main",
                           builder.getFunctionType({builder.getI64Type()}, {}));
  module.getBody()->push_back(function);
  Block* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value start = arith::ConstantIntOp::create(builder, loc, 0, 64);
  Value stop = arith::ConstantIntOp::create(builder, loc, 4, 64);
  OperationState state(loc, oq3::ForOp::getOperationName());
  state.addOperands({start, stop, entry->getArgument(0)});
  Region* body = state.addRegion();
  body->push_back(new Block());
  body->front().addArgument(builder.getI64Type(), loc);
  auto loop = cast<oq3::ForOp>(builder.create(state));
  builder.setInsertionPointToStart(&loop.getBody().front());
  oq3::YieldOp::create(builder, loc);
  builder.setInsertionPointToEnd(entry);
  func::ReturnOp::create(builder, loc);
  ASSERT_TRUE(succeeded(verify(module)));

  std::string diagnostic;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& value) {
    llvm::raw_string_ostream(diagnostic) << value;
    return success();
  });
  PassManager manager(context.get());
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(module)));
  EXPECT_NE(diagnostic.find("dynamic range step cannot be proven nonzero"),
            std::string::npos);
}

} // namespace
