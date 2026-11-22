/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"
#include "mqt/Support/Passes.h"

#include "Conversion/QCToQIR/QIRConversionCases.h"
#include "TestCaseUtils.h"
#include "qc_programs.h"
#include "qir_programs.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

using namespace mlir;

namespace {

using ::mqt::test::QIRConversionCase;

class QCToQIRBaseTest : public testing::TestWithParam<QIRConversionCase> {
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

class QCToQIRBaseNativeTest : public testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<qc::QCDialect, cbit::CBitDialect, arith::ArithDialect,
                        func::FuncDialect, LLVM::LLVMDialect,
                        memref::MemRefDialect, scf::SCFDialect,
                        cf::ControlFlowDialect, math::MathDialect>();
  }
};

} // namespace

static LogicalResult runQCToQIRBaseConversion(ModuleOp moduleOp) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(mlir::mqt::createUnrollModifiers());
  pm.addPass(createQCToQIRBase());
  return pm.run(moduleOp);
}

static void expectConversionFailure(ModuleOp moduleOp, StringRef expected) {
  ASSERT_TRUE(succeeded(verify(moduleOp)));
  bool diagnosed = false;
  ScopedDiagnosticHandler handler(
      moduleOp.getContext(), [&](Diagnostic& diagnostic) {
        diagnosed |= StringRef(diagnostic.str()).contains(expected);
        return success();
      });
  EXPECT_TRUE(failed(runQCToQIRBaseConversion(moduleOp)));
  EXPECT_TRUE(diagnosed) << expected.str();
}

static void expectFollowingXIsUncontrolled(
    const function_ref<void(qc::QCProgramBuilder&, Value, Value)>
        buildModifier) {
  MLIRContext context;
  context.loadDialect<qc::QCDialect, arith::ArithDialect, func::FuncDialect,
                      LLVM::LLVMDialect>();
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto control = builder.allocQubit();
  auto target = builder.allocQubit();
  buildModifier(builder, control, target);
  builder.x(target);
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQIRBaseConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t xCalls = 0;
  size_t controlledXCalls = 0;
  moduleOp->walk([&](LLVM::CallOp call) {
    xCalls += call.getCallee() == qir::QIR_X;
    controlledXCalls += call.getCallee() == qir::QIR_CX;
  });
  EXPECT_EQ(xCalls, 1);
  EXPECT_EQ(controlledXCalls, 0);
}

TEST_F(QCToQIRBaseNativeTest, RejectsExplicitRuntimeAssertions) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  cf::AssertOp::create(builder, builder.getUnknownLoc(),
                       builder.boolConstant(false),
                       "unsupported runtime precondition");
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  expectConversionFailure(*moduleOp, "cf.assert");
}

TEST_F(QCToQIRBaseNativeTest, RejectsMeasurementFeedbackDuringConversion) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto control = builder.allocQubit();
  auto target = builder.allocQubit();
  builder.h(control);
  auto result = builder.measure(control);
  builder.scfIf(result, [&] { builder.x(target); });
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(failed(runQCToQIRBaseConversion(*moduleOp)));
}

TEST_F(QCToQIRBaseNativeTest, CtrlBodyCardinalityIgnoresClassicalOps) {
  for (uint32_t numUnitaries : {0U, 1U, 2U}) {
    SCOPED_TRACE(numUnitaries);
    qc::QCProgramBuilder builder(&context);
    builder.initialize();
    auto control = builder.allocQubit();
    auto target = builder.allocQubit();
    builder.ctrl(control, target, [&](Value argument) {
      for (uint32_t i = 0; i <= numUnitaries; ++i) {
        arith::ConstantIntOp::create(builder, builder.getUnknownLoc(), i, 64);
        if (i < numUnitaries) {
          builder.x(argument);
        }
      }
    });
    builder.x(target);
    auto moduleOp = builder.finalize();
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));

    bool diagnosed = false;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
      diagnosed |= diagnostic.str().find("qc.ctrl") != std::string::npos;
      return success();
    });
    PassManager pm(&context);
    pm.addPass(createQCToQIRBase());
    if (numUnitaries == 2) {
      EXPECT_TRUE(failed(pm.run(*moduleOp)));
      EXPECT_TRUE(diagnosed);
      continue;
    }
    ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    size_t xCalls = 0;
    size_t controlledXCalls = 0;
    moduleOp->walk([&](LLVM::CallOp call) {
      xCalls += call.getCallee() == qir::QIR_X;
      controlledXCalls += call.getCallee() == qir::QIR_CX;
    });
    EXPECT_EQ(xCalls, 1U);
    EXPECT_EQ(controlledXCalls, numUnitaries);
  }
}

TEST_F(QCToQIRBaseNativeTest, RejectsReorderedOverlappingOutputStores) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto q0 = builder.allocQubit();
  auto q1 = builder.allocQubit();
  auto reg = builder.allocClassicalBitRegister(1);
  builder.x(q1);
  auto zero = builder.measure(q0);
  auto one = builder.measure(q1);
  builder.storeClassicalBit(one, reg, 0);
  builder.storeClassicalBit(zero, reg, 0);

  auto module = builder.finalize(reg);
  ASSERT_TRUE(module);
  expectConversionFailure(*module, "cannot fuse this measurement/store pair");
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(QCToQIRBaseNativeTest, RejectsMultiBlockEntryFunctionWithoutMutation) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  auto entryPoint = moduleOp->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(entryPoint);
  auto* extraBlock = &entryPoint.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(extraBlock);
  func::ReturnOp::create(builder, builder.getUnknownLoc());
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  expectConversionFailure(
      *moduleOp, "QIR Base Profile requires a single-block entry function");
  EXPECT_EQ(entryPoint.getBlocks().size(), 2);
}

static void expectMeasurementOrderRejected(
    function_ref<Value(qc::QCProgramBuilder&)> buildProgram,
    StringRef expectedDiagnostic =
        "QIR Base Profile forbids using a qubit after measurement") {
  MLIRContext context;
  context.loadDialect<qc::QCDialect, arith::ArithDialect, func::FuncDialect,
                      LLVM::LLVMDialect, memref::MemRefDialect>();
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto result = buildProgram(builder);

  auto moduleOp = builder.finalize(result);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  expectConversionFailure(*moduleOp, expectedDiagnostic);
}

TEST_F(QCToQIRBaseNativeTest, RejectsGateAfterMeasurementOnSameQubit) {
  expectMeasurementOrderRejected([](qc::QCProgramBuilder& builder) {
    auto qubit = builder.allocQubit();
    auto result = builder.measure(qubit);
    builder.x(qubit);
    return result;
  });
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func private @flip(%q: !qc.qubit) attributes {no_inline} {
      qc.x %q : !qc.qubit
      return
    }
    func.func @main() -> i1 attributes {mqt.entry_point} {
      %q = qc.static 7 : !qc.qubit
      %bit = qc.measure %q : !qc.qubit -> i1
      func.call @flip(%q) : (!qc.qubit) -> ()
      return %bit : i1
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  expectConversionFailure(
      *moduleOp, "QIR Base Profile forbids using a qubit after measurement");
}

TEST_F(QCToQIRBaseNativeTest, RejectsGatesWithMeasuredDirectOrPackedOperands) {
  expectMeasurementOrderRejected([](qc::QCProgramBuilder& builder) {
    auto control = builder.allocQubit();
    auto target = builder.allocQubit();
    auto result = builder.measure(control);
    builder.cx(control, target);
    return result;
  });
  for (const bool measureControl : {false, true}) {
    SCOPED_TRACE(measureControl);
    expectMeasurementOrderRejected([&](qc::QCProgramBuilder& builder) {
      auto qubits = builder.allocQubitRegister(4);
      auto result = builder.measure(qubits[measureControl ? 0 : 3]);
      auto alias = builder.loadQubit(
          qubits.value, builder.indexConstant(measureControl ? 0 : 3));
      builder.mcrx(0.25,
                   {measureControl ? alias : qubits[0], qubits[1], qubits[2]},
                   measureControl ? qubits[3] : alias);
      return result;
    });
  }
}

TEST_F(QCToQIRBaseNativeTest, RejectsGateOnMeasuredStaticAlias) {
  expectMeasurementOrderRejected([](qc::QCProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    auto result = builder.measure(qubit);
    auto alias = qc::StaticOp::create(builder, 0).getQubit();
    builder.x(alias);
    return result;
  });
}

TEST_F(QCToQIRBaseNativeTest, RejectsGateOnMeasuredRegisterElement) {
  expectMeasurementOrderRejected([](qc::QCProgramBuilder& builder) {
    auto qubits = builder.allocQubitRegister(2);
    auto result = builder.measure(qubits[0]);
    auto alias = builder.loadQubit(qubits.value, builder.indexConstant(0));
    builder.x(alias);
    return result;
  });
}

TEST_F(QCToQIRBaseNativeTest, RejectsRepeatedMeasurement) {
  expectMeasurementOrderRejected([](qc::QCProgramBuilder& builder) {
    auto qubit = builder.allocQubit();
    builder.measure(qubit);
    return builder.measure(qubit);
  });
}

TEST_F(QCToQIRBaseNativeTest, RejectsRepeatedMeasurementToSameBit) {
  expectMeasurementOrderRejected(qc::repeatedMeasurementToSameBit);
}

TEST_F(QCToQIRBaseNativeTest, RejectsRepeatedMeasurementToDifferentBits) {
  expectMeasurementOrderRejected([](qc::QCProgramBuilder& builder) {
    return qc::repeatedMeasurementToDifferentBits(builder).front();
  });
}

TEST_F(QCToQIRBaseNativeTest, RejectsRuntimeQubitRegisterIndex) {
  expectMeasurementOrderRejected(
      [](qc::QCProgramBuilder& builder) {
        auto qubits = builder.allocQubitRegister(2);
        auto unknown = LLVM::UndefOp::create(builder, builder.getI64Type());
        auto index = arith::IndexCastOp::create(builder, builder.getIndexType(),
                                                unknown);
        return builder.measure(builder.loadQubit(qubits.value, index));
      },
      "QIR Base Profile requires constant indices");
}

TEST_F(QCToQIRBaseNativeTest, RejectsOutOfBoundsQubitRegisterIndex) {
  for (const auto index : {-1, 2}) {
    SCOPED_TRACE(index);
    expectMeasurementOrderRejected(
        [index](qc::QCProgramBuilder& builder) {
          auto qubits = builder.allocQubitRegister(2);
          return builder.measure(
              builder.loadQubit(qubits.value, builder.indexConstant(index)));
        },
        "qubit-register index is out of bounds");
  }
}

TEST_F(QCToQIRBaseNativeTest, RejectsRepeatedMeasurementThroughRegisterAlias) {
  expectMeasurementOrderRejected([](qc::QCProgramBuilder& builder) {
    auto qubits = builder.allocQubitRegister(1);
    builder.measure(qubits[0]);
    return builder.measure(
        builder.loadQubit(qubits.value, builder.indexConstant(0)));
  });
}

TEST_F(QCToQIRBaseNativeTest, RegisterLoadsPreserveQubitIdentity) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto qubits = builder.allocQubitRegister(1);
  builder.x(qubits[0]);
  auto result = builder.measure(
      builder.loadQubit(qubits.value, builder.indexConstant(0)));

  auto moduleOp = builder.finalize(result);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQIRBaseConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  Value gateQubit;
  Value measuredQubit;
  moduleOp->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_X) {
      gateQubit = call.getOperand(0);
    } else if (call.getCallee() == qir::QIR_MEASURE) {
      measuredQubit = call.getOperand(0);
    }
  });
  ASSERT_TRUE(gateQubit);
  EXPECT_EQ(gateQubit, measuredQubit);
}

TEST_F(QCToQIRBaseNativeTest, AllowsGateAfterMeasurementOnIndependentQubit) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto qubits = builder.allocQubitRegister(4);
  auto result = builder.measure(qubits[0]);
  builder.x(qubits[1]);
  builder.mcrx(0.25, {qubits[1], qubits[2]}, qubits[3]);
  auto moduleOp = builder.finalize(result);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQIRBaseConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  SmallVector<LLVM::CallOp> quantumCalls;
  moduleOp->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_X || call.getCallee() == qir::QIR_CCRX ||
        call.getCallee() == qir::QIR_MEASURE) {
      quantumCalls.push_back(call);
    }
  });
  ASSERT_EQ(quantumCalls.size(), 3);
  EXPECT_EQ(quantumCalls[0].getCallee(), qir::QIR_X);
  EXPECT_EQ(quantumCalls[1].getCallee(), qir::QIR_CCRX);
  EXPECT_EQ(quantumCalls[2].getCallee(), qir::QIR_MEASURE);
  EXPECT_NE(quantumCalls[0].getOperand(0), quantumCalls[2].getOperand(0));
}

TEST_F(QCToQIRBaseNativeTest, ControlledBarrierDoesNotControlFollowingGate) {
  expectFollowingXIsUncontrolled(
      [](qc::QCProgramBuilder& builder, Value control, Value target) {
        builder.ctrl(control, target,
                     [&](Value bodyTarget) { builder.barrier(bodyTarget); });
      });
}

TEST_F(QCToQIRBaseNativeTest, RejectsResidualClassicalComputations) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto value = builder.intConstant(7);
  auto count = math::CtPopOp::create(builder, value);
  auto angle = arith::UIToFPOp::create(builder, builder.getF64Type(), count);
  builder.rx(angle, builder.allocQubit());
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(failed(runQCToQIRBaseConversion(*module)));
}

TEST_F(QCToQIRBaseNativeTest, SelectsControlledSpecializationsByArity) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto control0 = builder.allocQubit();
  auto control1 = builder.allocQubit();
  auto target = builder.allocQubit();
  builder.crx(0.25, control0, target);
  builder.mcrx(0.5, {control0, control1}, target);
  auto module = builder.finalize();

  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runQCToQIRBaseConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(qir::QIR_CRX));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(qir::QIR_CCRX));
}

TEST_F(QCToQIRBaseNativeTest, RejectsRuntimeControlArrays) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto control0 = builder.allocQubit();
  auto control1 = builder.allocQubit();
  auto control2 = builder.allocQubit();
  auto target = builder.allocQubit();
  builder.mcrx(0.75, {control0, control1, control2}, target);
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  EXPECT_TRUE(failed(runQCToQIRBaseConversion(*module)));
}

TEST_F(QCToQIRBaseNativeTest, RejectsHelperLocalClassicalResources) {
  for (const auto* body : {
           R"mlir(%zero = arith.constant 0 : index
             %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %bit = cbit.load %reg[%zero] : !cbit.reg<1>)mlir",
           R"mlir(%bit = qc.measure %q : !qc.qubit -> i1)mlir",
       }) {
    SCOPED_TRACE(body);
    const std::string source = R"mlir(module {
      func.func private @helper(%q: !qc.qubit) -> i1 attributes {no_inline} {
    )mlir" + std::string(body) +
                               R"mlir(
        return %bit : i1
      }
      func.func @main() attributes {mqt.entry_point} {
        %q = qc.static 0 : !qc.qubit
        %bit = func.call @helper(%q) : (!qc.qubit) -> i1
        return
      }
    })mlir";
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    expectConversionFailure(*module, "QIR conversion requires measurements and "
                                     "CBit allocations in the "
                                     "entry function");
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST_F(QCToQIRBaseNativeTest, RejectsUnsupportedEntryOutputs) {
  const std::array names{
      "selected register", "computed boolean",        "zero scalar",
      "nonzero scalar",    "overwritten measurement",
  };
  for (uint32_t shape = 0; shape < names.size(); ++shape) {
    SCOPED_TRACE(names[shape]);
    qc::QCProgramBuilder builder(&context);
    builder.initialize();
    SmallVector<Value> outputs;
    if (shape == 0) {
      auto first = builder.allocClassicalBitRegister(1);
      auto second = builder.allocClassicalBitRegister(1);
      outputs.push_back(arith::SelectOp::create(
          builder, builder.boolConstant(true), first, second));
    } else if (shape == 1) {
      outputs.push_back(builder.boolConstant(true));
    } else if (shape == 2 || shape == 3) {
      outputs.push_back(builder.intConstant(shape == 2 ? 0 : 7));
    } else {
      auto first = builder.allocQubit();
      auto second = builder.allocQubit();
      auto reg = builder.allocClassicalBitRegister(1);
      auto bit = builder.measure(first, reg, 0);
      builder.measure(second, reg, 0);
      outputs = {bit, reg};
    }

    auto module = builder.finalize(outputs);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    const StringRef expected =
        shape == 4 ? "returned scalar measurement cannot share overwritten "
                     "register storage"
                   : "QIR entry return supports only direct measurements and "
                     "CBit allocations";
    expectConversionFailure(*module, expected);
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST_F(QCToQIRBaseNativeTest, PreservesOutputOrderMultiplicityAndLabels) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  SmallVector<Value> registers;
  SmallVector<Value> bits;
  for (StringRef name : {"c", "c_0", "named_result"}) {
    auto qubit = builder.allocQubit();
    auto reg = builder.allocClassicalBitRegister(1, name.str());
    registers.push_back(reg);
    bits.push_back(builder.measure(qubit, reg, 0));
  }
  bits.push_back(builder.measure(builder.allocQubit()));
  SmallVector<Value> outputs{
      registers[0], registers[1], bits[3], bits[0],
      registers[2], bits[1],      bits[3], registers[1],
  };

  auto module = builder.finalize(outputs);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQIRBaseConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<Value> measurements;
  module->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_MEASURE) {
      measurements.push_back(call.getOperand(1));
    }
  });
  ASSERT_EQ(measurements.size(), 4U);
  SmallVector<Value> recorded;
  SmallVector<size_t> containerPositions;
  DenseSet<StringRef> labels;
  SmallVector<StringRef> containerLabels;
  module->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_RECORD_OUTPUT) {
      recorded.push_back(call.getOperand(0));
    } else if (call.getCallee() == qir::QIR_ARRAY_RECORD_OUTPUT) {
      containerPositions.push_back(recorded.size());
    } else {
      return;
    }
    auto address = call.getOperands().back().getDefiningOp<LLVM::AddressOfOp>();
    ASSERT_TRUE(address);
    auto global = module->lookupSymbol<LLVM::GlobalOp>(address.getGlobalName());
    ASSERT_TRUE(global);
    auto label = global->getAttrOfType<StringAttr>("value");
    ASSERT_TRUE(label);
    if (call.getCallee() != qir::QIR_RECORD_OUTPUT) {
      containerLabels.push_back(label.getValue().split('\0').first);
    }
    EXPECT_TRUE(labels.insert(label.getValue()).second)
        << label.getValue().str();
  });
  EXPECT_EQ(recorded, (SmallVector<Value>{measurements[0], measurements[1],
                                          measurements[3], measurements[0],
                                          measurements[2], measurements[1],
                                          measurements[3], measurements[1]}));
  EXPECT_EQ(containerPositions, (SmallVector<size_t>{0, 1, 4, 7}));
  ASSERT_EQ(containerLabels.size(), 4U);
  EXPECT_EQ(containerLabels[0], "c");
  EXPECT_EQ(containerLabels[1], "c_0");
  EXPECT_EQ(containerLabels[2], "named_result");
  EXPECT_TRUE(
      module->lookupSymbol<LLVM::GlobalOp>("qir.result_label_named_result"));
  auto main = module->lookupSymbol<LLVM::LLVMFuncOp>("main");
  ASSERT_TRUE(main);
  for (const StringRef key : {"required_num_qubits", "required_num_results"}) {
    EXPECT_TRUE(llvm::is_contained(main.getPassthroughAttr(),
                                   builder.getStrArrayAttr({key, "4"})));
  }
  EXPECT_TRUE(llvm::is_contained(
      main.getPassthroughAttr(),
      builder.getStrArrayAttr({"qir_profiles", "base_profile"})));
  EXPECT_TRUE(main.getFunctionType().getReturnType().isInteger(64));
  main.walk([&](LLVM::ReturnOp op) {
    ASSERT_EQ(op.getNumOperands(), 1U);
    auto code = op.getOperand(0).getDefiningOp<LLVM::ConstantOp>();
    ASSERT_TRUE(code);
    EXPECT_EQ(cast<IntegerAttr>(code.getValue()).getInt(), 0);
  });
}

TEST_F(QCToQIRBaseNativeTest, RejectsNonMeasurementClassicalStore) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto c = builder.allocClassicalBitRegister(1);
  builder.storeClassicalBit(builder.boolConstant(true), c, 0);

  auto module = builder.finalize(c);
  ASSERT_TRUE(module);

  expectConversionFailure(*module, "does not support non-measurement "
                                   "stores to returned CBit registers");
}

TEST_F(QCToQIRBaseNativeTest, RejectsUnmeasuredOutputBits) {
  for (auto initialization :
       {cbit::Initialization::Zero, cbit::Initialization::Undefined}) {
    for (bool partiallyMeasured : {false, true}) {
      qc::QCProgramBuilder builder(&context);
      builder.initialize();
      auto c = builder.allocClassicalBitRegister(2, {}, initialization);
      if (partiallyMeasured) {
        builder.measure(builder.allocQubit(), c, 1);
      }
      auto moduleOp = builder.finalize(c);
      ASSERT_TRUE(moduleOp);
      expectConversionFailure(*moduleOp,
                              "requires every output bit to be measured");
    }
  }
}

TEST_F(QCToQIRBaseNativeTest, RejectsNonMeasurementStoreAfterMeasurement) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto q = builder.allocQubit();
  auto c = builder.allocClassicalBitRegister(1);
  builder.measure(q, c, 0);
  builder.storeClassicalBit(builder.boolConstant(false), c, 0);

  auto module = builder.finalize(c);
  ASSERT_TRUE(module);

  expectConversionFailure(*module, "does not support non-measurement "
                                   "stores to returned CBit registers");
}

TEST_F(QCToQIRBaseNativeTest, RejectsClassicalMemrefOutput) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  const auto type = MemRefType::get({1}, builder.getI8Type());
  auto memref = memref::AllocOp::create(builder, type).getResult();

  auto module = builder.finalize(memref);
  ASSERT_TRUE(module);

  expectConversionFailure(*module, "QIR entry return supports only");
}

TEST_F(QCToQIRBaseNativeTest, RejectsDynamicClassicalRegisterIndex) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto q = builder.allocQubit();
  auto c = builder.allocClassicalBitRegister(1);
  auto unknown = LLVM::UndefOp::create(builder, builder.getI64Type());
  auto index = arith::IndexCastOp::create(builder, builder.getIndexType(),
                                          unknown.getResult());
  builder.measure(q, c, index.getResult());

  auto module = builder.finalize(c);
  ASSERT_TRUE(module);

  expectConversionFailure(
      *module, "requires constant classical-register measurement indices");
}

TEST_P(QCToQIRBaseTest, ProgramEquivalence) {
  const auto& [name, programBuilder, referenceBuilder] = GetParam();
  SCOPED_TRACE(name);

  auto program = ::mqt::test::buildMLIRProgram(context.get(), programBuilder);
  ASSERT_TRUE(program);
  ASSERT_TRUE(::mqt::test::verifyQIRTestStage(*program, "Original QC IR"));

  ASSERT_TRUE(runQCCleanupPipeline(program.get()).succeeded());
  ASSERT_TRUE(::mqt::test::verifyQIRTestStage(*program, "Canonicalized QC IR"));

  PassManager modifierLowering(context.get());
  modifierLowering.addPass(mlir::mqt::createLowerQuantumModifiers());
  ASSERT_TRUE(succeeded(modifierLowering.run(*program)));
  ASSERT_TRUE(succeeded(runQCToQIRBaseConversion(program.get())));
  ASSERT_TRUE(::mqt::test::verifyQIRTestStage(*program, "Converted QIR"));

  auto reference = ::mqt::test::buildMLIRProgram(
      context.get(), referenceBuilder, qir::QIRProgramBuilder::Profile::Base);
  ASSERT_TRUE(reference);
  ASSERT_TRUE(::mqt::test::verifyQIRTestStage(*reference, "Reference QIR IR"));

  ASSERT_TRUE(runQIRCleanupPipeline(reference.get(), false).succeeded());
  ASSERT_TRUE(::mqt::test::verifyQIRTestStage(
      *reference, "Canonicalized Reference QIR IR"));

  ::mqt::test::expectEquivalentQIR(*program, *reference);
}

namespace {
constexpr auto PROFILE_CASES = std::to_array<QIRConversionCase>({
    ::mqt::test::qirCase("HWithoutRegister", qc::hWithoutRegister,
                         qir::hWithoutRegisterBase),
    ::mqt::test::qirCase("MeasurementWithoutRegisters",
                         qc::measurementWithoutRegisters,
                         qir::measurementWithoutRegistersBase),
});
} // namespace

INSTANTIATE_TEST_SUITE_P(Common, QCToQIRBaseTest,
                         testing::ValuesIn(::mqt::test::COMMON_QIR_CASES),
                         [](const auto& info) { return info.param.name; });
INSTANTIATE_TEST_SUITE_P(BaseOnly, QCToQIRBaseTest,
                         testing::ValuesIn(PROFILE_CASES),
                         [](const auto& info) { return info.param.name; });
