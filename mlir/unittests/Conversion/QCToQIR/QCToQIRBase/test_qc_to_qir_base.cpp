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

#include "Support/IRVerification.h"
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
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>

using namespace mlir;

namespace {

struct QCToQIRBaseTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<qc::QCProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<qir::QIRProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QCToQIRBaseTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os, const QCToQIRBaseTestCase& info) {
  return os << "QCToQIRBase{" << info.name << ", original="
            << ::mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << ::mqt::test::displayName(info.referenceBuilder.name) << "}";
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
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  ::mqt::test::DeferredPrinter printer;

  auto program = ::mqt::test::buildMLIRProgram(context.get(), programBuilder);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCToQIRBaseConversion(program.get())));
  printer.record(program.get(), "Converted QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(program.get(), false).succeeded());
  printer.record(program.get(), "Canonicalized Converted QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = ::mqt::test::buildMLIRProgram(
      context.get(), referenceBuilder, qir::QIRProgramBuilder::Profile::Base);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QIR IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(reference.get(), false).succeeded());
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
                            MQT_NAMED_BUILDER(qir::alloc1QubitRegister<true>)},
        QCToQIRBaseTestCase{"BarrierTwoQubits",
                            MQT_NAMED_BUILDER(qc::barrierTwoQubits),
                            MQT_NAMED_BUILDER(qir::allocQubitRegister<true>)},
        QCToQIRBaseTestCase{"BarrierMultipleQubits",
                            MQT_NAMED_BUILDER(qc::barrierMultipleQubits),
                            MQT_NAMED_BUILDER(qir::alloc3QubitRegister<true>)},
        QCToQIRBaseTestCase{"SingleControlledBarrier",
                            MQT_NAMED_BUILDER(qc::singleControlledBarrier),
                            MQT_NAMED_BUILDER(qir::allocQubitRegister<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseDCXOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"DCX", MQT_NAMED_BUILDER(qc::dcx),
                                        MQT_NAMED_BUILDER(qir::dcx<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledDCX",
                        MQT_NAMED_BUILDER(qc::singleControlledDcx),
                        MQT_NAMED_BUILDER(qir::singleControlledDcx<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledDCX",
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx),
                        MQT_NAMED_BUILDER(qir::multipleControlledDcx<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseECROpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"ECR", MQT_NAMED_BUILDER(qc::ecr),
                                        MQT_NAMED_BUILDER(qir::ecr<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledECR",
                        MQT_NAMED_BUILDER(qc::singleControlledEcr),
                        MQT_NAMED_BUILDER(qir::singleControlledEcr<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledECR",
                        MQT_NAMED_BUILDER(qc::multipleControlledEcr),
                        MQT_NAMED_BUILDER(qir::multipleControlledEcr<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCToQIRBaseGPhaseOpTest, QCToQIRBaseTest,
                         testing::Values(QCToQIRBaseTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qc::globalPhase),
                             MQT_NAMED_BUILDER(qir::globalPhase<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseHOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                            MQT_NAMED_BUILDER(qir::h<true>)},
        QCToQIRBaseTestCase{"SingleControlledH",
                            MQT_NAMED_BUILDER(qc::singleControlledH),
                            MQT_NAMED_BUILDER(qir::singleControlledH<true>)},
        QCToQIRBaseTestCase{"MultipleControlledH",
                            MQT_NAMED_BUILDER(qc::multipleControlledH),
                            MQT_NAMED_BUILDER(qir::multipleControlledH<true>)},
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
                            MQT_NAMED_BUILDER(qir::identity<true>)},
        QCToQIRBaseTestCase{"SingleControlledIdentity",
                            MQT_NAMED_BUILDER(qc::singleControlledIdentity),
                            MQT_NAMED_BUILDER(qir::twoQubitsOneIdentity<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledIdentity",
            MQT_NAMED_BUILDER(qc::multipleControlledIdentity),
            MQT_NAMED_BUILDER(qir::threeQubitsOneIdentity<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseiSWAPOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"iSWAP", MQT_NAMED_BUILDER(qc::iswap),
                            MQT_NAMED_BUILDER(qir::iswap<true>)},
        QCToQIRBaseTestCase{
            "SingleControllediSWAP",
            MQT_NAMED_BUILDER(qc::singleControlledIswap),
            MQT_NAMED_BUILDER(qir::singleControlledIswap<true>)},
        QCToQIRBaseTestCase{
            "MultipleControllediSWAP",
            MQT_NAMED_BUILDER(qc::multipleControlledIswap),
            MQT_NAMED_BUILDER(qir::multipleControlledIswap<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBasePOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"P", MQT_NAMED_BUILDER(qc::p),
                            MQT_NAMED_BUILDER(qir::p<true>)},
        QCToQIRBaseTestCase{"SingleControlledP",
                            MQT_NAMED_BUILDER(qc::singleControlledP),
                            MQT_NAMED_BUILDER(qir::singleControlledP<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledP", MQT_NAMED_BUILDER(qc::multipleControlledP),
            MQT_NAMED_BUILDER(qir::multipleControlledP<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RCCXOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRCCXOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"RCCX", MQT_NAMED_BUILDER(qc::rccx),
                                        MQT_NAMED_BUILDER(qir::rccx<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledRCCX",
                        MQT_NAMED_BUILDER(qc::singleControlledRccx),
                        MQT_NAMED_BUILDER(qir::singleControlledRccx<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledRCCX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRccx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRccx<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseROpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"R", MQT_NAMED_BUILDER(qc::r),
                            MQT_NAMED_BUILDER(qir::r<true>)},
        QCToQIRBaseTestCase{"SingleControlledR",
                            MQT_NAMED_BUILDER(qc::singleControlledR),
                            MQT_NAMED_BUILDER(qir::singleControlledR<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledR", MQT_NAMED_BUILDER(qc::multipleControlledR),
            MQT_NAMED_BUILDER(qir::multipleControlledR<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRXOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RX", MQT_NAMED_BUILDER(qc::rx),
                            MQT_NAMED_BUILDER(qir::rx<true>)},
        QCToQIRBaseTestCase{"SingleControlledRX",
                            MQT_NAMED_BUILDER(qc::singleControlledRx),
                            MQT_NAMED_BUILDER(qir::singleControlledRx<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledRX", MQT_NAMED_BUILDER(qc::multipleControlledRx),
            MQT_NAMED_BUILDER(qir::multipleControlledRx<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRXXOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"RXX", MQT_NAMED_BUILDER(qc::rxx),
                                        MQT_NAMED_BUILDER(qir::rxx<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledRXX",
                        MQT_NAMED_BUILDER(qc::singleControlledRxx),
                        MQT_NAMED_BUILDER(qir::singleControlledRxx<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledRXX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRxx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRxx<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRYOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RY", MQT_NAMED_BUILDER(qc::ry),
                            MQT_NAMED_BUILDER(qir::ry<true>)},
        QCToQIRBaseTestCase{"SingleControlledRY",
                            MQT_NAMED_BUILDER(qc::singleControlledRy),
                            MQT_NAMED_BUILDER(qir::singleControlledRy<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledRY", MQT_NAMED_BUILDER(qc::multipleControlledRy),
            MQT_NAMED_BUILDER(qir::multipleControlledRy<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRYYOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"RYY", MQT_NAMED_BUILDER(qc::ryy),
                                        MQT_NAMED_BUILDER(qir::ryy<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledRYY",
                        MQT_NAMED_BUILDER(qc::singleControlledRyy),
                        MQT_NAMED_BUILDER(qir::singleControlledRyy<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledRYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRyy),
                        MQT_NAMED_BUILDER(qir::multipleControlledRyy<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRZOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz),
                            MQT_NAMED_BUILDER(qir::rz<true>)},
        QCToQIRBaseTestCase{"SingleControlledRZ",
                            MQT_NAMED_BUILDER(qc::singleControlledRz),
                            MQT_NAMED_BUILDER(qir::singleControlledRz<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledRZ", MQT_NAMED_BUILDER(qc::multipleControlledRz),
            MQT_NAMED_BUILDER(qir::multipleControlledRz<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRZXOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"RZX", MQT_NAMED_BUILDER(qc::rzx),
                                        MQT_NAMED_BUILDER(qir::rzx<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledRZX",
                        MQT_NAMED_BUILDER(qc::singleControlledRzx),
                        MQT_NAMED_BUILDER(qir::singleControlledRzx<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledRZX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRzx<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseRZZOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"RZZ", MQT_NAMED_BUILDER(qc::rzz),
                                        MQT_NAMED_BUILDER(qir::rzz<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::singleControlledRzz),
                        MQT_NAMED_BUILDER(qir::singleControlledRzz<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzz),
                        MQT_NAMED_BUILDER(qir::multipleControlledRzz<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"S", MQT_NAMED_BUILDER(qc::s),
                            MQT_NAMED_BUILDER(qir::s<true>)},
        QCToQIRBaseTestCase{"SingleControlledS",
                            MQT_NAMED_BUILDER(qc::singleControlledS),
                            MQT_NAMED_BUILDER(qir::singleControlledS<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledS", MQT_NAMED_BUILDER(qc::multipleControlledS),
            MQT_NAMED_BUILDER(qir::multipleControlledS<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSdgOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"Sdg", MQT_NAMED_BUILDER(qc::sdg),
                                        MQT_NAMED_BUILDER(qir::sdg<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledSdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSdg),
                        MQT_NAMED_BUILDER(qir::singleControlledSdg<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledSdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSdg),
                        MQT_NAMED_BUILDER(qir::multipleControlledSdg<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSWAPOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"SWAP", MQT_NAMED_BUILDER(qc::swap),
                                        MQT_NAMED_BUILDER(qir::swap<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::singleControlledSwap),
                        MQT_NAMED_BUILDER(qir::singleControlledSwap<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::multipleControlledSwap),
                        MQT_NAMED_BUILDER(qir::multipleControlledSwap<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSXOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"SX", MQT_NAMED_BUILDER(qc::sx),
                            MQT_NAMED_BUILDER(qir::sx<true>)},
        QCToQIRBaseTestCase{"SingleControlledSX",
                            MQT_NAMED_BUILDER(qc::singleControlledSx),
                            MQT_NAMED_BUILDER(qir::singleControlledSx<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledSX", MQT_NAMED_BUILDER(qc::multipleControlledSx),
            MQT_NAMED_BUILDER(qir::multipleControlledSx<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseSXdgOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"SXdg", MQT_NAMED_BUILDER(qc::sxdg),
                                        MQT_NAMED_BUILDER(qir::sxdg<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSxdg),
                        MQT_NAMED_BUILDER(qir::singleControlledSxdg<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSxdg),
                        MQT_NAMED_BUILDER(qir::multipleControlledSxdg<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseTOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"T", MQT_NAMED_BUILDER(qc::t_),
                            MQT_NAMED_BUILDER(qir::t_<true>)},
        QCToQIRBaseTestCase{"SingleControlledT",
                            MQT_NAMED_BUILDER(qc::singleControlledT),
                            MQT_NAMED_BUILDER(qir::singleControlledT<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledT", MQT_NAMED_BUILDER(qc::multipleControlledT),
            MQT_NAMED_BUILDER(qir::multipleControlledT<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseTdgOpTest, QCToQIRBaseTest,
    testing::Values(QCToQIRBaseTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg),
                                        MQT_NAMED_BUILDER(qir::tdg<true>)},
                    QCToQIRBaseTestCase{
                        "SingleControlledTdg",
                        MQT_NAMED_BUILDER(qc::singleControlledTdg),
                        MQT_NAMED_BUILDER(qir::singleControlledTdg<true>)},
                    QCToQIRBaseTestCase{
                        "MultipleControlledTdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledTdg),
                        MQT_NAMED_BUILDER(qir::multipleControlledTdg<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseU2OpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"U2", MQT_NAMED_BUILDER(qc::u2),
                            MQT_NAMED_BUILDER(qir::u2<true>)},
        QCToQIRBaseTestCase{"SingleControlledU2",
                            MQT_NAMED_BUILDER(qc::singleControlledU2),
                            MQT_NAMED_BUILDER(qir::singleControlledU2<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledU2", MQT_NAMED_BUILDER(qc::multipleControlledU2),
            MQT_NAMED_BUILDER(qir::multipleControlledU2<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseUOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                            MQT_NAMED_BUILDER(qir::u<true>)},
        QCToQIRBaseTestCase{"SingleControlledU",
                            MQT_NAMED_BUILDER(qc::singleControlledU),
                            MQT_NAMED_BUILDER(qir::singleControlledU<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledU", MQT_NAMED_BUILDER(qc::multipleControlledU),
            MQT_NAMED_BUILDER(qir::multipleControlledU<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseXOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"X", MQT_NAMED_BUILDER(qc::x),
                            MQT_NAMED_BUILDER(qir::x<true>)},
        QCToQIRBaseTestCase{"SingleControlledX",
                            MQT_NAMED_BUILDER(qc::singleControlledX),
                            MQT_NAMED_BUILDER(qir::singleControlledX<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledX", MQT_NAMED_BUILDER(qc::multipleControlledX),
            MQT_NAMED_BUILDER(qir::multipleControlledX<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseXXMinusYYOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qc::xxMinusYY),
                            MQT_NAMED_BUILDER(qir::xxMinusYY<true>)},
        QCToQIRBaseTestCase{
            "SingleControlledXXMinusYY",
            MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY),
            MQT_NAMED_BUILDER(qir::singleControlledXxMinusYY<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledXXMinusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY),
            MQT_NAMED_BUILDER(qir::multipleControlledXxMinusYY<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseXXPlusYYOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
                            MQT_NAMED_BUILDER(qir::xxPlusYY<true>)},
        QCToQIRBaseTestCase{
            "SingleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY),
            MQT_NAMED_BUILDER(qir::singleControlledXxPlusYY<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY),
            MQT_NAMED_BUILDER(qir::multipleControlledXxPlusYY<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseYOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                            MQT_NAMED_BUILDER(qir::y<true>)},
        QCToQIRBaseTestCase{"SingleControlledY",
                            MQT_NAMED_BUILDER(qc::singleControlledY),
                            MQT_NAMED_BUILDER(qir::singleControlledY<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledY", MQT_NAMED_BUILDER(qc::multipleControlledY),
            MQT_NAMED_BUILDER(qir::multipleControlledY<true>)}));
/// @}

/// \name QCToQIRBase/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCToQIRBaseZOpTest, QCToQIRBaseTest,
    testing::Values(
        QCToQIRBaseTestCase{"Z", MQT_NAMED_BUILDER(qc::z),
                            MQT_NAMED_BUILDER(qir::z<true>)},
        QCToQIRBaseTestCase{"SingleControlledZ",
                            MQT_NAMED_BUILDER(qc::singleControlledZ),
                            MQT_NAMED_BUILDER(qir::singleControlledZ<true>)},
        QCToQIRBaseTestCase{
            "MultipleControlledZ", MQT_NAMED_BUILDER(qc::multipleControlledZ),
            MQT_NAMED_BUILDER(qir::multipleControlledZ<true>)}));
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
                            MQT_NAMED_BUILDER(qir::allocQubit<true>)},
        QCToQIRBaseTestCase{"AllocQubitRegister",
                            MQT_NAMED_BUILDER(qc::allocQubitRegister),
                            MQT_NAMED_BUILDER(qir::allocQubitRegister<true>)},
        QCToQIRBaseTestCase{
            "AllocMultipleQubitRegisters",
            MQT_NAMED_BUILDER(qc::allocMultipleQubitRegisters),
            MQT_NAMED_BUILDER(qir::allocMultipleQubitRegisters<true>)},
        QCToQIRBaseTestCase{
            "AllocMultipleQubitRegistersWithOps",
            MQT_NAMED_BUILDER(qc::allocMultipleQubitRegistersWithOps),
            MQT_NAMED_BUILDER(qir::allocMultipleQubitRegistersWithOps<true>)},
        QCToQIRBaseTestCase{"AllocLargeRegister",
                            MQT_NAMED_BUILDER(qc::allocLargeRegister),
                            MQT_NAMED_BUILDER(qir::allocQubitRegister<true>)},
        QCToQIRBaseTestCase{"StaticQubits", MQT_NAMED_BUILDER(qc::staticQubits),
                            MQT_NAMED_BUILDER(qir::staticQubits)},
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
                            MQT_NAMED_BUILDER(qir::emptyQIR<true>)}));
/// @}

/// \name QCToQIRBase/Modifiers/CtrlOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCToQIRBaseCtrlOpTest, QCToQIRBaseTest,
                         testing::Values(QCToQIRBaseTestCase{
                             "CtrlTwo", MQT_NAMED_BUILDER(qc::ctrlTwo),
                             MQT_NAMED_BUILDER(qir::ctrlTwo<true>)}));
/// @}
