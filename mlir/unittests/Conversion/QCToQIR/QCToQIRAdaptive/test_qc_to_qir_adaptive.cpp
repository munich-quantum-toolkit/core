/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mqt/Conversion/QCToQIR/QIRCommon/QIRCommon.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
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
#include <utility>

using namespace mlir;

namespace {

struct QCToQIRAdaptiveTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<qc::QCProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<qir::QIRProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QCToQIRAdaptiveTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const QCToQIRAdaptiveTestCase& info) {
  return os << "QCToQIRAdaptive{" << info.name << ", original="
            << ::mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << ::mqt::test::displayName(info.referenceBuilder.name) << "}";
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

class QCToQIRAdaptiveNativeTest : public testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<qc::QCDialect, cbit::CBitDialect, arith::ArithDialect,
                        func::FuncDialect, LLVM::LLVMDialect,
                        memref::MemRefDialect, scf::SCFDialect,
                        cf::ControlFlowDialect, math::MathDialect,
                        tensor::TensorDialect>();
  }
};

} // namespace

static LogicalResult runQCToQIRAdaptiveConversion(ModuleOp moduleOp) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(mlir::mqt::createUnrollModifiers());
  pm.addPass(createQCToQIRAdaptive());
  return pm.run(moduleOp);
}

static LogicalResult runQCToQIRAdaptiveConversionSimple(ModuleOp moduleOp) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(createQCToQIRAdaptive());
  return pm.run(moduleOp);
}

static void expectConversionFailure(
    ModuleOp moduleOp, StringRef expected,
    LogicalResult (*convert)(ModuleOp) = runQCToQIRAdaptiveConversionSimple) {
  ASSERT_TRUE(succeeded(verify(moduleOp)));
  bool diagnosed = false;
  ScopedDiagnosticHandler handler(
      moduleOp.getContext(), [&](Diagnostic& diagnostic) {
        diagnosed |= StringRef(diagnostic.str()).contains(expected);
        return success();
      });
  EXPECT_TRUE(failed(convert(moduleOp)));
  EXPECT_TRUE(diagnosed) << expected.str();
}

TEST_F(QCToQIRAdaptiveNativeTest, LowersConstantTensorWithRuntimeIndex) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %angles = arith.constant dense<[0.25, 0.5]> : tensor<2xf64>
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %q = qc.alloc : !qc.qubit
      scf.for %index = %zero to %two step %one {
        %angle = tensor.extract %angles[%index] : tensor<2xf64>
        qc.rz(%angle) %q : !qc.qubit
      }
      qc.dealloc %q : !qc.qubit
      return
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t constantTables = 0;
  moduleOp->walk([&](LLVM::GlobalOp global) {
    if (global.getConstant() &&
        isa<LLVM::LLVMArrayType>(global.getGlobalType())) {
      ++constantTables;
    }
  });
  EXPECT_EQ(constantTables, 1);
  bool readsFloat = false;
  moduleOp->walk(
      [&](LLVM::LoadOp load) { readsFloat |= load.getType().isF64(); });
  EXPECT_TRUE(readsFloat);
  moduleOp->walk([&](Operation* operation) {
    EXPECT_NE(operation->getName().getDialectNamespace(), "tensor");
    EXPECT_NE(operation->getName().getDialectNamespace(), "memref");
    EXPECT_NE(operation->getName().getDialectNamespace(), "bufferization");
  });
}

TEST_F(QCToQIRAdaptiveNativeTest, ContiguousCopiesNeedOnlyArrayOperations) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    memref.global "private" constant @source : memref<2x3xf64>
      = dense<[[0.0, 0.25, 0.5], [0.75, 1.0, 1.25]]>
    memref.global "private" constant @scalar : memref<f64> = dense<0.5>
    func.func @main() attributes {mqt.entry_point} {
      %source = memref.get_global @source : memref<2x3xf64>
      %target = memref.alloca() : memref<2x3xf64>
      memref.copy %source, %target : memref<2x3xf64> to memref<2x3xf64>
      %row = memref.subview %source[1, 0][1, 3][1, 1]
        : memref<2x3xf64> to memref<1x3xf64, strided<[3, 1], offset: 3>>
      %first = memref.subview %target[0, 0][1, 3][1, 1]
        : memref<2x3xf64> to memref<1x3xf64, strided<[3, 1]>>
      memref.copy %row, %first
        : memref<1x3xf64, strided<[3, 1], offset: 3>>
        to memref<1x3xf64, strided<[3, 1]>>
      %scalar = memref.get_global @scalar : memref<f64>
      %scalar_target = memref.alloca() : memref<f64>
      memref.copy %scalar, %scalar_target : memref<f64> to memref<f64>
      %empty = memref.alloca() : memref<0x3xf64>
      memref.copy %empty, %empty : memref<0x3xf64> to memref<0x3xf64>
      return
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  SmallVector<uint64_t> copiedElements;
  moduleOp->walk([&](Operation* operation) {
    EXPECT_FALSE(
        (isa<LLVM::MemcpyOp, LLVM::MemmoveOp, LLVM::CondBrOp>(operation)));
    if (auto load = dyn_cast<LLVM::LoadOp>(operation)) {
      auto array = dyn_cast<LLVM::LLVMArrayType>(load.getType());
      ASSERT_TRUE(array);
      EXPECT_TRUE(array.getElementType().isF64());
      copiedElements.push_back(array.getNumElements());
    }
    if (auto function = dyn_cast<LLVM::LLVMFuncOp>(operation)) {
      EXPECT_NE(function.getName(), "memrefCopy");
    }
  });
  EXPECT_EQ(copiedElements, (SmallVector<uint64_t>{6, 3, 1}));
  ASSERT_TRUE(succeeded(runQIRCleanupPipeline(*moduleOp, true)));
  bool requiresDouble = false;
  moduleOp->walk([&](LLVM::ModuleFlagsOp flags) {
    for (const auto flag :
         flags.getFlags().getAsRange<LLVM::ModuleFlagAttr>()) {
      if (flag.getKey().getValue() == "float_computations") {
        requiresDouble =
            flag.getValue() == Builder(&context).getStrArrayAttr({"double"});
      }
    }
  });
  EXPECT_TRUE(requiresDouble);
}

TEST_F(QCToQIRAdaptiveNativeTest,
       UsesSharedAllocationVerifierForStandalonePass) {
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %condition = arith.constant false
      scf.if %condition {
        %q = qc.alloc : !qc.qubit
        qc.dealloc %q : !qc.qubit
      }
      return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_EQ(context.getLoadedDialect<mlir::mqt::MQTDialect>(), nullptr);
  expectConversionFailure(*module, "dynamic quantum allocations must be");
  EXPECT_TRUE(module->lookupSymbol<func::FuncOp>("main"));
}

TEST_F(QCToQIRAdaptiveNativeTest, RejectsExplicitRuntimeAssertions) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %condition = arith.constant false
      cf.assert %condition, "unsupported runtime precondition"
      return
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  expectConversionFailure(*moduleOp, "cf.assert");
}

TEST_F(QCToQIRAdaptiveNativeTest,
       RejectsMultipleReturnsBeforeOutputPreparation) {
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() -> i1 attributes {mqt.entry_point} {
      %q = qc.alloc : !qc.qubit
      %bit = qc.measure %q : !qc.qubit -> i1
      cf.cond_br %bit, ^left, ^right
    ^left:
      qc.dealloc %q : !qc.qubit
      return %bit : i1
    ^right:
      qc.dealloc %q : !qc.qubit
      return %bit : i1
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  expectConversionFailure(*module, "single return");
  size_t returns = 0;
  module->walk([&](func::ReturnOp op) {
    ++returns;
    EXPECT_EQ(op.getNumOperands(), 1);
  });
  EXPECT_EQ(returns, 2);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(QCToQIRAdaptiveNativeTest, PreservesConditionalReleases) {
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    func.func private @condition() -> i1
    func.func @main() attributes {mqt.entry_point} {
      %q = qc.alloc : !qc.qubit
      %reg = memref.alloc() : memref<1x!qc.qubit>
      %condition = func.call @condition() : () -> i1
      scf.if %condition {
        qc.dealloc %q : !qc.qubit
        memref.dealloc %reg : memref<1x!qc.qubit>
      } else {
        qc.dealloc %q : !qc.qubit
        memref.dealloc %reg : memref<1x!qc.qubit>
      }
      return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  SmallVector<LLVM::CallOp> releases;
  module->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_QUBIT_RELEASE ||
        call.getCallee() == qir::QIR_QUBIT_ARRAY_RELEASE) {
      releases.push_back(call);
    }
  });
  ASSERT_EQ(releases.size(), 4);
  EXPECT_EQ(releases[0]->getBlock(), releases[1]->getBlock());
  EXPECT_EQ(releases[2]->getBlock(), releases[3]->getBlock());
  EXPECT_NE(releases[0]->getBlock(), releases[2]->getBlock());
  for (auto release : releases) {
    EXPECT_TRUE(isa<LLVM::BrOp>(release->getBlock()->getTerminator()));
  }
}

TEST_F(QCToQIRAdaptiveNativeTest,
       LowersPureOutputIndicesComputedAfterMeasurement) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() -> !cbit.reg<4> attributes {mqt.entry_point} {
      %q = qc.alloc : !qc.qubit
      %r = cbit.alloc(#cbit.init<zero>) : !cbit.reg<4>
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %four = arith.constant 4 : index
      %three = arith.constant 3 : i32
      scf.for %i = %zero to %four step %one {
        %bit = qc.measure %q : !qc.qubit -> i1
        %i32 = arith.index_cast %i : index to i32
        %reversed = arith.subi %three, %i32 : i32
        %index = arith.index_cast %reversed : i32 to index
        %condition = arith.cmpi slt, %i, %four : index
        %selected = scf.if %condition -> index {
          scf.yield %index : index
        } else {
          scf.yield %zero : index
        }
        cbit.store %bit, %r[%selected] : !cbit.reg<4>
      }
      qc.dealloc %q : !qc.qubit
      return %r : !cbit.reg<4>
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  OwningOpRef<ModuleOp> converted = moduleOp->clone();
  qc::MeasureOp measure;
  cbit::StoreOp store;
  scf::ForOp loop;
  moduleOp->walk([&](qc::MeasureOp op) { measure = op; });
  moduleOp->walk([&](cbit::StoreOp op) { store = op; });
  moduleOp->walk([&](scf::ForOp op) { loop = op; });
  ASSERT_TRUE(measure);
  ASSERT_TRUE(store);
  ASSERT_TRUE(loop);
  auto index = store.getIndex();
  auto* producer = index.getDefiningOp();
  ASSERT_TRUE(producer);
  EXPECT_TRUE(measure->isBeforeInBlock(producer));

  LoweringState state;
  ASSERT_TRUE(succeeded(prepareClassicalResults(*moduleOp, state, true)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_EQ(state.cregMeasurements.at(measure),
            (std::pair<size_t, Value>{0, index}));
  EXPECT_EQ(measure->getParentOp(), loop.getOperation());
  EXPECT_TRUE(DominanceInfo(*moduleOp).dominates(index, measure));
  EXPECT_TRUE(loop.getBody()->getOps<cbit::StoreOp>().empty());

  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*converted)));
  EXPECT_TRUE(succeeded(verify(*converted)));
  LLVM::CallOp measurementCall;
  converted->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_MEASURE) {
      measurementCall = call;
    }
  });
  ASSERT_TRUE(measurementCall);
  auto resultLoad =
      measurementCall.getArgOperands()[1].getDefiningOp<LLVM::LoadOp>();
  ASSERT_TRUE(resultLoad);
  auto resultSlot = resultLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
  ASSERT_TRUE(resultSlot);
  ASSERT_EQ(resultSlot.getDynamicIndices().size(), 1U);
  EXPECT_TRUE(isa<BlockArgument>(resultSlot.getDynamicIndices().front()));
  EXPECT_TRUE(converted->lookupSymbol<LLVM::LLVMFuncOp>(
      qir::QIR_RESULT_ARRAY_RECORD_OUTPUT));
  EXPECT_FALSE(
      converted->lookupSymbol<LLVM::LLVMFuncOp>(qir::QIR_BOOL_RECORD_OUTPUT));
}

TEST_F(QCToQIRAdaptiveNativeTest,
       KeepsOutputIndexComputationsAfterMeasurements) {
  for (const auto* definition : {
           "%index = arith.index_cast %bit : i1 to index\n",
           "%old = cbit.load %scratch[%zero] : !cbit.reg<2>\n"
           "%index = arith.index_cast %old : i1 to index\n",
           "%index = func.call @index() : () -> index\n",
           "%index = arith.divui %one, %divisor : index\n",
           "%index = scf.if %condition -> index {\n"
           "  qc.x %q : !qc.qubit\n"
           "  scf.yield %zero : index\n"
           "} else { scf.yield %one : index }\n",
       }) {
    SCOPED_TRACE(definition);
    auto source = std::string(R"mlir(module {
      func.func private @index() -> index
      func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
        %q = qc.alloc : !qc.qubit
        %r = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %scratch = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %zero = arith.constant 0 : index
        %zero_i32 = arith.constant 0 : i32
        %one = arith.constant 1 : index
        %condition = arith.constant false
        %divisor = func.call @index() : () -> index
        %first = qc.measure %q : !qc.qubit -> i1
        %first_index = arith.index_cast %zero_i32 : i32 to index
        cbit.store %first, %r[%first_index] : !cbit.reg<2>
        %bit = qc.measure %q : !qc.qubit -> i1
    )mlir") + definition +
                  R"mlir(
        cbit.store %bit, %r[%index] : !cbit.reg<2>
        qc.dealloc %q : !qc.qubit
        return %r : !cbit.reg<2>
      }
    })mlir";
    auto moduleOp = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    qc::MeasureOp measurement;
    cbit::StoreOp store;
    moduleOp->walk([&](qc::MeasureOp op) { measurement = op; });
    moduleOp->walk([&](cbit::StoreOp op) { store = op; });
    auto index = store.getIndex();
    LoweringState state;
    ASSERT_TRUE(succeeded(prepareClassicalResults(*moduleOp, state, true)));
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    EXPECT_TRUE(state.cregMeasurements.empty());
    EXPECT_EQ(store.getIndex(), index);
    EXPECT_TRUE(measurement->isBeforeInBlock(index.getDefiningOp()));
  }
}

TEST_F(QCToQIRAdaptiveNativeTest, ReleasesResultArraysInAllocationOrder) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  SmallVector<Value> registers;
  for (size_t i = 0; i < 12; ++i) {
    registers.push_back(builder.allocClassicalBitRegister(1));
  }

  auto moduleOp = builder.finalize(registers);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  SmallVector<Value> allocated;
  SmallVector<Value> released;
  moduleOp->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_RESULT_ARRAY_ALLOC) {
      allocated.push_back(call.getOperand(1));
    } else if (call.getCallee() == qir::QIR_RESULT_ARRAY_RELEASE) {
      released.push_back(call.getOperand(1));
    }
  });
  ASSERT_EQ(allocated.size(), 12);
  EXPECT_EQ(released, allocated);
}

TEST_F(QCToQIRAdaptiveNativeTest, PreservesEntryBlockQubitAllocations) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func private @condition() -> i1
    func.func @main() attributes {mqt.entry_point} {
      %q = qc.alloc : !qc.qubit
      %c = func.call @condition() : () -> i1
      scf.if %c {
        qc.x %q : !qc.qubit
      }
      %reg = memref.alloc() : memref<1x!qc.qubit>
      qc.dealloc %q : !qc.qubit
      memref.dealloc %reg : memref<1x!qc.qubit>
      return
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST_F(QCToQIRAdaptiveNativeTest,
       NormalizesFactorableControlledGlobalPhaseBeforeLowering) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto control = builder.allocQubit();
  auto target = builder.allocQubit();
  builder.ctrl(control, target, [&](Value targetArg) {
    builder.x(targetArg);
    builder.gphase(0.317);
  });
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(succeeded(runQCToQIRAdaptiveConversion(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST_F(QCToQIRAdaptiveNativeTest, LowersPopulationCountThroughMathToLLVM) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto value = LLVM::UndefOp::create(builder, builder.getIntegerType(5));
  auto count = math::CtPopOp::create(builder, value);
  auto angle = arith::UIToFPOp::create(builder, builder.getF64Type(), count);
  builder.rx(angle, builder.allocQubit());
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  bool retainsMathPopulationCount = false;
  bool hasLLVMPopulationCount = false;
  module->walk([&](Operation* operation) {
    retainsMathPopulationCount |= isa<math::CtPopOp>(operation);
    hasLLVMPopulationCount |= isa<LLVM::CtPopOp>(operation);
  });
  EXPECT_FALSE(retainsMathPopulationCount);
  EXPECT_TRUE(hasLLVMPopulationCount);
}

TEST_F(QCToQIRAdaptiveNativeTest, LowersUnreturnedClassicalControlRegister) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto q = builder.allocQubit();
  auto c = builder.allocClassicalBitRegister(1);
  builder.measure(q, c, 0);
  builder.scfIf(c, 0, [&] { builder.x(q); });
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversion(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  EXPECT_FALSE(
      module->lookupSymbol<LLVM::LLVMFuncOp>(qir::QIR_RESULT_ARRAY_ALLOC));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(qir::QIR_READ_RESULT));
  EXPECT_FALSE(module->lookupSymbol<LLVM::LLVMFuncOp>(
      qir::QIR_RESULT_ARRAY_RECORD_OUTPUT));
}

TEST_F(QCToQIRAdaptiveNativeTest,
       LowersZeroInitializedClassicalControlRegister) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto q = builder.allocQubit();
  auto c = builder.allocClassicalBitRegister(1);
  builder.scfIf(c, 0, [&] { builder.x(q); });
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversion(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  EXPECT_FALSE(
      module->lookupSymbol<LLVM::LLVMFuncOp>(qir::QIR_RESULT_ARRAY_ALLOC));
  EXPECT_FALSE(module->lookupSymbol<LLVM::LLVMFuncOp>(qir::QIR_READ_RESULT));
}

TEST_F(QCToQIRAdaptiveNativeTest, LowersClassicalRegisterComparison) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto q = builder.allocQubit();
  auto c = builder.allocClassicalBitRegister(3);
  builder.measure(q, c, 0);
  auto rhs = builder.getIntegerAttr(builder.getIntegerType(3), 2);
  auto comparison =
      arith::CmpIOp::create(builder, arith::CmpIPredicate::ult,
                            cbit::ReadOp::create(builder, rhs.getType(), c),
                            arith::ConstantOp::create(builder, rhs));
  builder.scfIf(comparison, [&] { builder.x(q); });
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversion(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
  bool retainsComparison = false;
  module->walk([&](arith::CmpIOp) { retainsComparison = true; });
  EXPECT_FALSE(retainsComparison);
}

TEST_F(QCToQIRAdaptiveNativeTest, RejectsRegisterCallsAtTheSharedBoundary) {
  const auto sources = {
      R"mlir(module {
        func.func private @callee(!cbit.reg<1>)
        func.func @main() attributes {mqt.entry_point} {
          %c = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
          func.call @callee(%c) : (!cbit.reg<1>) -> ()
          return
        }
      })mlir",
      R"mlir(module {
        func.func private @callee() -> !cbit.reg<1>
        func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
          %c = func.call @callee() : () -> !cbit.reg<1>
          return %c : !cbit.reg<1>
        }
      })mlir",
      R"mlir(module {
        func.func private @callee(!cbit.reg<1>)
        func.func @main() attributes {mqt.entry_point} {
          %callee = func.constant @callee : (!cbit.reg<1>) -> ()
          %c = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
          func.call_indirect %callee(%c) : (!cbit.reg<1>) -> ()
          return
        }
      })mlir",
      R"mlir(module {
        func.func private @callee() -> !cbit.reg<1>
        func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
          %callee = func.constant @callee : () -> !cbit.reg<1>
          %c = func.call_indirect %callee() : () -> !cbit.reg<1>
          return %c : !cbit.reg<1>
        }
      })mlir",
  };
  for (const auto* source : sources) {
    SCOPED_TRACE(source);
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    expectConversionFailure(*module,
                            "does not support CBit registers in calls");
  }
}

TEST_F(QCToQIRAdaptiveNativeTest, PreservesScalarCalls) {
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    func.func private @callee(i64) -> i64
    func.func @main() attributes {mqt.entry_point} {
      %zero = arith.constant 0 : i64
      %callee = func.constant @callee : (i64) -> i64
      %direct = func.call @callee(%zero) : (i64) -> i64
      %indirect = func.call_indirect %callee(%direct) : (i64) -> i64
      return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(QCToQIRAdaptiveNativeTest,
       RejectsMixedClassicalRegisterRepresentations) {
  for (const auto* forwarding : {
           R"mlir(%selected = scf.if %true -> (!cbit.reg<1>) {
             scf.yield %returned : !cbit.reg<1>
           } else {
             scf.yield %local : !cbit.reg<1>
           })mlir",
           R"mlir(cf.cond_br %true, ^join(%returned : !cbit.reg<1>),
                                   ^join(%local : !cbit.reg<1>)
           ^join(%selected: !cbit.reg<1>):)mlir",
       }) {
    SCOPED_TRACE(forwarding);
    const std::string source = R"mlir(
    module {
      func.func private @consume(i1)
      func.func @main() -> !cbit.reg<1>
          attributes {mqt.entry_point} {
        %true = arith.constant true
        %c0 = arith.constant 0 : index
        %returned = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %local = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
    )mlir" + std::string(forwarding) +
                               R"mlir(
        %bit = cbit.load %selected[%c0] : !cbit.reg<1>
        %whole = cbit.read %selected : !cbit.reg<1> -> i1
        %matches_read = cbit.read %selected : !cbit.reg<1> -> i1
        %matches_rhs = arith.constant 0 : i1
        %matches = arith.cmpi eq, %matches_read, %matches_rhs : i1
        cbit.write %true, %selected : i1, !cbit.reg<1>
        %combined = arith.xori %bit, %whole : i1
        %result = arith.xori %combined, %matches : i1
        func.call @consume(%result) : (i1) -> ()
        return %returned : !cbit.reg<1>
      }
    }
  )mlir";
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));

    size_t mixedRepresentationDiagnostics = 0;
    const ScopedDiagnosticHandler handler(
        &context, [&](Diagnostic& diagnostic) {
          mixedRepresentationDiagnostics +=
              StringRef(diagnostic.str())
                  .contains("adaptive QIR conversion cannot merge returned and "
                            "local CBit "
                            "registers");
          return success();
        });
    EXPECT_TRUE(failed(runQCToQIRAdaptiveConversionSimple(*module)));
    EXPECT_EQ(mixedRepresentationDiagnostics, 4);
  }
}

TEST_F(QCToQIRAdaptiveNativeTest, LowersReturnedRegisterMerge) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func private @consume(i1)
      func.func @main() -> (!cbit.reg<1>, !cbit.reg<1>)
          attributes {mqt.entry_point} {
        %true = arith.constant true
        %c0 = arith.constant 0 : index
        %first = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %second = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %selected = scf.if %true -> (!cbit.reg<1>) {
          scf.yield %first : !cbit.reg<1>
        } else {
          scf.yield %second : !cbit.reg<1>
        }
        %bit = cbit.load %selected[%c0] : !cbit.reg<1>
        %whole = cbit.read %selected : !cbit.reg<1> -> i1
        %matches_read = cbit.read %selected : !cbit.reg<1> -> i1
        %matches_rhs = arith.constant 0 : i1
        %matches = arith.cmpi eq, %matches_read, %matches_rhs : i1
        %combined = arith.xori %bit, %whole : i1
        %result = arith.xori %combined, %matches : i1
        func.call @consume(%result) : (i1) -> ()
        return %first, %second : !cbit.reg<1>, !cbit.reg<1>
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
  size_t resultReads = 0;
  module->walk([&](LLVM::CallOp call) {
    resultReads += call.getCallee() == qir::QIR_READ_RESULT;
  });
  EXPECT_EQ(resultReads, 3);
}

TEST_F(QCToQIRAdaptiveNativeTest, RejectsWriteThroughReturnedRegisterMerge) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() -> (!cbit.reg<1>, !cbit.reg<1>)
          attributes {mqt.entry_point} {
        %true = arith.constant true
        %first = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %second = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %selected = scf.if %true -> (!cbit.reg<1>) {
          scf.yield %first : !cbit.reg<1>
        } else {
          scf.yield %second : !cbit.reg<1>
        }
        cbit.write %true, %selected : i1, !cbit.reg<1>
        return %first, %second : !cbit.reg<1>, !cbit.reg<1>
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  expectConversionFailure(*module, "does not support non-measurement "
                                   "writes to returned CBit registers");
}

TEST_F(QCToQIRAdaptiveNativeTest, RejectsHelperLocalClassicalResources) {
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

TEST_F(QCToQIRAdaptiveNativeTest, RejectsUnsupportedEntryOutputs) {
  const std::array names{
      "selected register",
      "zero scalar",
      "nonzero scalar",
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
    } else {
      outputs.push_back(builder.intConstant(shape == 1 ? 0 : 7));
    }

    auto module = builder.finalize(outputs);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    expectConversionFailure(*module, "QIR entry return supports only direct "
                                     "measurements and CBit allocations");
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST_F(QCToQIRAdaptiveNativeTest, PreservesOutputOrderMultiplicityAndLabels) {
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
      registers[2], registers[1], bits[3], bits[0],
      registers[0], bits[1],      bits[3], registers[1],
  };

  auto module = builder.finalize(outputs);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*module)));
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
  module->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_RECORD_OUTPUT) {
      recorded.push_back(call.getOperand(0));
    } else if (call.getCallee() == qir::QIR_ARRAY_RECORD_OUTPUT) {
      containerPositions.push_back(recorded.size());
    } else if (call.getCallee() == qir::QIR_RESULT_ARRAY_RECORD_OUTPUT) {
      containerPositions.push_back(recorded.size());
      bool found = false;
      for (auto measurement : measurements) {
        auto load = measurement.getDefiningOp<LLVM::LoadOp>();
        auto element =
            load ? load.getAddr().getDefiningOp<LLVM::GEPOp>() : LLVM::GEPOp{};
        if (element && element.getBase() == call.getOperand(1)) {
          recorded.push_back(measurement);
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found);
    } else {
      return;
    }
    auto address = call.getOperands().back().getDefiningOp<LLVM::AddressOfOp>();
    ASSERT_TRUE(address);
    auto global = module->lookupSymbol<LLVM::GlobalOp>(address.getGlobalName());
    ASSERT_TRUE(global);
    auto label = global->getAttrOfType<StringAttr>("value");
    ASSERT_TRUE(label);
    EXPECT_TRUE(labels.insert(label.getValue()).second)
        << label.getValue().str();
  });
  EXPECT_EQ(recorded, (SmallVector<Value>{measurements[2], measurements[1],
                                          measurements[3], measurements[0],
                                          measurements[0], measurements[1],
                                          measurements[3], measurements[1]}));
  EXPECT_EQ(containerPositions, (SmallVector<size_t>{0, 1, 4, 7}));
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
  SmallVector<Value> allocatedScalars;
  SmallVector<Value> releasedScalars;
  SmallVector<Value> allocatedArrays;
  SmallVector<Value> releasedArrays;
  module->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_RESULT_ALLOC) {
      allocatedScalars.push_back(call.getResult());
    } else if (call.getCallee() == qir::QIR_RESULT_RELEASE) {
      releasedScalars.push_back(call.getOperand(0));
    } else if (call.getCallee() == qir::QIR_RESULT_ARRAY_ALLOC) {
      allocatedArrays.push_back(call.getOperand(1));
    } else if (call.getCallee() == qir::QIR_RESULT_ARRAY_RELEASE) {
      releasedArrays.push_back(call.getOperand(1));
    }
  });
  EXPECT_EQ(allocatedScalars.size(), 1U);
  EXPECT_EQ(allocatedArrays.size(), 3U);
  EXPECT_EQ(releasedScalars, allocatedScalars);
  EXPECT_EQ(releasedArrays, allocatedArrays);
}

TEST_F(QCToQIRAdaptiveNativeTest, MaterializesBooleanOutputsBeforeRecording) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto measured = builder.measure(builder.allocQubit());
  auto first = builder.allocClassicalBitRegister(1);
  auto second = builder.allocClassicalBitRegister(1);
  builder.storeClassicalBit(builder.boolConstant(true), first, 0);
  builder.storeClassicalBit(builder.boolConstant(false), second, 0);
  auto moduleOp =
      builder.finalize({measured, builder.boolConstant(true), first, second});
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  bool recording = false;
  moduleOp->walk([&](Operation* operation) {
    if (auto call = dyn_cast<LLVM::CallOp>(operation);
        call && call.getCallee() &&
        call.getCallee()->ends_with("_record_output")) {
      recording = true;
    }
    if (isa<LLVM::LoadOp>(operation)) {
      EXPECT_FALSE(recording);
    }
  });
  EXPECT_TRUE(recording);
}

TEST_F(QCToQIRAdaptiveNativeTest, AcceptsZeroInitializedClassicalRegister) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto c = builder.allocClassicalBitRegister(1);

  auto module = builder.finalize(c);
  ASSERT_TRUE(module);

  EXPECT_TRUE(succeeded(runQCToQIRAdaptiveConversion(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(QCToQIRAdaptiveNativeTest,
       LowersInternalZeroInitializedRegisterStorage) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto c = builder.allocClassicalBitRegister(8192);
  auto first = builder.loadClassicalBit(c, 0);
  builder.storeClassicalBit(first, c, 1);
  auto module = builder.finalize();
  ASSERT_TRUE(module);

  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t allocs = 0;
  size_t loads = 0;
  size_t stores = 0;
  size_t initializations = 0;
  module->walk([&](LLVM::AllocaOp) { ++allocs; });
  module->walk([&](LLVM::LoadOp) { ++loads; });
  module->walk([&](LLVM::StoreOp store) {
    if (auto zero = store.getValue().getDefiningOp<LLVM::ZeroOp>()) {
      ++initializations;
      EXPECT_EQ(zero.getType(),
                LLVM::LLVMArrayType::get(IntegerType::get(&context, 1), 8192));
    } else {
      ++stores;
    }
  });
  EXPECT_EQ(allocs, 1);
  EXPECT_EQ(loads, 1);
  EXPECT_EQ(stores, 1);
  EXPECT_EQ(initializations, 1);
}

TEST_F(QCToQIRAdaptiveNativeTest, InitializesLocalRegisterInsideLoop) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func private @consume(i1)
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %false = arith.constant false
        %true = arith.constant true
        %answer = scf.for %i = %c0 to %c2 step %c1
            iter_args(%last = %false) -> i1 {
          %r = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
          %v = cbit.load %r[%c0] : !cbit.reg<1>
          cbit.store %true, %r[%c0] : !cbit.reg<1>
          scf.yield %v : i1
        }
        func.call @consume(%answer) : (i1) -> ()
        return
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  LLVM::StoreOp fill;
  LLVM::LoadOp load;
  moduleOp->walk([&](LLVM::StoreOp op) {
    if (op.getValue().getDefiningOp<LLVM::ZeroOp>()) {
      fill = op;
    }
  });
  moduleOp->walk([&](LLVM::LoadOp op) { load = op; });
  ASSERT_TRUE(fill);
  ASSERT_TRUE(load);
  ASSERT_EQ(fill->getBlock(), load->getBlock());
  EXPECT_TRUE(fill->isBeforeInBlock(load));
  auto main = moduleOp->lookupSymbol<LLVM::LLVMFuncOp>("main");
  ASSERT_TRUE(main);
  EXPECT_NE(fill->getBlock(), &main.getBody().front());
}

TEST_F(QCToQIRAdaptiveNativeTest, SupportsDynamicInternalRegisterIndices) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto c = builder.allocClassicalBitRegister(2);
  auto unknown = LLVM::UndefOp::create(builder, builder.getI64Type());
  auto index = arith::IndexCastOp::create(builder, builder.getIndexType(),
                                          unknown.getResult());
  builder.storeClassicalBit(builder.boolConstant(true), c, index.getResult());
  auto value = builder.loadClassicalBit(c, index.getResult());
  builder.storeClassicalBit(value, c, 0);
  auto module = builder.finalize();
  ASSERT_TRUE(module);

  EXPECT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(QCToQIRAdaptiveNativeTest, RecordsComputedStoreAfterMeasurement) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto q = builder.allocQubit();
  auto c = builder.allocClassicalBitRegister(1);
  auto first = builder.measure(q, c, 0);
  auto second = builder.measure(q);
  auto output = builder.measure(q);
  auto combined = arith::XOrIOp::create(builder, first, second);
  builder.storeClassicalBit(combined, c, 0);

  auto module = builder.finalize(ValueRange{c, output});
  ASSERT_TRUE(module);

  ASSERT_TRUE(succeeded(runQCToQIRAdaptiveConversionSimple(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(
      module->lookupSymbol<LLVM::LLVMFuncOp>(qir::QIR_BOOL_RECORD_OUTPUT));
  SmallVector<LLVM::CallOp> measurements;
  SmallVector<LLVM::CallOp> reads;
  SmallVector<Value> allocated;
  SmallVector<Value> released;
  module->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == qir::QIR_MEASURE) {
      measurements.push_back(call);
    } else if (call.getCallee() == qir::QIR_READ_RESULT) {
      reads.push_back(call);
    } else if (call.getCallee() == qir::QIR_RESULT_ALLOC) {
      allocated.push_back(call.getResult());
    } else if (call.getCallee() == qir::QIR_RESULT_RELEASE) {
      released.push_back(call.getOperand(0));
    }
  });
  ASSERT_EQ(measurements.size(), 3U);
  ASSERT_EQ(reads.size(), 2U);
  EXPECT_EQ(allocated.size(), 2U);
  EXPECT_EQ(released, allocated);
  EXPECT_EQ(measurements[0].getOperand(1), measurements[1].getOperand(1));
  EXPECT_NE(measurements[0].getOperand(1), measurements[2].getOperand(1));
  for (size_t i = 0; i < reads.size(); ++i) {
    EXPECT_EQ(reads[i].getOperand(0), measurements[i].getOperand(1));
    EXPECT_TRUE(measurements[i]->isBeforeInBlock(reads[i]));
    EXPECT_TRUE(reads[i]->isBeforeInBlock(measurements[i + 1]));
  }
  size_t combinations = 0;
  module->walk([&](LLVM::XOrOp op) {
    ++combinations;
    EXPECT_EQ(op.getLhs(), reads[0].getResult());
    EXPECT_EQ(op.getRhs(), reads[1].getResult());
    EXPECT_TRUE(measurements.back()->isBeforeInBlock(op));
  });
  EXPECT_EQ(combinations, 1U);
}

TEST_F(QCToQIRAdaptiveNativeTest, RejectsClassicalMemrefOutput) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  const auto type = MemRefType::get({1}, builder.getI8Type());
  auto memref = memref::AllocOp::create(builder, type).getResult();

  auto module = builder.finalize(memref);
  ASSERT_TRUE(module);

  expectConversionFailure(*module, "QIR entry return supports only",
                          runQCToQIRAdaptiveConversion);
}

TEST_P(QCToQIRAdaptiveTest, ProgramEquivalence) {
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

  EXPECT_TRUE(succeeded(runQCToQIRAdaptiveConversion(program.get())));
  printer.record(program.get(), "Converted QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(program.get(), true).succeeded());
  printer.record(program.get(), "Canonicalized Converted QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      ::mqt::test::buildMLIRProgram(context.get(), referenceBuilder,
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
        QCToQIRAdaptiveTestCase{"IfWithMeasurement",
                                MQT_NAMED_BUILDER(qc::ifWithMeasurement),
                                MQT_NAMED_BUILDER(qir::ifWithMeasurement)},
        QCToQIRAdaptiveTestCase{"IfWithCreg", MQT_NAMED_BUILDER(qc::ifWithCreg),
                                MQT_NAMED_BUILDER(qir::ifWithCreg)},
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
                             "CtrlTwo", MQT_NAMED_BUILDER(qc::ctrlTwo),
                             MQT_NAMED_BUILDER(qir::ctrlTwo<true>)}));
/// @}
