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
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <memory>
#include <ostream>
#include <string>
#include <tuple>

using namespace mlir;

namespace {

struct QCOToQCTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<qco::QCOProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<qc::QCProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QCOToQCTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os, const QCOToQCTestCase& info) {
  return os << "QCOToQC{" << info.name << ", original="
            << ::mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << ::mqt::test::displayName(info.referenceBuilder.name) << "}";
}

class QCOToQCTest : public testing::TestWithParam<QCOToQCTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                    arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

static LogicalResult runQCOToQCConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCOToQC());
  return pm.run(module);
}

TEST(QCOToQCPassContract, IsModuleAnchoredAndDeclaresCreatedDialects) {
  auto pass = createQCOToQC();
  ASSERT_TRUE(pass->getOpName());
  EXPECT_EQ(*pass->getOpName(), ModuleOp::getOperationName());

  DialectRegistry registry;
  pass->getDependentDialects(registry);
  EXPECT_TRUE(
      registry.getDialectAllocator(func::FuncDialect::getDialectNamespace()));
  EXPECT_TRUE(registry.getDialectAllocator(
      cf::ControlFlowDialect::getDialectNamespace()));
  EXPECT_TRUE(
      registry.getDialectAllocator(scf::SCFDialect::getDialectNamespace()));
}

TEST(QCOToQCRegressionTest, RejectsYieldPermutationWithoutMutation) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %out0, %out1 = scf.for %iv = %lb to %ub step %step
        iter_args(%left = %q0, %right = %q1)
        -> (!qco.qubit, !qco.qubit) {
      scf.yield %right, %left : !qco.qubit, !qco.qubit
    }
    qco.sink %out0 : !qco.qubit
    qco.sink %out1 : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OwningOpRef<ModuleOp> original(module->clone());

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str()).contains("preserve input order");
    return success();
  });
  EXPECT_TRUE(failed(runQCOToQCConversion(*module)));
  EXPECT_TRUE(sawExpectedDiagnostic);
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST(QCOToQCRegressionTest, RejectsMixedAllocationModesWithoutMutation) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %static = qco.static 0 : !qco.qubit
    %dynamic = qco.alloc : !qco.qubit
    qco.sink %static : !qco.qubit
    qco.sink %dynamic : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OwningOpRef<ModuleOp> original(module->clone());

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str()).contains("cannot mix static and dynamic");
    return success();
  });
  EXPECT_TRUE(failed(runQCOToQCConversion(*module)));
  EXPECT_TRUE(sawExpectedDiagnostic);
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST(QCOToQCRegressionTest, RejectsDuplicateStaticIndicesWithoutMutation) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %first = qco.static 7 : !qco.qubit
    %second = qco.static 7 : !qco.qubit
    qco.sink %first : !qco.qubit
    qco.sink %second : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OwningOpRef<ModuleOp> original(module->clone());

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str()).contains("found duplicate index 7");
    return success();
  });
  EXPECT_TRUE(failed(runQCOToQCConversion(*module)));
  EXPECT_TRUE(sawExpectedDiagnostic);
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST(QCOToQCRegressionTest,
     RejectsStaticIndexReacquisitionAfterSinkWithoutMutation) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %first = qco.static 7 : !qco.qubit
    %after_h = qco.h %first : !qco.qubit -> !qco.qubit
    qco.sink %after_h : !qco.qubit
    %second = qco.static 7 : !qco.qubit
    %after_x = qco.x %second : !qco.qubit -> !qco.qubit
    qco.sink %after_x : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OwningOpRef<ModuleOp> original(module->clone());

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str()).contains("found duplicate index 7");
    return success();
  });
  EXPECT_TRUE(failed(runQCOToQCConversion(*module)));
  EXPECT_TRUE(sawExpectedDiagnostic);
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST(QCOToQCRegressionTest, RejectsNonlinearQubitsWithoutMutation) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit
    %q2 = qco.x %q0 : !qco.qubit -> !qco.qubit
    qco.sink %q1 : !qco.qubit
    qco.sink %q2 : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OwningOpRef<ModuleOp> original(module->clone());

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str()).contains("exactly one use");
    return success();
  });
  EXPECT_TRUE(failed(runQCOToQCConversion(*module)));
  EXPECT_TRUE(sawExpectedDiagnostic);
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST(QCOToQCRegressionTest, PreservesQTensorInsertSlotUpdates) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
    %tensor1, %q0 = qtensor.extract %tensor0[%c0] : tensor<2x!qco.qubit>
    %tensor2, %q1 = qtensor.extract %tensor1[%c1] : tensor<2x!qco.qubit>
    %tensor3 = qtensor.insert %q0 into %tensor2[%c1] : tensor<2x!qco.qubit>
    %tensor4 = qtensor.insert %q1 into %tensor3[%c0] : tensor<2x!qco.qubit>
    %tensor5, %at0 = qtensor.extract %tensor4[%c0] : tensor<2x!qco.qubit>
    %tensor6, %at1 = qtensor.extract %tensor5[%c1] : tensor<2x!qco.qubit>
    qco.sink %at0 : !qco.qubit
    qco.sink %at1 : !qco.qubit
    qtensor.dealloc %tensor6 : tensor<2x!qco.qubit>
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<memref::StoreOp> stores;
  module->walk([&](memref::StoreOp store) { stores.push_back(store); });
  ASSERT_EQ(stores.size(), 2U);
  EXPECT_NE(stores[0].getValue(), stores[1].getValue());
  EXPECT_EQ(stores[0].getMemref(), stores[1].getMemref());

  SmallVector<memref::LoadOp> loads;
  module->walk([&](memref::LoadOp load) { loads.push_back(load); });
  ASSERT_EQ(loads.size(), 3U);
  EXPECT_TRUE(stores[1]->isBeforeInBlock(loads[2]));

  bool containsQTensorOperations = false;
  module->walk([&](Operation* operation) {
    containsQTensorOperations |=
        operation->getDialect() ==
        context.getLoadedDialect<qtensor::QTensorDialect>();
  });
  EXPECT_FALSE(containsQTensorOperations);
}

TEST(QCOToQCRegressionTest, InvalidatesQTensorCacheAcrossLoopSlotSwap) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
    %tensor1, %before = qtensor.extract %tensor0[%c0] : tensor<2x!qco.qubit>
    %tensor2 = qtensor.insert %before into %tensor1[%c0] : tensor<2x!qco.qubit>
    %tensor3 = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%tensor = %tensor2) -> (tensor<2x!qco.qubit>) {
      %tensor4, %left = qtensor.extract %tensor[%c0] : tensor<2x!qco.qubit>
      %tensor5, %right = qtensor.extract %tensor4[%c1] : tensor<2x!qco.qubit>
      %tensor6 = qtensor.insert %left into %tensor5[%c1] : tensor<2x!qco.qubit>
      %tensor7 = qtensor.insert %right into %tensor6[%c0] : tensor<2x!qco.qubit>
      scf.yield %tensor7 : tensor<2x!qco.qubit>
    }
    %tensor8, %at0 = qtensor.extract %tensor3[%c0] : tensor<2x!qco.qubit>
    %tensor9, %at1 = qtensor.extract %tensor8[%c1] : tensor<2x!qco.qubit>
    qco.sink %at0 : !qco.qubit
    qco.sink %at1 : !qco.qubit
    qtensor.dealloc %tensor9 : tensor<2x!qco.qubit>
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  auto function = *module->getOps<func::FuncOp>().begin();
  auto loops = llvm::to_vector(function.getBody().getOps<scf::ForOp>());
  ASSERT_EQ(loops.size(), 1U);
  EXPECT_EQ(llvm::range_size(loops[0].getBody()->getOps<memref::StoreOp>()),
            2U);

  SmallVector<memref::LoadOp> loadsBeforeLoop;
  SmallVector<memref::LoadOp> loadsAfterLoop;
  for (auto load : function.getBody().front().getOps<memref::LoadOp>()) {
    (load->isBeforeInBlock(loops[0]) ? loadsBeforeLoop : loadsAfterLoop)
        .push_back(load);
  }
  EXPECT_EQ(loadsBeforeLoop.size(), 1U);
  EXPECT_EQ(loadsAfterLoop.size(), 2U);
}

TEST(QCOToQCRegressionTest, RetainsQubitRegisterName) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                  qtensor::QTensorDialect, arith::ArithDialect,
                  func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  qco::QCOProgramBuilder builder(&context);
  builder.initialize();
  std::ignore = builder.allocQubitRegister(2, "named_qubits");
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));

  memref::AllocOp allocation;
  moduleOp->walk([&](memref::AllocOp op) {
    if (isa<qc::QubitType>(op.getType().getElementType())) {
      allocation = op;
    }
  });
  ASSERT_TRUE(allocation);
  const auto name = allocation->getAttrOfType<StringAttr>(
      mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_qubits");
}

TEST(QCOToQCRegressionTest, RetainsDynamicQubitRegisterName) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                  qtensor::QTensorDialect, arith::ArithDialect,
                  func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%size: index) attributes {mqt.entry_point} {
    %reg = qtensor.alloc(%size) {mqt.register_name = "named_qubits"} : tensor<?x!qco.qubit>
    qtensor.dealloc %reg : tensor<?x!qco.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));

  memref::AllocOp allocation;
  moduleOp->walk([&](memref::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  EXPECT_TRUE(allocation.getType().isDynamicDim(0));
  ASSERT_EQ(allocation.getDynamicSizes().size(), 1);
  EXPECT_EQ(allocation.getDynamicSizes().front(),
            allocation->getBlock()->getArgument(0));
  const auto name = allocation->getAttrOfType<StringAttr>(
      mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_qubits");
}

static Value
aliasSafeNestedForLoopCtrlOpWithExtractedQubit(qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto c0 = arith::ConstantIndexOp::create(b, 0);
  auto control = b.loadQubit(reg.value, c0);
  b.h(control);
  b.scfFor(1, 4, 1, [&](Value iv) {
    auto target = b.loadQubit(reg.value, iv);
    b.h(target);
    b.cx(b.loadQubit(reg.value, c0), target);
  });
  auto result = b.allocClassicalBitRegister(1);
  b.measure(control, result, 0);
  return result;
}

TEST(QCOToQCRegressionTest, PreservesClassicalIfResult) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%condition: i1) -> i64
      attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %then = arith.constant 1 : i64
    %else = arith.constant 2 : i64
    %result, %q1 = qco.if %condition args(%arg = %q0)
        -> (i64, !qco.qubit) {
      %q2 = qco.h %arg : !qco.qubit -> !qco.qubit
      qco.yield %then, %q2 : i64, !qco.qubit
    } else args(%arg = %q0) {
      %q2 = qco.x %arg : !qco.qubit -> !qco.qubit
      qco.yield %else, %q2 : i64, !qco.qubit
    }
    qco.sink %q1 : !qco.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::IfOp ifOp;
  module->walk([&](scf::IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getNumResults(), 1);
  EXPECT_TRUE(ifOp.getResult(0).getType().isInteger(64));

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), ifOp.getResult(0));

  bool containsQCOOperations = false;
  module->walk([&](Operation* operation) {
    containsQCOOperations |=
        operation->getDialect() == context.getLoadedDialect<qco::QCODialect>();
  });
  EXPECT_FALSE(containsQCOOperations);
}

TEST(QCOToQCRegressionTest, PreservesClassicalIndexSwitchResult) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%index: index) -> i64
      attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %result, %q1 = qco.index_switch %index -> (i64, !qco.qubit)
    case 0 args(%arg0 = %q0) {
      %q2 = qco.h %arg0 : !qco.qubit -> !qco.qubit
      %case = arith.constant 1 : i64
      qco.yield %case, %q2 : i64, !qco.qubit
    }
    default args(%arg0 = %q0) {
      %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
      %default = arith.constant 2 : i64
      qco.yield %default, %q2 : i64, !qco.qubit
    }
    qco.sink %q1 : !qco.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  ASSERT_TRUE(succeeded(runQCOToQCConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::IndexSwitchOp switchOp;
  module->walk([&](scf::IndexSwitchOp candidate) { switchOp = candidate; });
  ASSERT_TRUE(switchOp);
  ASSERT_EQ(switchOp.getNumResults(), 1);
  EXPECT_TRUE(switchOp.getResult(0).getType().isInteger(64));

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), switchOp.getResult(0));

  bool containsQCOOperations = false;
  module->walk([&](Operation* operation) {
    containsQCOOperations |=
        operation->getDialect() == context.getLoadedDialect<qco::QCODialect>();
  });
  EXPECT_FALSE(containsQCOOperations);
}

TEST(QCOToQCRegressionTest, PreservesClassicalForLoopState) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i64 attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    %initial = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %result, %q1 = scf.for %iv = %lb to %ub step %step
        iter_args(%value = %initial, %q = %q0) -> (i64, !qco.qubit) {
      %next = arith.addi %value, %one : i64
      %q2 = qco.h %q : !qco.qubit -> !qco.qubit
      scf.yield %next, %q2 : i64, !qco.qubit
    }
    qco.sink %q1 : !qco.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::ForOp loop;
  module->walk([&](scf::ForOp candidate) { loop = candidate; });
  ASSERT_TRUE(loop);
  ASSERT_EQ(loop.getInitArgs().size(), 1);
  EXPECT_TRUE(loop.getInitArgs().front().getType().isInteger(64));
  ASSERT_EQ(loop.getNumResults(), 1);
  EXPECT_TRUE(loop.getResult(0).getType().isInteger(64));
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  ASSERT_EQ(yield.getNumOperands(), 1);
  EXPECT_TRUE(yield.getOperand(0).getType().isInteger(64));
}

TEST(QCOToQCRegressionTest, PreservesTypeChangingClassicalWhileState) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i64 attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %initial = arith.constant 1.0 : f32
    %result, %q1 = scf.while (%input = %initial, %q = %q0)
        : (f32, !qco.qubit) -> (i64, !qco.qubit) {
      %q2 = qco.h %q : !qco.qubit -> !qco.qubit
      %condition = arith.constant false
      %next = arith.constant 7 : i64
      scf.condition(%condition) %next, %q2 : i64, !qco.qubit
    } do {
    ^bb0(%input: i64, %q: !qco.qubit):
      %q2 = qco.x %q : !qco.qubit -> !qco.qubit
      %next = arith.sitofp %input : i64 to f32
      scf.yield %next, %q2 : f32, !qco.qubit
    }
    qco.sink %q1 : !qco.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::WhileOp loop;
  module->walk([&](scf::WhileOp candidate) { loop = candidate; });
  ASSERT_TRUE(loop);
  ASSERT_EQ(loop.getInits().size(), 1);
  EXPECT_TRUE(loop.getInits().front().getType().isF32());
  ASSERT_EQ(loop.getNumResults(), 1);
  EXPECT_TRUE(loop.getResult(0).getType().isInteger(64));
  auto condition =
      cast<scf::ConditionOp>(loop.getBeforeBody()->getTerminator());
  ASSERT_EQ(condition.getArgs().size(), 1);
  EXPECT_TRUE(condition.getArgs().front().getType().isInteger(64));
  auto yield = cast<scf::YieldOp>(loop.getAfterBody()->getTerminator());
  ASSERT_EQ(yield.getNumOperands(), 1);
  EXPECT_TRUE(yield.getOperand(0).getType().isF32());
}

TEST_P(QCOToQCTest, ProgramEquivalence) {
  const auto& [nameStr, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + nameStr + ")";
  ::mqt::test::DeferredPrinter printer;

  auto program = ::mqt::test::buildMLIRProgram(context.get(), programBuilder);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCOToQCConversion(program.get())));
  printer.record(program.get(), "Converted QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized Converted QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      ::mqt::test::buildMLIRProgram(context.get(), referenceBuilder);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// \name QCOToQC/QubitManagement/QubitManagement.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOQubitManagementTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"StaticQubits", MQT_NAMED_BUILDER(qco::staticQubits),
                        MQT_NAMED_BUILDER(qc::staticQubits)},
        QCOToQCTestCase{"StaticQubitsWithOps",
                        MQT_NAMED_BUILDER(qco::staticQubitsWithOps),
                        MQT_NAMED_BUILDER(qc::staticQubitsWithOps)},
        QCOToQCTestCase{"StaticQubitsWithParametricOps",
                        MQT_NAMED_BUILDER(qco::staticQubitsWithParametricOps),
                        MQT_NAMED_BUILDER(qc::staticQubitsWithParametricOps)},
        QCOToQCTestCase{"StaticQubitsWithTwoTargetOps",
                        MQT_NAMED_BUILDER(qco::staticQubitsWithTwoTargetOps),
                        MQT_NAMED_BUILDER(qc::staticQubitsWithTwoTargetOps)},
        QCOToQCTestCase{"StaticQubitsWithCtrl",
                        MQT_NAMED_BUILDER(qco::staticQubitsWithCtrl),
                        MQT_NAMED_BUILDER(qc::staticQubitsWithCtrl)},
        QCOToQCTestCase{"StaticQubitsWithInv",
                        MQT_NAMED_BUILDER(qco::staticQubitsWithInv),
                        MQT_NAMED_BUILDER(qc::staticQubitsWithInv)},
        QCOToQCTestCase{"AllocDeallocPair",
                        MQT_NAMED_BUILDER(qco::allocSinkPair),
                        MQT_NAMED_BUILDER(qc::allocDeallocPair)}));
/// @}

/// \name QCOToQC/Modifiers/PowOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPowOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"CtrlPowSx",
                                    MQT_NAMED_BUILDER(qco::ctrlPowSx),
                                    MQT_NAMED_BUILDER(qc::ctrlPowSx)},
                    QCOToQCTestCase{"PowTwo", MQT_NAMED_BUILDER(qco::powTwo),
                                    MQT_NAMED_BUILDER(qc::powTwo)}));

/// @}

/// \name QCOToQC/Modifiers/CtrlOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOCtrlOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"CtrlTwo", MQT_NAMED_BUILDER(qco::ctrlTwo),
                                    MQT_NAMED_BUILDER(qc::ctrlTwo)},
                    QCOToQCTestCase{"CtrlTwoMixed",
                                    MQT_NAMED_BUILDER(qco::ctrlTwoMixed),
                                    MQT_NAMED_BUILDER(qc::ctrlTwoMixed)},
                    QCOToQCTestCase{"CtrlInvTwo",
                                    MQT_NAMED_BUILDER(qco::ctrlInvTwo),
                                    MQT_NAMED_BUILDER(qc::ctrlInvTwo)}));
/// @}

/// \name QCOToQC/Modifiers/InvOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOInvOpTest, QCOToQCTest,
    testing::Values(
        // iSWAP cannot be inverted with current canonicalization
        QCOToQCTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(qco::inverseIswap),
                        MQT_NAMED_BUILDER(qc::inverseIswap)},
        QCOToQCTestCase{"InverseMultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qco::inverseMultipleControlledIswap),
                        MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap)},
        // Inverse DCX is not canonicalized in QCO
        QCOToQCTestCase{"InverseDCX", MQT_NAMED_BUILDER(qco::inverseDcx),
                        MQT_NAMED_BUILDER(qc::dcx)},
        QCOToQCTestCase{"InverseMultipleControlledDCX",
                        MQT_NAMED_BUILDER(qco::inverseMultipleControlledDcx),
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx)},
        QCOToQCTestCase{"InvTwo", MQT_NAMED_BUILDER(qco::invTwo),
                        MQT_NAMED_BUILDER(qc::invTwo)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOBarrierOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Barrier", MQT_NAMED_BUILDER(qco::barrier),
                                    MQT_NAMED_BUILDER(qc::barrier)},
                    QCOToQCTestCase{"BarrierTwoQubits",
                                    MQT_NAMED_BUILDER(qco::barrierTwoQubits),
                                    MQT_NAMED_BUILDER(qc::barrierTwoQubits)},
                    QCOToQCTestCase{
                        "BarrierMultipleQubits",
                        MQT_NAMED_BUILDER(qco::barrierMultipleQubits),
                        MQT_NAMED_BUILDER(qc::barrierMultipleQubits)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCODCXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"DCX", MQT_NAMED_BUILDER(qco::dcx),
                                    MQT_NAMED_BUILDER(qc::dcx)},
                    QCOToQCTestCase{"SingleControlledDCX",
                                    MQT_NAMED_BUILDER(qco::singleControlledDcx),
                                    MQT_NAMED_BUILDER(qc::singleControlledDcx)},
                    QCOToQCTestCase{
                        "MultipleControlledDCX",
                        MQT_NAMED_BUILDER(qco::multipleControlledDcx),
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOECROpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"ECR", MQT_NAMED_BUILDER(qco::ecr),
                                    MQT_NAMED_BUILDER(qc::ecr)},
                    QCOToQCTestCase{"SingleControlledECR",
                                    MQT_NAMED_BUILDER(qco::singleControlledEcr),
                                    MQT_NAMED_BUILDER(qc::singleControlledEcr)},
                    QCOToQCTestCase{
                        "MultipleControlledECR",
                        MQT_NAMED_BUILDER(qco::multipleControlledEcr),
                        MQT_NAMED_BUILDER(qc::multipleControlledEcr)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCOGPhaseOpTest, QCOToQCTest,
                         testing::Values(QCOToQCTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qco::globalPhase),
                             MQT_NAMED_BUILDER(qc::globalPhase)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOHOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"H", MQT_NAMED_BUILDER(qco::h),
                                    MQT_NAMED_BUILDER(qc::h)},
                    QCOToQCTestCase{"SingleControlledH",
                                    MQT_NAMED_BUILDER(qco::singleControlledH),
                                    MQT_NAMED_BUILDER(qc::singleControlledH)},
                    QCOToQCTestCase{"MultipleControlledH",
                                    MQT_NAMED_BUILDER(qco::multipleControlledH),
                                    MQT_NAMED_BUILDER(qc::multipleControlledH)},
                    QCOToQCTestCase{"HWithoutRegister",
                                    MQT_NAMED_BUILDER(qco::hWithoutRegister),
                                    MQT_NAMED_BUILDER(qc::hWithoutRegister)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOiSWAPOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"iSWAP", MQT_NAMED_BUILDER(qco::iswap),
                        MQT_NAMED_BUILDER(qc::iswap)},
        QCOToQCTestCase{"SingleControllediSWAP",
                        MQT_NAMED_BUILDER(qco::singleControlledIswap),
                        MQT_NAMED_BUILDER(qc::singleControlledIswap)},
        QCOToQCTestCase{"MultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qco::multipleControlledIswap),
                        MQT_NAMED_BUILDER(qc::multipleControlledIswap)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"P", MQT_NAMED_BUILDER(qco::p),
                                    MQT_NAMED_BUILDER(qc::p)},
                    QCOToQCTestCase{"SingleControlledP",
                                    MQT_NAMED_BUILDER(qco::singleControlledP),
                                    MQT_NAMED_BUILDER(qc::singleControlledP)},
                    QCOToQCTestCase{
                        "MultipleControlledP",
                        MQT_NAMED_BUILDER(qco::multipleControlledP),
                        MQT_NAMED_BUILDER(qc::multipleControlledP)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RCCXOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORCCXOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"RCCX", MQT_NAMED_BUILDER(qco::rccx),
                        MQT_NAMED_BUILDER(qc::rccx)},
        QCOToQCTestCase{"SingleControlledRCCX",
                        MQT_NAMED_BUILDER(qco::singleControlledRccx),
                        MQT_NAMED_BUILDER(qc::singleControlledRccx)},
        QCOToQCTestCase{"MultipleControlledRCCX",
                        MQT_NAMED_BUILDER(qco::multipleControlledRccx),
                        MQT_NAMED_BUILDER(qc::multipleControlledRccx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOROpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"R", MQT_NAMED_BUILDER(qco::r),
                                    MQT_NAMED_BUILDER(qc::r)},
                    QCOToQCTestCase{"SingleControlledR",
                                    MQT_NAMED_BUILDER(qco::singleControlledR),
                                    MQT_NAMED_BUILDER(qc::singleControlledR)},
                    QCOToQCTestCase{
                        "MultipleControlledR",
                        MQT_NAMED_BUILDER(qco::multipleControlledR),
                        MQT_NAMED_BUILDER(qc::multipleControlledR)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RX", MQT_NAMED_BUILDER(qco::rx),
                                    MQT_NAMED_BUILDER(qc::rx)},
                    QCOToQCTestCase{"SingleControlledRX",
                                    MQT_NAMED_BUILDER(qco::singleControlledRx),
                                    MQT_NAMED_BUILDER(qc::singleControlledRx)},
                    QCOToQCTestCase{
                        "MultipleControlledRX",
                        MQT_NAMED_BUILDER(qco::multipleControlledRx),
                        MQT_NAMED_BUILDER(qc::multipleControlledRx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RXX", MQT_NAMED_BUILDER(qco::rxx),
                                    MQT_NAMED_BUILDER(qc::rxx)},
                    QCOToQCTestCase{"SingleControlledRXX",
                                    MQT_NAMED_BUILDER(qco::singleControlledRxx),
                                    MQT_NAMED_BUILDER(qc::singleControlledRxx)},
                    QCOToQCTestCase{
                        "MultipleControlledRXX",
                        MQT_NAMED_BUILDER(qco::multipleControlledRxx),
                        MQT_NAMED_BUILDER(qc::multipleControlledRxx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RY", MQT_NAMED_BUILDER(qco::ry),
                                    MQT_NAMED_BUILDER(qc::ry)},
                    QCOToQCTestCase{"SingleControlledRY",
                                    MQT_NAMED_BUILDER(qco::singleControlledRy),
                                    MQT_NAMED_BUILDER(qc::singleControlledRy)},
                    QCOToQCTestCase{
                        "MultipleControlledRY",
                        MQT_NAMED_BUILDER(qco::multipleControlledRy),
                        MQT_NAMED_BUILDER(qc::multipleControlledRy)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RYY", MQT_NAMED_BUILDER(qco::ryy),
                                    MQT_NAMED_BUILDER(qc::ryy)},
                    QCOToQCTestCase{"SingleControlledRYY",
                                    MQT_NAMED_BUILDER(qco::singleControlledRyy),
                                    MQT_NAMED_BUILDER(qc::singleControlledRyy)},
                    QCOToQCTestCase{
                        "MultipleControlledRYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledRyy),
                        MQT_NAMED_BUILDER(qc::multipleControlledRyy)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RZ", MQT_NAMED_BUILDER(qco::rz),
                                    MQT_NAMED_BUILDER(qc::rz)},
                    QCOToQCTestCase{"SingleControlledRZ",
                                    MQT_NAMED_BUILDER(qco::singleControlledRz),
                                    MQT_NAMED_BUILDER(qc::singleControlledRz)},
                    QCOToQCTestCase{
                        "MultipleControlledRZ",
                        MQT_NAMED_BUILDER(qco::multipleControlledRz),
                        MQT_NAMED_BUILDER(qc::multipleControlledRz)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RZX", MQT_NAMED_BUILDER(qco::rzx),
                                    MQT_NAMED_BUILDER(qc::rzx)},
                    QCOToQCTestCase{"SingleControlledRZX",
                                    MQT_NAMED_BUILDER(qco::singleControlledRzx),
                                    MQT_NAMED_BUILDER(qc::singleControlledRzx)},
                    QCOToQCTestCase{
                        "MultipleControlledRZX",
                        MQT_NAMED_BUILDER(qco::multipleControlledRzx),
                        MQT_NAMED_BUILDER(qc::multipleControlledRzx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZZOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RZZ", MQT_NAMED_BUILDER(qco::rzz),
                                    MQT_NAMED_BUILDER(qc::rzz)},
                    QCOToQCTestCase{"SingleControlledRZZ",
                                    MQT_NAMED_BUILDER(qco::singleControlledRzz),
                                    MQT_NAMED_BUILDER(qc::singleControlledRzz)},
                    QCOToQCTestCase{
                        "MultipleControlledRZZ",
                        MQT_NAMED_BUILDER(qco::multipleControlledRzz),
                        MQT_NAMED_BUILDER(qc::multipleControlledRzz)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"S", MQT_NAMED_BUILDER(qco::s),
                                    MQT_NAMED_BUILDER(qc::s)},
                    QCOToQCTestCase{"SingleControlledS",
                                    MQT_NAMED_BUILDER(qco::singleControlledS),
                                    MQT_NAMED_BUILDER(qc::singleControlledS)},
                    QCOToQCTestCase{
                        "MultipleControlledS",
                        MQT_NAMED_BUILDER(qco::multipleControlledS),
                        MQT_NAMED_BUILDER(qc::multipleControlledS)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSdgOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Sdg", MQT_NAMED_BUILDER(qco::sdg),
                                    MQT_NAMED_BUILDER(qc::sdg)},
                    QCOToQCTestCase{"SingleControlledSdg",
                                    MQT_NAMED_BUILDER(qco::singleControlledSdg),
                                    MQT_NAMED_BUILDER(qc::singleControlledSdg)},
                    QCOToQCTestCase{
                        "MultipleControlledSdg",
                        MQT_NAMED_BUILDER(qco::multipleControlledSdg),
                        MQT_NAMED_BUILDER(qc::multipleControlledSdg)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSWAPOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SWAP", MQT_NAMED_BUILDER(qco::swap),
                        MQT_NAMED_BUILDER(qc::swap)},
        QCOToQCTestCase{"SingleControlledSWAP",
                        MQT_NAMED_BUILDER(qco::singleControlledSwap),
                        MQT_NAMED_BUILDER(qc::singleControlledSwap)},
        QCOToQCTestCase{"MultipleControlledSWAP",
                        MQT_NAMED_BUILDER(qco::multipleControlledSwap),
                        MQT_NAMED_BUILDER(qc::multipleControlledSwap)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"SX", MQT_NAMED_BUILDER(qco::sx),
                                    MQT_NAMED_BUILDER(qc::sx)},
                    QCOToQCTestCase{"SingleControlledSX",
                                    MQT_NAMED_BUILDER(qco::singleControlledSx),
                                    MQT_NAMED_BUILDER(qc::singleControlledSx)},
                    QCOToQCTestCase{
                        "MultipleControlledSX",
                        MQT_NAMED_BUILDER(qco::multipleControlledSx),
                        MQT_NAMED_BUILDER(qc::multipleControlledSx)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXdgOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SXdg", MQT_NAMED_BUILDER(qco::sxdg),
                        MQT_NAMED_BUILDER(qc::sxdg)},
        QCOToQCTestCase{"SingleControlledSXdg",
                        MQT_NAMED_BUILDER(qco::singleControlledSxdg),
                        MQT_NAMED_BUILDER(qc::singleControlledSxdg)},
        QCOToQCTestCase{"MultipleControlledSXdg",
                        MQT_NAMED_BUILDER(qco::multipleControlledSxdg),
                        MQT_NAMED_BUILDER(qc::multipleControlledSxdg)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"T", MQT_NAMED_BUILDER(qco::t_),
                                    MQT_NAMED_BUILDER(qc::t_)},
                    QCOToQCTestCase{"SingleControlledT",
                                    MQT_NAMED_BUILDER(qco::singleControlledT),
                                    MQT_NAMED_BUILDER(qc::singleControlledT)},
                    QCOToQCTestCase{
                        "MultipleControlledT",
                        MQT_NAMED_BUILDER(qco::multipleControlledT),
                        MQT_NAMED_BUILDER(qc::multipleControlledT)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTdgOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Tdg", MQT_NAMED_BUILDER(qco::tdg),
                                    MQT_NAMED_BUILDER(qc::tdg)},
                    QCOToQCTestCase{"SingleControlledTdg",
                                    MQT_NAMED_BUILDER(qco::singleControlledTdg),
                                    MQT_NAMED_BUILDER(qc::singleControlledTdg)},
                    QCOToQCTestCase{
                        "MultipleControlledTdg",
                        MQT_NAMED_BUILDER(qco::multipleControlledTdg),
                        MQT_NAMED_BUILDER(qc::multipleControlledTdg)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOU2OpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"U2", MQT_NAMED_BUILDER(qco::u2),
                                    MQT_NAMED_BUILDER(qc::u2)},
                    QCOToQCTestCase{"SingleControlledU2",
                                    MQT_NAMED_BUILDER(qco::singleControlledU2),
                                    MQT_NAMED_BUILDER(qc::singleControlledU2)},
                    QCOToQCTestCase{
                        "MultipleControlledU2",
                        MQT_NAMED_BUILDER(qco::multipleControlledU2),
                        MQT_NAMED_BUILDER(qc::multipleControlledU2)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOUOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"U", MQT_NAMED_BUILDER(qco::u),
                                    MQT_NAMED_BUILDER(qc::u)},
                    QCOToQCTestCase{"SingleControlledU",
                                    MQT_NAMED_BUILDER(qco::singleControlledU),
                                    MQT_NAMED_BUILDER(qc::singleControlledU)},
                    QCOToQCTestCase{
                        "MultipleControlledU",
                        MQT_NAMED_BUILDER(qco::multipleControlledU),
                        MQT_NAMED_BUILDER(qc::multipleControlledU)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"X", MQT_NAMED_BUILDER(qco::x),
                                    MQT_NAMED_BUILDER(qc::x)},
                    QCOToQCTestCase{"SingleControlledX",
                                    MQT_NAMED_BUILDER(qco::singleControlledX),
                                    MQT_NAMED_BUILDER(qc::singleControlledX)},
                    QCOToQCTestCase{"MultipleControlledX",
                                    MQT_NAMED_BUILDER(qco::multipleControlledX),
                                    MQT_NAMED_BUILDER(qc::multipleControlledX)},
                    QCOToQCTestCase{
                        "RepeatedControlledX",
                        MQT_NAMED_BUILDER(qco::repeatedControlledX),
                        MQT_NAMED_BUILDER(qc::repeatedControlledX)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXMinusYYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qco::xxMinusYY),
                        MQT_NAMED_BUILDER(qc::xxMinusYY)},
        QCOToQCTestCase{"SingleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::singleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY)},
        QCOToQCTestCase{"MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXPlusYYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qco::xxPlusYY),
                        MQT_NAMED_BUILDER(qc::xxPlusYY)},
        QCOToQCTestCase{"SingleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY)},
        QCOToQCTestCase{"MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Y", MQT_NAMED_BUILDER(qco::y),
                                    MQT_NAMED_BUILDER(qc::y)},
                    QCOToQCTestCase{"SingleControlledY",
                                    MQT_NAMED_BUILDER(qco::singleControlledY),
                                    MQT_NAMED_BUILDER(qc::singleControlledY)},
                    QCOToQCTestCase{
                        "MultipleControlledY",
                        MQT_NAMED_BUILDER(qco::multipleControlledY),
                        MQT_NAMED_BUILDER(qc::multipleControlledY)}));
/// @}

/// \name QCOToQC/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOZOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Z", MQT_NAMED_BUILDER(qco::z),
                                    MQT_NAMED_BUILDER(qc::z)},
                    QCOToQCTestCase{"SingleControlledZ",
                                    MQT_NAMED_BUILDER(qco::singleControlledZ),
                                    MQT_NAMED_BUILDER(qc::singleControlledZ)},
                    QCOToQCTestCase{
                        "MultipleControlledZ",
                        MQT_NAMED_BUILDER(qco::multipleControlledZ),
                        MQT_NAMED_BUILDER(qc::multipleControlledZ)}));
/// @}

/// \name QCOToQC/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOMeasureOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SingleMeasurementToSingleBit",
                        MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit),
                        MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit)},
        QCOToQCTestCase{"RepeatedMeasurementToSameBit",
                        MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit),
                        MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit)},
        QCOToQCTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits)},
        QCOToQCTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qco::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements)},
        QCOToQCTestCase{"PartialMeasurementToRegister",
                        MQT_NAMED_BUILDER(qco::partialMeasurementToRegister),
                        MQT_NAMED_BUILDER(qc::partialMeasurementToRegister)},
        QCOToQCTestCase{"DynamicallyIndexedMeasurement",
                        MQT_NAMED_BUILDER(qco::dynamicallyIndexedMeasurement),
                        MQT_NAMED_BUILDER(qc::dynamicallyIndexedMeasurement)},
        QCOToQCTestCase{"MeasurementWithoutRegisters",
                        MQT_NAMED_BUILDER(qco::measurementWithoutRegisters),
                        MQT_NAMED_BUILDER(qc::measurementWithoutRegisters)}));
/// @}

/// \name QCOToQC/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"ResetQubitAfterSingleOp",
                        MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp),
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)},
        QCOToQCTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp)},
        QCOToQCTestCase{"RepeatedResetAfterSingleOp",
                        MQT_NAMED_BUILDER(qco::repeatedResetAfterSingleOp),
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)}));
/// @}

/// \name QCOToQC/Operations/IfOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOIfOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SimpleIfOp", MQT_NAMED_BUILDER(qco::simpleIf),
                        MQT_NAMED_BUILDER(qc::simpleIf)},
        QCOToQCTestCase{"IfElse", MQT_NAMED_BUILDER(qco::ifElse),
                        MQT_NAMED_BUILDER(qc::ifElse)},
        QCOToQCTestCase{"IfTwoQubits", MQT_NAMED_BUILDER(qco::ifTwoQubits),
                        MQT_NAMED_BUILDER(qc::ifTwoQubits)},
        QCOToQCTestCase{"IfWithMeasurement",
                        MQT_NAMED_BUILDER(qco::ifWithMeasurement),
                        MQT_NAMED_BUILDER(qc::ifWithMeasurement)},
        QCOToQCTestCase{"IfWithCreg", MQT_NAMED_BUILDER(qco::ifWithCreg),
                        MQT_NAMED_BUILDER(qc::ifWithCreg)},
        QCOToQCTestCase{"NestedIfOpForLoop",
                        MQT_NAMED_BUILDER(qco::nestedIfOpForLoop),
                        MQT_NAMED_BUILDER(qc::nestedIfOpForLoop)}));
/// @}

/// \name QCOToQC/Operations/IndexSwitchOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOIndexSwitchOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"SimpleIndexSwitchOp",
                                    MQT_NAMED_BUILDER(qco::simpleIndexSwitch),
                                    MQT_NAMED_BUILDER(qc::simpleIndexSwitch)},
                    QCOToQCTestCase{
                        "IndexSwitchMultiCase",
                        MQT_NAMED_BUILDER(qco::indexSwitchMultiCase),
                        MQT_NAMED_BUILDER(qc::indexSwitchMultiCase)}));
/// @}

/// \name QCOToQC/Operations/WhileOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFWhileOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SimpleWhile", MQT_NAMED_BUILDER(qco::simpleWhileReset),
                        MQT_NAMED_BUILDER(qc::simpleWhileReset)},
        QCOToQCTestCase{"SimpleDoWhile",
                        MQT_NAMED_BUILDER(qco::simpleDoWhileReset),
                        MQT_NAMED_BUILDER(qc::simpleDoWhileReset)}));
/// @}

/// \name QCOToQC/Operations/ForOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFForOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SimpleForLoop", MQT_NAMED_BUILDER(qco::simpleForLoop),
                        MQT_NAMED_BUILDER(qc::simpleForLoop)},
        QCOToQCTestCase{"NestedForLoopIfOp",
                        MQT_NAMED_BUILDER(qco::nestedForLoopIfOp),
                        MQT_NAMED_BUILDER(qc::nestedForLoopIfOp)},
        QCOToQCTestCase{"NestedForLoopWhileOp",
                        MQT_NAMED_BUILDER(qco::nestedForLoopWhileOp),
                        MQT_NAMED_BUILDER(qc::nestedForLoopWhileOp)},
        QCOToQCTestCase{"NestedForLoopSwitchOp",
                        MQT_NAMED_BUILDER(qco::nestedForLoopSwitchOp),
                        MQT_NAMED_BUILDER(qc::nestedForLoopSwitchOp)},
        QCOToQCTestCase{
            "NestedForLoopCtrlOpWithSeparateQubit",
            MQT_NAMED_BUILDER(qco::nestedForLoopCtrlOpWithSeparateQubit),
            MQT_NAMED_BUILDER(qc::nestedForLoopCtrlOpWithSeparateQubit)},
        QCOToQCTestCase{
            "NestedForLoopCtrlOpWithExtractedQubit",
            MQT_NAMED_BUILDER(qco::nestedForLoopCtrlOpWithExtractedQubit),
            MQT_NAMED_BUILDER(
                aliasSafeNestedForLoopCtrlOpWithExtractedQubit)}));
/// @}
