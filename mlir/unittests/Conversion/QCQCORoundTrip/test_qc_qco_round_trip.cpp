/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <string>

using namespace mlir;

namespace {

class QCQCORoundTripTest : public testing::Test {
protected:
  MLIRContext context;

  QCQCORoundTripTest() {
    DialectRegistry registry;
    registry
        .insert<cbit::CBitDialect, mqt::MQTDialect, qc::QCDialect,
                qco::QCODialect, qtensor::QTensorDialect, arith::ArithDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  [[nodiscard]] LogicalResult runRoundTrip(ModuleOp module) {
    PassManager pm(&context);
    pm.addPass(createQCToQCO());
    pm.addPass(createQCOToQC());
    return pm.run(module);
  }

  static void expectNoScratchStorage(ModuleOp module) {
    bool containsScratchStorage = false;
    module.walk([&](Operation* operation) {
      containsScratchStorage |=
          isa<memref::AllocaOp, memref::LoadOp, memref::StoreOp>(operation);
    });
    EXPECT_FALSE(containsScratchStorage);
  }
};

} // namespace

TEST_F(QCQCORoundTripTest, PreservesSharedMQTMetadata) {
  constexpr StringLiteral source = R"mlir(
module {
  func.func @main(%theta: f64 {mqt.input_name = "theta"})
      attributes {mqt.entry_point, mqt.source_name = "source"} {
    %reg = memref.alloc() {mqt.register_name = "q"}
        : memref<2x!qc.qubit>
    memref.dealloc %reg : memref<2x!qc.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runRoundTrip(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(function);
  EXPECT_TRUE(mqt::isEntryPoint(function));
  auto sourceName = function->getAttrOfType<StringAttr>(
      mqt::MQTDialect::SourceNameAttrHelper::getNameStr());
  ASSERT_TRUE(sourceName);
  EXPECT_EQ(sourceName.getValue(), "source");
  const auto inputName = function.getArgAttrOfType<StringAttr>(
      0, mqt::MQTDialect::InputNameAttrHelper::getNameStr());
  ASSERT_TRUE(inputName);
  EXPECT_EQ(inputName.getValue(), "theta");

  memref::AllocOp allocation;
  moduleOp->walk([&](memref::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  const auto registerName = allocation->getAttrOfType<StringAttr>(
      mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(registerName);
  EXPECT_EQ(registerName.getValue(), "q");
}

TEST_F(QCQCORoundTripTest, PreservesClassicalRegistersWithoutConversion) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (!cbit.reg<2>, !cbit.reg<1>)
      attributes {mqt.entry_point} {
    %c0 = arith.constant 0 : index
    %zero = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "zero"} : !cbit.reg<2>
    %undefined = cbit.alloc(#cbit.init<undefined>) {mqt.register_name = "undefined"} : !cbit.reg<1>
    %q = qc.alloc : !qc.qubit
    %measurement = qc.measure %q : !qc.qubit -> i1
    cbit.store %measurement, %zero[%c0] : !cbit.reg<2>
    %loaded = cbit.load %zero[%c0] : !cbit.reg<2>
    cbit.store %loaded, %undefined[%c0] : !cbit.reg<1>
    qc.dealloc %q : !qc.qubit
    return %zero, %undefined : !cbit.reg<2>, !cbit.reg<1>
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runRoundTrip(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  SmallVector<cbit::AllocOp> allocations;
  SmallVector<cbit::LoadOp> loads;
  SmallVector<cbit::StoreOp> stores;
  moduleOp->walk([&](cbit::AllocOp op) { allocations.push_back(op); });
  moduleOp->walk([&](cbit::LoadOp op) { loads.push_back(op); });
  moduleOp->walk([&](cbit::StoreOp op) { stores.push_back(op); });
  ASSERT_EQ(allocations.size(), 2);
  ASSERT_EQ(loads.size(), 1);
  ASSERT_EQ(stores.size(), 2);
  EXPECT_EQ(allocations[0].getInitialization(), cbit::Initialization::Zero);
  EXPECT_EQ(allocations[0]
                ->getAttrOfType<StringAttr>(
                    mqt::MQTDialect::RegisterNameAttrHelper::getNameStr())
                .getValue(),
            "zero");
  EXPECT_EQ(allocations[1].getInitialization(),
            cbit::Initialization::Undefined);
  EXPECT_EQ(allocations[1]
                ->getAttrOfType<StringAttr>(
                    mqt::MQTDialect::RegisterNameAttrHelper::getNameStr())
                .getValue(),
            "undefined");
  EXPECT_EQ(loads.front().getReg(), allocations.front().getResult());
  EXPECT_EQ(stores.front().getReg(), allocations.front().getResult());
  EXPECT_EQ(stores.back().getReg(), allocations.back().getResult());

  auto main = moduleOp->lookupSymbol<func::FuncOp>("main");
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), allocations[0].getResult());
  EXPECT_EQ(returnOp.getOperand(1), allocations[1].getResult());
}

TEST_F(QCQCORoundTripTest, PreservesClassicalIfResultWithoutScratch) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%condition: i1) -> i64
      attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %result = scf.if %condition -> i64 {
      qc.h %q : !qc.qubit
      %then = arith.constant 1 : i64
      scf.yield %then : i64
    } else {
      qc.x %q : !qc.qubit
      %else = arith.constant 2 : i64
      scf.yield %else : i64
    }
    qc.dealloc %q : !qc.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runRoundTrip(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::IfOp ifOp;
  module->walk([&](scf::IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getNumResults(), 1);

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), ifOp.getResult(0));
  expectNoScratchStorage(*module);
}

TEST_F(QCQCORoundTripTest, PreservesClassicalIndexSwitchResultWithoutScratch) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%index: index) -> i64
      attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %result = scf.index_switch %index -> i64
    case 0 {
      qc.h %q : !qc.qubit
      %case = arith.constant 1 : i64
      scf.yield %case : i64
    }
    default {
      qc.x %q : !qc.qubit
      %default = arith.constant 2 : i64
      scf.yield %default : i64
    }
    qc.dealloc %q : !qc.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runRoundTrip(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::IndexSwitchOp switchOp;
  module->walk([&](scf::IndexSwitchOp candidate) { switchOp = candidate; });
  ASSERT_TRUE(switchOp);
  ASSERT_EQ(switchOp.getNumResults(), 1);

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), switchOp.getResult(0));
  expectNoScratchStorage(*module);
}

TEST_F(QCQCORoundTripTest, PreservesDenseUnitaryMatrixAndQubitArity) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %q0 = qc.alloc : !qc.qubit
    %q1 = qc.alloc : !qc.qubit
    qc.unitary dense<[
        [(1.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0)],
        [(0.0,0.0), (0.0,0.0), (1.0,0.0), (0.0,0.0)],
        [(0.0,0.0), (1.0,0.0), (0.0,0.0), (0.0,0.0)],
        [(0.0,0.0), (0.0,0.0), (0.0,0.0), (1.0,0.0)]]>
        : tensor<4x4xcomplex<f64>> %q0, %q1 : !qc.qubit, !qc.qubit
    qc.dealloc %q0 : !qc.qubit
    qc.dealloc %q1 : !qc.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  ElementsAttr originalMatrix;
  module->walk(
      [&](qc::UnitaryOp unitary) { originalMatrix = unitary.getMatrix(); });
  ASSERT_TRUE(originalMatrix);

  std::string serialized;
  llvm::raw_string_ostream stream(serialized);
  module->print(stream);
  stream.flush();
  auto reparsed = parseSourceString<ModuleOp>(serialized, &context);
  ASSERT_TRUE(reparsed);
  ASSERT_TRUE(succeeded(verify(*reparsed)));
  qc::UnitaryOp reparsedUnitary;
  reparsed->walk([&](qc::UnitaryOp candidate) { reparsedUnitary = candidate; });
  ASSERT_TRUE(reparsedUnitary);
  EXPECT_EQ(reparsedUnitary.getQubits().size(), 2U);
  EXPECT_EQ(reparsedUnitary.getMatrix(), originalMatrix);

  ASSERT_TRUE(succeeded(runRoundTrip(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  qc::UnitaryOp unitary;
  module->walk([&](qc::UnitaryOp candidate) { unitary = candidate; });
  ASSERT_TRUE(unitary);
  EXPECT_EQ(unitary.getQubits().size(), 2U);
  EXPECT_EQ(unitary.getMatrix(), originalMatrix);
}
