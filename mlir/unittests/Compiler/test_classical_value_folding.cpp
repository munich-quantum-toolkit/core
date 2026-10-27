/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <string>
#include <utility>
#include <vector>

namespace {
TEST(ClassicalValueFolding, PreservesSuppliedQuantumSequence) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(OPENQASM 3;
include "stdgates.inc";
input float theta;
h $0; h $0; rx(0.0) $0; rx(1e-16) $0;
barrier $0; h $0; barrier $0; h $0;
)");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  ASSERT_TRUE(mlir::succeeded(qco->runPassPipeline("fold-classical-values")));
  ASSERT_TRUE(mlir::succeeded(mlir::qco::verifyLinearity(qco->module())));
  std::vector<std::string> gates;
  qco->module().walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::qco::HOp, mlir::qco::RXOp, mlir::qco::BarrierOp>(op)) {
      gates.push_back(op->getName().getStringRef().str());
    }
    if (auto rotation = mlir::dyn_cast<mlir::qco::RXOp>(op)) {
      EXPECT_TRUE(rotation.getTheta().getDefiningOp<mlir::arith::ConstantOp>());
    }
  });
  EXPECT_EQ(gates, (std::vector<std::string>{"qco.h", "qco.h", "qco.rx",
                                             "qco.rx", "qco.barrier", "qco.h",
                                             "qco.barrier", "qco.h"}));
  ASSERT_TRUE(mlir::succeeded(qco->runPassPipeline("mqt-qco-default")));
  unsigned hadamards = 0;
  qco->module().walk([&](mlir::qco::HOp) { ++hadamards; });
  EXPECT_EQ(hadamards, 2U);
}

TEST(ClassicalValueFolding, LeavesGateOnlyProgramUnchanged) {
  auto qc = mlir::QCProgram::fromOpenQASMString(
      "OPENQASM 3; include \"stdgates.inc\"; h $0; h $0;");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  const auto before = qc->str();
  mlir::PassManager manager(qc->module().getContext());
  manager.addPass(mlir::mqt::createFoldClassicalValues());
  ASSERT_TRUE(mlir::succeeded(manager.run(qc->module())));
  ASSERT_TRUE(mlir::succeeded(manager.run(qc->module())));
  EXPECT_EQ(qc->str(), before);
}

TEST(ClassicalValueFolding, PreservesRuntimeBranchAndGateOrder) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @test(%condition: i1) {
        %q = qc.static 0 : !qc.qubit
        %a = arith.constant 0.25 : f64
        %b = arith.constant 0.5 : f64
        scf.if %condition {
          %sum = arith.addf %a, %b : f64
          qc.h %q : !qc.qubit
          qc.h %q : !qc.qubit
          qc.rx(%sum) %q : !qc.qubit
        } else {
          qc.x %q : !qc.qubit
          qc.x %q : !qc.qubit
        }
        return
      }
    }
  )");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  mlir::PassManager manager(qc->module().getContext());
  manager.addPass(mlir::mqt::createFoldClassicalValues());
  ASSERT_TRUE(mlir::succeeded(manager.run(qc->module())));
  std::vector<std::string> operations;
  qc->module().walk([&](mlir::Operation* op) {
    const auto name = op->getName().getStringRef();
    if (name.starts_with("qc.") || name.starts_with("scf.")) {
      operations.push_back(name.str());
    }
  });
  EXPECT_EQ(operations, (std::vector<std::string>{
                            "qc.static", "qc.h", "qc.h", "qc.rx", "scf.yield",
                            "qc.x", "qc.x", "scf.yield", "scf.if"}));
  EXPECT_EQ(qc->str().find("arith.addf"), std::string::npos);
}

TEST(ClassicalValueFolding, DoesNotFoldQuantumHandleSelection) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %a = qc.static 0 : !qc.qubit
        %b = qc.static 1 : !qc.qubit
        %true = arith.constant true
        %selected = arith.select %true, %a, %b : !qc.qubit
        qc.h %selected : !qc.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  mlir::PassManager manager(qc->module().getContext());
  manager.addPass(mlir::mqt::createFoldClassicalValues());
  ASSERT_TRUE(mlir::succeeded(manager.run(qc->module())));
  EXPECT_NE(qc->str().find("arith.select"), std::string::npos);
}

TEST(ClassicalValueFolding, FoldsArraysAndCastsWhileRetainingRuntimeValues) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @test(%runtime: f64) -> (f64, f64) {
        %array = arith.constant dense<[0.25, -0.5]> : tensor<2xf64>
        %index = arith.constant 0 : index
        %element = tensor.extract %array[%index] : tensor<2xf64>
        %integer = arith.constant 3 : i64
        %converted = arith.uitofp %integer : i64 to f64
        %sum = arith.addf %element, %converted : f64
        %zero = arith.constant 0.0 : f64
        %sine = math.sin %zero : f64
        %offset = arith.addf %sum, %sine : f64
        %dynamic = arith.addf %runtime, %offset : f64
        return %offset, %dynamic : f64, f64
      }
    }
  )");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  mlir::PassManager manager(qc->module().getContext());
  manager.addPass(mlir::mqt::createFoldClassicalValues());
  ASSERT_TRUE(mlir::succeeded(manager.run(qc->module())));
  auto function = qc->module().lookupSymbol<mlir::func::FuncOp>("test");
  auto returned =
      mlir::cast<mlir::func::ReturnOp>(function.front().getTerminator());
  auto constant =
      returned.getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
  ASSERT_TRUE(constant);
  EXPECT_DOUBLE_EQ(
      mlir::cast<mlir::FloatAttr>(constant.getValue()).getValueAsDouble(),
      3.25);
  EXPECT_TRUE(returned.getOperand(1).getDefiningOp<mlir::arith::AddFOp>());
  EXPECT_EQ(qc->str().find("tensor.extract"), std::string::npos);
  EXPECT_EQ(qc->str().find("math.sin"), std::string::npos);
}

} // namespace
