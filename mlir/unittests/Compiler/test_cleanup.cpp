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
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace {
TEST(Cleanup, PreservesSuppliedQuantumSequence) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(OPENQASM 3;
include "stdgates.inc";
input float theta;
h $0; h $0; rx(0.0) $0; rx(1e-16) $0;
barrier $0; h $0; barrier $0; h $0;
)");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  ASSERT_TRUE(mlir::succeeded(qco->bindParameters({{"theta", 0.5}})));
  ASSERT_TRUE(mlir::succeeded(qco->cleanup()));
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

TEST(Cleanup, PreservesOverlappingBarrierDependencies) {
  auto qco = mlir::QCOProgram::fromMLIRString(R"(
    module {
      func.func @test(%a: !qco.qubit, %b: !qco.qubit, %c: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %x, %y = qco.barrier %a, %b : !qco.qubit, !qco.qubit
        %z, %w = qco.barrier %y, %c : !qco.qubit, !qco.qubit
        return %x, %z, %w : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )");
  ASSERT_TRUE(static_cast<bool>(qco)) << llvm::toString(qco.takeError());
  ASSERT_TRUE(mlir::succeeded(qco->runPassPipeline("canonicalize")));
  std::vector<mlir::qco::BarrierOp> barriers;
  qco->module().walk([&](mlir::qco::BarrierOp op) { barriers.push_back(op); });
  ASSERT_EQ(barriers.size(), 2U);
  EXPECT_EQ(barriers[0].getQubitsIn().size(), 2U);
  EXPECT_EQ(barriers[1].getQubitsIn()[0], barriers[0].getQubitsOut()[1]);
}

TEST(Cleanup, LeavesGateOnlyProgramUnchanged) {
  auto qc = mlir::QCProgram::fromOpenQASMString(
      "OPENQASM 3; include \"stdgates.inc\"; h $0; h $0;");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  const auto before = qc->str();
  mlir::PassManager manager(qc->module().getContext());
  manager.addPass(mlir::createCanonicalizerPass());
  ASSERT_TRUE(mlir::succeeded(manager.run(qc->module())));
  ASSERT_TRUE(mlir::succeeded(manager.run(qc->module())));
  EXPECT_EQ(qc->str(), before);
}

TEST(Cleanup, PreservesRuntimeBranchAndGateOrder) {
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
  manager.addPass(mlir::createCanonicalizerPass());
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

TEST(Cleanup, FoldsConstantQuantumHandleSelection) {
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
  manager.addPass(mlir::createCanonicalizerPass());
  ASSERT_TRUE(mlir::succeeded(manager.run(qc->module())));
  EXPECT_EQ(qc->str().find("arith.select"), std::string::npos);
}

TEST(Cleanup, FoldsArraysAndCastsWhileRetainingRuntimeValues) {
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
  manager.addPass(mlir::createCanonicalizerPass());
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

TEST(Cleanup, PreservesDeferredPhaseDiagnostics) {
  for (const auto* expression :
       {"arith.divf %one, %zero", "arith.mulf %large, %large"}) {
    for (bool reachable : {false, true}) {
      SCOPED_TRACE(expression);
      SCOPED_TRACE(reachable);
      const auto source = std::string(R"(
        module {
          func.func @test(%c: !qc.qubit, %q: !qc.qubit) {
            %one = arith.constant 1.0 : f64
            %zero = arith.constant 0.0 : f64
            %large = arith.constant 9000.0 : f64
            %condition = arith.constant )") +
                          (reachable ? "true" : "false") + R"(
            scf.if %condition {
              qc.ctrl(%c) targets(%t = %q) {
                %invalid = )" +
                          expression + R"( : f64
                qc.gphase(%invalid)
                qc.yield
              } : {!qc.qubit}, {!qc.qubit}
            }
            return
          }
        }
      )";
      auto program = mlir::QCProgram::fromMLIRString(source);
      ASSERT_TRUE(static_cast<bool>(program))
          << llvm::toString(program.takeError());
      EXPECT_EQ(mlir::succeeded(program->cleanup()), !reachable);
      if (!reachable) {
        EXPECT_EQ(program->str().find("qc.gphase"), std::string::npos);
        EXPECT_EQ(program->str().find("qc.p("), std::string::npos);
      }
    }
  }
}

TEST(Cleanup, CompactsSparseRegisters) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(OPENQASM 3;
qubit[100] q;
output bit[2] result;
result[0] = measure q[0];
result[1] = measure q[99];
)");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  auto qco = qc->copy().intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  ASSERT_TRUE(mlir::succeeded(qc->cleanup()));
  ASSERT_TRUE(mlir::succeeded(qco->cleanup()));
  mlir::memref::AllocOp qcAllocation;
  qc->module().walk([&](mlir::memref::AllocOp op) { qcAllocation = op; });
  ASSERT_TRUE(qcAllocation);
  EXPECT_EQ(qcAllocation.getType().getDimSize(0), 2);
  mlir::qtensor::AllocOp qcoAllocation;
  qco->module().walk([&](mlir::qtensor::AllocOp op) { qcoAllocation = op; });
  ASSERT_TRUE(qcoAllocation);
  EXPECT_EQ(
      mlir::cast<mlir::RankedTensorType>(qcoAllocation.getType()).getDimSize(0),
      2);
}

TEST(Cleanup, FoldsCommonExpressionsAndPreservesSymbolicPhase) {
  auto program = mlir::QCOProgram::fromMLIRString(R"(
    module {
      func.func @main(%theta: f64 {mqt.input_name = "theta"}) -> i1
          attributes {mqt.entry_point} {
        %a = math.sin %theta : f64
        %b = math.sin %theta : f64
        %ai = arith.bitcast %a : f64 to i64
        %bi = arith.bitcast %b : f64 to i64
        %same = arith.cmpi eq, %ai, %bi : i64
        scf.if %same {
          qco.gphase(%theta)
        } else {
          %other = arith.constant 0.7 : f64
          qco.gphase(%other)
        }
        qco.gphase(%theta)
        return %same : i1
      }
    }
  )");
  ASSERT_TRUE(static_cast<bool>(program))
      << llvm::toString(program.takeError());
  ASSERT_TRUE(mlir::succeeded(program->cleanup()));
  auto function = program->module().lookupSymbol<mlir::func::FuncOp>("main");
  auto returned =
      mlir::cast<mlir::func::ReturnOp>(function.front().getTerminator());
  llvm::APInt result;
  ASSERT_TRUE(
      mlir::matchPattern(returned.getOperand(0), mlir::m_ConstantInt(&result)));
  EXPECT_TRUE(result.isOne());
  EXPECT_EQ(program->str().find("scf.if"), std::string::npos);
  ASSERT_TRUE(mlir::succeeded(program->normalizeGlobalPhases()));
  unsigned phases = 0;
  program->module().walk([&](mlir::qco::GPhaseOp) { ++phases; });
  EXPECT_EQ(phases, 1U);

  for (double theta : {0.37, 6000.0, -6000.0}) {
    SCOPED_TRACE(theta);
    auto bound = program->copy();
    ASSERT_TRUE(mlir::succeeded(bound.bindParameters({{"theta", theta}})));
    ASSERT_TRUE(mlir::succeeded(bound.normalizeGlobalPhases()));
    mlir::qco::GPhaseOp phase;
    bound.module().walk([&](mlir::qco::GPhaseOp op) { phase = op; });
    ASSERT_TRUE(phase);
    auto angle = phase->getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
    ASSERT_TRUE(angle);
    const double actual =
        mlir::cast<mlir::FloatAttr>(angle.getValue()).getValueAsDouble();
    EXPECT_NEAR(std::cos(actual), std::cos(2 * theta), 1e-10);
    EXPECT_NEAR(std::sin(actual), std::sin(2 * theta), 1e-10);
  }
}

TEST(Cleanup, QuantumPassesLowerModifiersInsideRuntimeBranches) {
  for (const auto* pipeline : {"lower-quantum-modifiers", "simplify-quantum"}) {
    SCOPED_TRACE(pipeline);
    auto program = mlir::QCOProgram::fromMLIRString(R"(
      module {
        func.func @test(%condition: i1, %q: !qco.qubit) -> !qco.qubit {
          %a = arith.constant 0.25 : f64
          %b = arith.constant 0.5 : f64
          %r = qco.if %condition args(%t = %q) -> (!qco.qubit) {
            %v = qco.inv(%u = %t) {
              %sum = arith.addf %a, %b : f64
              %rotated = qco.rx(%sum) %u : !qco.qubit
              qco.yield %rotated : !qco.qubit
            } : {!qco.qubit} -> {!qco.qubit}
            qco.yield %v : !qco.qubit
          } else args(%t = %q) {
            %v = qco.x %t : !qco.qubit
            qco.yield %v : !qco.qubit
          }
          return %r : !qco.qubit
        }
      }
    )");
    ASSERT_TRUE(static_cast<bool>(program))
        << llvm::toString(program.takeError());
    ASSERT_TRUE(mlir::succeeded(program->runPassPipeline(pipeline)));
    ASSERT_TRUE(mlir::succeeded(mlir::qco::verifyLinearity(program->module())));
    unsigned branches = 0;
    unsigned rotations = 0;
    unsigned flips = 0;
    program->module().walk([&](mlir::Operation* op) {
      EXPECT_FALSE((mlir::isa<mlir::qco::InvOp, mlir::arith::AddFOp>(op)));
      if (auto branch = mlir::dyn_cast<mlir::qco::IfOp>(op)) {
        ++branches;
        EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(branch.getCondition()));
      }
      if (auto rotation = mlir::dyn_cast<mlir::qco::RXOp>(op)) {
        ++rotations;
        EXPECT_TRUE(mlir::isa<mlir::qco::IfOp>(op->getParentOp()));
        EXPECT_EQ(op->getParentRegion()->getRegionNumber(), 0U);
        auto angle =
            rotation.getTheta().getDefiningOp<mlir::arith::ConstantOp>();
        ASSERT_TRUE(angle);
        EXPECT_DOUBLE_EQ(
            mlir::cast<mlir::FloatAttr>(angle.getValue()).getValueAsDouble(),
            -0.75);
      }
      if (mlir::isa<mlir::qco::XOp>(op)) {
        ++flips;
        EXPECT_TRUE(mlir::isa<mlir::qco::IfOp>(op->getParentOp()));
        EXPECT_EQ(op->getParentRegion()->getRegionNumber(), 1U);
      }
    });
    EXPECT_EQ(branches, 1U);
    EXPECT_EQ(rotations, 1U);
    EXPECT_EQ(flips, 1U);
  }
}

} // namespace
