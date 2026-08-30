/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ConstantPropagation/ConstantPropagationAnalysis.hpp"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>

#include <cstddef>
#include <string>

using namespace mlir;
using namespace mlir::qco;

/// Runs the analysis over module and returns a "<op-name> -> <lattice>" line
/// for every operation, in walk order.
static std::string analyze(ModuleOp module, const size_t maxAmplitudes = 16,
                           const size_t maxHybridStates = 8) {
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<ConstantPropagationAnalysis>(maxAmplitudes, maxHybridStates);
  if (failed(solver.initializeAndRun(module))) {
    return "<analysis failed>";
  }

  std::string out;
  llvm::raw_string_ostream os(out);
  module.walk([&](Operation* op) {
    if (isa<ModuleOp>(op)) {
      return;
    }
    os << op->getName().getStringRef() << " -> ";
    if (const auto* lattice = solver.lookupState<UnionTableLattice>(
            solver.getProgramPointAfter(op))) {
      lattice->print(os);
    } else {
      os << "<none>";
    }
    os << "\n";
  });
  return out;
}

namespace {

class ConstantPropagationAnalysisTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;

  ConstantPropagationAnalysisTest() : builder(&context) {}

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    builder.initialize();
  }
};

TEST_F(ConstantPropagationAnalysisTest, allocSeedsZeroAndGateInterprets) {
  auto reg = builder.allocQubitRegister(1);
  builder.x(reg[0]);
  const auto module = builder.finalize();

  const std::string dump = analyze(*module);
  EXPECT_EQ(dump.find("<analysis failed>"), std::string::npos);
  EXPECT_NE(dump.find("qco.x -> "), std::string::npos);
  EXPECT_NE(dump.find("|1> -> 1.00"), std::string::npos);
}

TEST_F(ConstantPropagationAnalysisTest, uncalledHelperDoesNotDisturbEntry) {
  auto reg = builder.allocQubitRegister(1);
  builder.x(reg[0]);
  auto module = builder.finalize();

  // An uncalled helper function is tolerated: the entry stays precise.
  OpBuilder ob(module->getContext());
  ob.setInsertionPointToEnd(module->getBody());
  func::FuncOp::create(ob, module->getLoc(), "helper",
                       ob.getFunctionType({}, {}))
      .setPrivate();

  const std::string dump = analyze(*module);
  EXPECT_EQ(dump.find("<analysis failed>"), std::string::npos);
  EXPECT_NE(dump.find("|1> -> 1.00"), std::string::npos);
}

TEST_F(ConstantPropagationAnalysisTest, anyCallBailsToTop) {
  auto reg = builder.allocQubitRegister(1);
  builder.x(reg[0]);
  auto module = builder.finalize();

  OpBuilder ob(module->getContext());
  ob.setInsertionPointToEnd(module->getBody());
  auto callee = func::FuncOp::create(ob, module->getLoc(), "callee",
                                     ob.getFunctionType({}, {}));
  callee.setPrivate();
  auto entry = *module->getBody()->getOps<func::FuncOp>().begin();
  ob.setInsertionPointToStart(&entry.getBody().front());
  func::CallOp::create(ob, module->getLoc(), callee, ValueRange{});

  const std::string dump = analyze(*module);
  EXPECT_EQ(dump.find("<analysis failed>"), std::string::npos);
  EXPECT_NE(dump.find("qco.x -> <all top>"), std::string::npos);
}

TEST_F(ConstantPropagationAnalysisTest, independentGatesStayFactored) {
  auto reg = builder.allocQubitRegister(2);
  builder.x(reg[0]);
  builder.x(reg[1]);
  const auto module = builder.finalize();

  const std::string dump = analyze(*module);
  EXPECT_EQ(dump.find("<analysis failed>"), std::string::npos);
  EXPECT_EQ(dump.find("|11> -> 1.00"), std::string::npos);
  EXPECT_NE(dump.find("|1> -> 1.00"), std::string::npos);
}

TEST_F(ConstantPropagationAnalysisTest, entanglingGateMergedFactors) {
  auto reg = builder.allocQubitRegister(2);
  Value q0 = builder.x(reg[0]);
  builder.dcx(q0, reg[1]);
  const auto module = builder.finalize();

  const std::string dump = analyze(*module);
  EXPECT_EQ(dump.find("<analysis failed>"), std::string::npos);
  EXPECT_NE(dump.find("qco.dcx -> "), std::string::npos);
  EXPECT_NE(dump.find("|10> -> 1.00"), std::string::npos);
}

TEST_F(ConstantPropagationAnalysisTest, controlledGateFires) {
  auto reg = builder.allocQubitRegister(2);
  Value q0 = builder.h(reg[0]);
  builder.cx(q0, reg[1]);
  const auto module = builder.finalize();

  const std::string dump = analyze(*module);
  EXPECT_EQ(dump.find("<analysis failed>"), std::string::npos);
  EXPECT_NE(dump.find("|00> -> 0.71"), std::string::npos);
  EXPECT_NE(dump.find("|11> -> 0.71"), std::string::npos);
}

TEST_F(ConstantPropagationAnalysisTest, measuringSuperpositionTops) {
  auto reg = builder.allocQubitRegister(1);
  Value q0 = builder.h(reg[0]);
  builder.measure(q0);
  const auto module = builder.finalize();

  const std::string dump = analyze(*module);
  EXPECT_EQ(dump.find("<analysis failed>"), std::string::npos);
  // H builds a real superposition...
  EXPECT_NE(dump.find("[|0> -> 0.71, |1> -> 0.71]"), std::string::npos);
  // ...and measuring it tops that qubit's state (no v2.0 hybrid-state split);
  // the measured qubit prints as <top> from qco.measure onward.
  EXPECT_NE(dump.find("qco.measure ->"), std::string::npos);
  EXPECT_NE(dump.find("[<top>]"), std::string::npos);
}

TEST_F(ConstantPropagationAnalysisTest, deterministicMeasurementRecordsBit) {
  auto reg = builder.allocQubitRegister(1);
  Value q0 = builder.x(reg[0]);
  builder.measure(q0);
  const auto module = builder.finalize();

  const std::string dump = analyze(*module);
  EXPECT_EQ(dump.find("<analysis failed>"), std::string::npos);
  EXPECT_NE(dump.find("qco.measure -> "), std::string::npos);
  EXPECT_NE(dump.find("classical:"), std::string::npos);
}

} // namespace
