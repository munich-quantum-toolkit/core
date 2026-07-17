/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/QuantumComputation.hpp"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <numbers>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace qco;

namespace {

class QCODDFunctionalityTest : public testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] static func::FuncOp mainFunc(ModuleOp module) {
    return *module.getBody()->getOps<func::FuncOp>().begin();
  }

  template <typename BuildFn>
  [[nodiscard]] OwningOpRef<ModuleOp> buildModule(BuildFn&& buildFn) {
    return QCOProgramBuilder::build(context.get(),
                                    std::forward<BuildFn>(buildFn));
  }

  /// Compare `mlir::qco::{buildFunctionality,simulate}` to
  /// `dd::{buildFunctionality,simulate}` on an equivalent circuit.
  static void expectEqualToQc(func::FuncOp func,
                              const qc::QuantumComputation& qc) {
    const auto numQubits = qc.getNqubits();
    auto dd = std::make_unique<dd::Package>(numQubits);

    const auto fromQcFn = dd::buildFunctionality(qc, *dd);
    const auto fromQcoFn = buildFunctionality(func, *dd);
    ASSERT_TRUE(succeeded(fromQcoFn));
    EXPECT_EQ(fromQcoFn->getMatrix(numQubits), fromQcFn.getMatrix(numQubits));
    dd->decRef(*fromQcoFn);
    dd->decRef(fromQcFn);

    const auto fromQcSim =
        dd::simulate(qc, dd::makeZeroState(numQubits, *dd), *dd);
    const auto fromQcoSim =
        simulate(func, dd::makeZeroState(numQubits, *dd), *dd);
    ASSERT_TRUE(succeeded(fromQcoSim));
    EXPECT_EQ(fromQcoSim->getVector(), fromQcSim.getVector());
    dd->decRef(*fromQcoSim);
    dd->decRef(fromQcSim);
  }
};

TEST_F(QCODDFunctionalityTest, MatchesQuantumComputation) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    auto q2 = b.staticQubit(2);
    q0 = b.h(q0);
    q0 = b.rz(0.25, q0);
    std::tie(q0, q1) = b.cx(q0, q1);
    std::tie(q0, q1) = b.swap(q0, q1);
    std::tie(q1, q2) = b.cp(std::numbers::pi / 5.0, q1, q2);
    auto [controls, target] = b.mcx({q0, q1}, q2);
    q0 = controls[0];
    q1 = controls[1];
    q2 = target;
    q2 = b.inv(q2, [&](Value q) { return b.s(q); });
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  qc::QuantumComputation qc(3);
  qc.h(0);
  qc.rz(0.25, 0);
  qc.cx(0, 1);
  qc.swap(0, 1);
  qc.cp(std::numbers::pi / 5.0, 1, 2);
  qc.mcx({0, 1}, 2);
  qc.sdg(2);
  expectEqualToQc(mainFunc(*module), qc);
}

TEST_F(QCODDFunctionalityTest, QubitFuncArgsMapToWires) {
  // Synthesis helpers pass qubits as block args instead of `qco.static`.
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(module->getBody());
  const auto qubitTy = QubitType::get(context.get());
  auto func =
      func::FuncOp::create(builder, module->getLoc(), "main",
                           builder.getFunctionType({qubitTy}, {qubitTy}));
  auto* entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value out = HOp::create(builder, func.getLoc(), entry->getArgument(0));
  func::ReturnOp::create(builder, func.getLoc(), out);

  qc::QuantumComputation qc(1);
  qc.h(0);
  expectEqualToQc(func, qc);
}

TEST_F(QCODDFunctionalityTest, DenseMatrixPathComposesWithSparseGates) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    auto q2 = b.staticQubit(2);
    q1 = b.x(q1);
    std::tie(q0, q1) = b.ctrl(q0, q1, [&](Value t) { return b.h(b.t(t)); });
    (void)q2;
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  // Body `h(t(t))` is H∘T on the target → controlled-T then controlled-H.
  qc::QuantumComputation qc(3);
  qc.x(1);
  qc.ct(0, 1);
  qc.ch(0, 1);
  expectEqualToQc(mainFunc(*module), qc);
}

TEST_F(QCODDFunctionalityTest, GphaseScalesFunctionality) {
  // `QuantumComputation::gphase` is not applied by `dd::buildFunctionality`,
  // so compare QCO programs with and without `qco.gphase` directly.
  auto without = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    q0 = b.h(q0);
    return b.intConstant(0);
  });
  auto with = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    q0 = b.h(q0);
    b.gphase(0.25);
    return b.intConstant(0);
  });
  ASSERT_TRUE(without);
  ASSERT_TRUE(with);

  auto dd = std::make_unique<dd::Package>(1);
  const auto u0 = buildFunctionality(mainFunc(*without), *dd);
  const auto u1 = buildFunctionality(mainFunc(*with), *dd);
  ASSERT_TRUE(succeeded(u0));
  ASSERT_TRUE(succeeded(u1));
  const auto phase = std::polar(1.0, 0.25);
  const auto m0 = u0->getMatrix(1);
  const auto m1 = u1->getMatrix(1);
  for (std::size_t r = 0; r < 2; ++r) {
    for (std::size_t c = 0; c < 2; ++c) {
      EXPECT_TRUE(std::abs(m1[r][c] - (m0[r][c] * phase)) < 1e-10);
    }
  }
  dd->decRef(*u0);
  dd->decRef(*u1);
}

TEST_F(QCODDFunctionalityTest, UnsupportedOpFails) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    std::tie(q0, std::ignore) = b.measure(q0);
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*module), *dd)));
  EXPECT_TRUE(
      failed(simulate(mainFunc(*module), dd::makeZeroState(1, *dd), *dd)));
}

} // namespace
