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
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
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

TEST_F(QCODDFunctionalityTest, BarrierIsNoop) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    q0 = b.h(q0);
    q0 = b.barrier({q0})[0];
    b.sink(q0);
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  qc::QuantumComputation qc(1);
  qc.h(0);
  expectEqualToQc(mainFunc(*module), qc);
}

TEST_F(QCODDFunctionalityTest, ZeroQubitGphaseIsIdentityScaled) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    b.gphase(0.5);
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(0);
  const auto u = buildFunctionality(mainFunc(*module), *dd);
  ASSERT_TRUE(succeeded(u));
  EXPECT_TRUE(u->isTerminal());
  dd->decRef(*u);
}

TEST_F(QCODDFunctionalityTest, SoleStandardCtrlUsesSparsePath) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    std::tie(q0, q1) = b.ctrl(q0, q1, [&](Value t) { return b.x(t); });
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  qc::QuantumComputation qc(2);
  qc.cx(0, 1);
  expectEqualToQc(mainFunc(*module), qc);
}

TEST_F(QCODDFunctionalityTest, TwoQubitInvUsesDenseEmbedPath) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    auto outs = b.inv({q0, q1}, [&](ValueRange qs) -> SmallVector<Value> {
      auto [a, c] = b.swap(qs[0], qs[1]);
      return {a, c};
    });
    q0 = outs[0];
    q1 = outs[1];
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  // `inv(swap) = swap`.
  qc::QuantumComputation qc(2);
  qc.swap(0, 1);
  expectEqualToQc(mainFunc(*module), qc);
}

TEST_F(QCODDFunctionalityTest, FullWidthInvUsesDensePath) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    auto q2 = b.staticQubit(2);
    auto outs = b.inv({q0, q1, q2}, [&](ValueRange t) -> SmallVector<Value> {
      return {b.rx(0.2, t[0]), b.ry(0.3, t[1]), b.rz(0.4, t[2])};
    });
    q0 = outs[0];
    q1 = outs[1];
    q2 = outs[2];
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(3);
  const auto u = buildFunctionality(mainFunc(*module), *dd);
  ASSERT_TRUE(succeeded(u));
  dd->decRef(*u);
}

TEST_F(QCODDFunctionalityTest, PackageTooSmallFails) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    q0 = b.h(q0);
    (void)q1;
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*module), *dd)));
  EXPECT_TRUE(
      failed(simulate(mainFunc(*module), dd::makeZeroState(1, *dd), *dd)));
}

TEST_F(QCODDFunctionalityTest, UnmappedQubitFails) {
  // `qco.static` claims the wire map, so a qubit block arg stays unbound.
  constexpr auto mlirCode = R"mlir(
    module {
      func.func @main(%qarg: !qco.qubit) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.h %qarg : !qco.qubit -> !qco.qubit
        return
      }
    }
  )mlir";
  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*module), *dd)));
}

TEST_F(QCODDFunctionalityTest, UnmappedBarrierFails) {
  // `barrier` remaps without a prior `lookupRange`, so this hits
  // `remapUnitary`.
  constexpr auto mlirCode = R"mlir(
    module {
      func.func @main(%qarg: !qco.qubit) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.barrier %qarg : !qco.qubit -> !qco.qubit
        return
      }
    }
  )mlir";
  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*module), *dd)));
}

TEST_F(QCODDFunctionalityTest, NonConstantParameterFails) {
  constexpr auto mlirCode = R"mlir(
    module {
      func.func @main(%theta: f64) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.rz(%theta) %q : !qco.qubit -> !qco.qubit
        return
      }
    }
  )mlir";
  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*module), *dd)));
}

TEST_F(QCODDFunctionalityTest, NonConstantGphaseFails) {
  constexpr auto mlirCode = R"mlir(
    module {
      func.func @main(%theta: f64) {
        qco.gphase(%theta)
        return
      }
    }
  )mlir";
  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(0);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*module), *dd)));
}

TEST_F(QCODDFunctionalityTest, NonConstantInvMatrixFails) {
  constexpr auto mlirCode = R"mlir(
    module {
      func.func @main(%theta: f64) {
        %q = qco.static 0 : !qco.qubit
        %q_out = qco.inv (%q_in = %q) {
          %q1 = qco.rz(%theta) %q_in : !qco.qubit -> !qco.qubit
          qco.yield %q1 : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return
      }
    }
  )mlir";
  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*module), *dd)));
}

TEST_F(QCODDFunctionalityTest, MultiBlockFunctionFails) {
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(module->getBody());
  auto func = func::FuncOp::create(builder, module->getLoc(), "main",
                                   builder.getFunctionType({}, {}));
  auto* entry = func.addEntryBlock();
  func.addBlock();
  builder.setInsertionPointToStart(entry);
  func::ReturnOp::create(builder, func.getLoc());

  auto dd = std::make_unique<dd::Package>(0);
  EXPECT_TRUE(failed(buildFunctionality(func, *dd)));
}

TEST_F(QCODDFunctionalityTest, SkipsNonQubitBlockArgs) {
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(module->getBody());
  const auto qubitTy = QubitType::get(context.get());
  auto func = func::FuncOp::create(
      builder, module->getLoc(), "main",
      builder.getFunctionType({builder.getI32Type(), qubitTy}, {qubitTy}));
  auto* entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value out = XOp::create(builder, func.getLoc(), entry->getArgument(1));
  func::ReturnOp::create(builder, func.getLoc(), out);

  qc::QuantumComputation qc(1);
  qc.x(0);
  expectEqualToQc(func, qc);
}

TEST_F(QCODDFunctionalityTest, DenseFallbackRejectsMoreThan12Qubits) {
  auto module = buildModule([](QCOProgramBuilder& b) {
    SmallVector<Value, 13> qs;
    qs.reserve(13);
    for (int i = 0; i < 13; ++i) {
      qs.push_back(b.staticQubit(static_cast<std::int64_t>(i)));
    }
    // Compound `ctrl` forces the dense matrix path on a 13-qubit map.
    std::tie(qs[0], qs[1]) =
        b.ctrl(qs[0], qs[1], [&](Value t) { return b.h(b.t(t)); });
    return b.intConstant(0);
  });
  ASSERT_TRUE(module);

  auto dd = std::make_unique<dd::Package>(13);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*module), *dd)));
}

} // namespace
