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
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace qco;

namespace {

class QCODDFunctionalityTest : public testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<cbit::CBitDialect, QCODialect, arith::ArithDialect,
                    func::FuncDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] static func::FuncOp mainFunc(ModuleOp mod) {
    if (auto main = mod.lookupSymbol<func::FuncOp>("main")) {
      return main;
    }
    return *mod.getBody()->getOps<func::FuncOp>().begin();
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
    EXPECT_TRUE(*fromQcoFn == fromQcFn);
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

  void expectMlirFails(size_t numQubits, StringRef mlirCode) const {
    auto mod = parseSourceString<ModuleOp>(mlirCode, context.get());
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(numQubits);
    EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
  }

  static void expectSimulatesFromZero(func::FuncOp func, bool expectedOne) {
    auto dd = std::make_unique<dd::Package>(1);
    auto expected = dd::makeZeroState(1, *dd);
    if (expectedOne) {
      expected = dd->applyOperation(
          dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 0),
          expected);
    }
    const auto out = simulate(func, dd::makeZeroState(1, *dd), *dd);
    ASSERT_TRUE(succeeded(out));
    EXPECT_EQ(out->getVector(), expected.getVector());
    dd->decRef(*out);
    dd->decRef(expected);
  }

  static void expectSimulationFails(func::FuncOp func, size_t numQubits) {
    auto dd = std::make_unique<dd::Package>(numQubits);
    EXPECT_TRUE(failed(simulate(func, dd::makeZeroState(numQubits, *dd), *dd)));
    std::mt19937_64 rng(1);
    EXPECT_TRUE(failed(sample(func, *dd, 1, rng)));
  }

  void expectMlirSimulationFails(size_t numQubits, StringRef mlirCode) const {
    auto mod = parseSourceString<ModuleOp>(mlirCode, context.get());
    ASSERT_TRUE(mod);
    expectSimulationFails(mainFunc(*mod), numQubits);
  }
};

TEST_F(QCODDFunctionalityTest, MatchesQuantumComputation) {
  // Every `decodeStandardGate` branch once (distinct angles catch param-order
  // bugs), plus barrier / sparse ctrl / inv / sink.
  constexpr double theta = 0.31;
  constexpr double phi = 0.42;
  constexpr double lambda = 0.53;
  constexpr double beta = 0.64;

  auto mod = buildModule([&](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    auto q2 = b.staticQubit(2);
    q0 = b.id(q0);
    q0 = b.x(q0);
    q0 = b.y(q0);
    q0 = b.z(q0);
    q0 = b.h(q0);
    q0 = b.s(q0);
    q0 = b.sdg(q0);
    q0 = b.t(q0);
    q0 = b.tdg(q0);
    q0 = b.sx(q0);
    q0 = b.sxdg(q0);
    q0 = b.rx(theta, q0);
    q0 = b.ry(theta, q0);
    q0 = b.rz(theta, q0);
    q0 = b.p(theta, q0);
    q0 = b.r(theta, phi, q0);
    q0 = b.u2(phi, lambda, q0);
    q0 = b.u(theta, phi, lambda, q0);
    std::tie(q0, q1) = b.swap(q0, q1);
    std::tie(q0, q1) = b.iswap(q0, q1);
    std::tie(q0, q1) = b.dcx(q0, q1);
    std::tie(q0, q1) = b.ecr(q0, q1);
    std::tie(q0, q1) = b.rxx(theta, q0, q1);
    std::tie(q0, q1) = b.ryy(theta, q0, q1);
    std::tie(q0, q1) = b.rzz(theta, q0, q1);
    std::tie(q0, q1) = b.rzx(theta, q0, q1);
    std::tie(q0, q1) = b.xx_plus_yy(theta, beta, q0, q1);
    std::tie(q0, q1) = b.xx_minus_yy(theta, beta, q0, q1);
    q0 = b.barrier({q0})[0];
    std::tie(q0, q1) = b.cx(q0, q1);
    std::tie(q1, q2) = b.cp(std::numbers::pi / 5.0, q1, q2);
    auto [controls, target] = b.mcx({q0, q1}, q2);
    q0 = controls[0];
    q1 = controls[1];
    q2 = target;
    q2 = b.inv(q2, [&](Value q) { return b.s(q); });
    b.sink(q0);
    b.sink(q1);
    b.sink(q2);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  qc::QuantumComputation qc(3);
  qc.i(0);
  qc.x(0);
  qc.y(0);
  qc.z(0);
  qc.h(0);
  qc.s(0);
  qc.sdg(0);
  qc.t(0);
  qc.tdg(0);
  qc.sx(0);
  qc.sxdg(0);
  qc.rx(theta, 0);
  qc.ry(theta, 0);
  qc.rz(theta, 0);
  qc.p(theta, 0);
  qc.r(theta, phi, 0);
  qc.u2(phi, lambda, 0);
  qc.u(theta, phi, lambda, 0);
  qc.swap(0, 1);
  qc.iswap(0, 1);
  qc.dcx(0, 1);
  qc.ecr(0, 1);
  qc.rxx(theta, 0, 1);
  qc.ryy(theta, 0, 1);
  qc.rzz(theta, 0, 1);
  qc.rzx(theta, 0, 1);
  qc.xx_plus_yy(theta, beta, 0, 1);
  qc.xx_minus_yy(theta, beta, 0, 1);
  qc.cx(0, 1);
  qc.cp(std::numbers::pi / 5.0, 1, 2);
  qc.mcx({0, 1}, 2);
  qc.sdg(2);
  expectEqualToQc(mainFunc(*mod), qc);
}

TEST_F(QCODDFunctionalityTest, Rccx) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    auto q2 = b.staticQubit(2);
    auto q3 = b.staticQubit(3);
    std::tie(q2, q0, q3) = b.rccx(q2, q0, q3);
    auto [control, targets] = b.crccx(q1, q2, q0, q3);
    auto [q2Out, q0Out, q3Out] = targets;
    q1 = control;
    q2 = q2Out;
    q0 = q0Out;
    q3 = q3Out;
    b.sink(q0);
    b.sink(q1);
    b.sink(q2);
    b.sink(q3);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  qc::QuantumComputation qc(4);
  qc.rccx(2, 0, 3);
  qc.crccx(1, 2, 0, 3);
  expectEqualToQc(mainFunc(*mod), qc);
}

TEST_F(QCODDFunctionalityTest, DensePaths) {
  // Compound `ctrl` (dense) with sparse gates, 2-qubit `inv`, full-width `inv`,
  // and partial-width 3-qubit `inv`.
  {
    auto mod = buildModule([](QCOProgramBuilder& b) {
      auto q0 = b.staticQubit(0);
      auto q1 = b.staticQubit(1);
      auto q2 = b.staticQubit(2);
      q1 = b.x(q1);
      std::tie(q2, q0) = b.ctrl(q2, q0, [&](Value t) { return b.h(b.t(t)); });
      b.sink(q0);
      b.sink(q1);
      b.sink(q2);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    qc::QuantumComputation qc(3);
    qc.x(1);
    qc.ct(2, 0);
    qc.ch(2, 0);
    expectEqualToQc(mainFunc(*mod), qc);
  }
  {
    auto mod = buildModule([](QCOProgramBuilder& b) {
      auto q0 = b.staticQubit(0);
      auto q1 = b.staticQubit(1);
      auto outs = b.inv({q0, q1}, [&](ValueRange qs) -> SmallVector<Value> {
        auto [a, c] = b.swap(qs[0], qs[1]);
        return {a, c};
      });
      b.sink(outs[0]);
      b.sink(outs[1]);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    qc::QuantumComputation qc(2);
    qc.swap(0, 1);
    expectEqualToQc(mainFunc(*mod), qc);
  }
  {
    auto mod = buildModule([](QCOProgramBuilder& b) {
      auto q0 = b.staticQubit(0);
      auto q1 = b.staticQubit(1);
      auto q2 = b.staticQubit(2);
      auto outs = b.inv({q0, q1, q2}, [&](ValueRange t) -> SmallVector<Value> {
        return {b.rx(0.2, t[0]), b.ry(0.3, t[1]), b.rz(0.4, t[2])};
      });
      b.sink(outs[0]);
      b.sink(outs[1]);
      b.sink(outs[2]);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    qc::QuantumComputation qc(3);
    qc.rx(-0.2, 0);
    qc.ry(-0.3, 1);
    qc.rz(-0.4, 2);
    expectEqualToQc(mainFunc(*mod), qc);
  }
  {
    auto mod = buildModule([](QCOProgramBuilder& b) {
      auto q0 = b.staticQubit(0);
      auto q1 = b.staticQubit(1);
      auto q2 = b.staticQubit(2);
      auto q3 = b.staticQubit(3);
      auto outs = b.inv({q0, q1, q2}, [&](ValueRange t) -> SmallVector<Value> {
        return {b.rx(0.2, t[0]), b.ry(0.3, t[1]), b.rz(0.4, t[2])};
      });
      b.sink(outs[0]);
      b.sink(outs[1]);
      b.sink(outs[2]);
      b.sink(q3);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    qc::QuantumComputation qc(4);
    qc.rx(-0.2, 0);
    qc.ry(-0.3, 1);
    qc.rz(-0.4, 2);
    expectEqualToQc(mainFunc(*mod), qc);
  }
  {
    // Four-qubit dense `inv` on a non-contiguous wire subset (idle q3).
    auto mod = buildModule([](QCOProgramBuilder& b) {
      auto q0 = b.staticQubit(0);
      auto q1 = b.staticQubit(1);
      auto q2 = b.staticQubit(2);
      auto q3 = b.staticQubit(3);
      auto q4 = b.staticQubit(4);
      auto outs =
          b.inv({q0, q1, q2, q4}, [&](ValueRange t) -> SmallVector<Value> {
            return {b.rx(0.2, t[0]), b.ry(0.3, t[1]), b.rz(0.4, t[2]),
                    b.h(t[3])};
          });
      b.sink(outs[0]);
      b.sink(outs[1]);
      b.sink(outs[2]);
      b.sink(q3);
      b.sink(outs[3]);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    qc::QuantumComputation qc(5);
    qc.rx(-0.2, 0);
    qc.ry(-0.3, 1);
    qc.rz(-0.4, 2);
    qc.h(4);
    expectEqualToQc(mainFunc(*mod), qc);
  }
}

TEST_F(QCODDFunctionalityTest, TwoQubitDensePathBeyondFallbackLimit) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    SmallVector<Value, 13> qs;
    for (int i = 0; i < 13; ++i) {
      qs.push_back(b.staticQubit(static_cast<int64_t>(i)));
    }
    std::tie(qs[12], qs[0]) =
        b.ctrl(qs[12], qs[0], [&](Value t) { return b.h(b.t(t)); });
    for (Value q : qs) {
      b.sink(q);
    }
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(13);
  const auto functionality = buildFunctionality(mainFunc(*mod), *dd);
  ASSERT_TRUE(succeeded(functionality));
  dd->decRef(*functionality);
}

TEST_F(QCODDFunctionalityTest, Gphase) {
  auto without = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    q0 = b.h(q0);
    b.sink(q0);
    return b.intConstant(0);
  });
  auto with = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    q0 = b.h(q0);
    b.gphase(0.25);
    b.sink(q0);
    return b.intConstant(0);
  });
  auto zeroQubit = buildModule([](QCOProgramBuilder& b) {
    b.gphase(0.5);
    return b.intConstant(0);
  });
  ASSERT_TRUE(without);
  ASSERT_TRUE(with);
  ASSERT_TRUE(zeroQubit);

  auto dd = std::make_unique<dd::Package>(1);
  const auto u0 = buildFunctionality(mainFunc(*without), *dd);
  const auto u1 = buildFunctionality(mainFunc(*with), *dd);
  ASSERT_TRUE(succeeded(u0));
  ASSERT_TRUE(succeeded(u1));
  const auto phase = std::polar(1.0, 0.25);
  const auto m0 = u0->getMatrix(1);
  const auto m1 = u1->getMatrix(1);
  for (size_t r = 0; r < 2; ++r) {
    for (size_t c = 0; c < 2; ++c) {
      EXPECT_TRUE(std::abs(m1[r][c] - (m0[r][c] * phase)) < 1e-10);
    }
  }
  dd->decRef(*u0);
  dd->decRef(*u1);

  auto dd0 = std::make_unique<dd::Package>(0);
  const auto uZ = buildFunctionality(mainFunc(*zeroQubit), *dd0);
  ASSERT_TRUE(succeeded(uZ));
  EXPECT_TRUE(uZ->isTerminal());
  dd0->decRef(*uZ);
}

TEST_F(QCODDFunctionalityTest, FuncArgs) {
  // Qubit block args (no `qco.static`); non-qubit args are skipped.
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%c: i32, %q: !qco.qubit) -> !qco.qubit {
        %q1 = qco.h %q : !qco.qubit -> !qco.qubit
        return %q1 : !qco.qubit
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  qc::QuantumComputation qc(1);
  qc.h(0);
  expectEqualToQc(mainFunc(*mod), qc);
}

TEST_F(QCODDFunctionalityTest, ReturnedQubitsMustPreserveWireOrder) {
  auto canonical = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%q0: !qco.qubit, %q1: !qco.qubit)
          -> (!qco.qubit, !qco.qubit) {
        %q0_out = qco.h %q0 : !qco.qubit -> !qco.qubit
        %q1_out = qco.x %q1 : !qco.qubit -> !qco.qubit
        return %q0_out, %q1_out : !qco.qubit, !qco.qubit
      }
    }
  )mlir",
                                               context.get());
  auto swapped = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%q0: !qco.qubit, %q1: !qco.qubit)
          -> (!qco.qubit, !qco.qubit) {
        %q0_out = qco.h %q0 : !qco.qubit -> !qco.qubit
        %q1_out = qco.x %q1 : !qco.qubit -> !qco.qubit
        return %q1_out, %q0_out : !qco.qubit, !qco.qubit
      }
    }
  )mlir",
                                             context.get());
  ASSERT_TRUE(canonical);
  ASSERT_TRUE(swapped);

  auto dd = std::make_unique<dd::Package>(2);
  const auto canonicalFunctionality =
      buildFunctionality(mainFunc(*canonical), *dd);
  ASSERT_TRUE(succeeded(canonicalFunctionality));
  dd->decRef(*canonicalFunctionality);
  const auto canonicalSimulation =
      simulate(mainFunc(*canonical), dd::makeZeroState(2, *dd), *dd);
  ASSERT_TRUE(succeeded(canonicalSimulation));
  dd->decRef(*canonicalSimulation);

  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*swapped), *dd)));
  EXPECT_TRUE(
      failed(simulate(mainFunc(*swapped), dd::makeZeroState(2, *dd), *dd)));
}

TEST_F(QCODDFunctionalityTest, RejectsUnmappedReturnedQubit) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%unmapped: !qco.qubit) -> !qco.qubit {
        %q = qco.static 0 : !qco.qubit
        qco.sink %q : !qco.qubit
        return %unmapped : !qco.qubit
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
  EXPECT_TRUE(failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd)));
}

TEST_F(QCODDFunctionalityTest, SimulationConsumesInputReference) {
  auto valid = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    b.sink(q);
    return b.intConstant(0);
  });
  auto tooWide = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    b.sink(q0);
    b.sink(q1);
    return b.intConstant(0);
  });
  ASSERT_TRUE(valid);
  ASSERT_TRUE(tooWide);

  auto dd = std::make_unique<dd::Package>(1);
  auto& roots = dd->getRootSet<dd::vNode>();
  for (size_t i = 0; i < 3; ++i) {
    const auto output =
        simulate(mainFunc(*valid), dd::makeZeroState(1, *dd), *dd);
    ASSERT_TRUE(succeeded(output));
    EXPECT_EQ(roots.size(), 1U);
    EXPECT_EQ(roots.at(*output), 1U);
    dd->decRef(*output);
    EXPECT_TRUE(roots.empty());
  }
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(
        failed(simulate(mainFunc(*tooWide), dd::makeZeroState(1, *dd), *dd)));
    EXPECT_TRUE(roots.empty());
  }

  auto zeroQubitDd = std::make_unique<dd::Package>(0);
  EXPECT_TRUE(
      failed(simulate(mainFunc(*valid), dd::VectorDD::one(), *zeroQubitDd)));
  EXPECT_TRUE(zeroQubitDd->getRootSet<dd::vNode>().empty());
}

TEST_F(QCODDFunctionalityTest, SimulateMeasureCollapsesLikePackage) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.h(b.staticQubit(0));
    std::tie(q, std::ignore) = b.measure(q);
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  constexpr uint64_t seed = 42;
  auto dd = std::make_unique<dd::Package>(1);

  std::mt19937_64 refRng(seed);
  auto ref = dd::makeZeroState(1, *dd);
  ref = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::H), 0), ref);
  static_cast<void>(dd->measureOneCollapsing(ref, 0, refRng));
  const auto expected = ref.getVector();

  std::mt19937_64 rng(seed);
  const auto out =
      simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ(out->getVector(), expected);
  dd->decRef(*out);
  dd->decRef(ref);
}

TEST_F(QCODDFunctionalityTest, SimulateResetForcesZero) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    q = b.reset(q);
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(7);
  auto expected = dd::makeZeroState(1, *dd);
  const auto out =
      simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ(out->getVector(), expected.getVector());
  dd->decRef(*out);
  dd->decRef(expected);
}

TEST_F(QCODDFunctionalityTest, SimulateIfConstantBranches) {
  auto thenMod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    q = b.qcoIf(
        true, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  auto elseMod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    q = b.qcoIf(
        false, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(thenMod);
  ASSERT_TRUE(elseMod);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*thenMod), *dd)));
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*elseMod), *dd)));
  std::mt19937_64 rng(0);
  auto zero = dd::makeZeroState(1, *dd);
  auto one = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 0),
      dd::makeZeroState(1, *dd));

  const auto thenOut =
      simulate(mainFunc(*thenMod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(thenOut));
  EXPECT_EQ(thenOut->getVector(), one.getVector());

  const auto elseOut =
      simulate(mainFunc(*elseMod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(elseOut));
  EXPECT_EQ(elseOut->getVector(), zero.getVector());

  dd->decRef(*thenOut);
  dd->decRef(*elseOut);
  dd->decRef(zero);
  dd->decRef(one);
}

TEST_F(QCODDFunctionalityTest, SimulateIndexSwitchBranches) {
  auto caseMod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    q = b.qcoIndexSwitch(0, q, ArrayRef<int64_t>{0, 1},
                         SmallVector<function_ref<Value(Value)>>{
                             [&](Value arg) { return b.x(arg); },
                             [&](Value arg) { return arg; }},
                         [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  auto defaultMod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    q = b.qcoIndexSwitch(5, q, ArrayRef<int64_t>{0, 1},
                         SmallVector<function_ref<Value(Value)>>{
                             [&](Value arg) { return b.x(arg); },
                             [&](Value arg) { return b.x(arg); }},
                         [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(caseMod);
  ASSERT_TRUE(defaultMod);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*caseMod), *dd)));
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*defaultMod), *dd)));
  std::mt19937_64 rng(0);
  auto zero = dd::makeZeroState(1, *dd);
  auto one = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 0),
      dd::makeZeroState(1, *dd));

  const auto caseOut =
      simulate(mainFunc(*caseMod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(caseOut));
  EXPECT_EQ(caseOut->getVector(), one.getVector());

  const auto defaultOut =
      simulate(mainFunc(*defaultMod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(defaultOut));
  EXPECT_EQ(defaultOut->getVector(), zero.getVector());

  dd->decRef(*caseOut);
  dd->decRef(*defaultOut);
  dd->decRef(zero);
  dd->decRef(one);
}

TEST_F(QCODDFunctionalityTest, SimulateMeasureFeedsIf) {
  // |1> measure is deterministic; then-branch identity keeps |1>.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q);
    q = b.qcoIf(
        bit, q, [&](Value arg) { return arg; },
        [&](Value arg) { return b.x(arg); });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(99);
  auto one = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 0),
      dd::makeZeroState(1, *dd));
  const auto out =
      simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ(out->getVector(), one.getVector());
  dd->decRef(*out);
  dd->decRef(one);
}

TEST_F(QCODDFunctionalityTest, SimulateCBitConditionAndMeasurementUpdate) {
  auto zeroCondition = buildModule([](QCOProgramBuilder& b) {
    auto reg = b.allocClassicalBitRegister(1, "c");
    auto q = b.staticQubit(0);
    q = b.qcoIf(
        b.loadClassicalBit(reg, 0), q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  auto measurementCondition = buildModule([](QCOProgramBuilder& b) {
    auto reg =
        b.allocClassicalBitRegister(1, "c", cbit::Initialization::Undefined);
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q);
    b.storeClassicalBit(bit, reg, 0);
    q = b.qcoIf(
        b.loadClassicalBit(reg, 0), q, [&](Value arg) { return arg; },
        [&](Value arg) { return b.x(arg); });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(zeroCondition);
  ASSERT_TRUE(measurementCondition);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(99);
  auto zero = dd::makeZeroState(1, *dd);
  auto one = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 0),
      dd::makeZeroState(1, *dd));

  const auto zeroOut =
      simulate(mainFunc(*zeroCondition), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(zeroOut));
  EXPECT_EQ(zeroOut->getVector(), zero.getVector());

  const auto measurementOut = simulate(mainFunc(*measurementCondition),
                                       dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(measurementOut));
  EXPECT_EQ(measurementOut->getVector(), one.getVector());

  dd->decRef(*zeroOut);
  dd->decRef(*measurementOut);
  dd->decRef(zero);
  dd->decRef(one);
}

TEST_F(QCODDFunctionalityTest, RejectsUndefinedCBitLoad) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto reg =
        b.allocClassicalBitRegister(1, "c", cbit::Initialization::Undefined);
    auto q = b.staticQubit(0);
    q = b.qcoIf(
        b.loadClassicalBit(reg, 0), q, [&](Value arg) { return arg; },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(1);
  EXPECT_TRUE(
      failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
}

TEST_F(QCODDFunctionalityTest, SimulateMeasureFeedsIndexSwitch) {
  // |1> → measure → index_castui → index_switch case 1 applies X → |0>.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q);
    auto idx =
        arith::IndexCastUIOp::create(b, b.getIndexType(), bit).getResult();
    q = b.qcoIndexSwitch(idx, q, ArrayRef<int64_t>{0, 1},
                         SmallVector<function_ref<Value(Value)>>{
                             [&](Value arg) { return arg; },
                             [&](Value arg) { return b.x(arg); }},
                         [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(3);
  auto zero = dd::makeZeroState(1, *dd);
  const auto out =
      simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ(out->getVector(), zero.getVector());
  dd->decRef(*out);
  dd->decRef(zero);
}

TEST_F(QCODDFunctionalityTest, SimulateAndiOriShliClassical) {
  // Pack two measure bits (from |1>,|0>) as index = bit0 | (bit1 << 1) = 1,
  // then switch case 1 applies X on an idle |0> target → |1>.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.x(b.staticQubit(0));
    auto q1 = b.staticQubit(1);
    auto q2 = b.staticQubit(2);
    Value bit0;
    Value bit1;
    std::tie(q0, bit0) = b.measure(q0);
    std::tie(q1, bit1) = b.measure(q1);
    auto i0 =
        arith::IndexCastUIOp::create(b, b.getIndexType(), bit0).getResult();
    auto i1 =
        arith::IndexCastUIOp::create(b, b.getIndexType(), bit1).getResult();
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto shifted = arith::ShLIOp::create(b, i1, one).getResult();
    auto packed = arith::OrIOp::create(b, i0, shifted).getResult();
    // Exercise every supported boolean and index bitwise variant. Their
    // results need not feed the quantum path: the interpreter visits all SSA
    // operations in program order.
    auto t = b.boolConstant(true);
    static_cast<void>(arith::AndIOp::create(b, bit0, t));
    static_cast<void>(arith::OrIOp::create(b, bit1, t));
    static_cast<void>(arith::XOrIOp::create(b, bit0, bit1));
    static_cast<void>(arith::AndIOp::create(b, i0, one));
    static_cast<void>(arith::XOrIOp::create(b, i0, one));
    q2 = b.qcoIndexSwitch(packed, q2, ArrayRef<int64_t>{0, 1, 2},
                          SmallVector<function_ref<Value(Value)>>{
                              [&](Value arg) { return arg; },
                              [&](Value arg) { return b.x(arg); },
                              [&](Value arg) { return arg; }},
                          [&](Value arg) { return arg; });
    b.sink(q0);
    b.sink(q1);
    b.sink(q2);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(3);
  std::mt19937_64 rng(11);
  // Final computational basis: |1>|0>|1> after measures and case-1 X on q2.
  auto expected = dd::makeZeroState(3, *dd);
  expected = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 0),
      expected);
  expected = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 2),
      expected);

  const auto out =
      simulate(mainFunc(*mod), dd::makeZeroState(3, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ(out->getVector(), expected.getVector());
  dd->decRef(*out);
  dd->decRef(expected);
}

TEST_F(QCODDFunctionalityTest, AcceptsLargestValidShift) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto amount = arith::ConstantIndexOp::create(b, 63).getResult();
    auto shifted = arith::ShLIOp::create(b, one, amount).getResult();
    q = b.qcoIndexSwitch(shifted, q,
                         ArrayRef<int64_t>{std::numeric_limits<int64_t>::min()},
                         SmallVector<function_ref<Value(Value)>>{
                             [&](Value arg) { return b.x(arg); }},
                         [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(1);
  const auto out =
      simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(out));
  auto expected = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 0),
      dd::makeZeroState(1, *dd));
  EXPECT_EQ(out->getVector(), expected.getVector());
  dd->decRef(*out);
  dd->decRef(expected);
}

TEST_F(QCODDFunctionalityTest, RejectsOutOfRangeShift) {
  for (const int64_t amount : {-1, 64}) {
    auto mod = buildModule([amount](QCOProgramBuilder& b) {
      auto q = b.staticQubit(0);
      auto one = arith::ConstantIndexOp::create(b, 1).getResult();
      auto bad = arith::ConstantIndexOp::create(b, amount).getResult();
      auto shifted = arith::ShLIOp::create(b, one, bad).getResult();
      q = b.qcoIndexSwitch(shifted, q, ArrayRef<int64_t>{0},
                           SmallVector<function_ref<Value(Value)>>{
                               [&](Value arg) { return arg; }},
                           [&](Value arg) { return arg; });
      b.sink(q);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);

    auto dd = std::make_unique<dd::Package>(1);
    std::mt19937_64 rng(1);
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
  }
}

TEST_F(QCODDFunctionalityTest, SampleUnitaryXIsDeterministic) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(1);
  constexpr size_t shots = 64;
  const auto hist = sample(mainFunc(*mod), *dd, shots, rng);
  ASSERT_TRUE(succeeded(hist));
  ASSERT_EQ(hist->size(), 1U);
  EXPECT_EQ(hist->begin()->first, "1");
  EXPECT_EQ(hist->begin()->second, shots);
}

TEST_F(QCODDFunctionalityTest, SampleHadamardApproximatelyBalanced) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.h(b.staticQubit(0));
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(42);
  constexpr size_t shots = 2000;
  const auto hist = sample(mainFunc(*mod), *dd, shots, rng);
  ASSERT_TRUE(succeeded(hist));
  ASSERT_EQ(hist->size(), 2U);
  EXPECT_EQ(hist->at("0") + hist->at("1"), shots);
  EXPECT_NEAR(static_cast<double>(hist->at("0")), shots / 2.0, 150.0);
}

TEST_F(QCODDFunctionalityTest, SampleDefersTerminalMeasurement) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    std::tie(q, std::ignore) = b.measure(q);
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(7);
  const auto histogram = sample(mainFunc(*mod), *dd, 16, rng);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"1", 16}}));
}

TEST_F(QCODDFunctionalityTest, SampleRejectsUnmappedTerminalMeasurement) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%unmapped: !qco.qubit) {
        %q = qco.static 0 : !qco.qubit
        %out, %bit = qco.measure %unmapped : !qco.qubit
        qco.sink %q : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(7);
  EXPECT_TRUE(failed(sample(mainFunc(*mod), *dd, 1, rng)));
}

TEST_F(QCODDFunctionalityTest, SampleResetUsesDynamicSampling) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.reset(b.x(b.staticQubit(0)));
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(7);
  const auto histogram = sample(mainFunc(*mod), *dd, 16, rng);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"0", 16}}));
}

TEST_F(QCODDFunctionalityTest, SampleDynamicMeasureIf) {
  // |1> measure then identity branch; final measureAll is always "1".
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q);
    q = b.qcoIf(
        bit, q, [&](Value arg) { return arg; },
        [&](Value arg) { return b.x(arg); });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(7);
  constexpr size_t shots = 32;
  const auto hist = sample(mainFunc(*mod), *dd, shots, rng);
  ASSERT_TRUE(succeeded(hist));
  ASSERT_EQ(hist->size(), 1U);
  EXPECT_EQ(hist->begin()->first, "1");
  EXPECT_EQ(hist->begin()->second, shots);
}

TEST_F(QCODDFunctionalityTest, SampleHandlesZeroShotsAndSimulationFailure) {
  auto unitary = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(unitary);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(1);
  const auto empty = sample(mainFunc(*unitary), *dd, 0, rng);
  ASSERT_TRUE(succeeded(empty));
  EXPECT_TRUE(empty->empty());

  auto dynamic = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) {
        %q = qco.static 0 : !qco.qubit
        %out = qco.rz(%theta) %q : !qco.qubit -> !qco.qubit
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                             context.get());
  ASSERT_TRUE(dynamic);
  EXPECT_TRUE(failed(sample(mainFunc(*dynamic), *dd, 1, rng)));

  auto measuredDynamic = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) {
        %q = qco.static 0 : !qco.qubit
        %measured, %bit = qco.measure %q : !qco.qubit
        %out = qco.rz(%theta) %measured : !qco.qubit -> !qco.qubit
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                                     context.get());
  ASSERT_TRUE(measuredDynamic);
  EXPECT_TRUE(failed(sample(mainFunc(*measuredDynamic), *dd, 1, rng)));

  auto tooSmall = std::make_unique<dd::Package>(0);
  EXPECT_TRUE(failed(sample(mainFunc(*unitary), *tooSmall, 1, rng)));
}

TEST_F(QCODDFunctionalityTest, EmbedsWideLocalMatrixWithoutRegisterLimit) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    SmallVector<Value, 13> qs;
    for (int64_t i = 0; i < 13; ++i) {
      qs.push_back(b.staticQubit(i));
    }
    auto outs = b.inv(
        {qs[0], qs[4], qs[8], qs[12]}, [&](ValueRange t) -> SmallVector<Value> {
          return {b.rx(0.2, t[0]), b.ry(0.3, t[1]), b.rz(0.4, t[2]), b.h(t[3])};
        });
    for (size_t i = 0; i < qs.size(); ++i) {
      if (i == 0) {
        qs[i] = outs[0];
      } else if (i == 4) {
        qs[i] = outs[1];
      } else if (i == 8) {
        qs[i] = outs[2];
      } else if (i == 12) {
        qs[i] = outs[3];
      }
    }
    for (Value q : qs) {
      b.sink(q);
    }
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  qc::QuantumComputation qc(13);
  qc.rx(-0.2, 0);
  qc.ry(-0.3, 4);
  qc.rz(-0.4, 8);
  qc.h(12);
  expectEqualToQc(mainFunc(*mod), qc);
}

TEST_F(QCODDFunctionalityTest, RejectsInvalidClassicalOperations) {
  for (const StringRef source : {
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %c = arith.constant 0 : i64
               %bad = arith.index_castui %c : i64 to index
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %c = arith.constant 0 : index
               %bad = arith.index_castui %c : index to i64
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%c: i1) {
               %q = qco.static 0 : !qco.qubit
               %bad = arith.index_castui %c : i1 to index
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%unmapped: i1) {
               %q = qco.static 0 : !qco.qubit
               %true = arith.constant true
               %bad = arith.andi %unmapped, %true : i1
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%unmapped: index) {
               %q = qco.static 0 : !qco.qubit
               %one = arith.constant 1 : index
               %bad = arith.ori %unmapped, %one : index
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %one = arith.constant 1 : i64
               %bad = arith.andi %one, %one : i64
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %one = arith.constant 1 : i64
               %bad = arith.cmpi eq, %one, %one : i64
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %one = arith.constant 1 : i64
               %bad = arith.shli %one, %one : i64
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %true = arith.constant true
               %one = arith.constant 1 : i64
               %bad = arith.select %true, %one, %one : i64
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %one = arith.constant 1 : i64
               %bad = arith.maxsi %one, %one : i64
               qco.sink %q : !qco.qubit
               return
             }
           })mlir"}) {
    auto mod = parseSourceString<ModuleOp>(source, context.get());
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(1);
    std::mt19937_64 rng(1);
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
  }
}

TEST_F(QCODDFunctionalityTest, RejectsUnmappedClassicalControl) {
  for (const StringRef source : {R"mlir(
    module {
      func.func @main(%condition: i1) {
        %q = qco.static 0 : !qco.qubit
        %out = qco.if %condition args(%arg = %q) -> (!qco.qubit) {
          qco.yield %arg : !qco.qubit
        } else args(%arg = %q) {
          qco.yield %arg : !qco.qubit
        }
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                 R"mlir(
    module {
      func.func @main(%index: index) {
        %q = qco.static 0 : !qco.qubit
        %out = qco.index_switch %index -> !qco.qubit
        default args(%arg = %q) {
          qco.yield %arg : !qco.qubit
        }
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir"}) {
    auto mod = parseSourceString<ModuleOp>(source, context.get());
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(1);
    std::mt19937_64 rng(1);
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
  }
}

TEST_F(QCODDFunctionalityTest, BindsClassicalIfResults) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %true = arith.constant true
        %false = arith.constant false
        %flag, %q1 = qco.if %true args(%arg = %q) -> (i1, !qco.qubit) {
          qco.yield %false, %arg : i1, !qco.qubit
        } else args(%arg = %q) {
          qco.yield %true, %arg : i1, !qco.qubit
        }
        %q2 = qco.if %flag args(%arg = %q1) -> (!qco.qubit) {
          qco.yield %arg : !qco.qubit
        } else args(%arg = %q1) {
          %x = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.yield %x : !qco.qubit
        }
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(1);
  const auto out =
      simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(out));
  auto expected = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 0),
      dd::makeZeroState(1, *dd));
  EXPECT_EQ(out->getVector(), expected.getVector());
  dd->decRef(*out);
  dd->decRef(expected);
}

TEST_F(QCODDFunctionalityTest, BindsClassicalIndexResults) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %true = arith.constant true
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %index, %q1 = qco.if %true args(%arg = %q)
            -> (index, !qco.qubit) {
          qco.yield %one, %arg : index, !qco.qubit
        } else args(%arg = %q) {
          qco.yield %zero, %arg : index, !qco.qubit
        }
        %q2 = qco.index_switch %index -> !qco.qubit
        case 1 args(%arg = %q1) {
          %x = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.yield %x : !qco.qubit
        }
        default args(%arg = %q1) {
          qco.yield %arg : !qco.qubit
        }
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(1);
  const auto out =
      simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(out));
  auto expected = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X), 0),
      dd::makeZeroState(1, *dd));
  EXPECT_EQ(out->getVector(), expected.getVector());
  dd->decRef(*out);
  dd->decRef(expected);
}

TEST_F(QCODDFunctionalityTest, RejectsInvalidClassicalRegionResults) {
  for (const StringRef source : {
           R"mlir(module {
             func.func @main(%unmapped: i1) {
               %q = qco.static 0 : !qco.qubit
               %true = arith.constant true
               %result, %out = qco.if %true args(%arg = %q)
                   -> (i1, !qco.qubit) {
                 qco.yield %unmapped, %arg : i1, !qco.qubit
               } else args(%arg = %q) {
                 qco.yield %true, %arg : i1, !qco.qubit
               }
               qco.sink %out : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%unmapped: index) {
               %q = qco.static 0 : !qco.qubit
               %true = arith.constant true
               %zero = arith.constant 0 : index
               %result, %out = qco.if %true args(%arg = %q)
                   -> (index, !qco.qubit) {
                 qco.yield %unmapped, %arg : index, !qco.qubit
               } else args(%arg = %q) {
                 qco.yield %zero, %arg : index, !qco.qubit
               }
               qco.sink %out : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %true = arith.constant true
               %value = arith.constant 1 : i64
               %result, %out = qco.if %true args(%arg = %q)
                   -> (i64, !qco.qubit) {
                 qco.yield %value, %arg : i64, !qco.qubit
               } else args(%arg = %q) {
                 qco.yield %value, %arg : i64, !qco.qubit
               }
               qco.sink %out : !qco.qubit
               return
             }
           })mlir"}) {
    auto mod = parseSourceString<ModuleOp>(source, context.get());
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(1);
    std::mt19937_64 rng(1);
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
  }
}

TEST_F(QCODDFunctionalityTest, RejectsInvalidLinearRegionValues) {
  for (const StringRef source : {
           R"mlir(module {
             func.func @main(%unmapped: !qco.qubit) {
               %q = qco.static 0 : !qco.qubit
               %true = arith.constant true
               %out = qco.if %true args(%arg = %unmapped) -> (!qco.qubit) {
                 qco.yield %arg : !qco.qubit
               } else args(%arg = %unmapped) {
                 qco.yield %arg : !qco.qubit
               }
               qco.sink %q : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%unmapped: !qco.qubit) {
               %q = qco.static 0 : !qco.qubit
               %true = arith.constant true
               %out = qco.if %true args(%arg = %q) -> (!qco.qubit) {
                 qco.yield %unmapped : !qco.qubit
               } else args(%arg = %q) {
                 qco.yield %arg : !qco.qubit
               }
               qco.sink %out : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %true = arith.constant true
               %out = qco.if %true args(%arg = %q) -> (!qco.qubit) {
                 %one = arith.constant 1 : i64
                 %unsupported = arith.maxsi %one, %one : i64
                 qco.yield %arg : !qco.qubit
               } else args(%arg = %q) {
                 qco.yield %arg : !qco.qubit
               }
               qco.sink %out : !qco.qubit
               return
             }
           })mlir"}) {
    auto mod = parseSourceString<ModuleOp>(source, context.get());
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(1);
    std::mt19937_64 rng(1);
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
  }
}

TEST_F(QCODDFunctionalityTest, RejectsUnmappedMeasurementAndResetQubits) {
  for (const StringRef source : {R"mlir(module {
           func.func @main(%unmapped: !qco.qubit) {
             %q = qco.static 0 : !qco.qubit
             %out, %bit = qco.measure %unmapped : !qco.qubit
             qco.sink %q : !qco.qubit
             return
           }
         })mlir",
                                 R"mlir(module {
           func.func @main(%unmapped: !qco.qubit) {
             %q = qco.static 0 : !qco.qubit
             %out = qco.reset %unmapped : !qco.qubit -> !qco.qubit
             qco.sink %q : !qco.qubit
             return
           }
         })mlir"}) {
    auto mod = parseSourceString<ModuleOp>(source, context.get());
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(1);
    std::mt19937_64 rng(1);
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
  }
}

TEST_F(QCODDFunctionalityTest, Rejects) {
  {
    auto mod = buildModule([](QCOProgramBuilder& b) {
      auto q0 = b.staticQubit(0);
      std::tie(q0, std::ignore) = b.measure(q0);
      b.sink(q0);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(1);
    EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
    // Three-arg simulate has no RNG and must reject measure/reset.
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd)));
  }

  {
    auto mod = buildModule([](QCOProgramBuilder& b) {
      auto q = b.reset(b.staticQubit(0));
      b.sink(q);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(1);
    EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd)));
  }

  {
    auto mod = buildModule([](QCOProgramBuilder& b) {
      auto q0 = b.staticQubit(0);
      auto q1 = b.staticQubit(1);
      q0 = b.h(q0);
      b.sink(q0);
      b.sink(q1);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(1);
    EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd)));
  }

  expectMlirFails(1, R"mlir(
    module {
      func.func @main(%qarg: !qco.qubit) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.h %qarg : !qco.qubit -> !qco.qubit
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
    module {
      func.func @main(%qarg: !qco.qubit) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.barrier %qarg : !qco.qubit -> !qco.qubit
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
    module {
      func.func @main(%qarg: !qco.qubit) {
        %q = qco.static 0 : !qco.qubit
        %q_out = qco.inv (%q_in = %qarg) {
          %q1 = qco.x %q_in : !qco.qubit -> !qco.qubit
          qco.yield %q1 : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
    module {
      func.func @main(%qarg: !qco.qubit) {
        %q = qco.static 0 : !qco.qubit
        %c_out, %t_out = qco.ctrl(%qarg) targets(%t = %q) {
          %t1 = qco.x %t : !qco.qubit -> !qco.qubit
          qco.yield %t1 : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
    module {
      func.func @main(%theta: f64) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.rz(%theta) %q : !qco.qubit -> !qco.qubit
        return
      }
    }
  )mlir");
  expectMlirFails(0, R"mlir(
    module {
      func.func @main(%theta: f64) {
        qco.gphase(%theta)
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
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
  )mlir");
  expectMlirFails(2, R"mlir(
    module {
      func.func @main(%theta: f64) {
        %q0 = qco.static 0 : !qco.qubit
        %q1 = qco.static 1 : !qco.qubit
        %c_out, %t_out = qco.ctrl(%q0) targets(%t = %q1) {
          %t1 = qco.rz(%theta) %t : !qco.qubit -> !qco.qubit
          qco.yield %t1 : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
        return
      }
    }
  )mlir");

  OwningOpRef<ModuleOp> multi =
      ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(multi->getBody());
  auto func = func::FuncOp::create(builder, multi->getLoc(), "main",
                                   builder.getFunctionType({}, {}));
  auto* entry = func.addEntryBlock();
  func.addBlock();
  builder.setInsertionPointToStart(entry);
  func::ReturnOp::create(builder, func.getLoc());
  auto dd = std::make_unique<dd::Package>(0);
  EXPECT_TRUE(failed(buildFunctionality(func, *dd)));
}

TEST_F(QCODDFunctionalityTest, SimulateScfForAndFuncCallWithClassicalValues) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @flip_if(%q: !qco.qubit, %bit: i1, %i: index)
          -> (!qco.qubit, i1, index) {
        %c0 = arith.constant 0 : index
        %at_zero = arith.cmpi eq, %i, %c0 : index
        %cond = arith.andi %bit, %at_zero : i1
        %q1 = qco.if %cond args(%qin = %q) -> (!qco.qubit) {
          %qx = qco.x %qin : !qco.qubit -> !qco.qubit
          qco.yield %qx : !qco.qubit
        } else args(%qin = %q) {
          qco.yield %qin : !qco.qubit
        }
        return %q1, %bit, %i : !qco.qubit, i1, index
      }
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %true = arith.constant true
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %q1, %bit1, %i1 = scf.for %iv = %c0 to %c3 step %c1
            iter_args(%qarg = %q, %barg = %true, %iarg = %c0)
            -> (!qco.qubit, i1, index) {
          %q2, %bout, %iout = func.call @flip_if(%qarg, %barg, %iv)
              : (!qco.qubit, i1, index) -> (!qco.qubit, i1, index)
          scf.yield %q2, %bout, %iout : !qco.qubit, i1, index
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  expectSimulatesFromZero(mainFunc(*mod), true);
}

TEST_F(QCODDFunctionalityTest, RejectsInvalidFuncCalls) {
  auto recursive = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @rec(%q: !qco.qubit) -> !qco.qubit {
        %q1 = func.call @rec(%q) : (!qco.qubit) -> !qco.qubit
        return %q1 : !qco.qubit
      }
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %q1 = func.call @rec(%q) : (!qco.qubit) -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                               context.get());
  ASSERT_TRUE(recursive);
  expectSimulationFails(mainFunc(*recursive), 1);

  auto declaration = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func private @decl(%q: !qco.qubit) -> !qco.qubit
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %q1 = func.call @decl(%q) : (!qco.qubit) -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                                 context.get());
  ASSERT_TRUE(declaration);
  expectSimulationFails(mainFunc(*declaration), 1);

  auto unresolved =
      parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %q1 = func.call @missing(%q) : (!qco.qubit) -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                  ParserConfig(context.get(), false));
  ASSERT_TRUE(unresolved);
  expectSimulationFails(mainFunc(*unresolved), 1);
  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*unresolved), *dd)));

  OwningOpRef<func::FuncOp> standalone(
      cast<func::FuncOp>(mainFunc(*unresolved)->clone()));
  expectSimulationFails(*standalone, 1);
  EXPECT_TRUE(failed(buildFunctionality(*standalone, *dd)));
}

TEST_F(QCODDFunctionalityTest, HandlesScfForBounds) {
  for (const auto [lower, upper, step, succeeds] :
       {std::tuple<int64_t, int64_t, int64_t, bool>{3, 3, 1, true},
        {0, 10000, 1, true},
        {0, 10001, 1, false},
        {0, 3, 0, false},
        {0, 3, -1, false}}) {
    auto mod = buildModule([=](QCOProgramBuilder& b) {
      auto q = b.staticQubit(0);
      auto results =
          b.scfFor(lower, upper, step, ValueRange{q},
                   [&](Value /*iv*/, ValueRange args) -> SmallVector<Value> {
                     return {args[0]};
                   });
      b.sink(results[0]);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    if (succeeds) {
      expectSimulatesFromZero(mainFunc(*mod), false);
    } else {
      auto dd = std::make_unique<dd::Package>(1);
      EXPECT_TRUE(
          failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd)));
    }
  }

  expectMlirSimulationFails(0, R"mlir(
    module {
      func.func @main(%lower: index) {
        %upper = arith.constant 1 : index
        %step = arith.constant 1 : index
        scf.for %iv = %lower to %upper step %step {
        }
        return
      }
    }
  )mlir");

  auto unsignedLoop = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %lower = arith.constant -1 : index
        %upper = arith.constant 2 : index
        %step = arith.constant 1 : index
        %result = scf.for unsigned %iv = %lower to %upper step %step
            iter_args(%qarg = %q) -> !qco.qubit {
          %next = qco.x %qarg : !qco.qubit -> !qco.qubit
          scf.yield %next : !qco.qubit
        }
        qco.sink %result : !qco.qubit
        return
      }
    }
  )mlir",
                                                  context.get());
  ASSERT_TRUE(unsignedLoop);
  expectSimulatesFromZero(mainFunc(*unsignedLoop), false);
}

TEST_F(QCODDFunctionalityTest, SimulateRicherClassicalArithmetic) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto zero = arith::ConstantIndexOp::create(b, 0).getResult();
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto two = arith::ConstantIndexOp::create(b, 2).getResult();
    auto three = arith::ConstantIndexOp::create(b, 3).getResult();
    auto four = arith::ConstantIndexOp::create(b, 4).getResult();
    auto sum = arith::AddIOp::create(b, one, two).getResult();
    auto product = arith::MulIOp::create(b, sum, three).getResult();
    auto difference = arith::SubIOp::create(b, product, one).getResult();
    auto shifted = arith::ShRUIOp::create(b, difference, one).getResult();
    auto selected =
        arith::SelectOp::create(b, b.boolConstant(true), shifted, zero)
            .getResult();
    auto isFour =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, selected, four)
            .getResult();
    auto condition = arith::SelectOp::create(
                         b,
                         arith::CmpIOp::create(b, arith::CmpIPredicate::ne,
                                               isFour, b.boolConstant(false)),
                         isFour, b.boolConstant(false))
                         .getResult();
    q = b.qcoIf(
        condition, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);
  expectSimulatesFromZero(mainFunc(*mod), true);
}

TEST_F(QCODDFunctionalityTest, SimulateAllClassicalComparisons) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto condition = b.boolConstant(true);
    const auto require = [&](arith::CmpIPredicate predicate, Value lhs,
                             Value rhs) {
      auto compared = arith::CmpIOp::create(b, predicate, lhs, rhs).getResult();
      condition = arith::AndIOp::create(b, condition, compared).getResult();
    };

    auto zero = arith::ConstantIndexOp::create(b, 0).getResult();
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto minusOne = arith::ConstantIndexOp::create(b, -1).getResult();
    require(arith::CmpIPredicate::eq, one, one);
    require(arith::CmpIPredicate::ne, one, zero);
    require(arith::CmpIPredicate::slt, minusOne, zero);
    require(arith::CmpIPredicate::sle, minusOne, minusOne);
    require(arith::CmpIPredicate::sgt, one, zero);
    require(arith::CmpIPredicate::sge, one, one);
    require(arith::CmpIPredicate::ult, zero, minusOne);
    require(arith::CmpIPredicate::ule, one, minusOne);
    require(arith::CmpIPredicate::ugt, minusOne, zero);
    require(arith::CmpIPredicate::uge, minusOne, one);

    auto t = b.boolConstant(true);
    auto f = b.boolConstant(false);
    require(arith::CmpIPredicate::eq, t, t);
    require(arith::CmpIPredicate::ne, t, f);
    require(arith::CmpIPredicate::slt, t, f);
    require(arith::CmpIPredicate::sle, t, t);
    require(arith::CmpIPredicate::sgt, f, t);
    require(arith::CmpIPredicate::sge, f, t);
    require(arith::CmpIPredicate::ult, f, t);
    require(arith::CmpIPredicate::ule, t, t);
    require(arith::CmpIPredicate::ugt, t, f);
    require(arith::CmpIPredicate::uge, t, f);

    q = b.qcoIf(
        condition, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);
  expectSimulatesFromZero(mainFunc(*mod), true);
}

TEST_F(QCODDFunctionalityTest, SampleWithClassics) {
  auto unitary = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    b.sink(q);
    return b.intConstant(0);
  });
  auto dynamic = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q);
    q = b.qcoIf(
        bit, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(unitary);
  ASSERT_TRUE(dynamic);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(3);
  const auto staticResult = sampleWithClassics(mainFunc(*unitary), *dd, 8, rng);
  ASSERT_TRUE(succeeded(staticResult));
  EXPECT_EQ(staticResult->shots, (std::map<std::string, size_t>{{"0", 8}}));
  EXPECT_TRUE(staticResult->classical.empty());

  auto in = dd::makeBasisState(1, std::vector<bool>{true}, *dd);
  const auto inputResult =
      sampleWithClassics(mainFunc(*unitary), in, *dd, 8, rng);
  ASSERT_TRUE(succeeded(inputResult));
  EXPECT_EQ(inputResult->shots, (std::map<std::string, size_t>{{"1", 8}}));
  EXPECT_TRUE(inputResult->classical.empty());
  EXPECT_TRUE(dd->getRootSet<dd::vNode>().empty());

  auto dynamicInput = dd::makeBasisState(1, std::vector<bool>{true}, *dd);
  const auto dynamicInputResult =
      sample(mainFunc(*dynamic), dynamicInput, *dd, 8, rng);
  ASSERT_TRUE(succeeded(dynamicInputResult));
  EXPECT_EQ(*dynamicInputResult, (std::map<std::string, size_t>{{"0", 8}}));
  EXPECT_TRUE(dd->getRootSet<dd::vNode>().empty());

  auto zeroShotInput = dd::makeBasisState(1, std::vector<bool>{true}, *dd);
  const auto zeroShotResult =
      sample(mainFunc(*unitary), zeroShotInput, *dd, 0, rng);
  ASSERT_TRUE(succeeded(zeroShotResult));
  EXPECT_TRUE(zeroShotResult->empty());
  EXPECT_TRUE(dd->getRootSet<dd::vNode>().empty());

  const auto dynamicResult =
      sampleWithClassics(mainFunc(*dynamic), *dd, 8, rng);
  ASSERT_TRUE(succeeded(dynamicResult));
  EXPECT_EQ(dynamicResult->shots, (std::map<std::string, size_t>{{"0", 8}}));
  EXPECT_EQ(dynamicResult->classical,
            (std::map<std::string, size_t>{{"1", 8}}));
}

TEST_F(QCODDFunctionalityTest, SampleClassicalCBitRegister) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto reg = b.allocClassicalBitRegister(2);
    auto q = b.x(b.staticQubit(0));
    std::tie(q, std::ignore) = b.measure(q, reg, 1);
    auto results = b.qcoIf(
        reg, 1, ValueRange{q},
        [&](ValueRange args) { return SmallVector<Value>{b.x(args[0])}; },
        [&](ValueRange args) { return SmallVector<Value>{args[0]}; });
    b.sink(results[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(11);
  const auto result = sampleWithClassics(mainFunc(*mod), *dd, 8, rng);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->shots, (std::map<std::string, size_t>{{"0", 8}}));
  EXPECT_EQ(result->classical, (std::map<std::string, size_t>{{"1", 8}}));
}

TEST_F(QCODDFunctionalityTest, SampleConstantControlFlowUsesStaticPath) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    q = b.qcoIf(
        true, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    const auto identity = [](Value arg) { return arg; };
    q = b.qcoIndexSwitch(0, q, ArrayRef<int64_t>{0},
                         SmallVector<function_ref<Value(Value)>>{identity},
                         identity);
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(2);
  const auto histogram = sample(mainFunc(*mod), *dd, 16, rng);
  ASSERT_TRUE(succeeded(histogram));
  ASSERT_EQ(histogram->size(), 1U);
  EXPECT_EQ(histogram->at("1"), 16U);
  EXPECT_EQ(dd->matrixVectorMultiplication.getStats().lookups, 1U);
}

TEST_F(QCODDFunctionalityTest, FuncCallSharesClassicalCBitStorage) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @set(%reg: !cbit.reg<1>) {
        %true = arith.constant true
        %i0 = arith.constant 0 : index
        cbit.store %true, %reg[%i0] : !cbit.reg<1>
        return
      }
      func.func @main() {
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        func.call @set(%reg) : (!cbit.reg<1>) -> ()
        %i0 = arith.constant 0 : index
        %value = cbit.load %reg[%i0] : !cbit.reg<1>
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.if %value args(%qin = %q) -> (!qco.qubit) {
          %qx = qco.x %qin : !qco.qubit -> !qco.qubit
          qco.yield %qx : !qco.qubit
        } else args(%qin = %q) {
          qco.yield %qin : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  expectSimulatesFromZero(mainFunc(*mod), true);
}

TEST_F(QCODDFunctionalityTest, RejectsInvalidClassicalCBitAccesses) {
  for (const StringRef source : {
           R"mlir(module {
             func.func @main(%reg: !cbit.reg<1>) {
               %i0 = arith.constant 0 : index
               %value = cbit.load %reg[%i0] : !cbit.reg<1>
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%reg: !cbit.reg<1>) {
               %value = arith.constant true
               %i0 = arith.constant 0 : index
               cbit.store %value, %reg[%i0] : !cbit.reg<1>
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
               %value = arith.constant true
               %i1 = arith.constant 1 : index
               %i2 = arith.addi %i1, %i1 : index
               cbit.store %value, %reg[%i2] : !cbit.reg<1>
               return
             }
           })mlir"}) {
    expectMlirSimulationFails(0, source);
  }
}

TEST_F(QCODDFunctionalityTest,
       SampleExecutesCalleeMeasurementBeforeCallerGate) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @measure(%q: !qco.qubit) -> !qco.qubit {
        %q1, %bit = qco.measure %q : !qco.qubit
        return %q1 : !qco.qubit
      }
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.h %q : !qco.qubit -> !qco.qubit
        %q2 = func.call @measure(%q1) : (!qco.qubit) -> !qco.qubit
        %q3 = qco.h %q2 : !qco.qubit -> !qco.qubit
        qco.sink %q3 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(9);
  const auto histogram = sample(mainFunc(*mod), *dd, 128, rng);
  ASSERT_TRUE(succeeded(histogram));
  ASSERT_EQ(histogram->size(), 2U);
  EXPECT_EQ(histogram->at("0") + histogram->at("1"), 128U);

  const auto result = sampleWithClassics(mainFunc(*mod), *dd, 128, rng);
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->shots.size(), 2U);
  EXPECT_EQ(result->shots.at("0") + result->shots.at("1"), 128U);
  ASSERT_EQ(result->classical.size(), 2U);
  EXPECT_EQ(result->classical.at("0") + result->classical.at("1"), 128U);
}

} // namespace
