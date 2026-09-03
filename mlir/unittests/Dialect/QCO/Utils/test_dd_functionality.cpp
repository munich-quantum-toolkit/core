/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

struct ReferenceGate {
  dd::GateType type;
  dd::Targets targets;
  std::vector<dd::fp> params;
  dd::Controls controls;
};

class QCODDFunctionalityTest : public testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;
  std::mt19937_64 rng{0};

  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<cbit::CBitDialect, QCODialect, qtensor::QTensorDialect,
                arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
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

  static void
  expectEqualToReference(func::FuncOp func, const size_t numQubits,
                         const std::initializer_list<ReferenceGate> gates) {
    auto dd = std::make_unique<dd::Package>(numQubits);

    auto referenceFn = dd::MatrixDD::one();
    auto referenceSim = dd::makeZeroState(numQubits, *dd);
    for (const auto& gate : gates) {
      const auto operation = dd::getGateDD(*dd, gate.type, gate.params,
                                           gate.controls, gate.targets);
      referenceFn = dd->applyOperation(operation, referenceFn);
      referenceSim = dd->applyOperation(operation, referenceSim);
    }

    const auto fromQcoFn = buildFunctionality(func, *dd);
    ASSERT_TRUE(succeeded(fromQcoFn));
    EXPECT_TRUE(*fromQcoFn == referenceFn);
    dd->decRef(*fromQcoFn);
    dd->decRef(referenceFn);

    std::mt19937_64 rng(0);
    const auto fromQcoSim =
        simulate(func, dd::makeZeroState(numQubits, *dd), *dd, rng);
    ASSERT_TRUE(succeeded(fromQcoSim));
    EXPECT_EQ(fromQcoSim->getVector(), referenceSim.getVector());
    dd->decRef(*fromQcoSim);
    dd->decRef(referenceSim);
  }

  void expectMlirFails(size_t numQubits, StringRef mlirCode) const {
    auto mod = parseSourceString<ModuleOp>(mlirCode, context.get());
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(numQubits);
    EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
  }

  void expectSimulatesFromZero(func::FuncOp func, bool expectedOne) {
    auto dd = std::make_unique<dd::Package>(1);
    auto expected = dd::makeZeroState(1, *dd);
    if (expectedOne) {
      expected = dd->applyOperation(
          dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 0),
          expected);
    }
    const auto out = simulate(func, dd::makeZeroState(1, *dd), *dd, rng);
    ASSERT_TRUE(succeeded(out));
    EXPECT_EQ(out->getVector(), expected.getVector());
    dd->decRef(*out);
    dd->decRef(expected);
  }

  void expectSimulationFails(func::FuncOp func, size_t numQubits) {
    auto dd = std::make_unique<dd::Package>(numQubits);
    EXPECT_TRUE(
        failed(simulate(func, dd::makeZeroState(numQubits, *dd), *dd, rng)));
    EXPECT_TRUE(failed(sample(func, 1, 1)));
  }

  void expectMlirSimulationFails(size_t numQubits, StringRef mlirCode) {
    auto mod = parseSourceString<ModuleOp>(mlirCode, context.get());
    ASSERT_TRUE(mod);
    expectSimulationFails(mainFunc(*mod), numQubits);
  }
};

TEST_F(QCODDFunctionalityTest, ExercisesStandardGatePaths) {
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

  expectEqualToReference(
      mainFunc(*mod), 3,
      {{dd::GateType::I, {0}},
       {dd::GateType::X, {0}},
       {dd::GateType::Y, {0}},
       {dd::GateType::Z, {0}},
       {dd::GateType::H, {0}},
       {dd::GateType::S, {0}},
       {dd::GateType::Sdg, {0}},
       {dd::GateType::T, {0}},
       {dd::GateType::Tdg, {0}},
       {dd::GateType::SX, {0}},
       {dd::GateType::SXdg, {0}},
       {dd::GateType::RX, {0}, {theta}},
       {dd::GateType::RY, {0}, {theta}},
       {dd::GateType::RZ, {0}, {theta}},
       {dd::GateType::P, {0}, {theta}},
       {dd::GateType::R, {0}, {theta, phi}},
       {dd::GateType::U2, {0}, {phi, lambda}},
       {dd::GateType::U, {0}, {theta, phi, lambda}},
       {dd::GateType::SWAP, {0, 1}},
       {dd::GateType::iSWAP, {0, 1}},
       {dd::GateType::DCX, {0, 1}},
       {dd::GateType::ECR, {0, 1}},
       {dd::GateType::RXX, {0, 1}, {theta}},
       {dd::GateType::RYY, {0, 1}, {theta}},
       {dd::GateType::RZZ, {0, 1}, {theta}},
       {dd::GateType::RZX, {0, 1}, {theta}},
       {dd::GateType::XXplusYY, {0, 1}, {theta, beta}},
       {dd::GateType::XXminusYY, {0, 1}, {theta, beta}},
       {dd::GateType::X, {1}, {}, {{0}}},
       {dd::GateType::P, {2}, {std::numbers::pi / 5.0}, {{1}}},
       {dd::GateType::X, {2}, {}, {{0}, {1}}},
       {dd::GateType::Sdg, {2}}});
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

  expectEqualToReference(mainFunc(*mod), 4,
                         {{dd::GateType::RCCX, {2, 0, 3}},
                          {dd::GateType::RCCX, {2, 0, 3}, {}, {{1}}}});
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
    expectEqualToReference(mainFunc(*mod), 3,
                           {{dd::GateType::X, {1}},
                            {dd::GateType::T, {0}, {}, {{2}}},
                            {dd::GateType::H, {0}, {}, {{2}}}});
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
    expectEqualToReference(mainFunc(*mod), 2, {{dd::GateType::SWAP, {0, 1}}});
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
    expectEqualToReference(mainFunc(*mod), 3,
                           {{dd::GateType::RX, {0}, {-0.2}},
                            {dd::GateType::RY, {1}, {-0.3}},
                            {dd::GateType::RZ, {2}, {-0.4}}});
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
    expectEqualToReference(mainFunc(*mod), 4,
                           {{dd::GateType::RX, {0}, {-0.2}},
                            {dd::GateType::RY, {1}, {-0.3}},
                            {dd::GateType::RZ, {2}, {-0.4}}});
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
    expectEqualToReference(mainFunc(*mod), 5,
                           {{dd::GateType::RX, {0}, {-0.2}},
                            {dd::GateType::RY, {1}, {-0.3}},
                            {dd::GateType::RZ, {2}, {-0.4}},
                            {dd::GateType::H, {4}}});
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

  expectEqualToReference(mainFunc(*mod), 1, {{dd::GateType::H, {0}}});
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
      simulate(mainFunc(*canonical), dd::makeZeroState(2, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(canonicalSimulation));
  dd->decRef(*canonicalSimulation);

  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*swapped), *dd)));
  EXPECT_TRUE(failed(
      simulate(mainFunc(*swapped), dd::makeZeroState(2, *dd), *dd, rng)));
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
  EXPECT_TRUE(
      failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
}

TEST_F(QCODDFunctionalityTest, RejectsStaticQubitBeyondDDRange) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    b.sink(b.staticQubit(dd::Package::MAX_POSSIBLE_QUBITS));
    return b.intConstant(0);
  });

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
  EXPECT_TRUE(
      failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
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
        simulate(mainFunc(*valid), dd::makeZeroState(1, *dd), *dd, rng);
    ASSERT_TRUE(succeeded(output));
    EXPECT_EQ(roots.size(), 1U);
    EXPECT_EQ(roots.at(*output), 1U);
    dd->decRef(*output);
    EXPECT_TRUE(roots.empty());
  }
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(failed(
        simulate(mainFunc(*tooWide), dd::makeZeroState(1, *dd), *dd, rng)));
    EXPECT_TRUE(roots.empty());
  }

  auto twoQubitDd = std::make_unique<dd::Package>(2);
  EXPECT_TRUE(
      failed(simulate(mainFunc(*tooWide), dd::makeZeroState(1, *twoQubitDd),
                      *twoQubitDd, rng)));
  EXPECT_TRUE(twoQubitDd->getRootSet<dd::vNode>().empty());
  const auto widerOutput = simulate(
      mainFunc(*valid), dd::makeZeroState(2, *twoQubitDd), *twoQubitDd, rng);
  ASSERT_TRUE(succeeded(widerOutput));
  EXPECT_EQ(widerOutput->getVector().size(), 4U);
  twoQubitDd->decRef(*widerOutput);
  EXPECT_TRUE(twoQubitDd->getRootSet<dd::vNode>().empty());

  auto zeroQubitDd = std::make_unique<dd::Package>(0);
  EXPECT_TRUE(failed(
      simulate(mainFunc(*valid), dd::VectorDD::one(), *zeroQubitDd, rng)));
  EXPECT_TRUE(zeroQubitDd->getRootSet<dd::vNode>().empty());
}

TEST_F(QCODDFunctionalityTest,
       SimulationPreservesWiderInputAcrossRuntimeAllocation) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%q0: !qco.qubit) {
        %q1 = qco.alloc : !qco.qubit
        %q2 = qco.x %q1 : !qco.qubit -> !qco.qubit
        qco.sink %q0 : !qco.qubit
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(3);
  auto input = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 1),
      dd::makeZeroState(2, *dd));
  const auto output = simulate(mainFunc(*mod), input, *dd, rng);
  ASSERT_TRUE(succeeded(output));

  auto expected = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 1),
      dd::makeZeroState(3, *dd));
  expected = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 2),
      expected);
  EXPECT_EQ(output->getVector(), expected.getVector());
  dd->decRef(*output);
  dd->decRef(expected);
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
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::H), 0), ref);
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

  expectEqualToReference(mainFunc(*thenMod), 1, {{dd::GateType::X, {0}}});
  expectEqualToReference(mainFunc(*elseMod), 1, {});
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

  expectEqualToReference(mainFunc(*caseMod), 1, {{dd::GateType::X, {0}}});
  expectEqualToReference(mainFunc(*defaultMod), 1, {});
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
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 0),
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
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 0),
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

TEST_F(QCODDFunctionalityTest, SimulateCBitRegisterComparisons) {
  constexpr std::array comparisons{
      std::pair{arith::CmpIPredicate::eq, false},
      std::pair{arith::CmpIPredicate::ne, true},
      std::pair{arith::CmpIPredicate::ult, true},
      std::pair{arith::CmpIPredicate::ule, true},
      std::pair{arith::CmpIPredicate::ugt, false},
      std::pair{arith::CmpIPredicate::uge, false},
      std::pair{arith::CmpIPredicate::slt, false},
      std::pair{arith::CmpIPredicate::sle, false},
      std::pair{arith::CmpIPredicate::sgt, true},
      std::pair{arith::CmpIPredicate::sge, true},
  };
  for (const auto [predicate, expected] : comparisons) {
    auto mod = buildModule([&](QCOProgramBuilder& b) {
      auto reg = b.allocClassicalBitRegister(2, "c");
      auto rhs = b.getIntegerAttr(b.getIntegerType(2), 3);
      auto condition =
          cbit::CompareOp::create(b, b.getI1Type(), predicate, reg, rhs);
      auto q = b.staticQubit(0);
      q = b.qcoIf(
          condition, q, [&](Value arg) { return b.x(arg); },
          [&](Value arg) { return arg; });
      b.sink(q);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    expectSimulatesFromZero(mainFunc(*mod), expected);
  }
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

TEST_F(QCODDFunctionalityTest, RejectsUndefinedCBitRegisterComparison) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto reg =
        b.allocClassicalBitRegister(1, "c", cbit::Initialization::Undefined);
    auto rhs = b.getIntegerAttr(b.getIntegerType(1), 0);
    auto condition = cbit::CompareOp::create(
        b, b.getI1Type(), arith::CmpIPredicate::eq, reg, rhs);
    auto q = b.staticQubit(0);
    q = b.qcoIf(
        condition, q, [&](Value arg) { return arg; },
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

TEST_F(QCODDFunctionalityTest, SimulateWholeCBitRegisterReadAndWrite) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto reg = b.allocClassicalBitRegister(3, "c");
    auto five = arith::ConstantIntOp::create(b, 5, 3).getResult();
    cbit::WriteOp::create(b, five, reg);
    auto value = cbit::ReadOp::create(b, b.getIntegerType(3), reg).getResult();
    auto condition =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, value, five)
            .getResult();
    auto q = b.staticQubit(0);
    q = b.qcoIf(
        condition, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSimulatesFromZero(mainFunc(*mod), true);
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

TEST_F(QCODDFunctionalityTest, SimulateExtUI) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q);
    auto extended = arith::ExtUIOp::create(b, b.getI8Type(), bit).getResult();
    auto one = arith::ConstantIntOp::create(b, 1, 8).getResult();
    auto condition =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, extended, one)
            .getResult();
    q = b.qcoIf(
        condition, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSimulatesFromZero(mainFunc(*mod), false);
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
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 0),
      expected);
  expected = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 2),
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
    auto negative = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, shifted,
                                          arith::ConstantIndexOp::create(b, 0))
                        .getResult();
    q = b.qcoIf(
        negative, q, [&](Value arg) { return b.x(arg); },
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
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 0),
      dd::makeZeroState(1, *dd));
  EXPECT_EQ(out->getVector(), expected.getVector());
  dd->decRef(*out);
  dd->decRef(expected);
}

TEST_F(QCODDFunctionalityTest, RejectsOutOfRangeShift) {
  for (const auto [lhs, amount] :
       {std::pair<int64_t, int64_t>{1, -1}, {1, 64}, {0, 64}, {64, 64}}) {
    auto mod = buildModule([lhs, amount](QCOProgramBuilder& b) {
      auto q = b.staticQubit(0);
      auto value = arith::ConstantIndexOp::create(b, lhs).getResult();
      auto bad = lhs == amount
                     ? value
                     : arith::ConstantIndexOp::create(b, amount).getResult();
      auto shifted = arith::ShLIOp::create(b, value, bad).getResult();
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

  constexpr size_t shots = 64;
  const auto hist = sample(mainFunc(*mod), shots);
  ASSERT_TRUE(succeeded(hist));
  ASSERT_EQ(hist->size(), 1U);
  EXPECT_EQ(hist->begin()->first, "1");
  EXPECT_EQ(hist->begin()->second, shots);
}

TEST_F(QCODDFunctionalityTest, SamplePreservesDeclaredStaticWidth) {
  constexpr auto index = static_cast<int64_t>(dd::Package::DEFAULT_QUBITS);
  auto mod = buildModule([index](QCOProgramBuilder& b) {
    auto q = b.staticQubit(index);
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  const auto histogram = sample(mainFunc(*mod), 8, 1);
  ASSERT_TRUE(succeeded(histogram));
  const auto outcome = std::string(static_cast<size_t>(index + 1), '0');
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{outcome, 8}}));
}

TEST_F(QCODDFunctionalityTest, SampleHadamardApproximatelyBalanced) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.h(b.staticQubit(0));
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  constexpr size_t shots = 2000;
  const auto hist = sample(mainFunc(*mod), shots, 42);
  ASSERT_TRUE(succeeded(hist));
  ASSERT_EQ(hist->size(), 2U);
  EXPECT_EQ(hist->at("0") + hist->at("1"), shots);
  EXPECT_NEAR(static_cast<double>(hist->at("0")), shots / 2.0, 150.0);
}

TEST_F(QCODDFunctionalityTest, SampleResetUsesDynamicSampling) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.reset(b.x(b.staticQubit(0)));
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  const auto histogram = sample(mainFunc(*mod), 16, 7);
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

  constexpr size_t shots = 32;
  const auto hist = sample(mainFunc(*mod), shots, 7);
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

  const auto empty = sample(mainFunc(*unitary), 0, 1);
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
  EXPECT_TRUE(failed(sample(mainFunc(*dynamic), 1, 1)));

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
  EXPECT_TRUE(failed(sample(mainFunc(*measuredDynamic), 1, 1)));
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

  expectEqualToReference(mainFunc(*mod), 13,
                         {{dd::GateType::RX, {0}, {-0.2}},
                          {dd::GateType::RY, {4}, {-0.3}},
                          {dd::GateType::RZ, {8}, {-0.4}},
                          {dd::GateType::H, {12}}});
}

TEST_F(QCODDFunctionalityTest, RejectsUnsupportedOrUnboundClassicalOperations) {
  for (const StringRef source : {
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
               %bad = arith.constant 1.0 : f32
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %one = arith.constant 1 : i32
               %bad = arith.sitofp %one : i32 to f32
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %one = arith.constant 1.0 : f64
               %bad = math.erf %one : f64
               qco.sink %q : !qco.qubit
               return
             }
           })mlir"}) {
    expectMlirSimulationFails(1, source);
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
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 0),
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
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(dd::GateType::X), 0),
      dd::makeZeroState(1, *dd));
  EXPECT_EQ(out->getVector(), expected.getVector());
  dd->decRef(*out);
  dd->decRef(expected);
}

TEST_F(QCODDFunctionalityTest, RejectsUnboundClassicalRegionResults) {
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
           })mlir"}) {
    expectMlirSimulationFails(1, source);
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
    const auto functionality = buildFunctionality(mainFunc(*mod), *dd);
    ASSERT_TRUE(succeeded(functionality));
    EXPECT_EQ(dd->qubits(), 2U);
    dd->decRef(*functionality);
    EXPECT_TRUE(
        failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
  }

  expectMlirFails(1, R"mlir(
    module {
      func.func @main(%qarg: !qco.qubit) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.h %qarg : !qco.qubit -> !qco.qubit
        qco.sink %q : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
    module {
      func.func @main(%qarg: !qco.qubit) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.barrier %qarg : !qco.qubit -> !qco.qubit
        qco.sink %q : !qco.qubit
        qco.sink %q1 : !qco.qubit
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
        qco.sink %q : !qco.qubit
        qco.sink %q_out : !qco.qubit
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
        qco.sink %c_out : !qco.qubit
        qco.sink %t_out : !qco.qubit
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
    module {
      func.func @main(%theta: f64) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.rz(%theta) %q : !qco.qubit -> !qco.qubit
        qco.sink %q1 : !qco.qubit
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
        qco.sink %q_out : !qco.qubit
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
        qco.sink %c_out : !qco.qubit
        qco.sink %t_out : !qco.qubit
        return
      }
    }
  )mlir");

  for (const bool composed : {false, true}) {
    auto wideModifier = buildModule([composed](QCOProgramBuilder& b) {
      SmallVector<Value, 11> qubits;
      for (int64_t i = 0; i < 11; ++i) {
        qubits.push_back(b.staticQubit(i));
      }
      auto outputs = b.inv(qubits, [&](ValueRange args) -> SmallVector<Value> {
        SmallVector<Value> results(args.begin(), args.end());
        results[0] = b.x(results[0]);
        if (composed) {
          results[1] = b.x(results[1]);
        }
        return results;
      });
      for (Value output : outputs) {
        b.sink(output);
      }
      return b.intConstant(0);
    });
    ASSERT_TRUE(wideModifier);
    auto wideDD = std::make_unique<dd::Package>(11);
    EXPECT_TRUE(failed(buildFunctionality(mainFunc(*wideModifier), *wideDD)));
  }
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

TEST_F(QCODDFunctionalityTest, ScfForCarriesQubitsSimultaneously) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.static 0 : !qco.qubit
        %q1 = qco.static 1 : !qco.qubit
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %a, %b = scf.for %iv = %c0 to %c2 step %c1
            iter_args(%x = %q0, %y = %q1)
            -> (!qco.qubit, !qco.qubit) {
          scf.yield %y, %x : !qco.qubit, !qco.qubit
        }
        %flipped = qco.x %a : !qco.qubit -> !qco.qubit
        qco.sink %flipped : !qco.qubit
        qco.sink %b : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  const auto histogram = sample(mainFunc(*mod), 1, 1);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"01", 1}}));
}

TEST_F(QCODDFunctionalityTest, ScfForCarriesScalarsSimultaneously) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %zero = arith.constant 0 : i8
        %one = arith.constant 1 : i8
        %a, %b = scf.for %iv = %c0 to %c2 step %c1
            iter_args(%x = %zero, %y = %one) -> (i8, i8) {
          scf.yield %y, %x : i8, i8
        }
        %is_zero = arith.cmpi eq, %a, %zero : i8
        %out = qco.if %is_zero args(%arg = %q) -> (!qco.qubit) {
          %flipped = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.yield %flipped : !qco.qubit
        } else args(%arg = %q) {
          qco.yield %arg : !qco.qubit
        }
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  expectSimulatesFromZero(mainFunc(*mod), true);
}

TEST_F(QCODDFunctionalityTest, ScfForCarriesRegistersSimultaneously) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %r0 = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %r1 = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %true = arith.constant true
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        cbit.store %true, %r1[%c0] : !cbit.reg<1>
        %a, %b = scf.for %iv = %c0 to %c2 step %c1
            iter_args(%x = %r0, %y = %r1) -> (!cbit.reg<1>, !cbit.reg<1>) {
          scf.yield %y, %x : !cbit.reg<1>, !cbit.reg<1>
        }
        %bit = cbit.load %a[%c0] : !cbit.reg<1>
        %out = qco.if %bit args(%arg = %q) -> (!qco.qubit) {
          qco.yield %arg : !qco.qubit
        } else args(%arg = %q) {
          %flipped = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.yield %flipped : !qco.qubit
        }
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  expectSimulatesFromZero(mainFunc(*mod), true);
}

TEST_F(QCODDFunctionalityTest, ScfForSnapshotsYieldedInductionValue) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %cm1 = arith.constant -1 : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %out, %last = scf.for %iv = %c0 to %c2 step %c1
            iter_args(%qarg = %q, %prev = %cm1) -> (!qco.qubit, index) {
          %same = arith.cmpi eq, %prev, %iv : index
          %next = qco.if %same args(%arg = %qarg) -> (!qco.qubit) {
            %flipped = qco.x %arg : !qco.qubit -> !qco.qubit
            qco.yield %flipped : !qco.qubit
          } else args(%arg = %qarg) {
            qco.yield %arg : !qco.qubit
          }
          scf.yield %next, %iv : !qco.qubit, index
        }
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  expectSimulatesFromZero(mainFunc(*mod), false);
}

TEST_F(QCODDFunctionalityTest, RejectsUnsupportedFuncCalls) {
  auto selfRecursive = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%recurse: i1) {
        scf.if %recurse {
          %false = arith.constant false
          func.call @main(%false) : (i1) -> ()
        }
        return
      }
    }
  )mlir",
                                                   context.get());
  ASSERT_TRUE(selfRecursive);
  auto selfRecursiveFunc = mainFunc(*selfRecursive);
  DDArgumentBindings bindings;
  bindings[selfRecursiveFunc.getArgument(0)] =
      BoolAttr::get(context.get(), true);
  auto zeroQubitDd = std::make_unique<dd::Package>(0);
  EXPECT_TRUE(failed(simulate(selfRecursiveFunc, dd::VectorDD::one(),
                              *zeroQubitDd, rng, bindings)));

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
      EXPECT_TRUE(failed(
          simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
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

  auto signedExtreme = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto results =
        b.scfFor(std::numeric_limits<int64_t>::min(),
                 std::numeric_limits<int64_t>::max(),
                 std::numeric_limits<int64_t>::max(), ValueRange{q},
                 [&](Value /*iv*/, ValueRange args) -> SmallVector<Value> {
                   return {b.x(args[0])};
                 });
    b.sink(results[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(signedExtreme);
  expectSimulatesFromZero(mainFunc(*signedExtreme), true);

  auto unsignedExtreme = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %lower = arith.constant 9223372036854775807 : index
        %upper = arith.constant -1 : index
        %step = arith.constant 9223372036854775807 : index
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
  ASSERT_TRUE(unsignedExtreme);
  expectSimulatesFromZero(mainFunc(*unsignedExtreme), false);
}

TEST_F(QCODDFunctionalityTest, ScfForSharesExecutionBudget) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto outer = b.scfFor(
        0, 100, 1, ValueRange{q},
        [&](Value /*iv*/, ValueRange outerArgs) -> SmallVector<Value> {
          return b.scfFor(0, 100, 1, outerArgs,
                          [&](Value /*innerIv*/, ValueRange innerArgs)
                              -> SmallVector<Value> { return {innerArgs[0]}; });
        });
    b.sink(outer[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(
      failed(simulate(mainFunc(*mod), dd::makeZeroState(1, *dd), *dd, rng)));
  EXPECT_TRUE(dd->getRootSet<dd::vNode>().empty());
}

TEST_F(QCODDFunctionalityTest, ExecutionBudgetIncludesBranchesAndCalls) {
  for (const StringRef source : {
           R"mlir(module {
             func.func @main() {
               %true = arith.constant true
               %zero = arith.constant 0 : index
               %limit = arith.constant 10000 : index
               %one = arith.constant 1 : index
               scf.for %i = %zero to %limit step %one {
                 scf.if %true {
                 }
               }
               return
             }
           })mlir",
           R"mlir(module {
             func.func @noop() {
               return
             }
             func.func @main() {
               %zero = arith.constant 0 : index
               %limit = arith.constant 10000 : index
               %one = arith.constant 1 : index
               scf.for %i = %zero to %limit step %one {
                 func.call @noop() : () -> ()
               }
               return
             }
           })mlir"}) {
    expectMlirSimulationFails(0, source);
  }
}

TEST_F(QCODDFunctionalityTest, SimulateRicherClassicalArithmetic) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    const auto constant = [&](const int64_t value) {
      return arith::ConstantIntOp::create(b, value, 8).getResult();
    };
    auto zero = constant(0);
    auto one = constant(1);
    auto two = constant(2);
    auto three = constant(3);
    auto four = constant(4);
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

TEST_F(QCODDFunctionalityTest, SampleReturnsCBitRegistersInCountStringOrder) {
  auto mod = buildModule([](QCOProgramBuilder& b) -> SmallVector<Value> {
    auto wide =
        b.allocClassicalBitRegister(2, {}, cbit::Initialization::Undefined);
    auto narrow = b.allocClassicalBitRegister(1);
    b.storeClassicalBit(b.boolConstant(false), wide, 0);
    b.storeClassicalBit(b.boolConstant(true), wide, 1);
    b.storeClassicalBit(b.boolConstant(true), narrow, 0);
    return {wide, narrow};
  });
  ASSERT_TRUE(mod);

  const auto histogram = sample(mainFunc(*mod), 8, 3);
  ASSERT_TRUE(succeeded(histogram));
  // Count strings place the last returned register first: "1" + "10".
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"110", 8}}));
}

TEST_F(QCODDFunctionalityTest, SampleRejectsUndefinedAndMixedResults) {
  auto undefined = buildModule([](QCOProgramBuilder& b) {
    return b.allocClassicalBitRegister(1, {}, cbit::Initialization::Undefined);
  });
  auto mixed = buildModule([](QCOProgramBuilder& b) -> SmallVector<Value> {
    return {b.allocClassicalBitRegister(1), b.intConstant(0)};
  });
  ASSERT_TRUE(undefined);
  ASSERT_TRUE(mixed);

  EXPECT_TRUE(failed(sample(mainFunc(*undefined), 1, 3)));
  EXPECT_TRUE(failed(sample(mainFunc(*mixed), 1, 3)));
}

TEST_F(QCODDFunctionalityTest,
       SampleDefersReturnedMeasurementDespiteLaterUnrelatedOp) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto reg =
        b.allocClassicalBitRegister(1, {}, cbit::Initialization::Undefined);
    auto q0 = b.x(b.staticQubit(0));
    auto q1 = b.staticQubit(1);
    std::tie(q0, std::ignore) = b.measure(q0, reg, 0);
    q1 = b.x(q1);
    b.sink(q0);
    b.sink(q1);
    return reg;
  });
  ASSERT_TRUE(mod);

  const auto histogram = sample(mainFunc(*mod), 64, 11);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"1", 64}}));
}

TEST_F(QCODDFunctionalityTest, SampleDefersAllocatedQubitMeasurement) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto reg =
        b.allocClassicalBitRegister(1, {}, cbit::Initialization::Undefined);
    auto q = b.x(b.allocQubit());
    std::tie(q, std::ignore) = b.measure(q, reg, 0);
    b.sink(q);
    return reg;
  });
  ASSERT_TRUE(mod);

  const auto histogram = sample(mainFunc(*mod), 8, 1);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"1", 8}}));
}

TEST_F(QCODDFunctionalityTest, SampleExecutesControlMeasurementPerShot) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto reg =
        b.allocClassicalBitRegister(1, {}, cbit::Initialization::Undefined);
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q, reg, 0);
    q = b.qcoIf(
        bit, q, [&](Value arg) { return arg; }, [&](Value arg) { return arg; });
    q = b.x(q);
    b.sink(q);
    return reg;
  });
  ASSERT_TRUE(mod);

  constexpr size_t shots = 32;
  const auto histogram = sample(mainFunc(*mod), shots, 11);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"1", shots}}));
}

TEST_F(QCODDFunctionalityTest,
       SampleFallsBackWhenDeferredMeasurementIsStoredInMemRef) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() -> !cbit.reg<1> {
        %out = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<1>
        %memory = memref.alloc() : memref<1xi1>
        %zero = arith.constant 0 : index
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.h %q : !qco.qubit -> !qco.qubit
        %q2, %bit = qco.measure %q1 : !qco.qubit
        memref.store %bit, %memory[%zero] : memref<1xi1>
        %loaded = memref.load %memory[%zero] : memref<1xi1>
        cbit.store %loaded, %out[%zero] : !cbit.reg<1>
        memref.dealloc %memory : memref<1xi1>
        qco.sink %q2 : !qco.qubit
        return %out : !cbit.reg<1>
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  constexpr size_t shots = 128;
  const auto histogram = sample(mainFunc(*mod), shots, 1);
  ASSERT_TRUE(succeeded(histogram));
  ASSERT_EQ(histogram->size(), 2U);
  EXPECT_EQ(histogram->at("0") + histogram->at("1"), shots);
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

TEST_F(QCODDFunctionalityTest, RejectsUnsupportedClassicalMemRefs) {
  for (const StringRef source : {
           R"mlir(module {
             func.func @main(%reg: memref<i1>) {
               %value = memref.load %reg[] : memref<i1>
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%reg: memref<i1>) {
               %value = arith.constant true
               memref.store %value, %reg[] : memref<i1>
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%n: index) {
               %reg = memref.alloc(%n) : memref<?xi1>
               memref.dealloc %reg : memref<?xi1>
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %reg = memref.alloc() : memref<1xf32>
               memref.dealloc %reg : memref<1xf32>
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %reg = memref.alloc() : memref<1xi1>
               %value = arith.constant true
               %i2 = arith.constant 2 : index
               memref.store %value, %reg[%i2] : memref<1xi1>
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

  const auto histogram = sample(mainFunc(*mod), 128, 9);
  ASSERT_TRUE(succeeded(histogram));
  ASSERT_EQ(histogram->size(), 2U);
  EXPECT_EQ(histogram->at("0") + histogram->at("1"), 128U);
}

TEST_F(QCODDFunctionalityTest, SampleDefersNestedTerminalMeasurement) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() -> !cbit.reg<1> {
        %reg = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<1>
        %true = arith.constant true
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.h %q : !qco.qubit -> !qco.qubit
        %q2 = qco.if %true args(%qin = %q1) -> (!qco.qubit) {
          %measured, %bit = qco.measure %qin : !qco.qubit
          %i0 = arith.constant 0 : index
          cbit.store %bit, %reg[%i0] : !cbit.reg<1>
          qco.yield %measured : !qco.qubit
        } else args(%qin = %q1) {
          qco.yield %qin : !qco.qubit
        }
        qco.sink %q2 : !qco.qubit
        return %reg : !cbit.reg<1>
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  constexpr size_t shots = 128;
  const auto histogram = sample(mainFunc(*mod), shots, 9);
  ASSERT_TRUE(succeeded(histogram));
  ASSERT_EQ(histogram->size(), 2U);
  EXPECT_EQ(histogram->at("0") + histogram->at("1"), shots);
}

TEST_F(QCODDFunctionalityTest, SymbolicParametersUseBindings) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) {
        %q = qco.static 0 : !qco.qubit
        %twice = arith.addf %theta, %theta : f64
        %q1 = qco.rx(%twice) %q : !qco.qubit -> !qco.qubit
        qco.gphase(%theta)
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  auto concrete = buildModule([](QCOProgramBuilder& b) {
    auto q = b.rx(std::numbers::pi, b.staticQubit(0));
    b.gphase(std::numbers::pi / 2.0);
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);
  ASSERT_TRUE(concrete);

  auto func = mainFunc(*mod);
  DDArgumentBindings bindings;
  bindings[func.getArgument(0)] = FloatAttr::get(
      cast<FloatType>(func.getArgument(0).getType()), std::numbers::pi / 2.0);

  auto dd = std::make_unique<dd::Package>(1);
  auto actual = buildFunctionality(func, *dd, bindings);
  auto expected = buildFunctionality(mainFunc(*concrete), *dd);
  ASSERT_TRUE(succeeded(actual));
  ASSERT_TRUE(succeeded(expected));
  EXPECT_EQ(actual->getMatrix(1), expected->getMatrix(1));
  dd->decRef(*actual);
  dd->decRef(*expected);

  const auto histogram = sample(func, 8, 1, bindings);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"1", 8}}));

  const auto state = simulateStatevector(func, *dd, bindings);
  ASSERT_TRUE(succeeded(state));
  const auto vector = state->getVector();
  ASSERT_EQ(vector.size(), 2U);
  EXPECT_NEAR(std::norm(vector[0]), 0.0, 1e-12);
  EXPECT_NEAR(std::norm(vector[1]), 1.0, 1e-12);
  dd->decRef(*state);

  EXPECT_TRUE(failed(buildFunctionality(func, *dd)));
  bindings[func.getArgument(0)] =
      IntegerAttr::get(IntegerType::get(context.get(), 64), 1);
  EXPECT_TRUE(failed(buildFunctionality(func, *dd, bindings)));
  bindings[func.getArgument(0)] =
      FloatAttr::get(Float32Type::get(context.get()), 1.0);
  EXPECT_TRUE(failed(buildFunctionality(func, *dd, bindings)));
}

TEST_F(QCODDFunctionalityTest, RejectsNonFiniteParameters) {
  auto gate = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) {
        %q = qco.static 0 : !qco.qubit
        %out = qco.rx(%theta) %q : !qco.qubit -> !qco.qubit
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                          context.get());
  auto phase = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) {
        qco.gphase(%theta)
        return
      }
    }
  )mlir",
                                           context.get());
  ASSERT_TRUE(gate);
  ASSERT_TRUE(phase);

  for (auto [func, value] :
       {std::pair{mainFunc(*gate), std::numeric_limits<double>::infinity()},
        std::pair{mainFunc(*phase),
                  std::numeric_limits<double>::quiet_NaN()}}) {
    DDArgumentBindings bindings;
    bindings[func.getArgument(0)] =
        FloatAttr::get(Float64Type::get(context.get()), value);
    EXPECT_TRUE(failed(sample(func, 1, 1, bindings)));
  }
}

TEST_F(QCODDFunctionalityTest, BuildsThroughConcreteControlFlow) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    q = b.qcoIf(
        true, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    q = b.qcoIndexSwitch(1, q, ArrayRef<int64_t>{0, 1},
                         SmallVector<function_ref<Value(Value)>>{
                             [&](Value arg) { return b.h(arg); },
                             [&](Value arg) { return b.z(arg); }},
                         [&](Value arg) { return arg; });
    q = b.scfFor(0, 2, 1, ValueRange{q.value},
                 [&](Value /*index*/, ValueRange args) -> SmallVector<Value> {
                   return {b.h(args[0])};
                 })[0];
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectEqualToReference(mainFunc(*mod), 1,
                         {{dd::GateType::X, {0}},
                          {dd::GateType::Z, {0}},
                          {dd::GateType::H, {0}},
                          {dd::GateType::H, {0}}});
}

TEST_F(QCODDFunctionalityTest, StructuredScfAndWhileCarryValues) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.x %q : !qco.qubit -> !qco.qubit
        %true = arith.constant true
        %selector = scf.if %true -> index {
          %one = arith.constant 1 : index
          scf.yield %one : index
        } else {
          %zero = arith.constant 0 : index
          scf.yield %zero : index
        }
        %apply_z = scf.index_switch %selector -> i1
        case 1 {
          %yes = arith.constant true
          scf.yield %yes : i1
        }
        default {
          %no = arith.constant false
          scf.yield %no : i1
        }
        %q2 = qco.if %apply_z args(%qarg = %q1) -> (!qco.qubit) {
          %out = qco.z %qarg : !qco.qubit -> !qco.qubit
          qco.yield %out : !qco.qubit
        } else args(%qarg = %q1) {
          qco.yield %qarg : !qco.qubit
        }
        %zero = arith.constant 0 : index
        %result:2 = scf.while (%qarg = %q2, %i = %zero)
            : (!qco.qubit, index) -> (!qco.qubit, index) {
          %one = arith.constant 1 : index
          %condition = arith.cmpi slt, %i, %one : index
          scf.condition(%condition) %qarg, %i : !qco.qubit, index
        } do {
        ^bb0(%qarg: !qco.qubit, %i: index):
          %out = qco.x %qarg : !qco.qubit -> !qco.qubit
          %one = arith.constant 1 : index
          %next = arith.addi %i, %one : index
          scf.yield %out, %next : !qco.qubit, index
        }
        %false = arith.constant false
        %final = scf.while (%qarg = %result#0)
            : (!qco.qubit) -> !qco.qubit {
          scf.condition(%false) %qarg : !qco.qubit
        } do {
        ^bb0(%qarg: !qco.qubit):
          %unreachable = qco.h %qarg : !qco.qubit -> !qco.qubit
          scf.yield %unreachable : !qco.qubit
        }
        qco.sink %final : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  expectEqualToReference(
      mainFunc(*mod), 1,
      {{dd::GateType::X, {0}}, {dd::GateType::Z, {0}}, {dd::GateType::X, {0}}});
}

TEST_F(QCODDFunctionalityTest, DynamicAllocationsAndQTensorBookkeeping) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.x(b.allocQubit());
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto tensor = b.qtensorAlloc(one);
    Value remaining;
    Value q1;
    std::tie(remaining, q1) = b.qtensorExtract(tensor, 0);
    auto output = b.qtensorFromElements({q0, b.x(q1)});
    b.qtensorDealloc(remaining);
    b.qtensorDealloc(output);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  const auto histogram = sample(mainFunc(*mod), 8, 1);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"11", 8}}));

  auto invalidIndex = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %one = arith.constant 1 : index
        %tensor = qtensor.alloc(%one) : tensor<?x!qco.qubit>
        %remaining, %q = qtensor.extract %tensor[%one]
            : tensor<?x!qco.qubit>
        qco.sink %q : !qco.qubit
        qtensor.dealloc %remaining : tensor<?x!qco.qubit>
        return
      }
    }
  )mlir",
                                                  context.get());
  ASSERT_TRUE(invalidIndex);
  EXPECT_TRUE(failed(sample(mainFunc(*invalidIndex), 1, 1)));
}

TEST_F(QCODDFunctionalityTest, DynamicQTensorArgumentUsesBoundExtent) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%arg0: tensor<?x!qco.qubit>)
          -> tensor<?x!qco.qubit> {
        %one = arith.constant 1 : index
        %remaining, %q = qtensor.extract %arg0[%one]
            : tensor<?x!qco.qubit>
        %q1 = qco.x %q : !qco.qubit -> !qco.qubit
        %result = qtensor.insert %q1 into %remaining[%one]
            : tensor<?x!qco.qubit>
        return %result : tensor<?x!qco.qubit>
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  auto func = mainFunc(*mod);
  DDArgumentBindings bindings;
  bindings[func.getArgument(0)] =
      IntegerAttr::get(IndexType::get(context.get()), 2);

  auto dd = std::make_unique<dd::Package>(2);
  const auto histogram = sample(func, 4, 1, bindings);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"10", 4}}));

  EXPECT_TRUE(failed(buildFunctionality(func, *dd)));
  bindings[func.getArgument(0)] =
      IntegerAttr::get(IndexType::get(context.get()), -1);
  EXPECT_TRUE(failed(buildFunctionality(func, *dd, bindings)));
  bindings[func.getArgument(0)] =
      IntegerAttr::get(IntegerType::get(context.get(), 64), 2);
  EXPECT_TRUE(failed(buildFunctionality(func, *dd, bindings)));
}

TEST_F(QCODDFunctionalityTest, RejectsQTensorBeyondQubitRange) {
  expectMlirFails(1, R"mlir(
    module {
      func.func @main(%qubits: tensor<65537x!qco.qubit>) {
        return
      }
    }
  )mlir");
}

TEST_F(QCODDFunctionalityTest, QTensorFlowsThroughLoopAndCall) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @flip(%arg: tensor<1x!qco.qubit>)
          -> tensor<1x!qco.qubit> {
        %zero = arith.constant 0 : index
        %remaining, %q = qtensor.extract %arg[%zero]
            : tensor<1x!qco.qubit>
        %q1 = qco.x %q : !qco.qubit -> !qco.qubit
        %result = qtensor.insert %q1 into %remaining[%zero]
            : tensor<1x!qco.qubit>
        return %result : tensor<1x!qco.qubit>
      }
      func.func @main() {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %tensor = qtensor.alloc(%one) : tensor<1x!qco.qubit>
        %result = scf.for %i = %zero to %one step %one
            iter_args(%arg = %tensor) -> tensor<1x!qco.qubit> {
          %next = func.call @flip(%arg)
              : (tensor<1x!qco.qubit>) -> tensor<1x!qco.qubit>
          scf.yield %next : tensor<1x!qco.qubit>
        }
        qtensor.dealloc %result : tensor<1x!qco.qubit>
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  const auto histogram = sample(mainFunc(*mod), 4, 1);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"1", 4}}));
}

TEST_F(QCODDFunctionalityTest, WiderMemRefCallsShareStorage) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @set(%reg: memref<?xi16>, %value: i16) {
        %zero = arith.constant 0 : index
        memref.store %value, %reg[%zero] : memref<?xi16>
        return
      }
      func.func @main() {
        %one = arith.constant 1 : index
        %reg = memref.alloc(%one) : memref<?xi16>
        %three = arith.constant 3 : i16
        %four = arith.constant 4 : i16
        %seven = arith.addi %three, %four : i16
        %two = arith.constant 2 : i16
        %fourteen = arith.muli %seven, %two : i16
        %quotient = arith.divsi %fourteen, %two : i16
        %unsigned_quotient = arith.divui %fourteen, %two : i16
        %remainder = arith.remui %quotient, %two : i16
        %signed_remainder = arith.remsi %unsigned_quotient, %two : i16
        %shifted = arith.shli %remainder, %two : i16
        %restored = arith.shrui %shifted, %two : i16
        %signed_restored = arith.shrsi %shifted, %two : i16
        %wide = arith.extui %restored : i16 to i32
        %narrow = arith.trunci %wide : i32 to i16
        %as_float = arith.sitofp %narrow : i16 to f64
        %back = arith.fptosi %as_float : f64 to i16
        func.call @set(%reg, %quotient) : (memref<?xi16>, i16) -> ()
        %zero = arith.constant 0 : index
        %stored = memref.load %reg[%zero] : memref<?xi16>
        %expected = arith.constant 7 : i16
        %integer_ok = arith.cmpi eq, %stored, %expected : i16
        %casts_ok = arith.cmpi eq, %back, %remainder : i16
        %signed_remainder_ok = arith.cmpi eq, %signed_remainder, %remainder : i16
        %one_float = arith.constant 1.0 : f64
        %unsigned_float = arith.uitofp %narrow : i16 to f64
        %unsigned_float_ok = arith.cmpf oeq, %unsigned_float, %one_float : f64
        %negative_one_float = arith.negf %one_float : f64
        %two_float = arith.addf %one_float, %one_float : f64
        %three_float = arith.subf %two_float, %negative_one_float : f64
        %six_float = arith.mulf %three_float, %two_float : f64
        %four_float = arith.addf %two_float, %two_float : f64
        %half = arith.divf %four_float, %two_float : f64
        %float_remainder = arith.remf %half, %one_float : f64
        %zero_float = arith.constant 0.0 : f64
        %six = arith.constant 6.0 : f64
        %remainder_ok = arith.cmpf oeq, %float_remainder, %zero_float : f64
        %product_ok = arith.cmpf oeq, %six_float, %six : f64
        %integer_and_casts = arith.andi %integer_ok, %casts_ok : i1
        %all_integers_ok = arith.andi %integer_and_casts, %signed_remainder_ok : i1
        %float_results_ok = arith.andi %remainder_ok, %product_ok : i1
        %floats_ok = arith.andi %float_results_ok, %unsigned_float_ok : i1
        %condition = arith.andi %all_integers_ok, %floats_ok : i1
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.if %condition args(%qin = %q) -> (!qco.qubit) {
          %out = qco.x %qin : !qco.qubit -> !qco.qubit
          qco.yield %out : !qco.qubit
        } else args(%qin = %q) {
          qco.yield %qin : !qco.qubit
        }
        memref.dealloc %reg : memref<?xi16>
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  expectSimulatesFromZero(mainFunc(*mod), true);
}

TEST_F(QCODDFunctionalityTest, BindingsDriveObservableClassicalPath) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%idx: index, %word: i16, %flag: i1, %unbound: i1) {
        %one = arith.constant 1 : i16
        %expected = arith.constant 4 : i16
        %sum = arith.addi %word, %one : i16
        %integer_ok = arith.cmpi eq, %sum, %expected : i16
        %as_index = arith.index_cast %word : i16 to index
        %index_ok = arith.cmpi eq, %as_index, %idx : index
        %both = arith.andi %integer_ok, %index_ok : i1
        %condition = arith.select %flag, %both, %unbound : i1
        %q = qco.static 0 : !qco.qubit
        %out = qco.if %condition args(%arg = %q) -> (!qco.qubit) {
          %flipped = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.yield %flipped : !qco.qubit
        } else args(%arg = %q) {
          qco.yield %arg : !qco.qubit
        }
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  auto func = mainFunc(*mod);
  DDArgumentBindings bindings;
  bindings[func.getArgument(0)] =
      IntegerAttr::get(IndexType::get(context.get()), 3);
  bindings[func.getArgument(1)] =
      IntegerAttr::get(IntegerType::get(context.get(), 16), 3);
  bindings[func.getArgument(2)] =
      IntegerAttr::get(IntegerType::get(context.get(), 1), 1);
  const auto histogram = sample(func, 4, 1, bindings);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(*histogram, (std::map<std::string, size_t>{{"1", 4}}));

  bindings[func.getArgument(1)] =
      IntegerAttr::get(IntegerType::get(context.get(), 8), 3);
  EXPECT_TRUE(failed(sample(func, 1, 1, bindings)));
}

TEST_F(QCODDFunctionalityTest, RepeatedSimulationPreservesFoldableIR) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %zero = arith.constant 0 : i8
        %wide = arith.extui %zero : i8 to i16
        %narrow = arith.trunci %wide : i16 to i8
        %wide_again = arith.extsi %narrow : i8 to i16
        %equal = arith.cmpi eq, %wide, %wide_again : i16
        %q = qco.static 0 : !qco.qubit
        %out = qco.if %equal args(%arg = %q) -> (!qco.qubit) {
          %flipped = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.yield %flipped : !qco.qubit
        } else args(%arg = %q) {
          qco.yield %arg : !qco.qubit
        }
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  const auto print = [](ModuleOp module) {
    std::string result;
    llvm::raw_string_ostream stream(result);
    module.print(stream);
    stream.flush();
    return result;
  };
  const auto original = print(*mod);
  expectSimulatesFromZero(mainFunc(*mod), true);
  EXPECT_EQ(print(*mod), original);
  expectSimulatesFromZero(mainFunc(*mod), true);
  EXPECT_EQ(print(*mod), original);
}

TEST_F(QCODDFunctionalityTest, RejectsRepresentativeClassicalRuntimeErrors) {
  for (const StringRef source : {
           R"mlir(module {
             func.func @main(%unbound: i16) {
               %true = arith.constant true
               %zero = arith.constant 0 : i16
               %selected = arith.select %true, %unbound, %zero : i16
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%reg: memref<1xi16>) {
               %zero = arith.constant 0 : index
               %value = memref.load %reg[%zero] : memref<1xi16>
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %index = arith.constant 0 : index
               %reg = memref.alloc() : memref<1xi16>
               %value = memref.load %reg[%index] : memref<1xi16>
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %negative = arith.constant -1 : index
               %reg = memref.alloc(%negative) : memref<?xi16>
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %zero = arith.constant 0 : i8
               %invalid = arith.divui %zero, %zero : i8
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %zero = arith.constant 0 : i8
               %invalid = arith.divsi %zero, %zero : i8
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %zero = arith.constant 0 : i8
               %invalid = arith.remui %zero, %zero : i8
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %zero = arith.constant 0 : i8
               %invalid = arith.remsi %zero, %zero : i8
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %huge = arith.constant 1.0e+300 : f64
               %invalid = arith.fptosi %huge : f64 to i8
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main(%size: index) {
               %tensor = qtensor.alloc(%size) : tensor<?x!qco.qubit>
               qtensor.dealloc %tensor : tensor<?x!qco.qubit>
               return
             }
           })mlir"}) {
    expectMlirSimulationFails(0, source);
  }

  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %zero = arith.constant 0 : index
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  auto func = mainFunc(*mod);
  auto constant = *func.getBody().front().getOps<arith::ConstantOp>().begin();
  DDArgumentBindings bindings;
  bindings[constant.getResult()] =
      IntegerAttr::get(IndexType::get(context.get()), 0);
  auto dd = std::make_unique<dd::Package>(0);
  EXPECT_TRUE(failed(simulate(func, dd::VectorDD::one(), *dd, rng, bindings)));
  EXPECT_TRUE(dd->getRootSet<dd::vNode>().empty());
}

TEST_F(QCODDFunctionalityTest, BuildFunctionalityRestrictsRuntimeAllocations) {
  auto topLevel = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.alloc : !qco.qubit
        %out = qco.x %q : !qco.qubit -> !qco.qubit
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                              context.get());
  expectMlirFails(1, R"mlir(
    module {
      func.func @main() {
        %true = arith.constant true
        scf.if %true {
          %q = qco.alloc : !qco.qubit
          qco.sink %q : !qco.qubit
        }
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
    module {
      func.func @main() {
        %one = arith.constant 1 : index
        %tensor = qtensor.alloc(%one) : tensor<?x!qco.qubit>
        qtensor.dealloc %tensor : tensor<?x!qco.qubit>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(topLevel);

  auto dd = std::make_unique<dd::Package>(1);
  const auto functionality = buildFunctionality(mainFunc(*topLevel), *dd);
  ASSERT_TRUE(succeeded(functionality));
  dd->decRef(*functionality);
}

TEST_F(QCODDFunctionalityTest, RejectsMultiBlockAndExecuteRegion) {
  for (const StringRef source : {
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               cf.br ^next(%q : !qco.qubit)
             ^next(%arg: !qco.qubit):
               qco.sink %arg : !qco.qubit
               return
             }
           })mlir",
           R"mlir(module {
             func.func @main() {
               %q = qco.static 0 : !qco.qubit
               %out = scf.execute_region -> !qco.qubit {
                 scf.yield %q : !qco.qubit
               }
               qco.sink %out : !qco.qubit
               return
             }
           })mlir"}) {
    auto mod = parseSourceString<ModuleOp>(source, context.get());
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(1);
    EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
  }
}

TEST_F(QCODDFunctionalityTest, InterpretsMinMaxAndCommonMathOperations) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    Value minusOne = arith::ConstantIntOp::create(b, -1, 8);
    Value three = arith::ConstantIntOp::create(b, 3, 8);
    Value maxMatches = arith::CmpIOp::create(
        b, arith::CmpIPredicate::eq, arith::MaxSIOp::create(b, minusOne, three),
        three);
    Value zero =
        arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(0.0));
    Value sinMatches = arith::CmpFOp::create(
        b, arith::CmpFPredicate::OEQ, math::SinOp::create(b, zero), zero);
    Value all = arith::AndIOp::create(b, maxMatches, sinMatches);
    q = b.qcoIf(
        all, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectEqualToReference(mainFunc(*mod), 1, {{dd::GateType::X, {0}}});
}

TEST_F(QCODDFunctionalityTest, StatevectorSupportsTerminalMeasurements) {
  auto empty =
      buildModule([](QCOProgramBuilder& b) { return b.intConstant(0); });
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto reg =
        b.allocClassicalBitRegister(1, {}, cbit::Initialization::Undefined);
    auto q = b.h(b.staticQubit(0));
    std::tie(q, std::ignore) = b.measure(q, reg, 0);
    b.sink(q);
    return reg;
  });
  ASSERT_TRUE(empty);
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(0);
  const auto emptyState = simulateStatevector(mainFunc(*empty), *dd);
  ASSERT_TRUE(succeeded(emptyState));
  EXPECT_TRUE(emptyState->isTerminal());

  const auto state = simulateStatevector(mainFunc(*mod), *dd);
  ASSERT_TRUE(succeeded(state));
  EXPECT_EQ(dd->qubits(), 1U);
  const auto vector = state->getVector();
  ASSERT_EQ(vector.size(), 2U);
  EXPECT_NEAR(std::norm(vector[0]), 0.5, 1e-12);
  EXPECT_NEAR(std::norm(vector[1]), 0.5, 1e-12);
  dd->decRef(*state);
}

TEST_F(QCODDFunctionalityTest, StatevectorRejectsNonTerminalMeasurement) {
  auto reuse = buildModule([](QCOProgramBuilder& b) {
    auto q = b.h(b.staticQubit(0));
    std::tie(q, std::ignore) = b.measure(q);
    b.sink(b.h(q));
    return b.intConstant(0);
  });
  auto reset = buildModule([](QCOProgramBuilder& b) {
    b.sink(b.reset(b.staticQubit(0)));
    return b.intConstant(0);
  });
  auto call = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @measure(%q: !qco.qubit) -> !qco.qubit {
        %out, %bit = qco.measure %q : !qco.qubit
        return %out : !qco.qubit
      }
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %out = func.call @measure(%q) : (!qco.qubit) -> !qco.qubit
        qco.sink %out : !qco.qubit
        return
      }
    }
  )mlir",
                                          context.get());
  ASSERT_TRUE(reuse);
  ASSERT_TRUE(reset);
  ASSERT_TRUE(call);

  auto dd = std::make_unique<dd::Package>(1);
  EXPECT_TRUE(failed(simulateStatevector(mainFunc(*reuse), *dd)));
  EXPECT_TRUE(failed(simulateStatevector(mainFunc(*reset), *dd)));
  EXPECT_TRUE(failed(simulateStatevector(mainFunc(*call), *dd)));
}

TEST_F(QCODDFunctionalityTest, LifetimeMarkersPreserveEntangledState) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.h(b.staticQubit(0));
    auto q1 = b.staticQubit(1);
    std::tie(q0, q1) = b.cx(q0, q1);
    b.qtensorDealloc(b.qtensorFromElements({q0, q1}));
    b.sink(b.x(b.staticQubit(2)));
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(0);
  const auto state = simulateStatevector(mainFunc(*mod), *dd);
  ASSERT_TRUE(succeeded(state));
  const auto vector = state->getVector();
  ASSERT_EQ(vector.size(), 8U);
  EXPECT_NEAR(std::norm(vector[4]), 0.5, 1e-12);
  EXPECT_NEAR(std::norm(vector[7]), 0.5, 1e-12);
  dd->decRef(*state);

  const auto histogram = sample(mainFunc(*mod), 64, 1);
  ASSERT_TRUE(succeeded(histogram));
  EXPECT_EQ(histogram->at("100") + histogram->at("111"), 64U);
}

TEST_F(QCODDFunctionalityTest, StatevectorSkipsUnreachedAllocations) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %false = arith.constant false
        scf.if %false {
          %q0 = qco.alloc : !qco.qubit
          %q1 = qco.alloc : !qco.qubit
          %q2 = qco.alloc : !qco.qubit
          qco.sink %q0 : !qco.qubit
          qco.sink %q1 : !qco.qubit
          qco.sink %q2 : !qco.qubit
        }
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(0);
  const auto state = simulateStatevector(mainFunc(*mod), *dd);
  ASSERT_TRUE(succeeded(state));
  EXPECT_EQ(dd->qubits(), 0U);
  const auto vector = state->getVector();
  ASSERT_EQ(vector.size(), 1U);
  EXPECT_NEAR(std::norm(vector[0]), 1.0, 1e-12);
  dd->decRef(*state);
}

} // namespace
