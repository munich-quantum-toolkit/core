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
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <gtest/gtest.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
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
    registry
        .insert<QCODialect, qtensor::QTensorDialect, arith::ArithDialect,
                func::FuncDialect, scf::SCFDialect, memref::MemRefDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] static func::FuncOp mainFunc(ModuleOp mod) {
    if (auto main = mod.lookupSymbol<func::FuncOp>("main")) {
      return main;
    }
    auto funcs = mod.getBody()->getOps<func::FuncOp>();
    assert(funcs.begin() != funcs.end() && "module must contain a func.func");
    return *funcs.begin();
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

  void expectMlirFails(size_t numQubits, StringRef mlirCode) const {
    auto mod = parseSourceString<ModuleOp>(mlirCode, context.get());
    ASSERT_TRUE(mod);
    auto dd = std::make_unique<dd::Package>(numQubits);
    EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
  }

  [[nodiscard]] static dd::VectorDD
  basisState(size_t nQubits, llvm::ArrayRef<bool> bits, dd::Package& dd) {
    return dd::makeBasisState(nQubits,
                              std::vector<bool>(bits.begin(), bits.end()), dd);
  }

  [[nodiscard]] static dd::VectorDD oneQubitState(dd::Package& dd) {
    return basisState(1, {true}, dd);
  }

  static void
  expectSimulatesFromZero(func::FuncOp func, size_t nQubits,
                          llvm::ArrayRef<bool> expectedBits,
                          std::optional<std::uint64_t> seed = std::nullopt) {
    auto dd = std::make_unique<dd::Package>(nQubits);
    auto expected = basisState(nQubits, expectedBits, *dd);
    if (seed) {
      std::mt19937_64 rng(*seed);
      const auto out =
          simulate(func, dd::makeZeroState(nQubits, *dd), *dd, rng);
      ASSERT_TRUE(succeeded(out));
      EXPECT_EQ(out->getVector(), expected.getVector());
      dd->decRef(*out);
    } else {
      const auto out = simulate(func, dd::makeZeroState(nQubits, *dd), *dd);
      ASSERT_TRUE(succeeded(out));
      EXPECT_EQ(out->getVector(), expected.getVector());
      dd->decRef(*out);
    }
    dd->decRef(expected);
  }

  enum class SampleApi : std::uint8_t { Sample, SampleWithClassics };

  static void expectSampleHistogram(
      func::FuncOp func, size_t nQubits, std::size_t shots, std::uint64_t seed,
      StringRef expectedShotKey, SampleApi api = SampleApi::Sample,
      std::optional<StringRef> expectedClassicalKey = std::nullopt) {
    auto dd = std::make_unique<dd::Package>(nQubits);
    std::mt19937_64 rng(seed);
    if (api == SampleApi::Sample) {
      const auto hist = sample(func, *dd, shots, rng);
      ASSERT_TRUE(succeeded(hist));
      ASSERT_EQ(hist->size(), 1U);
      EXPECT_EQ(hist->begin()->first, expectedShotKey);
      EXPECT_EQ(hist->begin()->second, shots);
      return;
    }
    const auto hist = sampleWithClassics(func, *dd, shots, rng);
    ASSERT_TRUE(succeeded(hist));
    ASSERT_EQ(hist->shots.size(), 1U);
    EXPECT_EQ(hist->shots.begin()->first, expectedShotKey);
    EXPECT_EQ(hist->shots.begin()->second, shots);
    if (expectedClassicalKey) {
      ASSERT_EQ(hist->classical.size(), 1U);
      EXPECT_EQ(hist->classical.begin()->first, *expectedClassicalKey);
      EXPECT_EQ(hist->classical.begin()->second, shots);
    } else {
      EXPECT_TRUE(hist->classical.empty());
    }
  }

  static void expectBuildAndSimFail(func::FuncOp func, size_t nQubits) {
    auto dd = std::make_unique<dd::Package>(nQubits);
    EXPECT_TRUE(failed(buildFunctionality(func, *dd)));
    EXPECT_TRUE(failed(simulate(func, dd::makeZeroState(nQubits, *dd), *dd)));
  }

  static void expectSimulateFail(func::FuncOp func, size_t nQubits) {
    auto dd = std::make_unique<dd::Package>(nQubits);
    EXPECT_TRUE(failed(simulate(func, dd::makeZeroState(nQubits, *dd), *dd)));
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
    const auto& [q2Out, q0Out, q3Out] = targets;
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
      qs.push_back(b.staticQubit(static_cast<std::int64_t>(i)));
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
  for (std::size_t r = 0; r < 2; ++r) {
    for (std::size_t c = 0; c < 2; ++c) {
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

  constexpr std::uint64_t seed = 42;
  auto dd = std::make_unique<dd::Package>(1);

  std::mt19937_64 refRng(seed);
  auto ref = dd::makeZeroState(1, *dd);
  ref = dd->applyOperation(
      dd->makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::H), 0), ref);
  (void)dd->measureOneCollapsing(ref, 0, refRng);
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

  // Deterministic constant branches do not require an RNG.
  expectSimulatesFromZero(mainFunc(*thenMod), 1, {true});
  expectSimulatesFromZero(mainFunc(*elseMod), 1, {false});
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

  expectSimulatesFromZero(mainFunc(*caseMod), 1, {true});
  expectSimulatesFromZero(mainFunc(*defaultMod), 1, {false});
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

  const auto func = mainFunc(*mod);
  expectSimulatesFromZero(func, 1, {true}, 99);
  expectSampleHistogram(func, 1, 32, 7, "1");
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

  expectSimulatesFromZero(mainFunc(*mod), 1, {false}, 3);
}

TEST_F(QCODDFunctionalityTest, SimulateFuncCallAppliesCallee) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @apply_x(%q: !qco.qubit) -> !qco.qubit {
        %q1 = qco.x %q : !qco.qubit -> !qco.qubit
        return %q1 : !qco.qubit
      }
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %q1 = func.call @apply_x(%q) : (!qco.qubit) -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  auto main = mod->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);

  expectSimulatesFromZero(main, 1, {true});
}

TEST_F(QCODDFunctionalityTest, RejectsRecursiveFuncCall) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
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
  ASSERT_TRUE(mod);
  auto main = mod->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);

  expectSimulateFail(main, 1);
}

TEST_F(QCODDFunctionalityTest, SimulateScfForAppliesBodyTrips) {
  // Three X applications: |0> → |1>.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto results =
        b.scfFor(0, 3, 1, ValueRange{q},
                 [&](Value /*iv*/, ValueRange iterArgs) -> SmallVector<Value> {
                   return {b.x(iterArgs[0])};
                 });
    b.sink(results[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSimulatesFromZero(mainFunc(*mod), 1, {true});
}

TEST_F(QCODDFunctionalityTest, AcceptsScfForAtTripCountLimit) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto results =
        b.scfFor(0, 10000, 1, ValueRange{q},
                 [&](Value /*iv*/, ValueRange iterArgs) -> SmallVector<Value> {
                   return {iterArgs[0]};
                 });
    b.sink(results[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSimulatesFromZero(mainFunc(*mod), 1, {false});
}

TEST_F(QCODDFunctionalityTest, RejectsScfForTripCountLimit) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto results =
        b.scfFor(0, 10001, 1, ValueRange{q},
                 [&](Value /*iv*/, ValueRange iterArgs) -> SmallVector<Value> {
                   return {iterArgs[0]};
                 });
    b.sink(results[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectBuildAndSimFail(mainFunc(*mod), 1);
}

TEST_F(QCODDFunctionalityTest, SimulateScfWhileAppliesBodyTrips) {
  // Three X applications: |0> → |1>. The index is loop-carried alongside the
  // qubit and exercises concrete classical values across both regions.
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %c0 = arith.constant 0 : index
        %result:2 = scf.while (%arg0 = %q, %arg1 = %c0)
            : (!qco.qubit, index) -> (!qco.qubit, index) {
          %c3 = arith.constant 3 : index
          %cond = arith.cmpi slt, %arg1, %c3 : index
          scf.condition(%cond) %arg0, %arg1 : !qco.qubit, index
        } do {
        ^bb0(%arg0: !qco.qubit, %arg1: index):
          %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          %c1 = arith.constant 1 : index
          %next = arith.addi %arg1, %c1 : index
          scf.yield %q1, %next : !qco.qubit, index
        }
        qco.sink %result#0 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  expectSimulatesFromZero(mainFunc(*mod), 1, {true});
}

TEST_F(QCODDFunctionalityTest, SimulateScfWhileZeroTrips) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %result = scf.while (%arg0 = %q)
            : (!qco.qubit) -> !qco.qubit {
          %false = arith.constant false
          scf.condition(%false) %arg0 : !qco.qubit
        } do {
        ^bb0(%arg0: !qco.qubit):
          %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          scf.yield %q1 : !qco.qubit
        }
        qco.sink %result : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  expectSimulatesFromZero(mainFunc(*mod), 1, {false});
}

TEST_F(QCODDFunctionalityTest, SimulateMeasurementControlledScfWhile) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    auto results = b.scfWhile(
        ValueRange{q},
        [&](ValueRange args) {
          auto [measured, bit] = b.measure(args[0]);
          b.scfCondition(bit, measured);
          return SmallVector<Value>{measured};
        },
        [&](ValueRange args) { return SmallVector<Value>{b.x(args[0])}; });
    b.sink(results[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSimulatesFromZero(mainFunc(*mod), 1, {false}, 3);
}

TEST_F(QCODDFunctionalityTest, RejectsScfWhileTripCountLimit) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %result = scf.while (%arg0 = %q)
            : (!qco.qubit) -> !qco.qubit {
          %true = arith.constant true
          scf.condition(%true) %arg0 : !qco.qubit
        } do {
        ^bb0(%arg0: !qco.qubit):
          scf.yield %arg0 : !qco.qubit
        }
        qco.sink %result : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  expectBuildAndSimFail(mainFunc(*mod), 1);
}

TEST_F(QCODDFunctionalityTest, SimulateQTensorAllocationAndElementUpdates) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(2);
    Value q0;
    Value q1;
    std::tie(tensor, q0) = b.qtensorExtract(tensor, 0);
    std::tie(tensor, q1) = b.qtensorExtract(tensor, 1);
    tensor = b.qtensorInsert(b.x(q0), tensor, 0);
    tensor = b.qtensorInsert(b.x(q1), tensor, 1);
    b.qtensorDealloc(tensor);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 2, 8, 3, "11");
}

TEST_F(QCODDFunctionalityTest, QTensorAllocationExtendsInputWithZeroWires) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %c1 = arith.constant 1 : index
        %tensor = qtensor.alloc(%c1) : tensor<1x!qco.qubit>
        qco.sink %q : !qco.qubit
        qtensor.dealloc %tensor : tensor<1x!qco.qubit>
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(2);
  auto expected = basisState(2, {true, false}, *dd);
  const auto result = simulate(mainFunc(*mod), oneQubitState(*dd), *dd);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->getVector(), expected.getVector());
  dd->decRef(*result);
  dd->decRef(expected);

  EXPECT_TRUE(failed(buildFunctionality(mainFunc(*mod), *dd)));
}

TEST_F(QCODDFunctionalityTest, QTensorFromElementsSupportsMatrixAndSimulation) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.staticQubit(0);
    auto q1 = b.staticQubit(1);
    auto tensor = b.qtensorFromElements({q0, q1});
    std::tie(tensor, q1) = b.qtensorExtract(tensor, 1);
    tensor = b.qtensorInsert(b.x(q1), tensor, 1);
    b.qtensorDealloc(tensor);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  qc::QuantumComputation qc(2);
  qc.x(1);
  expectEqualToQc(mainFunc(*mod), qc);
}

TEST_F(QCODDFunctionalityTest, SimulateQTensorFromDynamicallyAllocatedQubits) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto q1 = b.allocQubit();
    auto tensor = b.qtensorFromElements({b.x(q0), q1});
    b.qtensorDealloc(tensor);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 2, 8, 3, "01");
}

TEST_F(QCODDFunctionalityTest, SimulateQTensorWithConcreteDynamicSize) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto size = arith::AddIOp::create(b, one, one).getResult();
    b.qtensorDealloc(b.qtensorAlloc(size));
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 2, 8, 3, "00");
}

TEST_F(QCODDFunctionalityTest, SimulateQTensorThroughScfFor) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(2);
    tensor = b.scfFor(0, 2, 1, tensor,
                      [&](Value index, ValueRange args) -> SmallVector<Value> {
                        auto [remaining, qubit] =
                            b.qtensorExtract(args[0], index);
                        return {b.qtensorInsert(b.x(qubit), remaining, index)};
                      })[0];
    b.qtensorDealloc(tensor);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 2, 8, 3, "11");
}

TEST_F(QCODDFunctionalityTest, SimulateQTensorThroughIf) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(1);
    tensor = b.qcoIf(b.boolConstant(true), tensor, [&](Value arg) {
      auto [remaining, qubit] = b.qtensorExtract(arg, 0);
      return b.qtensorInsert(b.x(qubit), remaining, 0);
    });
    b.qtensorDealloc(tensor);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 1, 8, 3, "1");
}

TEST_F(QCODDFunctionalityTest, SimulateQTensorThroughScfWhile) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(1);
    auto [remaining, qubit] = b.qtensorExtract(tensor, 0);
    tensor = b.qtensorInsert(b.x(qubit), remaining, 0);
    tensor = b.scfWhile(
        tensor,
        [&](ValueRange args) {
          auto [rest, q] = b.qtensorExtract(args[0], 0);
          auto [measured, bit] = b.measure(q);
          auto result = b.qtensorInsert(measured, rest, 0);
          b.scfCondition(bit, result);
          return SmallVector<Value>{result};
        },
        [&](ValueRange args) {
          auto [rest, q] = b.qtensorExtract(args[0], 0);
          return SmallVector<Value>{b.qtensorInsert(b.x(q), rest, 0)};
        })[0];
    b.qtensorDealloc(tensor);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 1, 8, 3, "0");
}

TEST_F(QCODDFunctionalityTest, SimulateQTensorThroughFuncCall) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @flip(%arg0: tensor<1x!qco.qubit>)
          -> tensor<1x!qco.qubit> {
        %c0 = arith.constant 0 : index
        %remaining, %q = qtensor.extract %arg0[%c0]
            : tensor<1x!qco.qubit>
        %q1 = qco.x %q : !qco.qubit -> !qco.qubit
        %result = qtensor.insert %q1 into %remaining[%c0]
            : tensor<1x!qco.qubit>
        return %result : tensor<1x!qco.qubit>
      }
      func.func @main() {
        %c1 = arith.constant 1 : index
        %tensor = qtensor.alloc(%c1) : tensor<1x!qco.qubit>
        %result = func.call @flip(%tensor)
            : (tensor<1x!qco.qubit>) -> tensor<1x!qco.qubit>
        qtensor.dealloc %result : tensor<1x!qco.qubit>
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 1, 8, 3, "1");
}

TEST_F(QCODDFunctionalityTest, QTensorFunctionArgumentMapsInputWires) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%arg0: tensor<2x!qco.qubit>)
          -> tensor<2x!qco.qubit> {
        %c0 = arith.constant 0 : index
        %remaining, %q = qtensor.extract %arg0[%c0]
            : tensor<2x!qco.qubit>
        %q1 = qco.x %q : !qco.qubit -> !qco.qubit
        %result = qtensor.insert %q1 into %remaining[%c0]
            : tensor<2x!qco.qubit>
        return %result : tensor<2x!qco.qubit>
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  qc::QuantumComputation qc(2);
  qc.x(0);
  expectEqualToQc(mainFunc(*mod), qc);
}

TEST_F(QCODDFunctionalityTest, RejectsQTensorAllocationBeyondPackage) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    b.qtensorDealloc(b.qtensorAlloc(2));
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(1);
  std::mt19937_64 rng(3);
  EXPECT_TRUE(failed(sample(mainFunc(*mod), *dd, 1, rng)));
}

TEST_F(QCODDFunctionalityTest, RejectsInvalidQTensorRuntimeIndex) {
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %c2 = arith.constant 2 : index
        %tensor = qtensor.alloc(%c2) : tensor<?x!qco.qubit>
        %remaining, %q = qtensor.extract %tensor[%c2]
            : tensor<?x!qco.qubit>
        qco.sink %q : !qco.qubit
        qtensor.dealloc %remaining : tensor<?x!qco.qubit>
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);

  auto dd = std::make_unique<dd::Package>(2);
  std::mt19937_64 rng(3);
  EXPECT_TRUE(failed(sample(mainFunc(*mod), *dd, 1, rng)));
}

TEST_F(QCODDFunctionalityTest, SimulateRicherClassicalArithmetic) {
  // idx = (1+2)*3 >> 1 = 4; select(true, idx, 0)=4; cmpi eq 4 → if applies X.
  // Also round-trip i1 via extui/trunci.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto two = arith::ConstantIndexOp::create(b, 2).getResult();
    auto three = arith::ConstantIndexOp::create(b, 3).getResult();
    auto four = arith::ConstantIndexOp::create(b, 4).getResult();
    auto zero = arith::ConstantIndexOp::create(b, 0).getResult();
    auto sum = arith::AddIOp::create(b, one, two).getResult();
    auto prod = arith::MulIOp::create(b, sum, three).getResult();
    auto shifted = arith::ShRUIOp::create(b, prod, one).getResult();
    auto t = b.boolConstant(true);
    auto selected = arith::SelectOp::create(b, t, shifted, zero).getResult();
    auto eq = arith::CmpIOp::create(b, arith::CmpIPredicate::eq, selected, four)
                  .getResult();
    auto asIndex = arith::ExtUIOp::create(b, b.getIndexType(), eq).getResult();
    auto asBool =
        arith::TruncIOp::create(b, b.getI1Type(), asIndex).getResult();
    q = b.qcoIf(
        asBool, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    // Exercise subi: 4-4=0 unused for branching but must succeed.
    (void)arith::SubIOp::create(b, four, four);
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSimulatesFromZero(mainFunc(*mod), 1, {true});
}

TEST_F(QCODDFunctionalityTest, SimulateAndiOriXoriShliClassical) {
  // Pack two measure bits (from |1>,|0>) as index = bit0 | (bit1 << 1) = 1,
  // then switch case 1 applies X on an idle |0> target → |1>.
  // Also exercise andi / xori on the measured bits.
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
    auto t = b.boolConstant(true);
    auto anded = arith::AndIOp::create(b, bit0, t).getResult();
    // bit0 ^ true flips the measured-1 bit to false; keep the value live.
    auto xored = arith::XOrIOp::create(b, anded, t).getResult();
    (void)xored;
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

  expectSimulatesFromZero(mainFunc(*mod), 3, {true, false, true}, 11);
}

TEST_F(QCODDFunctionalityTest, SampleUnitaryXIsDeterministic) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  const auto func = mainFunc(*mod);
  expectSampleHistogram(func, 1, 64, 1, "1");
  expectSampleHistogram(func, 1, 16, 1, "1", SampleApi::SampleWithClassics);
}

TEST_F(QCODDFunctionalityTest, SampleWithClassicsRecordsMeasureBits) {
  // |1> → measure (bit 1) → if then X → |0>. Classical key "1" every shot.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q);
    q = b.qcoIf(
        bit, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 1, 32, 9, "0",
                        SampleApi::SampleWithClassics, "1");
}

TEST_F(QCODDFunctionalityTest, SimulateClassicalMemRefRegister) {
  // measure into memref c[0], then qcoIf loads c[0] and applies X → |0>.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto c = b.allocClassicalBitRegister(1);
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q, c, 0);
    auto results = b.qcoIf(
        c, 0, ValueRange{q},
        [&](ValueRange args) { return SmallVector<Value>{b.x(args[0])}; },
        [&](ValueRange args) { return SmallVector<Value>{args[0]}; });
    b.sink(results[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 1, 16, 11, "0",
                        SampleApi::SampleWithClassics, "1");
}

TEST_F(QCODDFunctionalityTest, SampleCombinedForMeasureIfIndexSwitch) {
  // Drivers-style CF stack on static wires: three X in `scf.for` → |1>,
  // measure, keep via `qco.if`, then identity `index_switch`.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto forResults =
        b.scfFor(0, 3, 1, ValueRange{q},
                 [&](Value /*iv*/, ValueRange iterArgs) -> SmallVector<Value> {
                   return {b.x(iterArgs[0])};
                 });
    q = forResults[0];
    Value bit;
    std::tie(q, bit) = b.measure(q);
    q = b.qcoIf(
        bit, q, [&](Value arg) { return arg; },
        [&](Value arg) { return b.x(arg); });
    const auto identity = [](Value arg) { return arg; };
    q = b.qcoIndexSwitch(0, q, ArrayRef<int64_t>{0},
                         SmallVector<function_ref<Value(Value)>>{identity},
                         identity);
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 1, 24, 13, "1",
                        SampleApi::SampleWithClassics, "1");
}

TEST_F(QCODDFunctionalityTest, SampleFromInputStateConsumesReference) {
  auto unitary = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    b.sink(q);
    return b.intConstant(0);
  });
  auto withReset = buildModule([](QCOProgramBuilder& b) {
    auto q = b.reset(b.staticQubit(0));
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(unitary);
  ASSERT_TRUE(withReset);

  auto dd = std::make_unique<dd::Package>(1);
  auto& roots = dd->getRootSet<dd::vNode>();
  std::mt19937_64 rng(5);

  // Static path: input |1> sampled without mid-circuit collapse.
  for (size_t i = 0; i < 3; ++i) {
    auto in = oneQubitState(*dd);
    const auto hist = sample(mainFunc(*unitary), in, *dd, /*shots=*/8, rng);
    ASSERT_TRUE(succeeded(hist));
    ASSERT_EQ(hist->size(), 1U);
    EXPECT_EQ(hist->begin()->first, "1");
    EXPECT_EQ(hist->begin()->second, 8U);
    EXPECT_TRUE(roots.empty());
  }

  // Dynamic path: reset forces per-shot re-simulation from input |1|.
  for (size_t i = 0; i < 3; ++i) {
    auto in = oneQubitState(*dd);
    const auto hist = sample(mainFunc(*withReset), in, *dd, /*shots=*/4, rng);
    ASSERT_TRUE(succeeded(hist));
    ASSERT_EQ(hist->size(), 1U);
    EXPECT_EQ(hist->begin()->first, "0");
    EXPECT_EQ(hist->begin()->second, 4U);
    EXPECT_TRUE(roots.empty());
  }

  // shots == 0 still consumes the input reference.
  {
    auto in = dd::makeZeroState(1, *dd);
    const auto hist = sample(mainFunc(*unitary), in, *dd, /*shots=*/0, rng);
    ASSERT_TRUE(succeeded(hist));
    EXPECT_TRUE(hist->empty());
    EXPECT_TRUE(roots.empty());
  }
}

TEST_F(QCODDFunctionalityTest, SampleConstantIfUsesStaticPath) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    q = b.qcoIf(
        true, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectSampleHistogram(mainFunc(*mod), 1, 16, 2, "1");
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
  constexpr std::size_t shots = 2000;
  const auto hist = sample(mainFunc(*mod), *dd, shots, rng);
  ASSERT_TRUE(succeeded(hist));
  ASSERT_EQ(hist->size(), 2U);
  EXPECT_EQ(hist->at("0") + hist->at("1"), shots);
  EXPECT_NEAR(static_cast<double>(hist->at("0")), shots / 2.0, 150.0);
}

TEST_F(QCODDFunctionalityTest, RejectsOutOfRangeShift) {
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto bad = arith::ConstantIndexOp::create(b, 64).getResult();
    auto shifted = arith::ShLIOp::create(b, one, bad).getResult();
    q = b.qcoIndexSwitch(
        shifted, q, ArrayRef<int64_t>{0},
        SmallVector<function_ref<Value(Value)>>{[&](Value arg) { return arg; }},
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);

  expectBuildAndSimFail(mainFunc(*mod), 1);
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
    expectBuildAndSimFail(mainFunc(*mod), 1);
  }

  {
    auto mod = buildModule([](QCOProgramBuilder& b) {
      auto q0 = b.reset(b.staticQubit(0));
      b.sink(q0);
      return b.intConstant(0);
    });
    ASSERT_TRUE(mod);
    expectBuildAndSimFail(mainFunc(*mod), 1);
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
    expectBuildAndSimFail(mainFunc(*mod), 1);
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

TEST_F(QCODDFunctionalityTest, ClassicalCmpSelectAndIndexBitwise) {
  // Hit cmpi predicates (incl. i1), i1 select, and index andi/ori/xori.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto two = arith::ConstantIndexOp::create(b, 2).getResult();
    auto three = arith::ConstantIndexOp::create(b, 3).getResult();
    auto zero = arith::ConstantIndexOp::create(b, 0).getResult();
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();

    auto ne = arith::CmpIOp::create(b, arith::CmpIPredicate::ne, two, three)
                  .getResult();
    auto slt = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, two, three)
                   .getResult();
    auto sle = arith::CmpIOp::create(b, arith::CmpIPredicate::sle, two, three)
                   .getResult();
    auto sgt = arith::CmpIOp::create(b, arith::CmpIPredicate::sgt, three, two)
                   .getResult();
    auto sge = arith::CmpIOp::create(b, arith::CmpIPredicate::sge, three, two)
                   .getResult();
    auto ult = arith::CmpIOp::create(b, arith::CmpIPredicate::ult, two, three)
                   .getResult();
    auto ule = arith::CmpIOp::create(b, arith::CmpIPredicate::ule, two, three)
                   .getResult();
    auto ugt = arith::CmpIOp::create(b, arith::CmpIPredicate::ugt, three, two)
                   .getResult();
    auto uge = arith::CmpIOp::create(b, arith::CmpIPredicate::uge, three, two)
                   .getResult();
    auto all = arith::AndIOp::create(
                   b, ne,
                   arith::AndIOp::create(
                       b, slt,
                       arith::AndIOp::create(
                           b, sle,
                           arith::AndIOp::create(
                               b, sgt,
                               arith::AndIOp::create(
                                   b, sge,
                                   arith::AndIOp::create(
                                       b, ult,
                                       arith::AndIOp::create(
                                           b, ule,
                                           arith::AndIOp::create(b, ugt, uge)
                                               .getResult())
                                           .getResult())
                                       .getResult())
                                   .getResult())
                               .getResult())
                           .getResult())
                       .getResult())
                   .getResult();
    auto t = b.boolConstant(true);
    auto f = b.boolConstant(false);
    auto selected = arith::SelectOp::create(b, all, t, f).getResult();
    auto i1Ne = arith::CmpIOp::create(b, arith::CmpIPredicate::ne, selected, f)
                    .getResult();
    // i1 signed preds use sign-extension (true≡-1); unsigned use 0/1.
    auto i1Eq =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, t, t).getResult();
    auto i1Slt =
        arith::CmpIOp::create(b, arith::CmpIPredicate::slt, t, f).getResult();
    auto i1Sle =
        arith::CmpIOp::create(b, arith::CmpIPredicate::sle, t, t).getResult();
    auto i1Sgt =
        arith::CmpIOp::create(b, arith::CmpIPredicate::sgt, f, t).getResult();
    auto i1Sge =
        arith::CmpIOp::create(b, arith::CmpIPredicate::sge, f, t).getResult();
    auto i1Ult =
        arith::CmpIOp::create(b, arith::CmpIPredicate::ult, f, t).getResult();
    auto i1Ule =
        arith::CmpIOp::create(b, arith::CmpIPredicate::ule, t, t).getResult();
    auto i1Ugt =
        arith::CmpIOp::create(b, arith::CmpIPredicate::ugt, t, f).getResult();
    auto i1Uge =
        arith::CmpIOp::create(b, arith::CmpIPredicate::uge, t, f).getResult();
    auto i1All =
        arith::AndIOp::create(
            b, i1Ne,
            arith::AndIOp::create(
                b, i1Eq,
                arith::AndIOp::create(
                    b, i1Slt,
                    arith::AndIOp::create(
                        b, i1Sle,
                        arith::AndIOp::create(
                            b, i1Sgt,
                            arith::AndIOp::create(
                                b, i1Sge,
                                arith::AndIOp::create(
                                    b, i1Ult,
                                    arith::AndIOp::create(
                                        b, i1Ule,
                                        arith::AndIOp::create(b, i1Ugt, i1Uge)
                                            .getResult())
                                        .getResult())
                                    .getResult())
                                .getResult())
                            .getResult())
                        .getResult())
                    .getResult())
                .getResult())
            .getResult();
    auto orI1 = arith::OrIOp::create(b, i1All, f).getResult();

    auto masked = arith::AndIOp::create(b, three, one).getResult(); // 1
    auto ored = arith::OrIOp::create(b, masked, zero).getResult();  // 1
    auto xored = arith::XOrIOp::create(b, ored, zero).getResult();  // 1
    auto eqOne = arith::CmpIOp::create(b, arith::CmpIPredicate::eq, xored, one)
                     .getResult();
    auto cond = arith::AndIOp::create(b, orI1, eqOne).getResult();

    q = b.qcoIf(
        cond, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);
  expectSimulatesFromZero(mainFunc(*mod), 1, {true});
}

TEST_F(QCODDFunctionalityTest, ClassicalBindThroughScfForAndCall) {
  // Exercise ClassicalEnv::bindFrom via scf.for iter_args and func.call.
  auto mod = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @flip_if(%q: !qco.qubit, %bit: i1) -> (!qco.qubit, i1) {
        %q1 = qco.if %bit args(%qin = %q) -> (!qco.qubit) {
          %qx = qco.x %qin : !qco.qubit -> !qco.qubit
          qco.yield %qx : !qco.qubit
        } else args(%qin = %q) {
          qco.yield %qin : !qco.qubit
        }
        return %q1, %bit : !qco.qubit, i1
      }
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %true = arith.constant true
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %q1, %bit1 = scf.for %iv = %c0 to %c2 step %c1
            iter_args(%qarg = %q, %barg = %true) -> (!qco.qubit, i1) {
          %q2, %bout = func.call @flip_if(%qarg, %barg)
              : (!qco.qubit, i1) -> (!qco.qubit, i1)
          scf.yield %q2, %bout : !qco.qubit, i1
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(mod);
  // two flips with always-true bit: |0> -> |1> -> |0>
  expectSimulatesFromZero(mainFunc(*mod), 1, {false});
}

TEST_F(QCODDFunctionalityTest, ScfForZeroTripsAndRejectsBadStep) {
  auto zeroTrips = buildModule([](QCOProgramBuilder& b) {
    auto q = b.x(b.staticQubit(0));
    auto results =
        b.scfFor(3, 3, 1, ValueRange{q},
                 [&](Value /*iv*/, ValueRange iterArgs) -> SmallVector<Value> {
                   return {b.h(iterArgs[0])};
                 });
    b.sink(results[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(zeroTrips);
  expectSimulatesFromZero(mainFunc(*zeroTrips), 1, {true});

  auto badStep = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto results =
        b.scfFor(0, 3, 0, ValueRange{q},
                 [&](Value /*iv*/, ValueRange iterArgs) -> SmallVector<Value> {
                   return {iterArgs[0]};
                 });
    b.sink(results[0]);
    return b.intConstant(0);
  });
  ASSERT_TRUE(badStep);
  expectBuildAndSimFail(mainFunc(*badStep), 1);
}

TEST_F(QCODDFunctionalityTest, ClassicalMemRefErrorsAndDealloc) {
  auto ok = buildModule([](QCOProgramBuilder& b) {
    auto c = b.allocClassicalBitRegister(2);
    auto q = b.x(b.staticQubit(0));
    Value bit;
    std::tie(q, bit) = b.measure(q, c, 1);
    auto loaded = memref::LoadOp::create(
                      b, c, ValueRange{arith::ConstantIndexOp::create(b, 1)})
                      .getResult();
    q = b.qcoIf(
        loaded, q, [&](Value arg) { return b.x(arg); },
        [&](Value arg) { return arg; });
    memref::DeallocOp::create(b, c);
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(ok);
  expectSimulatesFromZero(mainFunc(*ok), 1, {false}, /*seed=*/5);

  // Wrong element type / rank rejected.
  expectMlirFails(0, R"mlir(
    module {
      func.func @main() {
        %c = memref.alloc() : memref<2xi32>
        memref.dealloc %c : memref<2xi32>
        return
      }
    }
  )mlir");

  auto oob = buildModule([](QCOProgramBuilder& b) {
    auto c = b.allocClassicalBitRegister(1);
    auto q = b.staticQubit(0);
    auto bit = b.boolConstant(true);
    memref::StoreOp::create(b, bit, c,
                            ValueRange{arith::ConstantIndexOp::create(b, 3)});
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(oob);
  expectSimulateFail(mainFunc(*oob), 1);

  auto badStoreRank = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %c = memref.alloc() : memref<2x2xi1>
        %t = arith.constant true
        %i0 = arith.constant 0 : index
        memref.store %t, %c[%i0, %i0] : memref<2x2xi1>
        memref.dealloc %c : memref<2x2xi1>
        return
      }
    }
  )mlir",
                                                  context.get());
  ASSERT_TRUE(badStoreRank);
  expectSimulateFail(mainFunc(*badStoreRank), 0);
}

TEST_F(QCODDFunctionalityTest, RejectsUnmappedClassicalAndBadControlFlow) {
  // if condition not concrete
  auto badIf = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%cond: i1) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.if %cond args(%qin = %q) -> (!qco.qubit) {
          qco.yield %qin : !qco.qubit
        } else args(%qin = %q) {
          qco.yield %qin : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                           context.get());
  ASSERT_TRUE(badIf);
  expectSimulateFail(mainFunc(*badIf), 1);

  auto badSwitch = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%idx: index) {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.index_switch %idx -> !qco.qubit
        default args(%arg0 = %q) {
          qco.yield %arg0 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                               context.get());
  ASSERT_TRUE(badSwitch);
  expectSimulateFail(mainFunc(*badSwitch), 1);

  // Declaration without a body (covers callee lookup / single-block checks).
  auto missingBody = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func private @missing(%q: !qco.qubit) -> !qco.qubit
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %q1 = func.call @missing(%q) : (!qco.qubit) -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                                 context.get());
  ASSERT_TRUE(missingBody);
  expectSimulateFail(mainFunc(*missingBody), 1);

  // Unsupported classical op
  auto div = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %d = arith.divui %c2, %c1 : index
        %q1 = qco.index_switch %d -> !qco.qubit
        default args(%arg0 = %q) {
          qco.yield %arg0 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                         context.get());
  ASSERT_TRUE(div);
  expectSimulateFail(mainFunc(*div), 1);

  auto shruiBad = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto bad = arith::ConstantIndexOp::create(b, 64).getResult();
    auto shifted = arith::ShRUIOp::create(b, one, bad).getResult();
    q = b.qcoIndexSwitch(
        shifted, q, ArrayRef<int64_t>{0},
        SmallVector<function_ref<Value(Value)>>{[&](Value arg) { return arg; }},
        [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(shruiBad);
  expectBuildAndSimFail(mainFunc(*shruiBad), 1);
}

TEST_F(QCODDFunctionalityTest, RejectsDenseFallbackAboveTwelveQubits) {
  // 4-qubit inv on a 13-qubit register forces the dense embed path and fails.
  auto mod = buildModule([](QCOProgramBuilder& b) {
    SmallVector<Value> qs;
    qs.reserve(13);
    for (unsigned i = 0; i < 13; ++i) {
      qs.push_back(b.staticQubit(i));
    }
    auto outs = b.inv({qs[0], qs[1], qs[2], qs[3]},
                      [&](ValueRange t) -> SmallVector<Value> {
                        return {b.x(t[0]), t[1], t[2], t[3]};
                      });
    for (unsigned i = 0; i < 4; ++i) {
      b.sink(outs[i]);
    }
    for (unsigned i = 4; i < 13; ++i) {
      b.sink(qs[i]);
    }
    return b.intConstant(0);
  });
  ASSERT_TRUE(mod);
  expectBuildAndSimFail(mainFunc(*mod), 13);
}

TEST_F(QCODDFunctionalityTest, ClassicalErrorPathsAndCalleeMeasureSample) {
  // Unmapped classical args fail in bindFrom through func.call.
  auto unmapped = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @use(%q: !qco.qubit, %b: i1, %i: index) -> !qco.qubit {
        return %q : !qco.qubit
      }
      func.func @main(%b: i1, %i: index) {
        %q = qco.static 0 : !qco.qubit
        %q1 = func.call @use(%q, %b, %i) : (!qco.qubit, i1, index) -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                              context.get());
  ASSERT_TRUE(unmapped);
  expectSimulateFail(mainFunc(*unmapped), 1);

  // Unsupported classical type for bindFrom (i64).
  auto badType = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @use(%q: !qco.qubit, %x: i64) -> !qco.qubit {
        return %q : !qco.qubit
      }
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %x = arith.constant 1 : i64
        %q1 = func.call @use(%q, %x) : (!qco.qubit, i64) -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                                             context.get());
  ASSERT_TRUE(badType);
  expectSimulateFail(mainFunc(*badType), 1);

  // Non-index shifts / bad select / bad trunci / bad cmpi result type.
  expectMlirFails(1, R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %t = arith.constant true
        %f = arith.constant false
        %s = arith.shli %t, %f : i1
        qco.sink %q : !qco.qubit
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %t = arith.constant true
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        %s = arith.select %t, %c0, %c1 : i64
        qco.sink %q : !qco.qubit
        return
      }
    }
  )mlir");
  expectMlirFails(1, R"mlir(
    module {
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %c = arith.constant 1 : i64
        %w = arith.trunci %c : i64 to i1
        qco.sink %q : !qco.qubit
        return
      }
    }
  )mlir");

  // Unmapped memref / dynamic alloc.
  auto unmappedMem = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%c: memref<1xi1>) {
        %i0 = arith.constant 0 : index
        %v = memref.load %c[%i0] : memref<1xi1>
        return
      }
    }
  )mlir",
                                                 context.get());
  ASSERT_TRUE(unmappedMem);
  expectSimulateFail(mainFunc(*unmappedMem), 0);

  auto dynAlloc = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%n: index) {
        %c = memref.alloc(%n) : memref<?xi1>
        memref.dealloc %c : memref<?xi1>
        return
      }
    }
  )mlir",
                                              context.get());
  ASSERT_TRUE(dynAlloc);
  expectSimulateFail(mainFunc(*dynAlloc), 0);

  // IntegerAttr i1 (non-BoolAttr) constant recording + index select.
  auto intAttrI1 = buildModule([](QCOProgramBuilder& b) {
    auto q = b.staticQubit(0);
    auto bit = arith::ConstantOp::create(b, IntegerAttr::get(b.getI1Type(), 1))
                   .getResult();
    auto zero = arith::ConstantIndexOp::create(b, 0).getResult();
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    auto idx = arith::SelectOp::create(b, bit, one, zero).getResult();
    q = b.qcoIndexSwitch(idx, q, ArrayRef<int64_t>{1},
                         SmallVector<function_ref<Value(Value)>>{
                             [&](Value arg) { return b.x(arg); }},
                         [&](Value arg) { return arg; });
    b.sink(q);
    return b.intConstant(0);
  });
  ASSERT_TRUE(intAttrI1);
  expectSimulatesFromZero(mainFunc(*intAttrI1), 1, {true});

  // Measure inside callee forces dynamic per-shot sampling.
  auto calleeMeasure = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @meas(%q: !qco.qubit) -> !qco.qubit {
        %q1, %b = qco.measure %q : !qco.qubit
        return %q1 : !qco.qubit
      }
      func.func @main() {
        %q = qco.static 0 : !qco.qubit
        %q1 = qco.x %q : !qco.qubit -> !qco.qubit
        %q2 = func.call @meas(%q1) : (!qco.qubit) -> !qco.qubit
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir",
                                                   context.get());
  ASSERT_TRUE(calleeMeasure);
  expectSampleHistogram(mainFunc(*calleeMeasure), 1, 8, /*seed=*/9, "1",
                        SampleApi::SampleWithClassics,
                        /*expectedClassicalKey=*/StringRef("1"));
}

} // namespace
