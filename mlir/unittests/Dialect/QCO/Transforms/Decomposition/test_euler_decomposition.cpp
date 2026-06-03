/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/IR/QCOUnitaryMatrixInterfaces.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <Eigen/QR>
#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::decomposition;

namespace {

struct SynthesisFixture {
  std::unique_ptr<MLIRContext> context;

  void setUp() {
    DialectRegistry registry;
    registry.insert<qco::QCODialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

struct SynthesizedCircuit {
  OwningOpRef<ModuleOp> module;
  func::FuncOp func;
};

} // namespace

template <typename MatrixType>
[[nodiscard]] static MatrixType randomUnitaryMatrix(std::mt19937& rng) {
  static_assert(MatrixType::RowsAtCompileTime != Eigen::Dynamic &&
                    MatrixType::ColsAtCompileTime != Eigen::Dynamic,
                "randomUnitaryMatrix requires fixed-size matrices");
  static_assert(MatrixType::RowsAtCompileTime == MatrixType::ColsAtCompileTime,
                "randomUnitaryMatrix requires square matrices");
  std::normal_distribution<double> normalDist(0.0, 1.0);
  MatrixType randomMatrix;
  for (auto& x : randomMatrix.reshaped()) {
    x = std::complex<double>(normalDist(rng), normalDist(rng));
  }
  Eigen::HouseholderQR<MatrixType> qr{};
  qr.compute(randomMatrix);
  const MatrixType qMatrix = qr.householderQ();
  const MatrixType rMatrix =
      qr.matrixQR().template triangularView<Eigen::Upper>();
  MatrixType dMatrix = MatrixType::Identity();
  constexpr Eigen::Index dim = MatrixType::RowsAtCompileTime;
  for (Eigen::Index i = 0; i < dim; ++i) {
    const auto rii = rMatrix(i, i);
    const auto absRii = std::abs(rii);
    dMatrix(i, i) =
        absRii > 0.0 ? (rii / absRii) : std::complex<double>{1.0, 0.0};
  }
  const MatrixType unitaryMatrix = qMatrix * dMatrix;
  assert(helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}

template <typename Fn> static void forEachBasis(Fn fn) {
  const std::array<const char*, 7> bases = {"zyz", "zxz", "xzx", "xyx",
                                            "u",   "zsx", "zsxx"};
  for (const char* basis : bases) {
    fn(StringRef{basis});
  }
}

static bool isAllowedBasisGate(Operation& op, StringRef basis) {
  // Always allow global phase as correction term.
  if (isa<GPhaseOp>(op)) {
    return true;
  }

  const auto b = basis.lower();
  if (b == "zyz") {
    return isa<RZOp, RYOp>(op);
  }
  if (b == "zxz") {
    return isa<RZOp, RXOp>(op);
  }
  if (b == "xzx") {
    return isa<RXOp, RZOp>(op);
  }
  if (b == "xyx") {
    return isa<RXOp, RYOp>(op);
  }
  if (b == "u") {
    return isa<UOp>(op);
  }
  if (b == "zsx") {
    return isa<RZOp, SXOp>(op);
  }
  if (b == "zsxx") {
    return isa<RZOp, SXOp, XOp>(op);
  }
  return false;
}

[[nodiscard]] static bool isTwoQubitGate(Operation& op) {
  if (auto u = dyn_cast<UnitaryOpInterface>(op)) {
    return u.isTwoQubit();
  }
  return false;
}

static void expectBasisGatesOnly(func::FuncOp funcOp, StringRef basis) {
  auto& block = funcOp.getBody().front();
  for (Operation& op : block.without_terminator()) {
    if (isa<arith::ConstantOp>(op)) {
      continue;
    }

    if (isTwoQubitGate(op)) {
      continue;
    }

    // Only check ops that claim to carry a unitary matrix (i.e., actual gates).
    if (isa<UnitaryMatrixOpInterface>(op)) {
      EXPECT_TRUE(isAllowedBasisGate(op, basis))
          << "basis=" << basis.str()
          << " unexpected gate: " << op.getName().getStringRef().str();
    }
  }
}

static Eigen::Matrix2cd compute1QMatrixFromFunction(func::FuncOp funcOp) {
  Eigen::Matrix2cd acc = Eigen::Matrix2cd::Identity();
  std::complex<double> global{1.0, 0.0};

  auto& block = funcOp.getBody().front();
  for (Operation& op : block.without_terminator()) {
    if (isa<arith::ConstantOp>(op)) {
      continue;
    }

    if (isTwoQubitGate(op)) {
      continue;
    }

    if (auto gphase = dyn_cast<GPhaseOp>(op)) {
      if (auto m = gphase.getUnitaryMatrix()) {
        global *= (*m)(0, 0);
      }
      continue;
    }

    if (auto iface = dyn_cast<UnitaryMatrixOpInterface>(op)) {
      // All ops in this test should be 1q ops after synthesis.
      const auto maybeM = iface.getUnitaryMatrix<Eigen::Matrix2cd>();
      if (!maybeM) {
        ADD_FAILURE() << "Expected constant unitary matrix for op: "
                      << op.getName().getStringRef().str();
        return Eigen::Matrix2cd::Zero();
      }
      acc = (*maybeM) * acc;
      continue;
    }
  }

  return global * acc;
}

static LogicalResult runFuse(ModuleOp module, StringRef basis) {
  PassManager pm(module.getContext());
  qco::FuseSingleQubitUnitaryRunsOptions opts;
  opts.basis = basis.str();
  pm.addPass(qco::createFuseSingleQubitUnitaryRuns(opts));
  return pm.run(module);
}

static void singleQubitRunWithSingleQubitGate(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  q[0] = b.rz(0.123, q[0]);
  // Keep `inv` inside the single-qubit run so it gets fused/resynthesized too.
  q[0] = b.inv({q[0]}, [&](ValueRange targets) -> SmallVector<Value> {
    return {b.sx(targets[0])};
  })[0];
  q[0] = b.ry(-0.456, q[0]);
}

static void singleQubitRunsSplitByTwoQGate(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  q[0] = b.rz(0.321, q[0]);
  q[0] = b.sx(q[0]);
}

static void singleQubitRunsSplitByBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  q[0] = b.barrier({q[0]})[0];
  q[0] = b.rz(0.321, q[0]);
  q[0] = b.sx(q[0]);
}

static OwningOpRef<ModuleOp> buildProgram(MLIRContext* ctx,
                                          void (*fn)(QCOProgramBuilder&)) {
  QCOProgramBuilder builder(ctx);
  builder.initialize();
  fn(builder);
  return builder.finalize();
}

static func::FuncOp lookupMain(ModuleOp module) {
  auto func = module.lookupSymbol<func::FuncOp>("main");
  EXPECT_TRUE(func) << "Expected a 'main' function";
  return func;
}

template <typename ChecksT>
static void runFuseOnProgramForAllBases(MLIRContext* ctx,
                                        void (*program)(QCOProgramBuilder&),
                                        ChecksT checksAfter) {
  forEachBasis([&](StringRef basis) {
    auto owned = buildProgram(ctx, program);
    if (!static_cast<bool>(owned)) {
      ADD_FAILURE() << "Failed to build program for basis=" << basis.str();
      return;
    }
    ModuleOp module = *owned;
    if (failed(verify(module))) {
      ADD_FAILURE() << "Verifier failed for basis=" << basis.str();
      return;
    }

    auto funcOp = lookupMain(module);
    if (!funcOp) {
      ADD_FAILURE() << "Missing 'main' for basis=" << basis.str();
      return;
    }

    const Eigen::Matrix2cd original = compute1QMatrixFromFunction(funcOp);

    if (failed(runFuse(module, basis))) {
      ADD_FAILURE() << "Fuse pass failed for basis=" << basis.str();
      return;
    }
    if (failed(verify(module))) {
      ADD_FAILURE() << "Verifier failed after fuse for basis=" << basis.str();
      return;
    }

    funcOp = lookupMain(module);
    if (!funcOp) {
      ADD_FAILURE() << "Missing 'main' after fuse for basis=" << basis.str();
      return;
    }

    checksAfter(funcOp, basis, original);
  });
}

template <typename RotationOp>
[[nodiscard]] static Eigen::Matrix2cd rotationMatrix(MLIRContext* ctx,
                                                     double theta) {
  OpBuilder builder(ctx);
  auto module = ModuleOp::create(UnknownLoc::get(ctx));
  builder.setInsertionPointToStart(module.getBody());
  const Location loc = module.getLoc();
  Value q = builder.create<AllocOp>(loc).getResult();
  auto op = builder.create<RotationOp>(loc, q, theta);
  return *cast<RotationOp>(op).getUnitaryMatrix();
}

[[nodiscard]] static SynthesizedCircuit
synthesizeMatrix(MLIRContext* ctx, const Eigen::Matrix2cd& matrix,
                 EulerBasis basis, bool simplify) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(ctx));
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(module->getBody());

  auto qubitTy = QubitType::get(ctx);
  auto funcTy = builder.getFunctionType({qubitTy}, {qubitTy});
  auto func = builder.create<func::FuncOp>(module->getLoc(), "main", funcTy);
  auto* entry = func.addEntryBlock();

  builder.setInsertionPointToStart(entry);
  Value q = entry->getArgument(0);
  q = synthesizeUnitary1QEuler(builder, module->getLoc(), q, matrix, basis,
                               simplify);
  builder.create<func::ReturnOp>(module->getLoc(), q);
  return SynthesizedCircuit{.module = std::move(module), .func = func};
}

template <typename OpTy>
[[nodiscard]] static std::size_t countOps(func::FuncOp funcOp) {
  std::size_t count = 0;
  funcOp.walk([&](OpTy) { ++count; });
  return count;
}

[[nodiscard]] static std::size_t countTwoQubitGates(func::FuncOp funcOp) {
  std::size_t count = 0;
  funcOp.walk([&](UnitaryOpInterface op) {
    if (op.isTwoQubit()) {
      ++count;
    }
  });
  return count;
}

[[nodiscard]] static std::size_t countUnitaryMatrixOps(func::FuncOp funcOp) {
  std::size_t count = 0;
  funcOp.walk([&](UnitaryMatrixOpInterface) { ++count; });
  return count;
}

TEST(EulerSynthesisTest, RandomReconstructionAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  std::mt19937 rng{12345678UL};
  constexpr int iterations = 200;

  for (int i = 0; i < iterations; ++i) {
    const auto original = randomUnitaryMatrix<Eigen::Matrix2cd>(rng);

    forEachBasis([&](StringRef basisStr) {
      const auto parsed = mlir::qco::decomposition::parseEulerBasis(basisStr);
      ASSERT_TRUE(parsed) << "basis=" << basisStr.str();

      auto module = ModuleOp::create(UnknownLoc::get(fx.context.get()));
      MLIRContext* ctx = module.getContext();

      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(module.getBody());

      auto qubitTy = QubitType::get(ctx);
      auto funcTy = builder.getFunctionType({qubitTy}, {qubitTy});
      auto func = builder.create<func::FuncOp>(module.getLoc(), "main", funcTy);
      auto* entry = func.addEntryBlock();

      builder.setInsertionPointToStart(entry);
      Value q = entry->getArgument(0);
      q = mlir::qco::decomposition::synthesizeUnitary1QEuler(
          builder, module.getLoc(), q, original, *parsed);
      builder.create<func::ReturnOp>(module.getLoc(), q);

      ASSERT_TRUE(succeeded(verify(module))) << "basis=" << basisStr.str();

      const auto restored = compute1QMatrixFromFunction(func);
      EXPECT_TRUE(restored.isApprox(original, mlir::utils::TOLERANCE))
          << "basis=" << basisStr.str();
    });
  }
}

TEST(FuseSingleQubitUnitaryRunsTest, ReconstructsOriginalRunAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunWithSingleQubitGate,
      /*checksAfter=*/
      [&](func::FuncOp funcOp, StringRef basis,
          const Eigen::Matrix2cd& original) {
        const auto restored = compute1QMatrixFromFunction(funcOp);
        EXPECT_TRUE(restored.isApprox(original, mlir::utils::TOLERANCE))
            << "basis=" << basis.str();
        expectBasisGatesOnly(funcOp, basis);
      });
}

TEST(EulerSynthesisTest, ZsxxPauliXUsesSingleXGate) {
  SynthesisFixture fx;
  fx.setUp();

  const Eigen::Matrix2cd pauliX = XOp::getUnitaryMatrix();
  const auto circuit =
      synthesizeMatrix(fx.context.get(), pauliX, EulerBasis::ZSXX,
                       /*simplify=*/true);

  ASSERT_TRUE(succeeded(verify(*circuit.module)));
  EXPECT_EQ(countUnitaryMatrixOps(circuit.func), 1U);
  EXPECT_EQ(countOps<XOp>(circuit.func), 1U);
  EXPECT_EQ(countOps<RZOp>(circuit.func), 0U);
  EXPECT_EQ(countOps<SXOp>(circuit.func), 0U);
  EXPECT_EQ(countOps<UOp>(circuit.func), 0U);
  EXPECT_TRUE(compute1QMatrixFromFunction(circuit.func)
                  .isApprox(pauliX, mlir::utils::TOLERANCE));
}

TEST(EulerSynthesisTest, UGateReconstruction) {
  SynthesisFixture fx;
  fx.setUp();

  std::mt19937 rng{99991};
  for (int i = 0; i < 32; ++i) {
    const auto u = randomUnitaryMatrix<Eigen::Matrix2cd>(rng);
    const auto circuit = synthesizeMatrix(fx.context.get(), u, EulerBasis::U,
                                          /*simplify=*/true);
    ASSERT_TRUE(succeeded(verify(*circuit.module)));
    EXPECT_LE(countOps<UOp>(circuit.func), 1U);
    EXPECT_TRUE(compute1QMatrixFromFunction(circuit.func)
                    .isApprox(u, mlir::utils::TOLERANCE));
  }
}

TEST(EulerDecompositionTest, ZYZAnglesFromUnitaryReconstructHadamard) {
  SynthesisFixture fx;
  fx.setUp();

  const Eigen::Matrix2cd hadamard = HOp::getUnitaryMatrix();
  const auto [theta, phi, lambda, phase] =
      EulerDecomposition::anglesFromUnitary(hadamard, EulerBasis::ZYZ);

  auto module = ModuleOp::create(UnknownLoc::get(fx.context.get()));
  OpBuilder builder(fx.context.get());
  builder.setInsertionPointToStart(module.getBody());
  const Location loc = module.getLoc();

  auto qubitTy = QubitType::get(fx.context.get());
  auto funcTy = builder.getFunctionType({qubitTy}, {qubitTy});
  auto func = builder.create<func::FuncOp>(loc, "main", funcTy);
  auto* entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value q = entry->getArgument(0);
  auto mkAngle = [&](double angle) -> Value {
    return builder
        .create<arith::ConstantOp>(loc, builder.getF64FloatAttr(angle))
        .getResult();
  };
  q = builder.create<RZOp>(loc, q, mkAngle(lambda)).getQubitOut();
  q = builder.create<RYOp>(loc, q, mkAngle(theta)).getQubitOut();
  q = builder.create<RZOp>(loc, q, mkAngle(phi)).getQubitOut();
  if (std::abs(phase) > mlir::utils::TOLERANCE) {
    Value phaseVal = mkAngle(phase);
    builder.create<GPhaseOp>(loc, phaseVal);
  }
  builder.create<func::ReturnOp>(loc, q);

  ASSERT_TRUE(succeeded(verify(module)));
  EXPECT_TRUE(compute1QMatrixFromFunction(func).isApprox(
      hadamard, mlir::utils::TOLERANCE));
}

// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_P` at global scope
class EulerSynthesisExactTest
    : public testing::TestWithParam<
          std::tuple<EulerBasis, Eigen::Matrix2cd (*)(MLIRContext*)>> {};

TEST_P(EulerSynthesisExactTest, WithoutSimplification) {
  SynthesisFixture fx;
  fx.setUp();

  const auto [basis, matrixFn] = GetParam();
  const Eigen::Matrix2cd original = matrixFn(fx.context.get());
  const auto circuit = synthesizeMatrix(fx.context.get(), original, basis,
                                        /*simplify=*/false);

  ASSERT_TRUE(succeeded(verify(*circuit.module)));
  EXPECT_TRUE(compute1QMatrixFromFunction(circuit.func)
                  .isApprox(original, mlir::utils::TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(
    SingleQubitMatrices, EulerSynthesisExactTest,
    testing::Combine(testing::Values(EulerBasis::XYX, EulerBasis::XZX,
                                     EulerBasis::ZYZ, EulerBasis::ZXZ),
                     testing::Values(
                         [](MLIRContext* /*ctx*/) -> Eigen::Matrix2cd {
                           return Eigen::Matrix2cd::Identity();
                         },
                         [](MLIRContext* ctx) -> Eigen::Matrix2cd {
                           return rotationMatrix<RYOp>(ctx, 2.0);
                         },
                         [](MLIRContext* ctx) -> Eigen::Matrix2cd {
                           return rotationMatrix<RXOp>(ctx, 0.5);
                         },
                         [](MLIRContext* ctx) -> Eigen::Matrix2cd {
                           return rotationMatrix<RZOp>(ctx, 3.14);
                         },
                         [](MLIRContext* /*ctx*/) -> Eigen::Matrix2cd {
                           return HOp::getUnitaryMatrix();
                         })));

TEST(FuseSingleQubitUnitaryRunsTest, DoesNotFuseAcrossTwoQGateAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunsSplitByTwoQGate,
      /*checksAfter=*/
      [&](func::FuncOp funcOp, StringRef basis,
          const Eigen::Matrix2cd& original) {
        EXPECT_EQ(countTwoQubitGates(funcOp), 1U) << "basis=" << basis.str();
        EXPECT_TRUE(compute1QMatrixFromFunction(funcOp).isApprox(
            original, mlir::utils::TOLERANCE))
            << "basis=" << basis.str();
        expectBasisGatesOnly(funcOp, basis);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, DoesNotFuseAcrossBarrierAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunsSplitByBarrier,
      /*checksAfter=*/
      [&](func::FuncOp funcOp, StringRef basis,
          const Eigen::Matrix2cd& original) {
        EXPECT_EQ(countOps<BarrierOp>(funcOp), 1U) << "basis=" << basis.str();
        EXPECT_TRUE(compute1QMatrixFromFunction(funcOp).isApprox(
            original, mlir::utils::TOLERANCE))
            << "basis=" << basis.str();
        expectBasisGatesOnly(funcOp, basis);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, InvalidBasisFailsPass) {
  SynthesisFixture fx;
  fx.setUp();

  auto owned =
      buildProgram(fx.context.get(), &singleQubitRunWithSingleQubitGate);
  ASSERT_TRUE(static_cast<bool>(owned));
  ModuleOp module = *owned;
  ASSERT_TRUE(succeeded(verify(module)));

  EXPECT_TRUE(failed(runFuse(module, "not-a-basis")));
}
