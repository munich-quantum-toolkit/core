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
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <numbers>
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
    registry.insert<qco::QCODialect, arith::ArithDialect, func::FuncDialect,
                    scf::SCFDialect>();
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

[[nodiscard]] static Matrix2x2 scaleMatrix(const Matrix2x2& matrix,
                                           Complex scale) {
  return Matrix2x2::fromElements(scale * matrix(0, 0), scale * matrix(0, 1),
                                 scale * matrix(1, 0), scale * matrix(1, 1));
}

[[nodiscard]] static bool hasConstant2x2Matrix(Operation& op) {
  if (isa<BarrierOp>(op)) {
    return false;
  }
  auto unitary = dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isSingleQubit()) {
    return false;
  }
  Matrix2x2 matrix;
  return unitary.getUnitaryMatrix2x2(matrix);
}

[[nodiscard]] static Matrix2x2 rzMatrix(double theta) {
  const auto m00 = std::polar(1.0, -theta / 2.0);
  const auto m11 = std::polar(1.0, theta / 2.0);
  return Matrix2x2::fromElements(m00, 0, 0, m11);
}

[[nodiscard]] static Matrix2x2 ryMatrix(double theta) {
  const auto m00 = std::cos(theta / 2.0);
  const auto m01 = -std::sin(theta / 2.0);
  return Matrix2x2::fromElements(m00, m01, -m01, m00);
}

[[nodiscard]] static Matrix2x2 randomUnitaryMatrix(std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(-std::numbers::pi,
                                              std::numbers::pi);
  const Matrix2x2 su2 =
      rzMatrix(dist(rng)) * ryMatrix(dist(rng)) * rzMatrix(dist(rng));
  const Complex globalPhase = std::polar(1.0, dist(rng));
  return Matrix2x2::fromElements(
      globalPhase * su2(0, 0), globalPhase * su2(0, 1), globalPhase * su2(1, 0),
      globalPhase * su2(1, 1));
}

template <typename Fn> static void forEachBasis(Fn fn) {
  const std::array<const char*, 6> bases = {"zyz", "zxz", "xzx",
                                            "xyx", "u",   "zsxx"};
  for (const char* basis : bases) {
    fn(StringRef{basis});
  }
}

static bool isAllowedBasisGate(Operation& op, StringRef basis) {
  // `gphase` is always allowed.
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

// Matrix-backed 1Q gate (not barrier, 2Q, or `gphase`).
[[nodiscard]] static bool isOneQubitGate(Operation& op) {
  if (isa<GPhaseOp>(op)) {
    return false;
  }
  return hasConstant2x2Matrix(op);
}

// At least one 1Q gate before and after the first `isBoundary` op in `main`.
template <typename BoundaryPred>
static void expectOneQubitGatesAroundBoundary(func::FuncOp funcOp,
                                              StringRef basis,
                                              BoundaryPred isBoundary) {
  auto& block = funcOp.getBody().front();
  std::size_t before = 0;
  std::size_t after = 0;
  bool seenBoundary = false;
  for (Operation& op : block.without_terminator()) {
    if (!seenBoundary && isBoundary(op)) {
      seenBoundary = true;
      continue;
    }
    if (!isOneQubitGate(op)) {
      continue;
    }
    if (seenBoundary) {
      ++after;
    } else {
      ++before;
    }
  }
  EXPECT_GE(before, 1U) << "basis=" << basis.str();
  EXPECT_GE(after, 1U) << "basis=" << basis.str();
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

    // Matrix-backed ops must be allowed basis gates.
    if (hasConstant2x2Matrix(op)) {
      EXPECT_TRUE(isAllowedBasisGate(op, basis))
          << "basis=" << basis.str()
          << " unexpected gate: " << op.getName().getStringRef().str();
    }
  }
}

static Matrix2x2 compute1QMatrixFromFunction(func::FuncOp funcOp) {
  Matrix2x2 acc = Matrix2x2::identity();
  std::complex<double> global{1.0, 0.0};
  bool failed = false;

  // Include nested regions (`scf.for`); skip `inv`/`ctrl` bodies after the
  // modifier op (combined matrix already counted).
  funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
    if (isa<arith::ConstantOp>(*op) || isTwoQubitGate(*op)) {
      return WalkResult::advance();
    }

    if (auto gphase = dyn_cast<GPhaseOp>(*op)) {
      if (auto m = gphase.getUnitaryMatrix()) {
        global *= (*m)(0, 0);
      }
      return WalkResult::advance();
    }

    if (isa<BarrierOp>(*op)) {
      return WalkResult::advance();
    }

    if (auto unitary = dyn_cast<UnitaryOpInterface>(*op)) {
      Matrix2x2 matrix;
      if (!unitary.getUnitaryMatrix2x2(matrix)) {
        return WalkResult::advance();
      }
      acc = matrix * acc;
      if (isa<InvOp, CtrlOp>(*op)) {
        return WalkResult::skip();
      }
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });

  if (failed) {
    return Matrix2x2::fromElements(0, 0, 0, 0);
  }
  return scaleMatrix(acc, global);
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
  // `inv` is part of the fusable run.
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

// Single `H` gate — not in any target basis; should still be resynthesized.
static void singleNonBasisGate(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
}

// Six `RZ`/`RY` gates in `zyz` basis — longer than canonical (3).
static void overlongZyzRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.3, q[0]);
  q[0] = b.ry(0.5, q[0]);
  q[0] = b.rz(0.7, q[0]);
  q[0] = b.ry(0.9, q[0]);
  q[0] = b.rz(1.1, q[0]);
  q[0] = b.ry(1.3, q[0]);
}

static void singleQubitRunInScfFor(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.scfFor(0, 1, 1, ValueRange{q[0]}, [&](Value, ValueRange iterArgs) {
    Value wire = iterArgs[0];
    wire = b.h(wire);
    wire = b.t(wire);
    wire = b.rz(0.123, wire);
    return SmallVector<Value>{wire};
  });
}

[[nodiscard]] static std::size_t countUOpsInScfFor(func::FuncOp funcOp) {
  std::size_t count = 0;
  funcOp.walk([&](UOp op) {
    for (Operation* parent = op->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (parent->getName().getStringRef() == "scf.for") {
        ++count;
        break;
      }
    }
  });
  return count;
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

    const Matrix2x2 original = compute1QMatrixFromFunction(funcOp);

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
[[nodiscard]] static Matrix2x2 rotationMatrix(MLIRContext* ctx, double theta) {
  OpBuilder builder(ctx);
  auto module = ModuleOp::create(UnknownLoc::get(ctx));
  builder.setInsertionPointToStart(module.getBody());
  const Location loc = module.getLoc();
  Value q = builder.create<AllocOp>(loc).getResult();
  auto op = builder.create<RotationOp>(loc, q, theta);
  const auto matrix = cast<RotationOp>(op).getUnitaryMatrix();
  EXPECT_TRUE(matrix.has_value());
  return *matrix;
}

[[nodiscard]] static SynthesizedCircuit
synthesizeMatrix(MLIRContext* ctx, const Matrix2x2& matrix, EulerBasis basis) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(ctx));
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(module->getBody());

  auto qubitTy = QubitType::get(ctx);
  auto funcTy = builder.getFunctionType({qubitTy}, {qubitTy});
  auto func = builder.create<func::FuncOp>(module->getLoc(), "main", funcTy);
  auto* entry = func.addEntryBlock();

  builder.setInsertionPointToStart(entry);
  Value q = entry->getArgument(0);
  q = synthesizeUnitary1QEuler(builder, module->getLoc(), q, matrix, basis);
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

TEST(EulerSynthesisTest, RandomReconstructionAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  std::mt19937 rng{12345678UL};
  constexpr int iterations = 200;

  for (int i = 0; i < iterations; ++i) {
    const auto original = randomUnitaryMatrix(rng);

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

TEST(FuseSingleQubitUnitaryRunsTest, FusesRunInScfForBody) {
  SynthesisFixture fx;
  fx.setUp();

  auto owned = buildProgram(fx.context.get(), &singleQubitRunInScfFor);
  ASSERT_TRUE(owned);
  ModuleOp module = *owned;
  ASSERT_TRUE(succeeded(verify(module)));

  auto funcOp = lookupMain(module);
  ASSERT_TRUE(funcOp);
  const Matrix2x2 original = compute1QMatrixFromFunction(funcOp);

  ASSERT_TRUE(succeeded(runFuse(module, "u")));
  ASSERT_TRUE(succeeded(verify(module)));

  funcOp = lookupMain(module);
  ASSERT_TRUE(funcOp);
  EXPECT_GE(countUOpsInScfFor(funcOp), 1U);
  EXPECT_TRUE(compute1QMatrixFromFunction(funcOp).isApprox(
      original, mlir::utils::TOLERANCE));
}

TEST(FuseSingleQubitUnitaryRunsTest, ReconstructsOriginalRunAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunWithSingleQubitGate,
      /*checksAfter=*/
      [&](func::FuncOp funcOp, StringRef basis, const Matrix2x2& original) {
        const auto restored = compute1QMatrixFromFunction(funcOp);
        EXPECT_TRUE(restored.isApprox(original, mlir::utils::TOLERANCE))
            << "basis=" << basis.str();
        expectBasisGatesOnly(funcOp, basis);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, ResynthesizesLoneNonBasisGateAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleNonBasisGate,
      /*checksAfter=*/
      [&](func::FuncOp funcOp, StringRef basis, const Matrix2x2& original) {
        EXPECT_EQ(countOps<HOp>(funcOp), 0U)
            << "basis=" << basis.str() << " left a non-basis gate";
        EXPECT_TRUE(compute1QMatrixFromFunction(funcOp).isApprox(
            original, mlir::utils::TOLERANCE))
            << "basis=" << basis.str();
        expectBasisGatesOnly(funcOp, basis);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesOverlongInBasisRun) {
  SynthesisFixture fx;
  fx.setUp();

  auto owned = buildProgram(fx.context.get(), &overlongZyzRun);
  ASSERT_TRUE(owned);
  ModuleOp module = *owned;
  ASSERT_TRUE(succeeded(verify(module)));

  auto funcOp = lookupMain(module);
  ASSERT_TRUE(funcOp);
  const Matrix2x2 original = compute1QMatrixFromFunction(funcOp);
  const std::size_t before = countOps<RZOp>(funcOp) + countOps<RYOp>(funcOp);
  ASSERT_EQ(before, 6U);

  ASSERT_TRUE(succeeded(runFuse(module, "zyz")));
  ASSERT_TRUE(succeeded(verify(module)));

  funcOp = lookupMain(module);
  ASSERT_TRUE(funcOp);
  const std::size_t after = countOps<RZOp>(funcOp) + countOps<RYOp>(funcOp);
  EXPECT_LE(after, 3U);
  EXPECT_LT(after, before);
  EXPECT_TRUE(compute1QMatrixFromFunction(funcOp).isApprox(
      original, mlir::utils::TOLERANCE));
  expectBasisGatesOnly(funcOp, "zyz");
}

TEST(EulerSynthesisTest, ZsxxPauliXUsesXGateShortcut) {
  SynthesisFixture fx;
  fx.setUp();

  const Matrix2x2 pauliX = XOp::getUnitaryMatrix();
  const auto circuit =
      synthesizeMatrix(fx.context.get(), pauliX, EulerBasis::ZSXX);

  ASSERT_TRUE(succeeded(verify(*circuit.module)));
  EXPECT_EQ(countOps<XOp>(circuit.func), 1U);
  EXPECT_EQ(countOps<SXOp>(circuit.func), 0U);
  EXPECT_TRUE(compute1QMatrixFromFunction(circuit.func)
                  .isApprox(pauliX, mlir::utils::TOLERANCE));
}

TEST(EulerSynthesisTest, UGateReconstruction) {
  SynthesisFixture fx;
  fx.setUp();

  std::mt19937 rng{99991};
  for (int i = 0; i < 32; ++i) {
    const auto u = randomUnitaryMatrix(rng);
    const auto circuit = synthesizeMatrix(fx.context.get(), u, EulerBasis::U);
    ASSERT_TRUE(succeeded(verify(*circuit.module)));
    EXPECT_LE(countOps<UOp>(circuit.func), 1U);
    EXPECT_TRUE(compute1QMatrixFromFunction(circuit.func)
                    .isApprox(u, mlir::utils::TOLERANCE));
  }
}

TEST(EulerDecompositionTest, ZYZAnglesFromUnitaryReconstructHadamard) {
  SynthesisFixture fx;
  fx.setUp();

  const Matrix2x2 hadamard = HOp::getUnitaryMatrix();
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
          std::tuple<EulerBasis, Matrix2x2 (*)(MLIRContext*)>> {};

TEST_P(EulerSynthesisExactTest, ReconstructsReferenceMatrices) {
  SynthesisFixture fx;
  fx.setUp();

  const auto [basis, matrixFn] = GetParam();
  const Matrix2x2 original = matrixFn(fx.context.get());
  const auto circuit = synthesizeMatrix(fx.context.get(), original, basis);

  ASSERT_TRUE(succeeded(verify(*circuit.module)));
  EXPECT_TRUE(compute1QMatrixFromFunction(circuit.func)
                  .isApprox(original, mlir::utils::TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(
    SingleQubitMatrices, EulerSynthesisExactTest,
    testing::Combine(
        testing::Values(EulerBasis::XYX, EulerBasis::XZX, EulerBasis::ZYZ,
                        EulerBasis::ZXZ, EulerBasis::U, EulerBasis::ZSXX),
        testing::Values([](MLIRContext* /*ctx*/)
                            -> Matrix2x2 { return Matrix2x2::identity(); },
                        [](MLIRContext* ctx) -> Matrix2x2 {
                          return rotationMatrix<RYOp>(ctx, 2.0);
                        },
                        // RY(pi/2): ZSXX single-SX branch.
                        [](MLIRContext* ctx) -> Matrix2x2 {
                          return rotationMatrix<RYOp>(ctx,
                                                      std::numbers::pi / 2.0);
                        },
                        [](MLIRContext* ctx) -> Matrix2x2 {
                          return rotationMatrix<RXOp>(ctx, 0.5);
                        },
                        [](MLIRContext* ctx) -> Matrix2x2 {
                          return rotationMatrix<RZOp>(ctx, 3.14);
                        },
                        [](MLIRContext* /*ctx*/) -> Matrix2x2 {
                          return HOp::getUnitaryMatrix();
                        })));

TEST(FuseSingleQubitUnitaryRunsTest, DoesNotFuseAcrossTwoQGateAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunsSplitByTwoQGate,
      /*checksAfter=*/
      [&](func::FuncOp funcOp, StringRef basis, const Matrix2x2& original) {
        EXPECT_EQ(countTwoQubitGates(funcOp), 1U) << "basis=" << basis.str();
        EXPECT_TRUE(compute1QMatrixFromFunction(funcOp).isApprox(
            original, mlir::utils::TOLERANCE))
            << "basis=" << basis.str();
        expectBasisGatesOnly(funcOp, basis);
        expectOneQubitGatesAroundBoundary(
            funcOp, basis, [](Operation& op) { return isTwoQubitGate(op); });
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, DoesNotFuseAcrossBarrierAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunsSplitByBarrier,
      /*checksAfter=*/
      [&](func::FuncOp funcOp, StringRef basis, const Matrix2x2& original) {
        EXPECT_EQ(countOps<BarrierOp>(funcOp), 1U) << "basis=" << basis.str();
        EXPECT_TRUE(compute1QMatrixFromFunction(funcOp).isApprox(
            original, mlir::utils::TOLERANCE))
            << "basis=" << basis.str();
        expectBasisGatesOnly(funcOp, basis);
        expectOneQubitGatesAroundBoundary(
            funcOp, basis, [](Operation& op) { return isa<BarrierOp>(op); });
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
