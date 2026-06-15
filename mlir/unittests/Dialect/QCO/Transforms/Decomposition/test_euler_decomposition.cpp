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
#include <llvm/ADT/SmallVector.h>
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
#include <functional>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <string_view>
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

struct ZSXXShortcutCase {
  std::string_view label;
  std::function<Matrix2x2(MLIRContext*)> makeMatrix;
  std::size_t expectedSynthesisCount;
  std::size_t expectedRZ;
  std::size_t expectedSX;
  std::size_t expectedX;
};

class ZSXXShortcutTest : public testing::TestWithParam<ZSXXShortcutCase> {};

} // namespace

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

template <typename RotationOp>
[[nodiscard]] static Matrix2x2 rotationMatrix(MLIRContext* ctx, double theta) {
  OpBuilder builder(ctx);
  auto module = ModuleOp::create(UnknownLoc::get(ctx));
  builder.setInsertionPointToStart(module.getBody());
  const Location loc = module.getLoc();
  Value q = builder.create<AllocOp>(loc).getResult();
  auto op = builder.create<RotationOp>(loc, q, theta);
  const auto matrix = cast<RotationOp>(op).getUnitaryMatrix();
  if (!matrix) {
    ADD_FAILURE() << "Expected constant unitary matrix";
    return Matrix2x2::identity();
  }
  return *matrix;
}

[[nodiscard]] static bool isTwoQubitGate(Operation& op) {
  if (auto u = dyn_cast<UnitaryOpInterface>(op)) {
    return u.isTwoQubit();
  }
  return false;
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

template <typename Fn> static void forEachBasis(Fn fn) {
  const std::array<const char*, 6> bases = {"zyz", "zxz", "xzx",
                                            "xyx", "u",   "zsxx"};
  for (const char* basis : bases) {
    fn(StringRef{basis});
  }
}

template <typename WalkRange>
static Matrix2x2 compute1QUnitaryMatrix(WalkRange& range) {
  Matrix2x2 acc = Matrix2x2::identity();
  std::complex<double> global{1.0, 0.0};
  bool failed = false;

  range.template walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
    if (isa<arith::ConstantOp>(*op)) {
      return WalkResult::advance();
    }

    if (isa<BarrierOp>(*op)) {
      return WalkResult::advance();
    }

    if (auto gphase = dyn_cast<GPhaseOp>(*op)) {
      if (auto m = gphase.getUnitaryMatrix()) {
        global *= (*m)(0, 0);
      }
      return WalkResult::advance();
    }

    if (isa<InvOp, CtrlOp>(*op)) {
      auto unitary = cast<UnitaryOpInterface>(*op);
      if (unitary.isSingleQubit()) {
        Matrix2x2 matrix;
        if (!unitary.getUnitaryMatrix2x2(matrix)) {
          ADD_FAILURE() << "Expected constant unitary matrix for op: "
                        << op->getName().getStringRef().str();
          failed = true;
          return WalkResult::interrupt();
        }
        acc = matrix * acc;
      }
      return WalkResult::skip();
    }

    if (isTwoQubitGate(*op)) {
      return WalkResult::advance();
    }

    if (auto unitary = dyn_cast<UnitaryOpInterface>(*op)) {
      if (!unitary.isSingleQubit()) {
        return WalkResult::advance();
      }
      Matrix2x2 matrix;
      if (!unitary.getUnitaryMatrix2x2(matrix)) {
        ADD_FAILURE() << "Expected constant unitary matrix for op: "
                      << op->getName().getStringRef().str();
        failed = true;
        return WalkResult::interrupt();
      }
      acc = matrix * acc;
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });

  if (failed) {
    return Matrix2x2::fromElements(0, 0, 0, 0);
  }
  return acc * global;
}

static Matrix2x2 compute1QMatrixFromFunction(func::FuncOp funcOp) {
  return compute1QUnitaryMatrix(funcOp);
}

[[nodiscard]] static Matrix2x2
compute1QMatrixFromCtrlBody(func::FuncOp funcOp) {
  for (CtrlOp ctrl : funcOp.getOps<CtrlOp>()) {
    return compute1QUnitaryMatrix(ctrl.getRegion());
  }
  ADD_FAILURE() << "Expected CtrlOp in function";
  return Matrix2x2::fromElements(0, 0, 0, 0);
}

static void expectMatrixPreserved(func::FuncOp funcOp,
                                  const Matrix2x2& original, StringRef label) {
  EXPECT_TRUE(compute1QMatrixFromFunction(funcOp).isApprox(
      original, mlir::utils::TOLERANCE))
      << label.str();
}

static void expectCtrlBodyMatrixPreserved(func::FuncOp funcOp,
                                          const Matrix2x2& original) {
  EXPECT_TRUE(compute1QMatrixFromCtrlBody(funcOp).isApprox(
      original, mlir::utils::TOLERANCE));
}

static void expectBasisGatesOnly(func::FuncOp funcOp, StringRef basis) {
  funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
    if (isa<arith::ConstantOp, GPhaseOp, BarrierOp>(*op)) {
      return WalkResult::advance();
    }
    if (isTwoQubitGate(*op)) {
      return WalkResult::advance();
    }
    if (isa<InvOp, CtrlOp>(*op)) {
      return WalkResult::skip();
    }

    auto unitaryOp = dyn_cast<UnitaryOpInterface>(*op);
    if (!unitaryOp || !unitaryOp.isSingleQubit()) {
      return WalkResult::advance();
    }

    Matrix2x2 matrix;
    if (!unitaryOp.getUnitaryMatrix2x2(matrix)) {
      ADD_FAILURE() << "basis=" << basis.str()
                    << " missing constant matrix for: "
                    << op->getName().getStringRef().str();
      return WalkResult::interrupt();
    }
    EXPECT_TRUE(isAllowedBasisGate(*op, basis))
        << "basis=" << basis.str()
        << " unexpected gate: " << op->getName().getStringRef().str();
    return WalkResult::advance();
  });
}

/// Composed unitary of the `h; t` segment in `singleQubitRunsSplitByBarrier`,
/// `singleQubitRunsSplitByTwoQGate`, and `singleQubitRunsSplitByScfFor`.
[[nodiscard]] static Matrix2x2 splitFixtureHTSegmentMatrix() {
  return TOp::getUnitaryMatrix() * HOp::getUnitaryMatrix();
}

/// Composed unitary of the `rz(0.321); sx` segment in the same split fixtures.
[[nodiscard]] static Matrix2x2 splitFixtureRZSXSegmentMatrix() {
  return SXOp::getUnitaryMatrix() * rzMatrix(0.321);
}

/// Composed unitary of `overlongZSXXMixedPureZRun` (`sx; rz(pi); sx` → pure Z).
[[nodiscard]] static Matrix2x2 overlongZSXXPureZRunMatrix() {
  return SXOp::getUnitaryMatrix() * rzMatrix(std::numbers::pi) *
         SXOp::getUnitaryMatrix();
}

[[nodiscard]] static std::size_t
expectedFusedGateCount(const Matrix2x2& segment, EulerBasis basis) {
  return synthesisGateCount(segment, basis);
}

template <typename BoundaryPred>
static void expectOneQubitGatesAroundBoundary(func::FuncOp funcOp,
                                              StringRef basis,
                                              BoundaryPred isBoundary,
                                              std::size_t expectedBefore,
                                              std::size_t expectedAfter) {
  auto& block = funcOp.getBody().front();
  std::size_t before = 0;
  std::size_t after = 0;
  bool seenBoundary = false;
  for (Operation& op : block.without_terminator()) {
    if (!seenBoundary && isBoundary(op)) {
      seenBoundary = true;
      continue;
    }
    if (isa<GPhaseOp, BarrierOp>(op)) {
      continue;
    }
    auto unitaryOp = dyn_cast<UnitaryOpInterface>(op);
    if (!unitaryOp || !unitaryOp.isSingleQubit()) {
      continue;
    }
    Matrix2x2 matrix;
    if (!unitaryOp.getUnitaryMatrix2x2(matrix)) {
      continue;
    }
    if (seenBoundary) {
      ++after;
    } else {
      ++before;
    }
  }
  EXPECT_EQ(before, expectedBefore) << "basis=" << basis.str();
  EXPECT_EQ(after, expectedAfter) << "basis=" << basis.str();
}

template <typename OpTy>
[[nodiscard]] static std::size_t countOps(func::FuncOp funcOp) {
  std::size_t count = 0;
  funcOp.walk([&](OpTy) { ++count; });
  return count;
}

[[nodiscard]] static std::size_t countZSXXBasisGates(func::FuncOp funcOp) {
  return countOps<RZOp>(funcOp) + countOps<SXOp>(funcOp) +
         countOps<XOp>(funcOp);
}

[[nodiscard]] static bool isInsideScfFor(Operation* op) {
  return op != nullptr && op->getParentOfType<scf::ForOp>() != nullptr;
}

[[nodiscard]] static bool isInsideInv(Operation* op) {
  return op != nullptr && op->getParentOfType<InvOp>() != nullptr;
}

[[nodiscard]] static bool isInsideCtrl(Operation* op) {
  return op != nullptr && op->getParentOfType<CtrlOp>() != nullptr;
}

template <typename OpTy>
[[nodiscard]] static std::size_t countOpsInScfFor(func::FuncOp funcOp) {
  std::size_t count = 0;
  funcOp.walk([&](OpTy op) {
    if (isInsideScfFor(op.getOperation())) {
      ++count;
    }
  });
  return count;
}

template <typename OpTy, typename InRegionPred>
[[nodiscard]] static std::size_t countOpsInRegion(func::FuncOp funcOp,
                                                  InRegionPred inRegion) {
  std::size_t count = 0;
  funcOp.walk([&](OpTy op) {
    if (inRegion(op.getOperation())) {
      ++count;
    }
  });
  return count;
}

static void expectOneQubitGatesInAndOutsideScfFor(func::FuncOp funcOp,
                                                  StringRef basis,
                                                  std::size_t expectedOutside,
                                                  std::size_t expectedInside) {
  std::size_t outside = 0;
  std::size_t inside = 0;
  funcOp.walk([&](Operation* op) {
    if (isa<arith::ConstantOp, GPhaseOp, BarrierOp>(*op)) {
      return;
    }
    auto unitary = dyn_cast<UnitaryOpInterface>(op);
    if (!unitary || !unitary.isSingleQubit()) {
      return;
    }
    Matrix2x2 matrix;
    if (!unitary.getUnitaryMatrix2x2(matrix)) {
      return;
    }
    if (isInsideScfFor(op)) {
      ++inside;
    } else {
      ++outside;
    }
  });
  EXPECT_EQ(outside, expectedOutside) << "basis=" << basis.str();
  EXPECT_EQ(inside, expectedInside) << "basis=" << basis.str();
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
static void overlongZYZRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.3, q[0]);
  q[0] = b.ry(0.5, q[0]);
  q[0] = b.rz(0.7, q[0]);
  q[0] = b.ry(0.9, q[0]);
  q[0] = b.rz(1.1, q[0]);
  q[0] = b.ry(1.3, q[0]);
}

// `SX`/`RZ(pi)`/`SX` in `zsxx` basis — composes to `Z` (pure-Z, theta = 0).
// Consecutive-`RZ` merge patterns do not apply across `SX` gates.
static void overlongZSXXMixedPureZRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.rz(std::numbers::pi, q[0]);
  q[0] = b.sx(q[0]);
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

static void xInverseTwoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  q[0] = b.inv({q[0]}, [&](ValueRange targets) {
    Value wire = b.x(targets[0]);
    wire = b.x(wire);
    return SmallVector{wire};
  })[0];
  q[0] = b.x(q[0]);
}

static void controlledInverseHT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&](ValueRange targets) {
    auto wire = b.inv({targets[0]}, [&](ValueRange innerTargets) {
      auto inner = b.h(innerTargets[0]);
      inner = b.t(inner);
      return SmallVector{inner};
    })[0];
    return SmallVector{wire};
  });
}

static void controlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1],
         [&](ValueRange targets) { return SmallVector{b.h(targets[0])}; });
}

static void singleQubitRunsSplitByScfFor(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  b.scfFor(0, 1, 1, ValueRange{q[0]}, [&](Value, ValueRange iterArgs) {
    Value wire = iterArgs[0];
    wire = b.rz(0.321, wire);
    wire = b.sx(wire);
    return SmallVector<Value>{wire};
  });
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

template <typename ExtraChecksT>
static void expectSynthesizedMatrix(MLIRContext* ctx, const Matrix2x2& matrix,
                                    EulerBasis basis,
                                    ExtraChecksT extraChecks) {
  const auto circuit = synthesizeMatrix(ctx, matrix, basis);
  ASSERT_TRUE(succeeded(verify(*circuit.module)));
  extraChecks(circuit.func, matrix);
  expectMatrixPreserved(circuit.func, matrix, "synthesis");
}

static LogicalResult runFuse(ModuleOp module, StringRef basis) {
  PassManager pm(module.getContext());
  qco::FuseSingleQubitUnitaryRunsOptions opts;
  opts.basis = basis.str();
  pm.addPass(qco::createFuseSingleQubitUnitaryRuns(opts));
  return pm.run(module);
}

template <typename BeforeT, typename AfterT>
static void
runFuseOnProgram(MLIRContext* ctx, void (*program)(QCOProgramBuilder&),
                 StringRef basis, BeforeT beforeFuse, AfterT afterFuse) {
  auto owned = QCOProgramBuilder::build(ctx, program);
  ASSERT_TRUE(owned);
  ModuleOp module = *owned;
  ASSERT_TRUE(succeeded(verify(module)));

  auto funcOp = module.lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(funcOp) << "Expected a 'main' function";
  const Matrix2x2 original = compute1QMatrixFromFunction(funcOp);
  beforeFuse(funcOp, original);

  ASSERT_TRUE(succeeded(runFuse(module, basis)));
  ASSERT_TRUE(succeeded(verify(module)));

  funcOp = module.lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(funcOp) << "Expected a 'main' function";
  afterFuse(funcOp, original);
}

template <typename ChecksT>
static void runFuseOnProgramForAllBases(MLIRContext* ctx,
                                        void (*program)(QCOProgramBuilder&),
                                        ChecksT checksAfter) {
  forEachBasis([&](StringRef basis) {
    runFuseOnProgram(
        ctx, program, basis, [](func::FuncOp, const Matrix2x2&) {},
        [&](func::FuncOp funcOp, const Matrix2x2& original) {
          checksAfter(funcOp, basis, original);
        });
  });
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

TEST(EulerSynthesisTest, ProfitabilityAndIdentityGateCount) {
  SynthesisFixture fx;
  fx.setUp();

  const Matrix2x2 identity = Matrix2x2::identity();
  EXPECT_TRUE(wouldShortenInBasisRun(6, identity, EulerBasis::ZYZ));
  EXPECT_TRUE(wouldShortenInBasisRun(2, identity, EulerBasis::ZYZ));
  EXPECT_FALSE(
      wouldShortenInBasisRun(1, XOp::getUnitaryMatrix(), EulerBasis::ZSXX));
  EXPECT_EQ(synthesisGateCount(identity, EulerBasis::ZYZ), 0U);
}

TEST(EulerSynthesisTest, ClassifyZSXXMiddleFromZYZThetaBoundaries) {
  using decomposition::classifyZSXXMiddleFromZYZTheta;
  using decomposition::ZSXXMiddleGate;

  constexpr double halfPi = std::numbers::pi / 2.0;
  constexpr double pi = std::numbers::pi;
  constexpr double tol = 0.5 * mlir::utils::TOLERANCE;

  EXPECT_EQ(classifyZSXXMiddleFromZYZTheta(tol), ZSXXMiddleGate::OnlyRZ);
  EXPECT_EQ(classifyZSXXMiddleFromZYZTheta(mlir::utils::TOLERANCE),
            ZSXXMiddleGate::OnlyRZ);
  EXPECT_EQ(classifyZSXXMiddleFromZYZTheta(halfPi + tol),
            ZSXXMiddleGate::OneSX);
  EXPECT_EQ(classifyZSXXMiddleFromZYZTheta(pi - tol), ZSXXMiddleGate::X);
  EXPECT_EQ(classifyZSXXMiddleFromZYZTheta(pi), ZSXXMiddleGate::X);
}

TEST_P(ZSXXShortcutTest, SynthesisMatchesGateCount) {
  SynthesisFixture fx;
  fx.setUp();

  const auto& testCase = GetParam();
  const Matrix2x2 matrix = testCase.makeMatrix(fx.context.get());
  EXPECT_EQ(synthesisGateCount(matrix, EulerBasis::ZSXX),
            testCase.expectedSynthesisCount);

  expectSynthesizedMatrix(
      fx.context.get(), matrix, EulerBasis::ZSXX,
      [&](func::FuncOp funcOp, const Matrix2x2& original) {
        EXPECT_EQ(countOps<RZOp>(funcOp), testCase.expectedRZ);
        EXPECT_EQ(countOps<SXOp>(funcOp), testCase.expectedSX);
        EXPECT_EQ(countOps<XOp>(funcOp), testCase.expectedX);
        EXPECT_EQ(countZSXXBasisGates(funcOp),
                  expectedFusedGateCount(original, EulerBasis::ZSXX));
      });
}

INSTANTIATE_TEST_SUITE_P(
    ZSXXShortcuts, ZSXXShortcutTest,
    testing::Values(ZSXXShortcutCase{"PauliX",
                                     [](MLIRContext*) -> Matrix2x2 {
                                       return XOp::getUnitaryMatrix();
                                     },
                                     1, 0, 0, 1},
                    ZSXXShortcutCase{"PureZ",
                                     [](MLIRContext*) -> Matrix2x2 {
                                       return rzMatrix(0.3) * rzMatrix(0.7);
                                     },
                                     2, 2, 0, 0},
                    ZSXXShortcutCase{"RYHalfPi",
                                     [](MLIRContext* ctx) -> Matrix2x2 {
                                       return rotationMatrix<RYOp>(
                                           ctx, std::numbers::pi / 2.0);
                                     },
                                     3, 2, 1, 0},
                    ZSXXShortcutCase{"RYNearHalfPi",
                                     [](MLIRContext* ctx) -> Matrix2x2 {
                                       return rotationMatrix<RYOp>(
                                           ctx,
                                           (std::numbers::pi / 2.0) +
                                               (0.5 * mlir::utils::TOLERANCE));
                                     },
                                     3, 2, 1, 0},
                    ZSXXShortcutCase{"RYNearZero",
                                     [](MLIRContext* ctx) -> Matrix2x2 {
                                       return rotationMatrix<RYOp>(
                                           ctx, 0.5 * mlir::utils::TOLERANCE);
                                     },
                                     0, 0, 0, 0},
                    ZSXXShortcutCase{"RYNearPi",
                                     [](MLIRContext* ctx) -> Matrix2x2 {
                                       return rotationMatrix<RYOp>(
                                           ctx,
                                           std::numbers::pi -
                                               (0.5 * mlir::utils::TOLERANCE));
                                     },
                                     2, 1, 0, 1}),
    [](const testing::TestParamInfo<ZSXXShortcutCase>& info) {
      return std::string(info.param.label);
    });

// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_P` at global scope
class EulerSynthesisExactTest
    : public testing::TestWithParam<
          std::tuple<EulerBasis, Matrix2x2 (*)(MLIRContext*)>> {};

TEST_P(EulerSynthesisExactTest, ReconstructsReferenceMatrices) {
  SynthesisFixture fx;
  fx.setUp();

  const auto [basis, matrixFn] = GetParam();
  const Matrix2x2 original = matrixFn(fx.context.get());
  expectSynthesizedMatrix(fx.context.get(), original, basis,
                          [&](func::FuncOp funcOp, const Matrix2x2& matrix) {
                            if (basis == EulerBasis::U) {
                              EXPECT_EQ(countOps<UOp>(funcOp),
                                        expectedFusedGateCount(matrix, basis));
                            }
                            if (basis == EulerBasis::ZYZ &&
                                matrix.isApprox(Matrix2x2::identity())) {
                              EXPECT_EQ(countOps<RZOp>(funcOp), 0U);
                              EXPECT_EQ(countOps<RYOp>(funcOp), 0U);
                            }
                          });
}

INSTANTIATE_TEST_SUITE_P(
    SingleQubitMatrices, EulerSynthesisExactTest,
    testing::Combine(
        testing::Values(EulerBasis::ZYZ, EulerBasis::ZXZ, EulerBasis::XZX,
                        EulerBasis::XYX, EulerBasis::U, EulerBasis::ZSXX),
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

TEST(EulerSynthesisTest, RandomReconstructionAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  std::mt19937 rng{12345678UL};
  constexpr int iterations = 200;

  for (int i = 0; i < iterations; ++i) {
    const auto original = randomUnitaryMatrix(rng);

    forEachBasis([&](StringRef basisStr) {
      const auto parsed = parseEulerBasis(basisStr);
      ASSERT_TRUE(parsed) << "basis=" << basisStr.str();

      const auto circuit =
          synthesizeMatrix(fx.context.get(), original, *parsed);
      ASSERT_TRUE(succeeded(verify(*circuit.module)))
          << "basis=" << basisStr.str();
      expectMatrixPreserved(circuit.func, original, basisStr);
    });
  }
}

TEST(FuseSingleQubitUnitaryRunsTest, InvalidBasisFailsPass) {
  SynthesisFixture fx;
  fx.setUp();

  auto owned = QCOProgramBuilder::build(fx.context.get(),
                                        &singleQubitRunWithSingleQubitGate);
  ASSERT_TRUE(static_cast<bool>(owned));
  ModuleOp module = *owned;
  ASSERT_TRUE(succeeded(verify(module)));

  EXPECT_TRUE(failed(runFuse(module, "not-a-basis")));
}

TEST(FuseSingleQubitUnitaryRunsTest, ReconstructsOriginalRunAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunWithSingleQubitGate,
      [&](func::FuncOp funcOp, StringRef basis, const Matrix2x2& original) {
        EXPECT_EQ(countOps<InvOp>(funcOp), 0U) << "basis=" << basis.str();
        expectMatrixPreserved(funcOp, original, basis);
        expectBasisGatesOnly(funcOp, basis);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, ResynthesizesLoneNonBasisGateAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleNonBasisGate,
      [&](func::FuncOp funcOp, StringRef basis, const Matrix2x2& original) {
        EXPECT_EQ(countOps<HOp>(funcOp), 0U)
            << "basis=" << basis.str() << " left a non-basis gate";
        expectMatrixPreserved(funcOp, original, basis);
        expectBasisGatesOnly(funcOp, basis);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesOverlongInBasisRun) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgram(
      fx.context.get(), &overlongZYZRun, "zyz",
      [](func::FuncOp funcOp, const Matrix2x2&) {
        ASSERT_EQ(countOps<RZOp>(funcOp) + countOps<RYOp>(funcOp), 6U);
      },
      [](func::FuncOp funcOp, const Matrix2x2& original) {
        EXPECT_EQ(countOps<RZOp>(funcOp) + countOps<RYOp>(funcOp),
                  expectedFusedGateCount(original, EulerBasis::ZYZ));
        expectMatrixPreserved(funcOp, original, "zyz");
        expectBasisGatesOnly(funcOp, "zyz");
      });
}

TEST(FuseSingleQubitUnitaryRunsTest,
     FusesOverlongZSXXMixedRunComposingToPureZ) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgram(
      fx.context.get(), &overlongZSXXMixedPureZRun, "zsxx",
      [](func::FuncOp funcOp, const Matrix2x2& /*original*/) {
        EXPECT_EQ(expectedFusedGateCount(overlongZSXXPureZRunMatrix(),
                                         EulerBasis::ZSXX),
                  1U);
        ASSERT_EQ(countZSXXBasisGates(funcOp), 3U);
        EXPECT_EQ(countOps<SXOp>(funcOp), 2U);
        EXPECT_EQ(countOps<RZOp>(funcOp), 1U);
        EXPECT_EQ(countOps<XOp>(funcOp), 0U);
      },
      [](func::FuncOp funcOp, const Matrix2x2& original) {
        EXPECT_EQ(countOps<RZOp>(funcOp), 1U);
        EXPECT_EQ(countOps<SXOp>(funcOp), 0U);
        EXPECT_EQ(countOps<XOp>(funcOp), 0U);
        EXPECT_EQ(countZSXXBasisGates(funcOp),
                  expectedFusedGateCount(overlongZSXXPureZRunMatrix(),
                                         EulerBasis::ZSXX));
        expectMatrixPreserved(funcOp, original, "zsxx");
        expectBasisGatesOnly(funcOp, "zsxx");
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, DoesNotFuseAcrossTwoQGateAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunsSplitByTwoQGate,
      [&](func::FuncOp funcOp, StringRef basis, const Matrix2x2& original) {
        std::size_t twoQubitGates = 0;
        funcOp.walk([&](UnitaryOpInterface op) {
          if (op.isTwoQubit()) {
            ++twoQubitGates;
          }
        });
        EXPECT_EQ(twoQubitGates, 1U) << "basis=" << basis.str();
        expectMatrixPreserved(funcOp, original, basis);
        expectBasisGatesOnly(funcOp, basis);
        const auto parsed = parseEulerBasis(basis);
        ASSERT_TRUE(parsed) << "basis=" << basis.str();
        expectOneQubitGatesAroundBoundary(
            funcOp, basis, [](Operation& op) { return isTwoQubitGate(op); },
            expectedFusedGateCount(splitFixtureHTSegmentMatrix(), *parsed),
            expectedFusedGateCount(splitFixtureRZSXSegmentMatrix(), *parsed));
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, DoesNotFuseAcrossBarrierAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunsSplitByBarrier,
      [&](func::FuncOp funcOp, StringRef basis, const Matrix2x2& original) {
        EXPECT_EQ(countOps<BarrierOp>(funcOp), 1U) << "basis=" << basis.str();
        expectMatrixPreserved(funcOp, original, basis);
        expectBasisGatesOnly(funcOp, basis);
        const auto parsed = parseEulerBasis(basis);
        ASSERT_TRUE(parsed) << "basis=" << basis.str();
        expectOneQubitGatesAroundBoundary(
            funcOp, basis, [](Operation& op) { return isa<BarrierOp>(op); },
            expectedFusedGateCount(splitFixtureHTSegmentMatrix(), *parsed),
            expectedFusedGateCount(splitFixtureRZSXSegmentMatrix(), *parsed));
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, EliminatesIdentityInvMultiOpBody) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgram(
      fx.context.get(), xInverseTwoX, "u",
      [](func::FuncOp funcOp, const Matrix2x2&) {
        EXPECT_EQ(countOps<XOp>(funcOp), 4U);
        EXPECT_EQ(countOps<InvOp>(funcOp), 1U);
        EXPECT_EQ(countOps<UOp>(funcOp), 0U);
      },
      [](func::FuncOp funcOp, const Matrix2x2& original) {
        EXPECT_EQ(countOps<InvOp>(funcOp), 0U);
        EXPECT_EQ(countOps<XOp>(funcOp), 0U);
        EXPECT_EQ(countOps<UOp>(funcOp),
                  expectedFusedGateCount(original, EulerBasis::U));
        expectMatrixPreserved(funcOp, original, "x-inv-xx-x");
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesSingleNonBasisGateInCtrlBody) {
  SynthesisFixture fx;
  fx.setUp();

  Matrix2x2 ctrlBodyBefore;
  runFuseOnProgram(
      fx.context.get(), controlledH, "u",
      [&](func::FuncOp funcOp, const Matrix2x2&) {
        EXPECT_EQ(countOps<CtrlOp>(funcOp), 1U);
        EXPECT_EQ(countOpsInRegion<HOp>(funcOp, isInsideCtrl), 1U);
        EXPECT_EQ(countOpsInRegion<UOp>(funcOp, isInsideCtrl), 0U);
        ctrlBodyBefore = compute1QMatrixFromCtrlBody(funcOp);
      },
      [&](func::FuncOp funcOp, const Matrix2x2&) {
        EXPECT_EQ(countOps<CtrlOp>(funcOp), 1U);
        EXPECT_EQ(countOpsInRegion<HOp>(funcOp, isInsideCtrl), 0U);
        EXPECT_EQ(countOpsInRegion<UOp>(funcOp, isInsideCtrl), 1U);
        expectCtrlBodyMatrixPreserved(funcOp, ctrlBodyBefore);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesRunInCtrlBody) {
  SynthesisFixture fx;
  fx.setUp();

  Matrix2x2 ctrlBodyBefore;
  runFuseOnProgram(
      fx.context.get(), controlledInverseHT, "u",
      [&](func::FuncOp funcOp, const Matrix2x2&) {
        EXPECT_EQ(countOps<CtrlOp>(funcOp), 1U);
        EXPECT_EQ(countOpsInRegion<InvOp>(funcOp, isInsideCtrl), 1U);
        EXPECT_EQ(countOpsInRegion<HOp>(funcOp, isInsideInv), 1U);
        EXPECT_EQ(countOpsInRegion<TOp>(funcOp, isInsideInv), 1U);
        EXPECT_EQ(countOpsInRegion<UOp>(funcOp, isInsideCtrl), 0U);
        ctrlBodyBefore = compute1QMatrixFromCtrlBody(funcOp);
      },
      [&](func::FuncOp funcOp, const Matrix2x2&) {
        EXPECT_EQ(countOps<CtrlOp>(funcOp), 1U);
        EXPECT_EQ(countOpsInRegion<InvOp>(funcOp, isInsideCtrl), 0U);
        EXPECT_EQ(countOpsInRegion<HOp>(funcOp, isInsideInv), 0U);
        EXPECT_EQ(countOpsInRegion<TOp>(funcOp, isInsideInv), 0U);
        EXPECT_EQ(countOpsInRegion<UOp>(funcOp, isInsideCtrl), 1U);
        expectCtrlBodyMatrixPreserved(funcOp, ctrlBodyBefore);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesRunInScfForBody) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgram(
      fx.context.get(), &singleQubitRunInScfFor, "u",
      [](func::FuncOp funcOp, const Matrix2x2&) {
        EXPECT_EQ(countOpsInScfFor<HOp>(funcOp), 1U);
        EXPECT_EQ(countOpsInScfFor<TOp>(funcOp), 1U);
        EXPECT_EQ(countOpsInScfFor<RZOp>(funcOp), 1U);
        EXPECT_EQ(countOpsInScfFor<UOp>(funcOp), 0U);
      },
      [](func::FuncOp funcOp, const Matrix2x2& original) {
        EXPECT_EQ(countOpsInScfFor<UOp>(funcOp), 1U);
        EXPECT_EQ(countOpsInScfFor<HOp>(funcOp), 0U);
        EXPECT_EQ(countOpsInScfFor<TOp>(funcOp), 0U);
        EXPECT_EQ(countOpsInScfFor<RZOp>(funcOp), 0U);
        expectMatrixPreserved(funcOp, original, "scf.for");
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, DoesNotFuseAcrossScfForAllBases) {
  SynthesisFixture fx;
  fx.setUp();

  runFuseOnProgramForAllBases(
      fx.context.get(), &singleQubitRunsSplitByScfFor,
      [&](func::FuncOp funcOp, StringRef basis, const Matrix2x2& original) {
        EXPECT_EQ(countOps<scf::ForOp>(funcOp), 1U) << "basis=" << basis.str();
        expectMatrixPreserved(funcOp, original, basis);
        expectBasisGatesOnly(funcOp, basis);
        const auto parsed = parseEulerBasis(basis);
        ASSERT_TRUE(parsed) << "basis=" << basis.str();
        expectOneQubitGatesInAndOutsideScfFor(
            funcOp, basis,
            expectedFusedGateCount(splitFixtureHTSegmentMatrix(), *parsed),
            expectedFusedGateCount(splitFixtureRZSXSegmentMatrix(), *parsed));
      });
}
