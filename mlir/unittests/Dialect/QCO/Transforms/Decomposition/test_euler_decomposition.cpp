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
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <ios>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using enum EulerBasis;

// File layout:
//   1. Fixtures and parametric test types
//   2. Euler synthesis support + tests
//   3. FuseSingleQubitUnitaryRuns support + tests

namespace {

struct TestFixture {
  std::unique_ptr<MLIRContext> context;

  void setUp() {
    DialectRegistry registry;
    registry.insert<qco::QCODialect, arith::ArithDialect, func::FuncDialect,
                    scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] MLIRContext* ctx() const { return context.get(); }
};

struct ZSXXShortcutCase {
  std::string_view label;
  std::function<Matrix2x2(MLIRContext*)> makeMatrix;
  std::size_t expectedRZ;
  std::size_t expectedSX;
  std::size_t expectedX;
};

class ZSXXShortcutTest : public testing::TestWithParam<ZSXXShortcutCase> {};

struct SynthesizedCircuit {
  OwningOpRef<ModuleOp> mlirModule;
  func::FuncOp func;
};

class EulerSynthesisExactTest
    : public testing::TestWithParam<
          std::tuple<EulerBasis, Matrix2x2 (*)(MLIRContext*)>> {};

} // namespace

//===----------------------------------------------------------------------===//
// Euler synthesis support
//===----------------------------------------------------------------------===//

[[nodiscard]] static Matrix2x2 randomUnitaryMatrix(std::mt19937& rng) {
  std::uniform_real_distribution dist(-std::numbers::pi, std::numbers::pi);
  const Matrix2x2 su2 =
      rzMatrix(dist(rng)) * ryMatrix(dist(rng)) * rzMatrix(dist(rng));
  const Complex globalPhase = std::polar(1.0, dist(rng));
  return Matrix2x2::fromElements(
      globalPhase * su2(0, 0), globalPhase * su2(0, 1), globalPhase * su2(1, 0),
      globalPhase * su2(1, 1));
}

template <typename RotationOp>
[[nodiscard]] static Matrix2x2 rotationMatrix(MLIRContext* ctx,
                                              const double theta) {
  OpBuilder builder(ctx);
  auto mlirModule = ModuleOp::create(UnknownLoc::get(ctx));
  builder.setInsertionPointToStart(mlirModule.getBody());
  const Location loc = mlirModule.getLoc();
  Value q = AllocOp::create(builder, loc).getResult();
  auto op = RotationOp::create(builder, loc, q, theta);
  const auto matrix = op.getUnitaryMatrix();
  if (!matrix) {
    ADD_FAILURE() << "Expected constant unitary matrix";
    return Matrix2x2::identity();
  }
  return *matrix;
}

template <typename Fn> static void forEachBasis(Fn fn) {
  const std::array<const char*, 6> bases = {"zyz", "zxz", "xzx",
                                            "xyx", "u",   "zsxx"};
  for (const char* basis : bases) {
    fn(StringRef{basis});
  }
}
[[nodiscard]] static WalkResult failMissingUnitaryMatrix(Operation* op,
                                                         bool& failed) {
  ADD_FAILURE() << "Expected constant unitary matrix for op: "
                << op->getName().getStringRef().str();
  failed = true;
  return WalkResult::interrupt();
}

[[nodiscard]] static WalkResult
accumulateConstantSingleQubit(UnitaryOpInterface unitary, Operation* op,
                              Matrix2x2& acc, bool& failed) {
  if (Matrix2x2 matrix; unitary.getUnitaryMatrix2x2(matrix)) {
    acc = matrix * acc;
    return WalkResult::advance();
  }
  return failMissingUnitaryMatrix(op, failed);
}

static WalkResult visit1QUnitaryOp(Operation* op, Matrix2x2& acc,
                                   std::complex<double>& global, bool& failed) {
  if (isa<arith::ConstantOp, BarrierOp>(*op)) {
    return WalkResult::advance();
  }
  if (auto gphase = dyn_cast<GPhaseOp>(*op)) {
    if (auto matrix = gphase.getUnitaryMatrix()) {
      global *= (*matrix)(0, 0);
    }
    return WalkResult::advance();
  }
  auto unitary = dyn_cast<UnitaryOpInterface>(*op);
  if (!unitary) {
    return WalkResult::advance();
  }
  if (isa<InvOp, CtrlOp>(*op)) {
    if (!unitary.isSingleQubit()) {
      return WalkResult::skip();
    }
    const WalkResult result =
        accumulateConstantSingleQubit(unitary, op, acc, failed);
    return failed ? result : WalkResult::skip();
  }
  if (unitary.isTwoQubit()) {
    return WalkResult::advance();
  }
  const WalkResult result =
      accumulateConstantSingleQubit(unitary, op, acc, failed);
  return failed ? result : WalkResult::advance();
}
template <typename WalkRange>
static Matrix2x2 compute1QUnitaryMatrix(WalkRange& range) {
  Matrix2x2 acc = Matrix2x2::identity();
  std::complex<double> global{1.0, 0.0};
  bool failed = false;

  range.template walk<WalkOrder::PreOrder>(
      [&acc, &global, &failed](Operation* op) {
        return visit1QUnitaryOp(op, acc, global, failed);
      });

  if (failed) {
    return Matrix2x2::fromElements(0, 0, 0, 0);
  }
  return acc * global;
}
static void expectMatrixPreserved(func::FuncOp funcOp,
                                  const Matrix2x2& original,
                                  StringRef label = {}) {
  // Logging of the matrices
  auto printMatrix = [](const Matrix2x2& matrix) {
    std::ostringstream oss;
    oss.precision(4);
    oss << std::fixed << "[[" << matrix(0, 0) << ", " << matrix(0, 1) << "],\n"
        << " [" << matrix(1, 0) << ", " << matrix(1, 1) << "]]";
    return oss.str();
  };
  const auto printOriginal = printMatrix(original);
  const auto actual = compute1QUnitaryMatrix(funcOp.getBody());
  const auto printActual = printMatrix(actual);
  EXPECT_TRUE(actual.isApprox(original))
      << "Matrix not preserved for " << label.str() << ":\nOriginal:\n"
      << printOriginal << "\nActual:\n"
      << printActual;
}
template <typename OpTy>
[[nodiscard]] static std::size_t countOps(func::FuncOp funcOp) {
  std::size_t count = 0;
  funcOp.walk([&count](OpTy) { ++count; });
  return count;
}

[[nodiscard]] static std::size_t countZYZGates(func::FuncOp funcOp) {
  return countOps<RZOp>(funcOp) + countOps<RYOp>(funcOp);
}

[[nodiscard]] static std::size_t countZSXXGates(func::FuncOp funcOp) {
  return countOps<RZOp>(funcOp) + countOps<SXOp>(funcOp) +
         countOps<XOp>(funcOp);
}

[[nodiscard]] static std::size_t countBasisGates(func::FuncOp funcOp,
                                                 EulerBasis basis) {
  switch (basis) {
  case ZYZ:
    return countZYZGates(funcOp);
  case ZXZ:
    return countOps<RZOp>(funcOp) + countOps<RXOp>(funcOp);
  case XZX:
    return countOps<RXOp>(funcOp) + countOps<RZOp>(funcOp);
  case XYX:
    return countOps<RXOp>(funcOp) + countOps<RYOp>(funcOp);
  case U:
    return countOps<UOp>(funcOp);
  case ZSXX:
    return countZSXXGates(funcOp);
  case R:
    return countOps<ROp>(funcOp);
  }
  return 0;
}

[[nodiscard]] static SynthesizedCircuit
synthesizeMatrix(MLIRContext* ctx, const Matrix2x2& matrix, EulerBasis basis) {
  OwningOpRef mlirModule = ModuleOp::create(UnknownLoc::get(ctx));
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(mlirModule->getBody());

  auto qubitTy = QubitType::get(ctx);
  auto funcTy = builder.getFunctionType({qubitTy}, {qubitTy});
  const Location loc = mlirModule->getLoc();
  auto func = func::FuncOp::create(builder, loc, "main", funcTy);
  auto* entry = func.addEntryBlock();

  builder.setInsertionPointToStart(entry);
  Value q = entry->getArgument(0);
  const std::optional<Value> qubitOut =
      synthesizeUnitary1QEuler(builder, loc, q, matrix, 0, true, basis);
  if (!qubitOut) {
    llvm::report_fatal_error(
        "synthesizeUnitary1QEuler failed during test synthesis");
  }
  func::ReturnOp::create(builder, loc, *qubitOut);
  return SynthesizedCircuit{.mlirModule = std::move(mlirModule), .func = func};
}

[[nodiscard]] static std::size_t expectedGateCount(MLIRContext* ctx,
                                                   const Matrix2x2& segment,
                                                   EulerBasis basis) {
  return countBasisGates(synthesizeMatrix(ctx, segment, basis).func, basis);
}

static void checkSynthesizedReferenceExtras(MLIRContext* ctx,
                                            func::FuncOp funcOp,
                                            EulerBasis basis,
                                            const Matrix2x2& matrix) {
  if (basis == U) {
    EXPECT_EQ(countOps<UOp>(funcOp), expectedGateCount(ctx, matrix, basis));
  }
  if (!matrix.isApprox(Matrix2x2::identity())) {
    return;
  }
  if (basis == ZYZ) {
    EXPECT_EQ(countZYZGates(funcOp), 0U);
  }
  if (basis == U) {
    EXPECT_EQ(countOps<UOp>(funcOp), 0U);
  }
}

template <typename ExtraChecksT>
static void expectSynthesizedMatrix(MLIRContext* ctx, const Matrix2x2& matrix,
                                    EulerBasis basis,
                                    ExtraChecksT extraChecks) {
  const auto circuit = synthesizeMatrix(ctx, matrix, basis);
  ASSERT_TRUE(succeeded(verify(*circuit.mlirModule)));
  extraChecks(circuit.func, matrix);
  expectMatrixPreserved(circuit.func, matrix, "synthesis");
}

//===----------------------------------------------------------------------===//
// Euler synthesis tests
//===----------------------------------------------------------------------===//

TEST_P(ZSXXShortcutTest, SynthesisMatchesGateCount) {
  TestFixture fx;
  fx.setUp();
  const auto& testCase = GetParam();
  const Matrix2x2 matrix = testCase.makeMatrix(fx.ctx());

  expectSynthesizedMatrix(
      fx.ctx(), matrix, ZSXX,
      [&testCase, &fx](func::FuncOp funcOp, const Matrix2x2& original) {
        EXPECT_EQ(countOps<RZOp>(funcOp), testCase.expectedRZ);
        EXPECT_EQ(countOps<SXOp>(funcOp), testCase.expectedSX);
        EXPECT_EQ(countOps<XOp>(funcOp), testCase.expectedX);
        EXPECT_EQ(countZSXXGates(funcOp),
                  expectedGateCount(fx.ctx(), original, ZSXX));
      });
}

INSTANTIATE_TEST_SUITE_P(
    ZSXXShortcuts, ZSXXShortcutTest,
    testing::Values(
        ZSXXShortcutCase{
            "Identity",
            [](MLIRContext*) -> Matrix2x2 { return Matrix2x2::identity(); }, 0,
            0, 0},
        ZSXXShortcutCase{
            "PauliX",
            [](MLIRContext*) -> Matrix2x2 { return XOp::getUnitaryMatrix(); },
            0, 0, 1},
        ZSXXShortcutCase{"PureZ",
                         [](MLIRContext*) -> Matrix2x2 {
                           return rzMatrix(0.3) * rzMatrix(0.7);
                         },
                         1, 0, 0},
        ZSXXShortcutCase{"ZYZNearZeroTheta",
                         [](MLIRContext*) -> Matrix2x2 {
                           constexpr double tol = 0.5 * mlir::utils::TOLERANCE;
                           return rzMatrix(0.4) * ryMatrix(tol) * rzMatrix(0.3);
                         },
                         1, 0, 0},
        ZSXXShortcutCase{"RYHalfPi",
                         [](MLIRContext* ctx) -> Matrix2x2 {
                           return rotationMatrix<RYOp>(ctx,
                                                       std::numbers::pi / 2.0);
                         },
                         2, 1, 0},
        ZSXXShortcutCase{"RYNearHalfPi",
                         [](MLIRContext* ctx) -> Matrix2x2 {
                           return rotationMatrix<RYOp>(
                               ctx, (std::numbers::pi / 2.0) +
                                        (0.5 * mlir::utils::TOLERANCE));
                         },
                         2, 1, 0},
        ZSXXShortcutCase{"RYNearZero",
                         [](MLIRContext* ctx) -> Matrix2x2 {
                           return rotationMatrix<RYOp>(
                               ctx, 0.5 * mlir::utils::TOLERANCE);
                         },
                         0, 0, 0},
        ZSXXShortcutCase{"RYNearPi",
                         [](MLIRContext* ctx) -> Matrix2x2 {
                           return rotationMatrix<RYOp>(
                               ctx, std::numbers::pi -
                                        (0.5 * mlir::utils::TOLERANCE));
                         },
                         1, 0, 1}),
    [](const testing::TestParamInfo<ZSXXShortcutCase>& info) {
      return std::string(info.param.label);
    });

TEST_P(EulerSynthesisExactTest, ReconstructsReferenceMatrices) {
  TestFixture fx;
  fx.setUp();
  const auto [basis, matrixFn] = GetParam();
  const Matrix2x2 original = matrixFn(fx.ctx());
  expectSynthesizedMatrix(
      fx.ctx(), original, basis,
      [&fx, basis](func::FuncOp funcOp, const Matrix2x2& matrix) {
        checkSynthesizedReferenceExtras(fx.ctx(), funcOp, basis, matrix);
      });
}

INSTANTIATE_TEST_SUITE_P(
    SingleQubitMatrices, EulerSynthesisExactTest,
    testing::Combine(testing::Values(ZYZ, ZXZ, XZX, XYX, U, ZSXX),
                     testing::Values(
                         [](MLIRContext* /*ctx*/) -> Matrix2x2 {
                           return Matrix2x2::identity();
                         },
                         [](MLIRContext* ctx) -> Matrix2x2 {
                           return rotationMatrix<RYOp>(ctx, 2.0);
                         },
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
  TestFixture fx;
  fx.setUp();
  std::mt19937 rng{12345678UL};

  for (int i = 0; i < 200; ++i) {
    const auto original = randomUnitaryMatrix(rng);
    forEachBasis([&fx, &original](StringRef basisStr) {
      const auto parsed = parseEulerBasis(basisStr);
      ASSERT_TRUE(parsed) << "basis=" << basisStr.str();
      const auto circuit = synthesizeMatrix(fx.ctx(), original, *parsed);
      ASSERT_TRUE(succeeded(verify(*circuit.mlirModule)))
          << "basis=" << basisStr.str();
      expectMatrixPreserved(circuit.func, original, basisStr);
    });
  }
}

//===----------------------------------------------------------------------===//
// FuseSingleQubitUnitaryRuns support
//===----------------------------------------------------------------------===//

[[nodiscard]] static bool isAllowedBasisGate(const Operation& op,
                                             EulerBasis basis) {
  switch (basis) {
  case ZYZ:
    return isa<RZOp, RYOp>(op);
  case ZXZ:
    return isa<RZOp, RXOp>(op);
  case XZX:
    return isa<RXOp, RZOp>(op);
  case XYX:
    return isa<RXOp, RYOp>(op);
  case U:
    return isa<UOp>(op);
  case ZSXX:
    return isa<RZOp, SXOp, XOp>(op);
  case R:
    return isa<ROp>(op);
  }
  return false;
}

template <typename ParentOp> [[nodiscard]] static bool inParent(Operation* op) {
  return op != nullptr && op->getParentOfType<ParentOp>() != nullptr;
}

static WalkResult visitBasisGateOp(Operation* op, StringRef basis,
                                   EulerBasis parsedBasis) {
  if (isa<arith::ConstantOp, GPhaseOp, BarrierOp>(*op)) {
    return WalkResult::advance();
  }
  if (auto unitary = dyn_cast<UnitaryOpInterface>(*op)) {
    if (unitary.isTwoQubit() || isa<InvOp, CtrlOp>(*op)) {
      return unitary.isTwoQubit() ? WalkResult::advance() : WalkResult::skip();
    }
    if (Matrix2x2 matrix; unitary.getUnitaryMatrix2x2(matrix)) {
      EXPECT_TRUE(isAllowedBasisGate(*op, parsedBasis) || isa<GPhaseOp>(*op))
          << "basis=" << basis.str()
          << " unexpected gate: " << op->getName().getStringRef().str();
      return WalkResult::advance();
    }
    ADD_FAILURE() << "basis=" << basis.str() << " missing constant matrix for: "
                  << op->getName().getStringRef().str();
    return WalkResult::interrupt();
  }
  return WalkResult::advance();
}

static void skipBeforeFuse(func::FuncOp /*funcOp*/,
                           const Matrix2x2& /*original*/) {
  // Pre-fuse checks are not required for this scenario.
}

template <typename ParentOp>
[[nodiscard]] static Matrix2x2 matrixInParent(func::FuncOp funcOp) {
  auto parents = funcOp.getOps<ParentOp>();
  if (parents.begin() == parents.end()) {
    ADD_FAILURE() << "Expected parent op in function";
    return Matrix2x2::fromElements(0, 0, 0, 0);
  }
  return compute1QUnitaryMatrix((*parents.begin()).getRegion());
}

static void expectBasisGatesOnly(func::FuncOp funcOp, StringRef basis) {
  const auto parsed = parseEulerBasis(basis);
  ASSERT_TRUE(parsed) << basis.str();

  funcOp.walk<WalkOrder::PreOrder>(
      [basis, parsedBasis = *parsed](Operation* op) {
        return visitBasisGateOp(op, basis, parsedBasis);
      });
}

static void expectFusePreserved(func::FuncOp funcOp, const Matrix2x2& original,
                                StringRef basis) {
  expectMatrixPreserved(funcOp, original, basis);
  expectBasisGatesOnly(funcOp, basis);
}
[[nodiscard]] static Matrix2x2 splitFixtureHTSegmentMatrix() {
  return TOp::getUnitaryMatrix() * HOp::getUnitaryMatrix();
}

[[nodiscard]] static Matrix2x2 splitFixtureRZSXSegmentMatrix() {
  return SXOp::getUnitaryMatrix() * rzMatrix(0.321);
}

[[nodiscard]] static Matrix2x2 overlongZSXXPureZRunMatrix() {
  return SXOp::getUnitaryMatrix() * rzMatrix(std::numbers::pi) *
         SXOp::getUnitaryMatrix();
}
template <typename OpTy, typename ParentOp>
[[nodiscard]] static std::size_t countInParent(func::FuncOp funcOp) {
  std::size_t count = 0;
  funcOp.walk([&count](OpTy op) {
    if (inParent<ParentOp>(op.getOperation())) {
      ++count;
    }
  });
  return count;
}
static void expectSplitFixtureSegments(func::FuncOp funcOp, StringRef basis,
                                       MLIRContext* ctx) {
  const auto parsed = parseEulerBasis(basis);
  ASSERT_TRUE(parsed) << basis.str();
  const std::size_t ht =
      expectedGateCount(ctx, splitFixtureHTSegmentMatrix(), *parsed);
  const std::size_t rzsx =
      expectedGateCount(ctx, splitFixtureRZSXSegmentMatrix(), *parsed);

  std::size_t outside = 0;
  std::size_t inside = 0;
  funcOp.walk([&outside, &inside](Operation* op) {
    if (isa<arith::ConstantOp, GPhaseOp, BarrierOp>(*op)) {
      return;
    }
    auto unitary = dyn_cast<UnitaryOpInterface>(op);
    if (Matrix2x2 matrix; unitary && unitary.isSingleQubit() &&
                          unitary.getUnitaryMatrix2x2(matrix)) {
      if (inParent<scf::ForOp>(op)) {
        ++inside;
      } else {
        ++outside;
      }
    }
  });
  EXPECT_EQ(outside, ht) << "basis=" << basis.str();
  EXPECT_EQ(inside, rzsx) << "basis=" << basis.str();
}

template <typename BoundaryPred>
static void expectSplitFixtureSegments(func::FuncOp funcOp, StringRef basis,
                                       MLIRContext* ctx,
                                       BoundaryPred isBoundary) {
  const auto parsed = parseEulerBasis(basis);
  ASSERT_TRUE(parsed) << basis.str();
  const std::size_t ht =
      expectedGateCount(ctx, splitFixtureHTSegmentMatrix(), *parsed);
  const std::size_t rzsx =
      expectedGateCount(ctx, splitFixtureRZSXSegmentMatrix(), *parsed);

  std::size_t before = 0;
  std::size_t after = 0;
  bool seenBoundary = false;
  for (Operation& op : funcOp.getBody().front().without_terminator()) {
    if (!seenBoundary && isBoundary(op)) {
      seenBoundary = true;
      continue;
    }
    if (isa<GPhaseOp, BarrierOp>(op)) {
      continue;
    }
    auto unitary = dyn_cast<UnitaryOpInterface>(op);
    if (Matrix2x2 matrix; unitary && unitary.isSingleQubit() &&
                          unitary.getUnitaryMatrix2x2(matrix)) {
      if (seenBoundary) {
        ++after;
      } else {
        ++before;
      }
    }
  }
  EXPECT_EQ(before, ht) << "basis=" << basis.str();
  EXPECT_EQ(after, rzsx) << "basis=" << basis.str();
}

static LogicalResult runFuse(ModuleOp mlirModule, StringRef basis) {
  PassManager pm(mlirModule.getContext());
  qco::FuseSingleQubitUnitaryRunsOptions opts;
  opts.basis = basis.str();
  pm.addPass(qco::createFuseSingleQubitUnitaryRuns(opts));
  return pm.run(mlirModule);
}

template <typename ProgramT, typename BeforeT, typename AfterT>
static void runFuseOnProgram(MLIRContext* ctx, ProgramT program,
                             StringRef basis, BeforeT beforeFuse,
                             AfterT afterFuse) {
  auto owned = QCOProgramBuilder::build(ctx, program);
  ASSERT_TRUE(owned);
  ModuleOp mlirModule = *owned;
  ASSERT_TRUE(succeeded(verify(mlirModule)));

  auto funcOp = mlirModule.lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(funcOp);
  const Matrix2x2 original = compute1QUnitaryMatrix(funcOp);
  beforeFuse(funcOp, original);

  ASSERT_TRUE(succeeded(runFuse(mlirModule, basis)));
  ASSERT_TRUE(succeeded(verify(mlirModule)));

  funcOp = mlirModule.lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(funcOp);
  afterFuse(funcOp, original);
}

template <typename ProgramT, typename ChecksT>
static void runFuseForAllBases(MLIRContext* ctx, ProgramT program,
                               ChecksT checksAfter) {
  forEachBasis([&ctx, program, &checksAfter](StringRef basis) {
    runFuseOnProgram(
        ctx, program, basis, skipBeforeFuse,
        [basis, &checksAfter](func::FuncOp funcOp, const Matrix2x2& original) {
          checksAfter(funcOp, basis, original);
        });
  });
}

template <typename ParentOp, typename ProgramT, typename BeforeT,
          typename AfterT>
static void runFuseInParent(MLIRContext* ctx, ProgramT program,
                            BeforeT checkBefore, AfterT checkAfter) {
  Matrix2x2 bodyBefore;
  runFuseOnProgram(
      ctx, program, "u",
      [&checkBefore, &bodyBefore](func::FuncOp funcOp, const Matrix2x2&) {
        checkBefore(funcOp);
        bodyBefore = matrixInParent<ParentOp>(funcOp);
      },
      [&checkAfter, &bodyBefore](func::FuncOp funcOp, const Matrix2x2&) {
        checkAfter(funcOp);
        EXPECT_TRUE(matrixInParent<ParentOp>(funcOp).isApprox(
            bodyBefore, MATRIX_TOLERANCE));
      });
}

// --- Fuse program fixtures --- //

static void singleQubitRunWithSingleQubitGate(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  q[0] = b.rz(0.123, q[0]);
  q[0] = b.inv({q[0]}, [&b](ValueRange targets) -> SmallVector<Value> {
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

static void singleNonBasisGate(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
}

static void singlePauliX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
}

static void canonicalZYZRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.3, q[0]);
  q[0] = b.ry(0.5, q[0]);
  q[0] = b.rz(0.7, q[0]);
}

static void overlongZYZRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.3, q[0]);
  q[0] = b.ry(0.5, q[0]);
  q[0] = b.rz(0.7, q[0]);
  q[0] = b.ry(0.9, q[0]);
  q[0] = b.rz(1.1, q[0]);
  q[0] = b.ry(1.3, q[0]);
}

static void overlongZSXXMixedPureZRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.rz(std::numbers::pi, q[0]);
  q[0] = b.sx(q[0]);
}

static void singleQubitRunInScfFor(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.scfFor(0, 1, 1, ValueRange{q[0]}, [&b](Value, ValueRange iterArgs) {
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
  q[0] = b.inv({q[0]}, [&b](ValueRange targets) {
    Value wire = b.x(targets[0]);
    wire = b.x(wire);
    return SmallVector{wire};
  })[0];
  q[0] = b.x(q[0]);
}

static void inverseMultiQubitBodySingleQubitRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto outs =
      b.inv({q[0], q[1]}, [&b](ValueRange targets) -> SmallVector<Value> {
        Value wire = b.h(targets[0]);
        wire = b.t(wire);
        return {wire, targets[1]};
      });
  q[0] = outs[0];
  q[1] = outs[1];
}

static void controlledInverseHT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&b](ValueRange targets) {
    auto wire = b.inv({targets[0]}, [&b](ValueRange innerTargets) {
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
         [&b](ValueRange targets) { return SmallVector{b.h(targets[0])}; });
}

static void singleQubitRunsSplitByScfFor(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  b.scfFor(0, 1, 1, ValueRange{q[0]}, [&b](Value, ValueRange iterArgs) {
    Value wire = iterArgs[0];
    wire = b.rz(0.321, wire);
    wire = b.sx(wire);
    return SmallVector<Value>{wire};
  });
}

//===----------------------------------------------------------------------===//
// FuseSingleQubitUnitaryRuns tests
//===----------------------------------------------------------------------===//

TEST(FuseSingleQubitUnitaryRunsTest, InvalidBasisFailsPass) {
  TestFixture fx;
  fx.setUp();
  auto owned =
      QCOProgramBuilder::build(fx.ctx(), &singleQubitRunWithSingleQubitGate);
  ASSERT_TRUE(owned);
  EXPECT_TRUE(failed(runFuse(*owned, "not-a-basis")));
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesProgramsAllBases) {
  TestFixture fx;
  fx.setUp();

  struct Case {
    void (*program)(QCOProgramBuilder&);
    void (*extra)(func::FuncOp, StringRef);
  };
  const std::array<Case, 2> cases = {{
      {.program = &singleQubitRunWithSingleQubitGate,
       .extra =
           [](func::FuncOp funcOp, StringRef basis) {
             EXPECT_EQ(countOps<InvOp>(funcOp), 0U) << basis.str();
           }},
      {.program = &singleNonBasisGate,
       .extra =
           [](func::FuncOp funcOp, StringRef basis) {
             EXPECT_EQ(countOps<HOp>(funcOp), 0U) << basis.str();
           }},
  }};

  for (const Case& testCase : cases) {
    runFuseForAllBases(fx.ctx(), testCase.program,
                       [&testCase](func::FuncOp funcOp, StringRef basis,
                                   const Matrix2x2& original) {
                         testCase.extra(funcOp, basis);
                         expectFusePreserved(funcOp, original, basis);
                       });
  }
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesOverlongInBasisRun) {
  TestFixture fx;
  fx.setUp();
  runFuseOnProgram(
      fx.ctx(), &overlongZYZRun, "zyz",
      [](func::FuncOp funcOp, const Matrix2x2&) {
        ASSERT_EQ(countZYZGates(funcOp), 6U);
      },
      [&fx](func::FuncOp funcOp, const Matrix2x2& original) {
        EXPECT_EQ(countZYZGates(funcOp),
                  expectedGateCount(fx.ctx(), original, ZYZ));
        expectFusePreserved(funcOp, original, "zyz");
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, DoesNotFuseCanonicalInBasisRun) {
  TestFixture fx;
  fx.setUp();

  runFuseOnProgram(fx.ctx(), &singlePauliX, "zsxx", skipBeforeFuse,
                   [](func::FuncOp funcOp, const Matrix2x2& original) {
                     EXPECT_EQ(countOps<XOp>(funcOp), 1U);
                     expectFusePreserved(funcOp, original, "zsxx");
                   });

  runFuseOnProgram(fx.ctx(), &canonicalZYZRun, "zyz", skipBeforeFuse,
                   [](func::FuncOp funcOp, const Matrix2x2& original) {
                     EXPECT_EQ(countZYZGates(funcOp), 3U);
                     expectFusePreserved(funcOp, original, "zyz");
                   });
}

TEST(FuseSingleQubitUnitaryRunsTest,
     FusesOverlongZSXXMixedRunComposingToPureZ) {
  TestFixture fx;
  fx.setUp();
  runFuseOnProgram(
      fx.ctx(), &overlongZSXXMixedPureZRun, "zsxx",
      [](func::FuncOp funcOp, const Matrix2x2&) {
        ASSERT_EQ(countZSXXGates(funcOp), 3U);
      },
      [&fx](func::FuncOp funcOp, const Matrix2x2& original) {
        EXPECT_EQ(
            countZSXXGates(funcOp),
            expectedGateCount(fx.ctx(), overlongZSXXPureZRunMatrix(), ZSXX));
        expectFusePreserved(funcOp, original, "zsxx");
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, DoesNotFuseAcrossBoundariesAllBases) {
  TestFixture fx;
  fx.setUp();

  struct Case {
    void (*program)(QCOProgramBuilder&);
    void (*check)(func::FuncOp, StringRef, MLIRContext*);
  };
  const std::array<Case, 3> cases = {{
      {.program = &singleQubitRunsSplitByTwoQGate,
       .check =
           [](func::FuncOp funcOp, StringRef basis, MLIRContext* ctx) {
             std::size_t twoQ = 0;
             funcOp.walk([&twoQ](UnitaryOpInterface op) {
               if (op.isTwoQubit()) {
                 ++twoQ;
               }
             });
             EXPECT_EQ(twoQ, 1U) << basis.str();
             expectSplitFixtureSegments(
                 funcOp, basis, ctx, [](const Operation& op) {
                   if (auto unitary = dyn_cast<UnitaryOpInterface>(op)) {
                     return unitary.isTwoQubit();
                   }
                   return false;
                 });
           }},
      {.program = &singleQubitRunsSplitByBarrier,
       .check =
           [](func::FuncOp funcOp, StringRef basis, MLIRContext* ctx) {
             EXPECT_EQ(countOps<BarrierOp>(funcOp), 1U) << basis.str();
             expectSplitFixtureSegments(
                 funcOp, basis, ctx,
                 [](const Operation& op) { return isa<BarrierOp>(op); });
           }},
      {.program = &singleQubitRunsSplitByScfFor,
       .check =
           [](func::FuncOp funcOp, StringRef basis, MLIRContext* ctx) {
             EXPECT_EQ(countOps<scf::ForOp>(funcOp), 1U) << basis.str();
             expectSplitFixtureSegments(funcOp, basis, ctx);
           }},
  }};

  for (const Case& testCase : cases) {
    runFuseForAllBases(fx.ctx(), testCase.program,
                       [&testCase, &fx](func::FuncOp funcOp, StringRef basis,
                                        const Matrix2x2& original) {
                         testCase.check(funcOp, basis, fx.ctx());
                         expectFusePreserved(funcOp, original, basis);
                       });
  }
}

TEST(FuseSingleQubitUnitaryRunsTest, EliminatesIdentityInvMultiOpBody) {
  TestFixture fx;
  fx.setUp();
  runFuseOnProgram(
      fx.ctx(), xInverseTwoX, "u",
      [](func::FuncOp funcOp, const Matrix2x2&) {
        EXPECT_EQ(countOps<XOp>(funcOp), 4U);
        EXPECT_EQ(countOps<InvOp>(funcOp), 1U);
      },
      [&fx](func::FuncOp funcOp, const Matrix2x2& original) {
        EXPECT_EQ(countOps<InvOp>(funcOp), 0U);
        EXPECT_EQ(countOps<XOp>(funcOp), 0U);
        EXPECT_EQ(countOps<UOp>(funcOp),
                  expectedGateCount(fx.ctx(), original, U));
        expectMatrixPreserved(funcOp, original, "x-inv-xx-x");
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesRunInMultiQubitInvBody) {
  TestFixture fx;
  fx.setUp();
  runFuseInParent<InvOp>(
      fx.ctx(), inverseMultiQubitBodySingleQubitRun,
      [](func::FuncOp funcOp) {
        EXPECT_EQ(countOps<InvOp>(funcOp), 1U);
        EXPECT_EQ((countInParent<HOp, InvOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<TOp, InvOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<UOp, InvOp>(funcOp)), 0U);
      },
      [](func::FuncOp funcOp) {
        EXPECT_EQ(countOps<InvOp>(funcOp), 1U);
        EXPECT_EQ((countInParent<UOp, InvOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<HOp, InvOp>(funcOp)), 0U);
        EXPECT_EQ((countInParent<TOp, InvOp>(funcOp)), 0U);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesInCtrlBody) {
  TestFixture fx;
  fx.setUp();

  runFuseInParent<CtrlOp>(
      fx.ctx(), controlledH,
      [](func::FuncOp funcOp) {
        EXPECT_EQ((countInParent<HOp, CtrlOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<UOp, CtrlOp>(funcOp)), 0U);
      },
      [](func::FuncOp funcOp) {
        EXPECT_EQ((countInParent<UOp, CtrlOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<HOp, CtrlOp>(funcOp)), 0U);
      });

  runFuseInParent<CtrlOp>(
      fx.ctx(), controlledInverseHT,
      [](func::FuncOp funcOp) {
        EXPECT_EQ((countInParent<InvOp, CtrlOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<HOp, InvOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<TOp, InvOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<UOp, CtrlOp>(funcOp)), 0U);
      },
      [](func::FuncOp funcOp) {
        EXPECT_EQ((countInParent<InvOp, CtrlOp>(funcOp)), 0U);
        EXPECT_EQ((countInParent<UOp, CtrlOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<HOp, InvOp>(funcOp)), 0U);
        EXPECT_EQ((countInParent<TOp, InvOp>(funcOp)), 0U);
      });
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesRunInScfForBody) {
  TestFixture fx;
  fx.setUp();
  runFuseInParent<scf::ForOp>(
      fx.ctx(), &singleQubitRunInScfFor,
      [](func::FuncOp funcOp) {
        EXPECT_EQ((countInParent<HOp, scf::ForOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<TOp, scf::ForOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<RZOp, scf::ForOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<UOp, scf::ForOp>(funcOp)), 0U);
      },
      [](func::FuncOp funcOp) {
        EXPECT_EQ((countInParent<UOp, scf::ForOp>(funcOp)), 1U);
        EXPECT_EQ((countInParent<HOp, scf::ForOp>(funcOp)), 0U);
        EXPECT_EQ((countInParent<TOp, scf::ForOp>(funcOp)), 0U);
        EXPECT_EQ((countInParent<RZOp, scf::ForOp>(funcOp)), 0U);
      });
}
