/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <ios>
#include <limits>
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
using enum SingleQubitBasis;

// File layout:
//   1. Fixtures and parametric test types
//   2. Euler synthesis support + tests
//   3. FuseSingleQubitUnitaryRuns support + tests

namespace {

struct TestFixture {
  std::unique_ptr<MLIRContext> context;

  void setUp() {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
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
          std::tuple<SingleQubitBasis, Matrix2x2 (*)(MLIRContext*)>> {};

} // namespace

/**
 * @brief Measures the given qubits and returns the measurement outcomes.
 * @param b The `ProgramBuilder` used to perform the measurements.
 * @param qubits The qubits to be measured.
 * @return The result values.
 */
static SmallVector<Value> measureAndReturn(QCOProgramBuilder& b,
                                           ValueRange qubits) {
  return to_vector(
      llvm::map_range(qubits, [&](Value q) { return b.measure(q).second; }));
}

//===----------------------------------------------------------------------===//
// Euler synthesis support
//===----------------------------------------------------------------------===//

[[nodiscard]] static Matrix2x2 randomUnitaryMatrix(std::mt19937& rng) {
  std::uniform_real_distribution dist(-std::numbers::pi, std::numbers::pi);
  const Matrix2x2 su2 = RZOp::unitaryMatrix(dist(rng)) *
                        RYOp::unitaryMatrix(dist(rng)) *
                        RZOp::unitaryMatrix(dist(rng));
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
  constexpr std::array<const char*, 7> bases = {"zyz", "zxz",  "xzx", "xyx",
                                                "u",   "zsxx", "r"};
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

[[nodiscard]] static bool valueDependsOn(Value value, Value target) {
  llvm::DenseSet<Value> visited;
  SmallVector<Value> worklist{value};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (current == target) {
      return true;
    }
    if (!visited.insert(current).second) {
      continue;
    }
    if (Operation* definingOp = current.getDefiningOp()) {
      worklist.append(definingOp->operand_begin(), definingOp->operand_end());
    }
  }
  return false;
}

static void bindLeadingArguments(func::FuncOp funcOp, ArrayRef<double> values) {
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  for (const auto [index, value] : llvm::enumerate(values)) {
    Value constant = arith::ConstantOp::create(builder, funcOp.getLoc(),
                                               builder.getF64FloatAttr(value));
    funcOp.getArgument(index).replaceAllUsesWith(constant);
  }
}

static LogicalResult canonicalizeBoundValues(ModuleOp mlirModule) {
  PassManager pm(mlirModule.getContext());
  pm.addPass(createCanonicalizerPass());
  return pm.run(mlirModule);
}

[[nodiscard]] static std::size_t countZYZGates(func::FuncOp funcOp) {
  return countOps<RZOp>(funcOp) + countOps<RYOp>(funcOp);
}

[[nodiscard]] static std::size_t countZSXXGates(func::FuncOp funcOp) {
  return countOps<RZOp>(funcOp) + countOps<SXOp>(funcOp) +
         countOps<XOp>(funcOp);
}

[[nodiscard]] static std::size_t countBasisGates(func::FuncOp funcOp,
                                                 SingleQubitBasis basis) {
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
synthesizeMatrix(MLIRContext* ctx, const Matrix2x2& matrix,
                 SingleQubitBasis basis) {
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
  const auto synthesized =
      synthesizeUnitary1QEuler(builder, loc, q, matrix, 0, true, basis);
  if (!synthesized) {
    llvm::report_fatal_error(
        "synthesizeUnitary1QEuler failed during test synthesis");
  }
  emitGPhaseIfNeeded(builder, loc, synthesized->globalPhase);
  func::ReturnOp::create(builder, loc, synthesized->qubit);
  return SynthesizedCircuit{.mlirModule = std::move(mlirModule), .func = func};
}

[[nodiscard]] static std::size_t expectedGateCount(MLIRContext* ctx,
                                                   const Matrix2x2& segment,
                                                   SingleQubitBasis basis) {
  return countBasisGates(synthesizeMatrix(ctx, segment, basis).func, basis);
}

static void checkSynthesizedReferenceExtras(MLIRContext* ctx,
                                            func::FuncOp funcOp,
                                            SingleQubitBasis basis,
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
                                    SingleQubitBasis basis,
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
                           return RZOp::unitaryMatrix(0.3) *
                                  RZOp::unitaryMatrix(0.7);
                         },
                         1, 0, 0},
        ZSXXShortcutCase{"ZYZNearZeroTheta",
                         [](MLIRContext*) -> Matrix2x2 {
                           constexpr double tol =
                               0.5 * mlir::mqt::PARAMETER_COMPARISON_TOLERANCE;
                           return RZOp::unitaryMatrix(0.4) *
                                  RYOp::unitaryMatrix(tol) *
                                  RZOp::unitaryMatrix(0.3);
                         },
                         1, 0, 0},
        ZSXXShortcutCase{"RYHalfPi",
                         [](MLIRContext* ctx) -> Matrix2x2 {
                           return rotationMatrix<RYOp>(ctx,
                                                       std::numbers::pi / 2.0);
                         },
                         2, 1, 0},
        ZSXXShortcutCase{
            "RYNearHalfPi",
            [](MLIRContext* ctx) -> Matrix2x2 {
              return rotationMatrix<RYOp>(
                  ctx, (std::numbers::pi / 2.0) +
                           (0.5 * mlir::mqt::PARAMETER_COMPARISON_TOLERANCE));
            },
            2, 1, 0},
        ZSXXShortcutCase{"RYNearZero",
                         [](MLIRContext* ctx) -> Matrix2x2 {
                           return rotationMatrix<RYOp>(
                               ctx,
                               0.5 * mlir::mqt::PARAMETER_COMPARISON_TOLERANCE);
                         },
                         0, 0, 0},
        ZSXXShortcutCase{
            "RYNearPi",
            [](MLIRContext* ctx) -> Matrix2x2 {
              return rotationMatrix<RYOp>(
                  ctx, std::numbers::pi -
                           (0.5 * mlir::mqt::PARAMETER_COMPARISON_TOLERANCE));
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
      const auto parsed = parseSingleQubitBasis(basisStr);
      ASSERT_TRUE(parsed) << "basis=" << basisStr.str();
      const auto circuit = synthesizeMatrix(fx.ctx(), original, *parsed);
      ASSERT_TRUE(succeeded(verify(*circuit.mlirModule)))
          << "basis=" << basisStr.str();
      expectMatrixPreserved(circuit.func, original, basisStr);
    });
  }
}

TEST(EulerAnglesCoverageTest, ParamsZYZUsesOffDiagonal01When10IsNearZero) {
  Matrix2x2 matrix = RXOp::unitaryMatrix(0.4);
  matrix(1, 0) = Complex{0.0, 0.0};
  ASSERT_GT(std::abs(matrix(0, 1)), mlir::mqt::PARAMETER_COMPARISON_TOLERANCE);
  const EulerAngles angles = anglesFromUnitary(matrix, ZYZ);
  EXPECT_TRUE(std::isfinite(angles.theta));
  EXPECT_TRUE(std::isfinite(angles.phi));
  EXPECT_TRUE(std::isfinite(angles.lambda));
}

TEST(EulerAnglesCoverageTest, PhaseOnlyDecompositionSkipsRotationGates) {
  TestFixture fx;
  fx.setUp();
  constexpr double scale = 1.0 + 1e-10;
  const Matrix2x2 matrix = Matrix2x2::fromElements(scale, 0, 0, scale);
  ASSERT_FALSE(matrix.isApprox(Matrix2x2::identity()));
  const EulerAngles angles = anglesFromUnitary(matrix, ZYZ);
  EXPECT_LE(std::abs(angles.theta), mlir::mqt::PARAMETER_COMPARISON_TOLERANCE);
  EXPECT_LE(std::abs(angles.phi), mlir::mqt::PARAMETER_COMPARISON_TOLERANCE);
  EXPECT_LE(std::abs(angles.lambda), mlir::mqt::PARAMETER_COMPARISON_TOLERANCE);
  const auto circuit = synthesizeMatrix(fx.ctx(), matrix, ZYZ);
  ASSERT_TRUE(succeeded(verify(*circuit.mlirModule)));
  EXPECT_EQ(countZYZGates(circuit.func), 0U);
}

TEST(EulerAnglesCoverageTest, UBasisZeroThetaEmitsSingleUGate) {
  TestFixture fx;
  fx.setUp();
  const Matrix2x2 matrix = RZOp::unitaryMatrix(0.7);
  expectSynthesizedMatrix(fx.ctx(), matrix, U,
                          [](func::FuncOp funcOp, const Matrix2x2& /*matrix*/) {
                            EXPECT_EQ(countOps<UOp>(funcOp), 1U);
                            EXPECT_EQ(countZYZGates(funcOp), 0U);
                          });
}

TEST(EulerAnglesCoverageTest, UBasisNonzeroThetaEmitsSingleUGate) {
  TestFixture fx;
  fx.setUp();
  const Matrix2x2 matrix = RYOp::unitaryMatrix(1.2);
  const EulerAngles angles = anglesFromUnitary(matrix, U);
  ASSERT_GT(std::abs(angles.theta), mlir::mqt::PARAMETER_COMPARISON_TOLERANCE);
  expectSynthesizedMatrix(fx.ctx(), matrix, U,
                          [](func::FuncOp funcOp, const Matrix2x2& /*matrix*/) {
                            EXPECT_EQ(countOps<UOp>(funcOp), 1U);
                            EXPECT_EQ(countZYZGates(funcOp), 0U);
                          });
}

TEST(EulerAnglesCoverageTest, RBasisNonzeroThetaEmitsThreeRGates) {
  TestFixture fx;
  fx.setUp();
  const Matrix2x2 matrix = HOp::getUnitaryMatrix();
  const EulerAngles angles = anglesFromUnitary(matrix, R);
  ASSERT_GT(std::abs(angles.theta), mlir::mqt::PARAMETER_COMPARISON_TOLERANCE);
  expectSynthesizedMatrix(fx.ctx(), matrix, R,
                          [](func::FuncOp funcOp, const Matrix2x2& /*matrix*/) {
                            EXPECT_EQ(countOps<ROp>(funcOp), 3U);
                          });
}

TEST(EulerAnglesCoverageTest, Mod2PiMapsPiBoundaryThroughSynthesis) {
  TestFixture fx;
  fx.setUp();
  constexpr double eps = 0.5 * mlir::mqt::PARAMETER_COMPARISON_TOLERANCE;
  const Complex global = std::polar(1.0, std::numbers::pi - eps);
  const Matrix2x2 matrix = Matrix2x2::fromElements(global, 0, 0, global);
  expectSynthesizedMatrix(fx.ctx(), matrix, U,
                          [](func::FuncOp funcOp, const Matrix2x2& /*matrix*/) {
                            EXPECT_EQ(countOps<UOp>(funcOp), 1U);
                            EXPECT_EQ(countOps<GPhaseOp>(funcOp), 1U);
                          });
}

TEST(EulerAnglesCoverageTest, Mod2PiPreservesNonFinitePhase) {
  TestFixture fx;
  fx.setUp();
  const Matrix2x2 matrix = Matrix2x2::fromElements(
      Complex{std::numeric_limits<double>::quiet_NaN(), 0}, 0, 0, 1);
  EXPECT_NO_FATAL_FAILURE(std::ignore =
                              synthesizeMatrix(fx.ctx(), matrix, ZYZ));
}

//===----------------------------------------------------------------------===//
// FuseSingleQubitUnitaryRuns support
//===----------------------------------------------------------------------===//

[[nodiscard]] static bool isAllowedBasisGate(Operation& op,
                                             SingleQubitBasis basis) {
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
                                   SingleQubitBasis parsedBasis) {
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
  const auto parsed = parseSingleQubitBasis(basis);
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
  return SXOp::getUnitaryMatrix() * RZOp::unitaryMatrix(0.321);
}

[[nodiscard]] static Matrix2x2 overlongZSXXPureZRunMatrix() {
  return SXOp::getUnitaryMatrix() * RZOp::unitaryMatrix(std::numbers::pi) *
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
  const auto parsed = parseSingleQubitBasis(basis);
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
  const auto parsed = parseSingleQubitBasis(basis);
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

namespace {

struct DynamicGateCase {
  StringRef name;
  size_t numParameters = 0;
  Value (*build)(QCOProgramBuilder&, Value) = nullptr;
};

const std::array DYNAMIC_GATE_CASES = {
    DynamicGateCase{
        .name = "rx",
        .numParameters = 1,
        .build = [](QCOProgramBuilder& b, Value q) { return b.rx(0.11, q); }},
    DynamicGateCase{
        .name = "ry",
        .numParameters = 1,
        .build = [](QCOProgramBuilder& b, Value q) { return b.ry(0.13, q); }},
    DynamicGateCase{
        .name = "rz",
        .numParameters = 1,
        .build = [](QCOProgramBuilder& b, Value q) { return b.rz(0.17, q); }},
    DynamicGateCase{
        .name = "p",
        .numParameters = 1,
        .build = [](QCOProgramBuilder& b, Value q) { return b.p(0.19, q); }},
    DynamicGateCase{.name = "r",
                    .numParameters = 2,
                    .build = [](QCOProgramBuilder& b,
                                Value q) { return b.r(0.23, -0.29, q); }},
    DynamicGateCase{.name = "u2",
                    .numParameters = 2,
                    .build = [](QCOProgramBuilder& b,
                                Value q) { return b.u2(0.31, -0.37, q); }},
    DynamicGateCase{.name = "u",
                    .numParameters = 3,
                    .build = [](QCOProgramBuilder& b,
                                Value q) { return b.u(0.41, -0.43, 0.47, q); }},
};

struct DirectSynthesisCounts {
  size_t u = 0;
  size_t rz = 0;
  size_t ry = 0;
  size_t rx = 0;
  size_t sx = 0;
  bool gphase = false;
};

} // namespace

[[nodiscard]] static DirectSynthesisCounts
expectedDirectSynthesisCounts(StringRef gate, SingleQubitBasis basis) {
  switch (basis) {
  case U:
    return {.u = 1, .gphase = gate == "rz"};
  case ZYZ:
    if (gate == "ry") {
      return {.ry = 1};
    }
    if (gate == "rz" || gate == "p") {
      return {.rz = 1, .gphase = gate == "p"};
    }
    return {.rz = 2, .ry = 1, .gphase = gate == "u" || gate == "u2"};
  case ZXZ:
    if (gate == "rx") {
      return {.rx = 1};
    }
    if (gate == "rz" || gate == "p") {
      return {.rz = 1, .gphase = gate == "p"};
    }
    return {.rz = 2, .rx = 1, .gphase = gate == "u" || gate == "u2"};
  case ZSXX:
    if (gate == "rz" || gate == "p") {
      return {.rz = 1, .gphase = gate == "p"};
    }
    if (gate == "u2") {
      return {.rz = 2, .sx = 1, .gphase = true};
    }
    if (gate == "ry") {
      return {.rz = 2, .sx = 2, .gphase = true};
    }
    return {.rz = 3, .sx = 2, .gphase = true};
  case XZX:
  case XYX:
  case R:
    llvm_unreachable("basis does not use direct synthesis");
  }
  llvm_unreachable("invalid single-qubit synthesis basis");
}

static void expectDirectSynthesisCounts(func::FuncOp funcOp,
                                        const DirectSynthesisCounts& expected) {
  EXPECT_EQ(countOps<UOp>(funcOp), expected.u);
  EXPECT_EQ(countOps<RZOp>(funcOp), expected.rz);
  EXPECT_EQ(countOps<RYOp>(funcOp), expected.ry);
  EXPECT_EQ(countOps<RXOp>(funcOp), expected.rx);
  EXPECT_EQ(countOps<SXOp>(funcOp), expected.sx);
  EXPECT_EQ(countOps<GPhaseOp>(funcOp), static_cast<size_t>(expected.gphase));
  EXPECT_EQ(countOps<U2Op>(funcOp), 0U);
  EXPECT_EQ(countOps<POp>(funcOp), 0U);
  EXPECT_EQ(countOps<ROp>(funcOp), 0U);

  EXPECT_EQ(countOps<math::SinOp>(funcOp), 0U);
  EXPECT_EQ(countOps<math::CosOp>(funcOp), 0U);
  EXPECT_EQ(countOps<math::AbsFOp>(funcOp), 0U);
  EXPECT_EQ(countOps<math::FloorOp>(funcOp), 0U);
  EXPECT_EQ(countOps<math::AcosOp>(funcOp), 0U);
  EXPECT_EQ(countOps<math::Atan2Op>(funcOp), 0U);
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

static SmallVector<Value>
singleQubitRunWithSingleQubitGate(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  q[0] = b.rz(0.123, q[0]);
  q[0] = b.inv(q[0], [&b](Value qubit) { return b.sx(qubit); });
  q[0] = b.ry(-0.456, q[0]);
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value> singleQubitRunsSplitByTwoQGate(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  q[0] = b.rz(0.321, q[0]);
  q[0] = b.sx(q[0]);
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value> singleQubitRunsSplitByBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  q[0] = b.barrier({q[0]})[0];
  q[0] = b.rz(0.321, q[0]);
  q[0] = b.sx(q[0]);
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value> singleNonBasisGate(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value> singlePauliX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value> canonicalZYZRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.3, q[0]);
  q[0] = b.ry(0.5, q[0]);
  q[0] = b.rz(0.7, q[0]);
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value> overlongZYZRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.3, q[0]);
  q[0] = b.ry(0.5, q[0]);
  q[0] = b.rz(0.7, q[0]);
  q[0] = b.ry(0.9, q[0]);
  q[0] = b.rz(1.1, q[0]);
  q[0] = b.ry(1.3, q[0]);
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value> overlongZSXXMixedPureZRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.rz(std::numbers::pi, q[0]);
  q[0] = b.sx(q[0]);
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value> singleQubitRunInScfFor(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res =
      b.scfFor(0, 1, 1, ValueRange{q[0]}, [&b](Value, ValueRange iterArgs) {
        Value wire = iterArgs[0];
        wire = b.h(wire);
        wire = b.t(wire);
        wire = b.rz(0.123, wire);
        return SmallVector<Value>{wire};
      });
  return measureAndReturn(b, res);
}

static SmallVector<Value> xInverseTwoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  q[0] = b.inv(q[0], [&b](Value qubit) {
    qubit = b.x(qubit);
    return b.x(qubit);
  });
  q[0] = b.x(q[0]);
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value>
inverseMultiQubitBodySingleQubitRun(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto outs =
      b.inv({q[0], q[1]}, [&b](ValueRange targets) -> SmallVector<Value> {
        Value wire = b.h(targets[0]);
        wire = b.t(wire);
        return {wire, targets[1]};
      });
  q[0] = outs[0];
  q[1] = outs[1];
  return measureAndReturn(b, q.qubits);
}

static SmallVector<Value> controlledInverseHT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ctrl(q[0], q[1], [&b](Value target) {
    return b.inv(target, [&b](Value qubit) {
      qubit = b.h(qubit);
      return b.t(qubit);
    });
  });
  return measureAndReturn(b, {res.second});
}

static SmallVector<Value> controlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ctrl(q[0], q[1], [&b](Value target) { return b.h(target); });
  return measureAndReturn(b, {res.second});
}

static Value dynamicPowX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto powOut = b.pow(0.5, q[0], [&](Value qubit) { return b.x(qubit); });
  return b.measure(powOut).second;
}

static SmallVector<Value> singleQubitRunsSplitByScfFor(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.t(q[0]);
  auto res =
      b.scfFor(0, 1, 1, ValueRange{q[0]}, [&b](Value, ValueRange iterArgs) {
        Value wire = iterArgs[0];
        wire = b.rz(0.321, wire);
        wire = b.sx(wire);
        return SmallVector<Value>{wire};
      });
  return measureAndReturn(b, res);
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

TEST(FuseSingleQubitUnitaryRunsTest, IgnoresDynamicPowerExponent) {
  TestFixture fx;
  fx.setUp();
  auto owned = QCOProgramBuilder::build(fx.ctx(), &dynamicPowX);
  ASSERT_TRUE(owned);
  auto funcOp = owned->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(funcOp);
  funcOp.insertArgument(0, Float64Type::get(fx.ctx()), {}, funcOp.getLoc());
  auto powOp = *funcOp.getBody().getOps<PowOp>().begin();
  powOp->setOperand(0, funcOp.getArgument(0));
  ASSERT_TRUE(succeeded(verify(*owned)));
  ASSERT_FALSE(powOp.hasCompileTimeKnownUnitaryMatrix());

  EXPECT_TRUE(succeeded(runFuse(*owned, "zyz")));
  EXPECT_EQ(countOps<PowOp>(funcOp), 1U);
}

TEST(FuseSingleQubitUnitaryRunsTest, MergesShortDynamicSameAxisRun) {
  TestFixture fx;
  fx.setUp();
  auto owned = QCOProgramBuilder::build(fx.ctx(), [](QCOProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    q[0] = b.rz(0.3, q[0]);
    q[0] = b.rz(0.4, q[0]);
    return measureAndReturn(b, q.qubits);
  });
  ASSERT_TRUE(owned);

  auto funcOp = owned->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(funcOp);
  funcOp.insertArgument(0, Float64Type::get(fx.ctx()), {}, funcOp.getLoc());
  funcOp.insertArgument(1, Float64Type::get(fx.ctx()), {}, funcOp.getLoc());
  SmallVector<RZOp> rotations;
  funcOp.walk([&](RZOp op) { rotations.push_back(op); });
  ASSERT_EQ(rotations.size(), 2U);
  rotations[0].getThetaMutable().assign(funcOp.getArgument(0));
  rotations[1].getThetaMutable().assign(funcOp.getArgument(1));

  ASSERT_TRUE(succeeded(runFuse(*owned, "zyz")));
  rotations.clear();
  funcOp.walk([&](RZOp op) { rotations.push_back(op); });
  ASSERT_EQ(rotations.size(), 1U);
  EXPECT_TRUE(
      valueDependsOn(rotations.front().getTheta(), funcOp.getArgument(0)));
  EXPECT_TRUE(
      valueDependsOn(rotations.front().getTheta(), funcOp.getArgument(1)));

  bindLeadingArguments(funcOp, {0.3, 0.4});
  ASSERT_TRUE(succeeded(canonicalizeBoundValues(*owned)));
  ASSERT_TRUE(succeeded(verify(*owned)));
  expectMatrixPreserved(
      funcOp, RZOp::unitaryMatrix(0.4) * RZOp::unitaryMatrix(0.3), "zyz");
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesNamedDynamicGatesInAllBases) {
  TestFixture fx;
  fx.setUp();
  constexpr std::array boundValues = {0.37, -0.61, 0.83};

  for (const auto& gateCase : DYNAMIC_GATE_CASES) {
    SCOPED_TRACE(gateCase.name.str());
    forEachBasis([&](StringRef basis) {
      SCOPED_TRACE(basis.str());
      auto owned = QCOProgramBuilder::build(
          fx.ctx(), [build = gateCase.build](QCOProgramBuilder& b) {
            auto q = b.allocQubitRegister(1);
            q[0] = b.h(q[0]);
            q[0] = build(b, q[0]);
            return measureAndReturn(b, q.qubits);
          });
      ASSERT_TRUE(owned);

      ModuleOp mlirModule = *owned;
      auto funcOp = mlirModule.lookupSymbol<func::FuncOp>("main");
      ASSERT_TRUE(funcOp);
      for (size_t i = 0; i < gateCase.numParameters; ++i) {
        funcOp.insertArgument(i, Float64Type::get(fx.ctx()), {},
                              funcOp.getLoc());
      }

      SmallVector<UnitaryOpInterface> parameterizedGates;
      funcOp.walk([&](UnitaryOpInterface unitary) {
        if (unitary.getNumParams() != 0) {
          parameterizedGates.push_back(unitary);
        }
      });
      ASSERT_EQ(parameterizedGates.size(), 1U);
      UnitaryOpInterface parameterizedGate = parameterizedGates.front();
      ASSERT_EQ(parameterizedGate.getBaseSymbol(), gateCase.name);
      ASSERT_EQ(parameterizedGate.getNumParams(), gateCase.numParameters);
      SmallVector<Value> parameters(parameterizedGate.getParameters().begin(),
                                    parameterizedGate.getParameters().end());
      for (auto [index, parameter] : llvm::enumerate(parameters)) {
        parameter.replaceAllUsesWith(funcOp.getArgument(index));
      }
      ASSERT_TRUE(succeeded(verify(mlirModule)));

      OwningOpRef<ModuleOp> original = cast<ModuleOp>(mlirModule->clone());
      ASSERT_TRUE(succeeded(runFuse(mlirModule, basis)));
      ASSERT_TRUE(succeeded(verify(mlirModule)));

      funcOp = mlirModule.lookupSymbol<func::FuncOp>("main");
      ASSERT_TRUE(funcOp);
      const auto parsedBasis = parseSingleQubitBasis(basis);
      ASSERT_TRUE(parsedBasis);
      EXPECT_EQ(countOps<HOp>(funcOp), 0U);
      EXPECT_GT(countBasisGates(funcOp, *parsedBasis), 0U);

      SmallVector<bool> dependsOnParameter(gateCase.numParameters, false);
      auto recordDependencies = [&](ValueRange emittedParameters) {
        for (Value emittedParameter : emittedParameters) {
          for (size_t i = 0; i < gateCase.numParameters; ++i) {
            dependsOnParameter[i] |=
                valueDependsOn(emittedParameter, funcOp.getArgument(i));
          }
        }
      };
      funcOp.walk([&](UnitaryOpInterface unitary) {
        if (!unitary.isSingleQubit()) {
          return;
        }
        EXPECT_TRUE(isAllowedBasisGate(*unitary, *parsedBasis));
        recordDependencies(unitary.getParameters());
      });
      funcOp.walk(
          [&](GPhaseOp phase) { recordDependencies(phase.getParameters()); });
      EXPECT_TRUE(llvm::all_of(dependsOnParameter,
                               [](bool depends) { return depends; }));

      ArrayRef values{boundValues.data(), gateCase.numParameters};
      auto originalFunc = original->lookupSymbol<func::FuncOp>("main");
      ASSERT_TRUE(originalFunc);
      bindLeadingArguments(originalFunc, values);
      bindLeadingArguments(funcOp, values);
      ASSERT_TRUE(succeeded(canonicalizeBoundValues(*original)));
      ASSERT_TRUE(succeeded(canonicalizeBoundValues(mlirModule)));
      ASSERT_TRUE(succeeded(verify(*original)));
      ASSERT_TRUE(succeeded(verify(mlirModule)));
      const Matrix2x2 expected = compute1QUnitaryMatrix(originalFunc.getBody());
      expectMatrixPreserved(funcOp, expected, basis);
    });
  }
}

TEST(FuseSingleQubitUnitaryRunsTest,
     DirectlySynthesizesParameterizedGatesInNativeEulerBases) {
  TestFixture fx;
  fx.setUp();
  constexpr std::array boundValues = {0.37, -0.61, 0.83};
  const std::array directBases = {
      std::pair{StringRef{"u"}, U},
      std::pair{StringRef{"zyz"}, ZYZ},
      std::pair{StringRef{"zxz"}, ZXZ},
      std::pair{StringRef{"zsxx"}, ZSXX},
  };

  for (const auto& gateCase : DYNAMIC_GATE_CASES) {
    SCOPED_TRACE(gateCase.name.str());
    for (const auto& [basisName, basis] : directBases) {
      SCOPED_TRACE(basisName.str());
      auto owned = QCOProgramBuilder::build(
          fx.ctx(), [build = gateCase.build](QCOProgramBuilder& b) {
            auto q = b.allocQubitRegister(1);
            q[0] = build(b, q[0]);
            return measureAndReturn(b, q.qubits);
          });
      ASSERT_TRUE(owned);

      ModuleOp mlirModule = *owned;
      auto funcOp = mlirModule.lookupSymbol<func::FuncOp>("main");
      ASSERT_TRUE(funcOp);
      for (size_t i = 0; i < gateCase.numParameters; ++i) {
        funcOp.insertArgument(i, Float64Type::get(fx.ctx()), {},
                              funcOp.getLoc());
      }

      UnitaryOpInterface parameterizedGate;
      funcOp.walk([&](UnitaryOpInterface unitary) {
        if (unitary.getBaseSymbol() == gateCase.name) {
          parameterizedGate = unitary;
        }
      });
      ASSERT_TRUE(parameterizedGate);
      SmallVector<Value> parameters(parameterizedGate.getParameters().begin(),
                                    parameterizedGate.getParameters().end());
      ASSERT_EQ(parameters.size(), gateCase.numParameters);
      for (auto [index, parameter] : llvm::enumerate(parameters)) {
        parameter.replaceAllUsesWith(funcOp.getArgument(index));
      }
      ASSERT_TRUE(succeeded(verify(mlirModule)));

      OwningOpRef<ModuleOp> original = cast<ModuleOp>(mlirModule->clone());
      IRRewriter rewriter(fx.ctx());
      synthesizeParameterizedUnitary1Q(rewriter,
                                       parameterizedGate.getOperation(), basis);
      ASSERT_TRUE(succeeded(verify(mlirModule)));
      expectDirectSynthesisCounts(
          funcOp, expectedDirectSynthesisCounts(gateCase.name, basis));

      auto originalFunc = original->lookupSymbol<func::FuncOp>("main");
      ASSERT_TRUE(originalFunc);
      ArrayRef values{boundValues.data(), gateCase.numParameters};
      bindLeadingArguments(originalFunc, values);
      bindLeadingArguments(funcOp, values);
      ASSERT_TRUE(succeeded(canonicalizeBoundValues(*original)));
      ASSERT_TRUE(succeeded(canonicalizeBoundValues(mlirModule)));
      ASSERT_TRUE(succeeded(verify(*original)));
      ASSERT_TRUE(succeeded(verify(mlirModule)));
      const Matrix2x2 expected = compute1QUnitaryMatrix(originalFunc.getBody());
      expectMatrixPreserved(funcOp, expected, basisName);
    }
  }
}

TEST(FuseSingleQubitUnitaryRunsTest,
     FusesStandaloneDynamicUAtTransformedBasisSingularities) {
  TestFixture fx;
  fx.setUp();
  constexpr std::array<const char*, 3> bases = {"xzx", "xyx", "r"};
  constexpr std::array<std::array<double, 3>, 2> singularParameterSets{{
      {0.0, 0.0, 0.0},
      {std::numbers::pi, 0.0, 0.0},
  }};
  constexpr size_t numParameters = singularParameterSets.front().size();

  for (const StringRef basis : bases) {
    SCOPED_TRACE(basis.str());
    auto owned = QCOProgramBuilder::build(fx.ctx(), [](QCOProgramBuilder& b) {
      auto q = b.allocQubitRegister(1);
      q[0] = b.u(0.1, 0.2, 0.3, q[0]);
      return measureAndReturn(b, q.qubits);
    });
    ASSERT_TRUE(owned);

    ModuleOp mlirModule = *owned;
    auto funcOp = mlirModule.lookupSymbol<func::FuncOp>("main");
    ASSERT_TRUE(funcOp);
    for (size_t i = 0; i < numParameters; ++i) {
      funcOp.insertArgument(i, Float64Type::get(fx.ctx()), {}, funcOp.getLoc());
    }

    UOp uOp = nullptr;
    funcOp.walk([&](UOp op) { uOp = op; });
    ASSERT_TRUE(uOp);
    SmallVector<Value> parameters(uOp.getParameters().begin(),
                                  uOp.getParameters().end());
    ASSERT_EQ(parameters.size(), numParameters);
    for (auto [index, parameter] : llvm::enumerate(parameters)) {
      parameter.replaceAllUsesWith(funcOp.getArgument(index));
    }
    ASSERT_TRUE(succeeded(verify(mlirModule)));

    OwningOpRef<ModuleOp> original = cast<ModuleOp>(mlirModule->clone());
    ASSERT_TRUE(succeeded(runFuse(mlirModule, basis)));
    ASSERT_TRUE(succeeded(verify(mlirModule)));

    for (const auto& values : singularParameterSets) {
      SCOPED_TRACE(values.front());
      OwningOpRef<ModuleOp> boundOriginal = cast<ModuleOp>(original->clone());
      OwningOpRef<ModuleOp> boundActual = cast<ModuleOp>(mlirModule->clone());
      auto originalFunc = boundOriginal->lookupSymbol<func::FuncOp>("main");
      auto actualFunc = boundActual->lookupSymbol<func::FuncOp>("main");
      ASSERT_TRUE(originalFunc);
      ASSERT_TRUE(actualFunc);
      bindLeadingArguments(originalFunc, values);
      bindLeadingArguments(actualFunc, values);
      ASSERT_TRUE(succeeded(canonicalizeBoundValues(*boundOriginal)));
      ASSERT_TRUE(succeeded(canonicalizeBoundValues(*boundActual)));
      ASSERT_TRUE(succeeded(verify(*boundOriginal)));
      ASSERT_TRUE(succeeded(verify(*boundActual)));
      const Matrix2x2 expected = compute1QUnitaryMatrix(originalFunc.getBody());
      expectMatrixPreserved(actualFunc, expected, basis);
    }
  }
}

TEST(FuseSingleQubitUnitaryRunsTest, FusesProgramsAllBases) {
  TestFixture fx;
  fx.setUp();

  struct Case {
    SmallVector<Value> (*program)(QCOProgramBuilder&);
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
    SmallVector<Value> (*program)(QCOProgramBuilder&);
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
             expectSplitFixtureSegments(funcOp, basis, ctx, [](Operation& op) {
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
             expectSplitFixtureSegments(funcOp, basis, ctx, [](Operation& op) {
               return isa<BarrierOp>(op);
             });
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
