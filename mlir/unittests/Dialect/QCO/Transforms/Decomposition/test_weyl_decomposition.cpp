/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeProfile.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mqt::test;

using QubitId = std::size_t;

static constexpr double K_SANITY_CHECK_PRECISION = 1e-12;

static const Matrix2x2& hGate() {
  static const Matrix2x2 MATRIX = Matrix2x2::fromElements(
      FRAC1_SQRT2, FRAC1_SQRT2, FRAC1_SQRT2, -FRAC1_SQRT2);
  return MATRIX;
}

static Matrix4x4 randomUnitary4x4(std::mt19937& rng) {
  std::normal_distribution<double> normalDist(0.0, 1.0);
  std::vector<std::vector<std::complex<double>>> columns(
      4, std::vector<std::complex<double>>(4));
  for (auto& column : columns) {
    for (auto& entry : column) {
      entry = std::complex<double>(normalDist(rng), normalDist(rng));
    }
  }
  for (std::size_t j = 0; j < 4; ++j) {
    for (std::size_t k = 0; k < j; ++k) {
      std::complex<double> projection{0.0, 0.0};
      for (std::size_t i = 0; i < 4; ++i) {
        projection += std::conj(columns[k][i]) * columns[j][i];
      }
      for (std::size_t i = 0; i < 4; ++i) {
        columns[j][i] -= projection * columns[k][i];
      }
    }
    double norm = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
      norm += std::norm(columns[j][i]);
    }
    norm = std::sqrt(norm);
    for (std::size_t i = 0; i < 4; ++i) {
      columns[j][i] /= norm;
    }
  }
  const Matrix4x4 unitary = Matrix4x4::fromElements(
      columns[0][0], columns[1][0], columns[2][0], columns[3][0], columns[0][1],
      columns[1][1], columns[2][1], columns[3][1], columns[0][2], columns[1][2],
      columns[2][2], columns[3][2], columns[0][3], columns[1][3], columns[2][3],
      columns[3][3]);
  assert((unitary.adjoint() * unitary).isIdentity(1e-12));
  return unitary;
}

static Matrix4x4 restoreWeyl(const TwoQubitWeylDecomposition& decomposition) {
  return kron(decomposition.k1l(), decomposition.k1r()) *
         decomposition.getCanonicalMatrix() *
         kron(decomposition.k2l(), decomposition.k2r()) *
         globalPhaseFactor(decomposition.globalPhase());
}

static Matrix4x4 restoreBasis(const TwoQubitNativeDecomposition& decomposition,
                              const Matrix4x4& entangler) {
  const auto& factors = decomposition.singleQubitFactors;
  const auto layer = [&](std::size_t i) {
    return kron(factors[(2 * i) + 1], factors[2 * i]);
  };
  Matrix4x4 matrix = layer(0);
  for (std::uint8_t i = 0; i < decomposition.numBasisUses; ++i) {
    matrix = entangler * matrix;
    matrix = layer(static_cast<std::size_t>(i) + 1) * matrix;
  }
  return matrix * globalPhaseFactor(decomposition.globalPhase);
}

static auto productMatrixCases() {
  return ::testing::Values(
      []() -> Matrix4x4 { return Matrix4x4::identity(); },
      []() -> Matrix4x4 { return kron(rzMatrix(1.0), ryMatrix(3.1)); },
      []() -> Matrix4x4 { return kron(Matrix2x2::identity(), rxMatrix(0.1)); });
}

static auto entangledMatrixCases() {
  return ::testing::Values(
      []() -> Matrix4x4 { return rzzMatrix(2.0); },
      []() -> Matrix4x4 {
        return ryyMatrix(1.0) * rzzMatrix(3.0) * rxxMatrix(2.0);
      },
      []() -> Matrix4x4 {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(1.5, -0.2, 0.0) *
               kron(rxMatrix(1.0), Matrix2x2::identity());
      },
      []() -> Matrix4x4 {
        return kron(rxMatrix(1.0), ryMatrix(1.0)) *
               TwoQubitWeylDecomposition::getCanonicalMatrix(1.1, 0.2, 3.0) *
               kron(rxMatrix(1.0), Matrix2x2::identity());
      },
      []() -> Matrix4x4 {
        return kron(hGate(), iPauliZ()) * twoQubitControlledX01() *
               kron(iPauliX(), iPauliY());
      });
}

static auto cxBasisCases() {
  return ::testing::Values(
      []() -> Matrix4x4 { return twoQubitControlledX01(); },
      []() -> Matrix4x4 { return twoQubitControlledX10(); });
}

static bool extractSingleQubitMatrix(UnitaryOpInterface op, Matrix2x2& out) {
  if (op.getUnitaryMatrix2x2(out)) {
    return true;
  }
  DynamicMatrix dynamic;
  if (!op.getUnitaryMatrixDynamic(dynamic) || dynamic.rows() != 2 ||
      dynamic.cols() != 2) {
    return false;
  }
  out = Matrix2x2::fromElements(dynamic(0, 0), dynamic(0, 1), dynamic(1, 0),
                                dynamic(1, 1));
  return true;
}

static bool extractTwoQubitMatrix(UnitaryOpInterface op, Matrix4x4& out) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op.getOperation())) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    Operation* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      out = twoQubitControlledX01();
      return true;
    }
    if (llvm::isa<ZOp>(body)) {
      out = twoQubitControlledZ();
      return true;
    }
    return false;
  }
  return op.getUnitaryMatrix4x4(out);
}

static std::optional<Matrix4x4>
computeTwoQubitUnitaryFromFunc(func::FuncOp funcOp) {
  Matrix4x4 unitary = Matrix4x4::identity();
  std::complex<double> global{1.0, 0.0};
  llvm::DenseMap<Value, QubitId> wireIds;
  wireIds[funcOp.getArgument(0)] = 0;
  wireIds[funcOp.getArgument(1)] = 1;

  auto wireId = [&](Value qubit) -> std::optional<QubitId> {
    const auto it = wireIds.find(qubit);
    if (it == wireIds.end()) {
      return std::nullopt;
    }
    return it->second;
  };

  for (Operation& rawOp : funcOp.getBody().front()) {
    if (llvm::isa<func::ReturnOp>(&rawOp)) {
      continue;
    }
    if (auto gphase = llvm::dyn_cast<GPhaseOp>(&rawOp)) {
      if (const auto matrix = gphase.getUnitaryMatrix()) {
        global *= (*matrix)(0, 0);
      }
      continue;
    }
    auto op = llvm::dyn_cast<UnitaryOpInterface>(&rawOp);
    if (!op) {
      continue;
    }

    if (op.isSingleQubit()) {
      const auto qIn = op->getOperand(0);
      const auto qid = wireId(qIn);
      if (!qid) {
        return std::nullopt;
      }
      Matrix2x2 oneQ;
      if (!extractSingleQubitMatrix(op, oneQ)) {
        return std::nullopt;
      }
      unitary = embedSingleQubitInTwoQubit(oneQ, *qid) * unitary;
      wireIds[op->getResult(0)] = *qid;
      continue;
    }

    if (op.isTwoQubit()) {
      const auto q0In = op->getOperand(0);
      const auto q1In = op->getOperand(1);
      const auto q0id = wireId(q0In);
      const auto q1id = wireId(q1In);
      if (!q0id || !q1id) {
        return std::nullopt;
      }
      Matrix4x4 twoQ;
      if (!extractTwoQubitMatrix(op, twoQ)) {
        return std::nullopt;
      }
      const llvm::SmallVector<QubitId, 2> ids{*q0id, *q1id};
      unitary = reorderTwoQubitMatrix(twoQ, ids[0], ids[1]) * unitary;
      wireIds[op->getResult(0)] = *q0id;
      wireIds[op->getResult(1)] = *q1id;
    }
  }

  return unitary * global;
}

static func::FuncOp synthesize2QIntoFunc(MLIRContext* ctx,
                                         const Matrix4x4& target,
                                         const NativeProfileSpec& spec,
                                         OwningOpRef<ModuleOp>& moduleOut) {
  moduleOut = ModuleOp::create(UnknownLoc::get(ctx));
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(moduleOut->getBody());

  const auto qubitTy = QubitType::get(ctx);
  const auto funcTy =
      builder.getFunctionType({qubitTy, qubitTy}, {qubitTy, qubitTy});
  const Location loc = moduleOut->getLoc();
  auto func = func::FuncOp::create(builder, loc, "main", funcTy);
  auto* entry = func.addEntryBlock();

  builder.setInsertionPointToStart(entry);
  Value out0;
  Value out1;
  if (failed(synthesizeUnitary2QWeyl(builder, loc, entry->getArgument(0),
                                     entry->getArgument(1), target, spec, out0,
                                     out1))) {
    llvm::report_fatal_error(
        "synthesizeUnitary2QWeyl failed during test synthesis");
  }
  func::ReturnOp::create(builder, loc, ValueRange{out0, out1});
  return func;
}

static void expectSynthesized2QMatrix(MLIRContext* ctx, const Matrix4x4& target,
                                      const NativeProfileSpec& spec) {
  OwningOpRef<ModuleOp> module;
  const auto func = synthesize2QIntoFunc(ctx, target, spec, module);
  ASSERT_TRUE(succeeded(verify(module.get())));
  const auto actual = computeTwoQubitUnitaryFromFunc(func);
  ASSERT_TRUE(actual.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*actual, target));
}

TEST(DecompositionHelpersTest, MatrixUtilitySanity) {
  EXPECT_DOUBLE_EQ(remEuclid(-1.0, 3.0), 2.0);
  EXPECT_DOUBLE_EQ(traceToFidelity(std::complex<double>{3.0, 4.0}),
                   (4.0 + 25.0) / 20.0);
  EXPECT_NEAR(std::abs(globalPhaseFactor(1.25)), 1.0, 1e-14);
  EXPECT_FALSE(isUnitaryMatrix(Matrix2x2::fromElements(2.0, 0.0, 0.0, 2.0)));
  EXPECT_TRUE(isUnitaryMatrix(Matrix2x2::identity()));
}

namespace {

struct MlirTestContext {
  std::unique_ptr<MLIRContext> context;

  void setUp() {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] MLIRContext* ctx() const { return context.get(); }
};

class WeylDecompositionTest : public testing::TestWithParam<Matrix4x4 (*)()> {};

class BasisDecomposerTest : public testing::TestWithParam<
                                std::tuple<Matrix4x4 (*)(), Matrix4x4 (*)()>> {
protected:
  void SetUp() override {
    basisMatrix = std::get<0>(GetParam())();
    target = std::get<1>(GetParam())();
    targetDecomposition = std::make_unique<TwoQubitWeylDecomposition>(
        TwoQubitWeylDecomposition::create(target, 1.0));
  }

  Matrix4x4 target;
  Matrix4x4 basisMatrix;
  std::unique_ptr<TwoQubitWeylDecomposition> targetDecomposition;
};

struct WeylSynthesisCase {
  const char* name;
  const char* nativeGates;
  Matrix4x4 (*target)();
};

class WeylSynthesisTest : public testing::TestWithParam<WeylSynthesisCase> {};

} // namespace

TEST_P(WeylDecompositionTest, ReconstructsWithinRequestedFidelity) {
  const Matrix4x4 originalMatrix = GetParam()();
  for (const double fidelity : {1.0, 1.0 - 1e-12}) {
    const auto decomposition =
        TwoQubitWeylDecomposition::create(originalMatrix, fidelity);
    EXPECT_TRUE(restoreWeyl(decomposition).isApprox(originalMatrix));
  }
}

TEST(WeylDecompositionStandalone,
     CnotProducesValidWeylParametersAndUnitaryLocals) {
  const Matrix4x4 cnot =
      Matrix4x4::fromElements(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0);
  const auto decomp = TwoQubitWeylDecomposition::create(cnot, std::nullopt);
  constexpr double piOver4 = 0.7853981633974483;
  for (const double angle : {decomp.a(), decomp.b(), decomp.c()}) {
    EXPECT_GE(angle, -1e-10);
    EXPECT_LE(angle, piOver4 + 1e-10);
  }
  EXPECT_TRUE(isUnitaryMatrix(decomp.k1l()));
  EXPECT_TRUE(isUnitaryMatrix(decomp.k2l()));
  EXPECT_TRUE(isUnitaryMatrix(decomp.k1r()));
  EXPECT_TRUE(isUnitaryMatrix(decomp.k2r()));
}

TEST(WeylDecompositionStandalone, Random) {
  std::mt19937 rng{1234567UL};
  for (int i = 0; i < 5000; ++i) {
    const Matrix4x4 originalMatrix = randomUnitary4x4(rng);
    const auto decomposition = TwoQubitWeylDecomposition::create(
        originalMatrix, std::optional<double>{1.0 - 1e-12});
    EXPECT_TRUE(restoreWeyl(decomposition)
                    .isApprox(originalMatrix, K_SANITY_CHECK_PRECISION));
  }
}

INSTANTIATE_TEST_SUITE_P(ProductTwoQubitMatrices, WeylDecompositionTest,
                         productMatrixCases());
INSTANTIATE_TEST_SUITE_P(TwoQubitMatrices, WeylDecompositionTest,
                         entangledMatrixCases());

TEST_P(BasisDecomposerTest, ReconstructsWithinRequestedFidelity) {
  for (const double fidelity : {1.0, 1.0 - 1e-12}) {
    const auto decomposer =
        TwoQubitBasisDecomposer::create(basisMatrix, fidelity);
    const auto decomposed =
        decomposer.twoQubitDecompose(*targetDecomposition, std::nullopt);
    ASSERT_TRUE(decomposed.has_value());
    EXPECT_TRUE(restoreBasis(*decomposed, basisMatrix).isApprox(target));
  }
}

TEST(BasisDecomposerTest, Random) {
  std::mt19937 rng{123456UL};
  const llvm::SmallVector<Matrix4x4, 2> basisMatrices{twoQubitControlledX01(),
                                                      twoQubitControlledX10()};
  std::uniform_int_distribution<std::size_t> distBasisGate{0, 1};

  for (int i = 0; i < 2000; ++i) {
    const Matrix4x4 originalMatrix = randomUnitary4x4(rng);
    const auto targetDecomposition = TwoQubitWeylDecomposition::create(
        originalMatrix, std::optional<double>{1.0});
    const Matrix4x4 basisMatrix = basisMatrices[distBasisGate(rng)];
    const auto decomposer = TwoQubitBasisDecomposer::create(basisMatrix, 1.0);
    const auto decomposed =
        decomposer.twoQubitDecompose(targetDecomposition, std::nullopt);
    ASSERT_TRUE(decomposed.has_value());
    EXPECT_TRUE(restoreBasis(*decomposed, basisMatrix)
                    .isApprox(originalMatrix, K_SANITY_CHECK_PRECISION));
  }
}

TEST(BasisDecomposerNumBasisTest, ForcesZeroBasisUsesForIdentityTarget) {
  const Matrix4x4 basis = twoQubitControlledX01();
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const Matrix4x4 target = Matrix4x4::identity();
  const auto weyl = TwoQubitWeylDecomposition::create(target, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{0});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 0);
  EXPECT_TRUE(restoreBasis(*decomposed, basis).isApprox(target));
}

INSTANTIATE_TEST_SUITE_P(ProductTwoQubitMatrices, BasisDecomposerTest,
                         testing::Combine(cxBasisCases(),
                                          productMatrixCases()));
INSTANTIATE_TEST_SUITE_P(TwoQubitMatrices, BasisDecomposerTest,
                         testing::Combine(cxBasisCases(),
                                          entangledMatrixCases()));

TEST_P(WeylSynthesisTest, PreservesTargetUnitary) {
  MlirTestContext fx;
  fx.setUp();
  const auto spec = parseNativeSpec(GetParam().nativeGates);
  ASSERT_TRUE(spec);
  expectSynthesized2QMatrix(fx.ctx(), GetParam().target(), *spec);
}

INSTANTIATE_TEST_SUITE_P(
    Profiles, WeylSynthesisTest,
    testing::Values(
        WeylSynthesisCase{"CxGeneric", "u,cx",
                          [] { return twoQubitControlledX01(); }},
        WeylSynthesisCase{"ProductGeneric", "u,cx",
                          [] { return kron(rzMatrix(1.0), ryMatrix(0.3)); }},
        WeylSynthesisCase{"IbmBasic", "x,sx,rz,cx",
                          [] {
                            return kron(hGate(), Matrix2x2::identity()) *
                                   twoQubitControlledX01() *
                                   kron(rzMatrix(0.2), ryMatrix(0.1));
                          }}),
    [](const testing::TestParamInfo<WeylSynthesisCase>& info) {
      return info.param.name;
    });

TEST(WeylSynthesisTest, IdentityRequiresNoEntanglers) {
  const auto spec = parseNativeSpec("u,cx");
  ASSERT_TRUE(spec);
  const auto count = twoQubitEntanglerCount(Matrix4x4::identity(), *spec);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 0U);
}

TEST(WeylSynthesisTest, FailsWithoutEntanglerInSpec) {
  MlirTestContext fx;
  fx.setUp();
  const auto spec = parseNativeSpec("u");
  ASSERT_TRUE(spec);
  EXPECT_TRUE(spec->entanglerBases.empty());
  OpBuilder builder(fx.ctx());
  const auto qubitTy = QubitType::get(fx.ctx());
  const auto funcTy =
      builder.getFunctionType({qubitTy, qubitTy}, {qubitTy, qubitTy});
  auto func =
      func::FuncOp::create(builder, UnknownLoc::get(fx.ctx()), "main", funcTy);
  auto* entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value out0;
  Value out1;
  EXPECT_TRUE(failed(synthesizeUnitary2QWeyl(
      builder, func.getLoc(), entry->getArgument(0), entry->getArgument(1),
      twoQubitControlledX01(), *spec, out0, out1)));
}

TEST(NativeSpecTest, ParsesAndRejectsMenus) {
  const auto ibm = parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(ibm);
  EXPECT_TRUE(ibm->allowedGates.contains(NativeGateKind::Cx));
  EXPECT_TRUE(ibm->allowedGates.contains(NativeGateKind::X));
  EXPECT_FALSE(ibm->allowRzz);
  EXPECT_FALSE(parseNativeSpec("x,sx,rz,not-a-gate").has_value());

  const auto pMenu = parseNativeSpec("x,sx,p,cx");
  const auto rzMenu = parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(pMenu);
  ASSERT_TRUE(rzMenu);
  EXPECT_EQ(pMenu->allowedGates, rzMenu->allowedGates);

  const auto cxOnly = parseNativeSpec("u,cx");
  ASSERT_TRUE(cxOnly);
  EXPECT_TRUE(llvm::is_contained(cxOnly->entanglerBases, EntanglerBasis::Cx));
  EXPECT_FALSE(llvm::is_contained(cxOnly->entanglerBases, EntanglerBasis::Cz));

  const auto both = parseNativeSpec("u,cx,cz");
  ASSERT_TRUE(both);
  EXPECT_TRUE(llvm::is_contained(both->entanglerBases, EntanglerBasis::Cx));
  EXPECT_TRUE(llvm::is_contained(both->entanglerBases, EntanglerBasis::Cz));

  const auto generic = parseNativeSpec("u,cx");
  ASSERT_TRUE(generic);
  EXPECT_TRUE(generic->allowedGates.contains(NativeGateKind::U));
  EXPECT_FALSE(generic->allowedGates.contains(NativeGateKind::X));
}
