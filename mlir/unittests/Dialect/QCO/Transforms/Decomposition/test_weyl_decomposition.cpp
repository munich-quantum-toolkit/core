/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Package.hpp"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeGateset.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
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
using namespace mlir::qco;
using namespace mlir::qco::decomposition;

static constexpr Matrix4x4 TWO_QUBIT_CONTROLLED_X01 =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, //
                            0.0, 1.0, 0.0, 0.0, //
                            0.0, 0.0, 0.0, 1.0, //
                            0.0, 0.0, 1.0, 0.0);

static constexpr Matrix4x4 TWO_QUBIT_CONTROLLED_X10 =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, //
                            0.0, 0.0, 0.0, 1.0, //
                            0.0, 0.0, 1.0, 0.0, //
                            0.0, 1.0, 0.0, 0.0);

static const Matrix4x4 TWO_QUBIT_CONTROLLED_Z =
    Matrix4x4::fromDiagonal({1, 1, 1, -1});

[[nodiscard]] static bool
isUnitaryMatrix(const auto& matrix, const double tolerance = MATRIX_TOLERANCE) {
  return (matrix.adjoint() * matrix).isIdentity(tolerance);
}

static Matrix4x4 randomUnitary4x4(std::mt19937& rng) {
  std::normal_distribution normalDist(0.0, 1.0);
  std::vector columns(4, std::vector(4, std::complex{0.0, 0.0}));
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
  const auto unitary = Matrix4x4::fromElements(
      columns[0][0], columns[1][0], columns[2][0], columns[3][0], columns[0][1],
      columns[1][1], columns[2][1], columns[3][1], columns[0][2], columns[1][2],
      columns[2][2], columns[3][2], columns[0][3], columns[1][3], columns[2][3],
      columns[3][3]);
  assert(isUnitaryMatrix(unitary, WEYL_TOLERANCE));
  return unitary;
}

static auto productMatrixCases() {
  return ::testing::Values([]() { return Matrix4x4::identity(); },
                           []() {
                             return Matrix4x4::kron(RZOp::unitaryMatrix(1.0),
                                                    RYOp::unitaryMatrix(3.1));
                           },
                           []() {
                             return Matrix4x4::kron(Matrix2x2::identity(),
                                                    RXOp::unitaryMatrix(0.1));
                           });
}

static auto entangledMatrixCases() {
  return ::testing::Values(
      []() { return RZZOp::unitaryMatrix(2.0); },
      []() {
        return RYYOp::unitaryMatrix(1.0) * RZZOp::unitaryMatrix(3.0) *
               RXXOp::unitaryMatrix(2.0);
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(1.5, -0.2, 0.0) *
               Matrix4x4::kron(RXOp::unitaryMatrix(1.0), Matrix2x2::identity());
      },
      []() {
        return Matrix4x4::kron(RXOp::unitaryMatrix(1.0),
                               RYOp::unitaryMatrix(1.0)) *
               TwoQubitWeylDecomposition::getCanonicalMatrix(1.1, 0.2, 3.0) *
               Matrix4x4::kron(RXOp::unitaryMatrix(1.0), Matrix2x2::identity());
      },
      []() {
        return Matrix4x4::kron(HOp::getUnitaryMatrix(),
                               Complex{0.0, 1.0} * ZOp::getUnitaryMatrix()) *
               TWO_QUBIT_CONTROLLED_X01 *
               Matrix4x4::kron(Complex{0.0, 1.0} * XOp::getUnitaryMatrix(),
                               Complex{0.0, 1.0} * YOp::getUnitaryMatrix());
      });
}

static auto cxBasisCases() {
  return ::testing::Values([]() { return TWO_QUBIT_CONTROLLED_X01; },
                           []() { return TWO_QUBIT_CONTROLLED_X10; });
}

static auto specializedMatrixCases() {
  return ::testing::Values(
      []() {
        return TWO_QUBIT_CONTROLLED_X01 * TWO_QUBIT_CONTROLLED_X10 *
               TWO_QUBIT_CONTROLLED_X01;
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, 0.5);
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, -0.5);
      },
      []() { return TWO_QUBIT_CONTROLLED_X01 * TWO_QUBIT_CONTROLLED_X10; },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, 0.1);
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.1, 0.1);
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.1, -0.1);
      });
}

TEST(DecompositionHelpersTest, MatrixUtilitySanity) {
  EXPECT_NEAR(std::abs(std::polar(1.0, 1.25)), 1.0, 1e-14);
  EXPECT_FALSE(isUnitaryMatrix(Matrix2x2::fromElements(2.0, 0.0, 0.0, 2.0)));
  EXPECT_TRUE(isUnitaryMatrix(Matrix2x2::identity()));
}

TEST(DecompositionHelpersTest, GateMatrixFactoriesMatchCanonicalForm) {
  for (const double theta : {0.0, 0.25, 1.0, 2.5, -1.3}) {
    EXPECT_TRUE(RXXOp::unitaryMatrix(theta).isApprox(
        TwoQubitWeylDecomposition::getCanonicalMatrix(-theta / 2.0, 0.0, 0.0),
        WEYL_TOLERANCE));
    EXPECT_TRUE(RYYOp::unitaryMatrix(theta).isApprox(
        TwoQubitWeylDecomposition::getCanonicalMatrix(0.0, -theta / 2.0, 0.0),
        WEYL_TOLERANCE));
    EXPECT_TRUE(RZZOp::unitaryMatrix(theta).isApprox(
        TwoQubitWeylDecomposition::getCanonicalMatrix(0.0, 0.0, -theta / 2.0),
        WEYL_TOLERANCE));
  }
}

TEST(DecompositionHelpersTest, CanonicalMatrixMatchesGateProduct) {
  for (const auto& [a, b, c] : {std::tuple{0.3, 0.2, 0.1},
                                {0.5, 0.5, 0.5},
                                {0.5, 0.1, -0.1},
                                {1.1, 0.2, 3.0},
                                {-0.2, 0.3, 0.4}}) {
    const auto fromGates = RZZOp::unitaryMatrix(-2.0 * c) *
                           RYYOp::unitaryMatrix(-2.0 * b) *
                           RXXOp::unitaryMatrix(-2.0 * a);
    EXPECT_TRUE(TwoQubitWeylDecomposition::getCanonicalMatrix(a, b, c).isApprox(
        fromGates, WEYL_TOLERANCE));
  }
}

namespace {

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

} // namespace

TEST_P(WeylDecompositionTest, ReconstructsWithinRequestedFidelity) {
  const Matrix4x4 originalMatrix = GetParam()();
  for (const double fidelity : {1.0, WEYL_DEFAULT_FIDELITY}) {
    const auto decomposition =
        TwoQubitWeylDecomposition::create(originalMatrix, fidelity);
    EXPECT_TRUE(
        decomposition.unitaryMatrix().isApprox(originalMatrix, WEYL_TOLERANCE));
  }
}

TEST(WeylDecompositionStandalone,
     CnotProducesValidWeylParametersAndUnitaryLocals) {
  const auto decomp =
      TwoQubitWeylDecomposition::create(TWO_QUBIT_CONTROLLED_X01, std::nullopt);
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
        originalMatrix, std::optional<double>{WEYL_DEFAULT_FIDELITY});
    EXPECT_TRUE(
        decomposition.unitaryMatrix().isApprox(originalMatrix, WEYL_TOLERANCE));
  }
}

INSTANTIATE_TEST_SUITE_P(ProductTwoQubitMatrices, WeylDecompositionTest,
                         productMatrixCases());
INSTANTIATE_TEST_SUITE_P(TwoQubitMatrices, WeylDecompositionTest,
                         entangledMatrixCases());
INSTANTIATE_TEST_SUITE_P(SpecializedMatrices, WeylDecompositionTest,
                         specializedMatrixCases());

TEST_P(BasisDecomposerTest, ReconstructsWithinRequestedFidelity) {
  for (const double fidelity : {1.0, WEYL_DEFAULT_FIDELITY}) {
    const auto decomposer =
        TwoQubitBasisDecomposer::create(basisMatrix, fidelity);
    const auto decomposed =
        decomposer.twoQubitDecompose(*targetDecomposition, std::nullopt);
    ASSERT_TRUE(decomposed.has_value());
    EXPECT_TRUE(unitaryMatrix(*decomposed, basisMatrix)
                    .isApprox(target, WEYL_TOLERANCE));
  }
}

TEST(BasisDecomposerTest, Random) {
  std::mt19937 rng{123456UL};
  const mlir::SmallVector<Matrix4x4, 2> basisMatrices{TWO_QUBIT_CONTROLLED_X01,
                                                      TWO_QUBIT_CONTROLLED_X10};
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
    EXPECT_TRUE(unitaryMatrix(*decomposed, basisMatrix)
                    .isApprox(originalMatrix, WEYL_TOLERANCE));
  }
}

TEST(BasisDecomposerNumBasisTest, ForcesZeroBasisUsesForIdentityTarget) {
  const Matrix4x4 basis = TWO_QUBIT_CONTROLLED_X01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const Matrix4x4 target = Matrix4x4::identity();
  const auto weyl = TwoQubitWeylDecomposition::create(target, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{0});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 0);
  EXPECT_TRUE(
      unitaryMatrix(*decomposed, basis).isApprox(target, WEYL_TOLERANCE));
}

TEST(BasisDecomposerTest, DecomposeTwoQubitWithBasisReconstructsTarget) {
  const Matrix4x4 basis = TWO_QUBIT_CONTROLLED_X01;
  const Matrix4x4 target =
      Matrix4x4::kron(RXOp::unitaryMatrix(0.4), RYOp::unitaryMatrix(0.6)) *
      TwoQubitWeylDecomposition::getCanonicalMatrix(0.3, 0.2, 0.1) *
      Matrix4x4::kron(RZOp::unitaryMatrix(0.2), Matrix2x2::identity());
  const auto decomposed = decomposeTwoQubitWithBasis(target, basis);
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_TRUE(
      unitaryMatrix(*decomposed, basis).isApprox(target, WEYL_TOLERANCE));
}

TEST(BasisDecomposerTest, CachedDecomposerMatchesOneShotAcrossTargets) {
  const Matrix4x4 basis = TWO_QUBIT_CONTROLLED_X01;
  const auto cachedDecomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const mlir::SmallVector<Matrix4x4, 3> targets{
      Matrix4x4::identity(),
      TWO_QUBIT_CONTROLLED_X01,
      Matrix4x4::kron(RXOp::unitaryMatrix(0.2), RYOp::unitaryMatrix(0.3)) *
          TwoQubitWeylDecomposition::getCanonicalMatrix(0.1, 0.2, 0.3) *
          Matrix4x4::kron(RZOp::unitaryMatrix(0.1), Matrix2x2::identity()),
  };
  for (const Matrix4x4& target : targets) {
    const auto oneShot = decomposeTwoQubitWithBasis(target, basis);
    const auto cached = cachedDecomposer.decomposeTarget(target);
    ASSERT_TRUE(oneShot.has_value());
    ASSERT_TRUE(cached.has_value());
    EXPECT_TRUE(
        unitaryMatrix(*oneShot, basis).isApprox(target, WEYL_TOLERANCE));
    EXPECT_TRUE(unitaryMatrix(*cached, basis).isApprox(target, WEYL_TOLERANCE));
    EXPECT_EQ(cached->numBasisUses, oneShot->numBasisUses);
    EXPECT_EQ(cached->singleQubitFactors.size(),
              oneShot->singleQubitFactors.size());
  }
}

TEST(BasisDecomposerTest, RejectsMultipleBasisUsesForNonSuperControlledBasis) {
  const Matrix4x4 basis = RZZOp::unitaryMatrix(1.0);
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(Matrix4x4::identity(), 1.0);
  EXPECT_FALSE(decomposer.twoQubitDecompose(weyl, std::uint8_t{2}).has_value());
}

TEST(BasisDecomposerTest, RejectsInvalidBasisGateUseCount) {
  const Matrix4x4 basis = TWO_QUBIT_CONTROLLED_X01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(TWO_QUBIT_CONTROLLED_X01, 1.0);
  EXPECT_FALSE(decomposer.twoQubitDecompose(weyl, std::uint8_t{4}).has_value());
}

TEST(BasisDecomposerForcedCountTest, OneBasisUseProducesFactors) {
  const Matrix4x4 basis = TWO_QUBIT_CONTROLLED_X01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(TWO_QUBIT_CONTROLLED_X01, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{1});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 1);
  EXPECT_EQ(decomposed->singleQubitFactors.size(),
            singleQubitFactorCount(decomposed->numBasisUses));
}

TEST(BasisDecomposerForcedCountTest, TwoBasisUsesProducesFactors) {
  const Matrix4x4 basis = TWO_QUBIT_CONTROLLED_X01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(TWO_QUBIT_CONTROLLED_X01, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{2});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 2);
  EXPECT_EQ(decomposed->singleQubitFactors.size(),
            singleQubitFactorCount(decomposed->numBasisUses));
}

TEST(BasisDecomposerForcedCountTest, ThreeBasisUsesProducesFactors) {
  const Matrix4x4 basis = TWO_QUBIT_CONTROLLED_X01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(TWO_QUBIT_CONTROLLED_X01, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{3});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 3);
  EXPECT_EQ(decomposed->singleQubitFactors.size(),
            singleQubitFactorCount(decomposed->numBasisUses));
}

TEST(WeylDecompositionStandalone, SwapNegativeCSpecializationReconstructs) {
  constexpr double piOver4 = std::numbers::pi / 4.0;
  const Matrix4x4 swapNegativeC =
      TwoQubitWeylDecomposition::getCanonicalMatrix(piOver4, piOver4, -piOver4);
  const auto decomposition =
      TwoQubitWeylDecomposition::create(swapNegativeC, 1.0);
  EXPECT_TRUE(
      decomposition.unitaryMatrix().isApprox(swapNegativeC, WEYL_TOLERANCE));
}

TEST(WeylDecompositionStandalone, ControlledSpecializationReconstructs) {
  const Matrix4x4 controlledLike =
      Matrix4x4::kron(RXOp::unitaryMatrix(0.3), RYOp::unitaryMatrix(0.4)) *
      TwoQubitWeylDecomposition::getCanonicalMatrix(0.6, 0.0, 0.0) *
      Matrix4x4::kron(Matrix2x2::identity(), RZOp::unitaryMatrix(0.2));
  const auto decomposition =
      TwoQubitWeylDecomposition::create(controlledLike, 1.0);
  EXPECT_TRUE(
      decomposition.unitaryMatrix().isApprox(controlledLike, WEYL_TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(ProductTwoQubitMatrices, BasisDecomposerTest,
                         testing::Combine(cxBasisCases(),
                                          productMatrixCases()));
INSTANTIATE_TEST_SUITE_P(TwoQubitMatrices, BasisDecomposerTest,
                         testing::Combine(cxBasisCases(),
                                          entangledMatrixCases()));

namespace {

struct Synthesized2QCircuit {
  OwningOpRef<ModuleOp> mlirModule;
  func::FuncOp func;
};

} // namespace

[[nodiscard]] static FailureOr<Matrix4x4>
computeTwoQubitUnitaryFromFunc(func::FuncOp funcOp) {
  auto dd = std::make_unique<dd::Package>(2);
  auto u = buildFunctionality(funcOp, *dd);
  if (failed(u)) {
    return failure();
  }
  // `getMatrix` is DD/LSB-first; QCO is MSB-first — index `1↔2` swaps the
  // middle basis states (`|01>` ↔ `|10>`).
  const auto& m = u->getMatrix(2);
  const Matrix4x4 matrix = Matrix4x4::fromElements(
      m[0][0], m[0][2], m[0][1], m[0][3], m[2][0], m[2][2], m[2][1], m[2][3],
      m[1][0], m[1][2], m[1][1], m[1][3], m[3][0], m[3][2], m[3][1], m[3][3]);
  dd->decRef(*u);
  return matrix;
}

[[nodiscard]] static Synthesized2QCircuit
synthesize2QMatrix(MLIRContext* ctx, const Matrix4x4& target,
                   const NativeGateset& spec) {
  OwningOpRef mlirModule = ModuleOp::create(UnknownLoc::get(ctx));
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(mlirModule->getBody());

  const auto qubitTy = QubitType::get(ctx);
  const auto funcTy =
      builder.getFunctionType({qubitTy, qubitTy}, {qubitTy, qubitTy});
  const Location loc = mlirModule->getLoc();
  auto func = func::FuncOp::create(builder, loc, "main", funcTy);
  auto* entry = func.addEntryBlock();

  builder.setInsertionPointToStart(entry);
  Value out0;
  if (Value out1; failed(synthesizeUnitary2QWeyl(
          builder, loc, entry->getArgument(0), entry->getArgument(1), target,
          spec, out0, out1))) {
    ADD_FAILURE() << "synthesizeUnitary2QWeyl failed during test synthesis";
  } else {
    func::ReturnOp::create(builder, loc, ValueRange{out0, out1});
  }
  return {.mlirModule = std::move(mlirModule), .func = func};
}

static void expectSynthesized2QMatrix(MLIRContext* ctx, const Matrix4x4& target,
                                      const NativeGateset& spec) {
  const auto circuit = synthesize2QMatrix(ctx, target, spec);
  ASSERT_TRUE(succeeded(verify(*circuit.mlirModule)));
  const auto actual = computeTwoQubitUnitaryFromFunc(circuit.func);
  ASSERT_TRUE(succeeded(actual));
  EXPECT_TRUE(actual->isApprox(target, WEYL_TOLERANCE));
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

struct WeylSynthesisCase {
  const char* name;
  const char* nativeGates;
  Matrix4x4 (*target)();
};

class WeylSynthesisTest : public testing::TestWithParam<WeylSynthesisCase> {
protected:
  MlirTestContext mlir;

  void SetUp() override { mlir.setUp(); }
};

class NativeGatesetMlirTest : public testing::Test {
protected:
  MlirTestContext mlir;

  void SetUp() override { mlir.setUp(); }
};

} // namespace

TEST_P(WeylSynthesisTest, PreservesTargetUnitary) {
  const auto spec = NativeGateset::parse(GetParam().nativeGates);
  ASSERT_TRUE(spec);
  expectSynthesized2QMatrix(mlir.ctx(), GetParam().target(), *spec);
}

INSTANTIATE_TEST_SUITE_P(
    Gatesets, WeylSynthesisTest,
    testing::Values(
        WeylSynthesisCase{"CxGeneric", "u,cx",
                          [] { return TWO_QUBIT_CONTROLLED_X01; }},
        WeylSynthesisCase{"ProductGeneric", "u,cx",
                          [] {
                            return Matrix4x4::kron(RZOp::unitaryMatrix(1.0),
                                                   RYOp::unitaryMatrix(0.3));
                          }},
        WeylSynthesisCase{"IbmBasic", "x,sx,rz,cx",
                          [] {
                            return Matrix4x4::kron(HOp::getUnitaryMatrix(),
                                                   Matrix2x2::identity()) *
                                   TWO_QUBIT_CONTROLLED_X01 *
                                   Matrix4x4::kron(RZOp::unitaryMatrix(0.2),
                                                   RYOp::unitaryMatrix(0.1));
                          }},
        WeylSynthesisCase{
            "RxxGeneric", "u,rxx",
            [] { return RXXOp::unitaryMatrix(std::numbers::pi / 2.0); }},
        WeylSynthesisCase{
            "RyyGeneric", "u,ryy",
            [] { return RYYOp::unitaryMatrix(std::numbers::pi / 2.0); }},
        WeylSynthesisCase{
            "RzxGeneric", "u,rzx",
            [] { return RZXOp::unitaryMatrix(std::numbers::pi / 2.0); }},
        WeylSynthesisCase{
            "RzzGeneric", "u,rzz",
            [] { return RZZOp::unitaryMatrix(std::numbers::pi / 2.0); }},
        WeylSynthesisCase{"IswapGeneric", "u,iswap",
                          [] { return iSWAPOp::getUnitaryMatrix(); }},
        WeylSynthesisCase{"CzGeneric", "u,cz",
                          [] { return TWO_QUBIT_CONTROLLED_Z; }},
        WeylSynthesisCase{"EcrGeneric", "u,ecr",
                          [] { return ECROp::getUnitaryMatrix(); }}),
    [](const testing::TestParamInfo<WeylSynthesisCase>& info) {
      return info.param.name;
    });

TEST(WeylSynthesisTest, IdentityRequiresNoEntanglers) {
  for (const char* gateset : {"u,rxx", "u,ryy", "u,rzx", "u,rzz", "u,iswap",
                              "u,cz", "u,cx", "u,ecr"}) {
    const auto spec = NativeGateset::parse(gateset);
    ASSERT_TRUE(spec) << gateset;
    const auto native = spec->decomposeTarget(Matrix4x4::identity());
    ASSERT_TRUE(native.has_value()) << gateset;
    EXPECT_EQ(native->numBasisUses, 0U) << gateset;
  }
}

TEST(WeylSynthesisTest, RejectsGatesetWithoutEntangler) {
  EXPECT_FALSE(NativeGateset::parse("u").has_value());
}

TEST_F(NativeGatesetMlirTest, ReconstructionRejectsUnhandledOps) {
  OpBuilder builder(mlir.ctx());
  const Location loc = UnknownLoc::get(mlir.ctx());
  const auto qubitTy = QubitType::get(mlir.ctx());
  const auto funcTy =
      builder.getFunctionType({qubitTy, qubitTy}, {qubitTy, qubitTy});
  auto func = func::FuncOp::create(builder, loc, "main", funcTy);
  auto* entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value q0 = entry->getArgument(0);
  Value q1 = entry->getArgument(1);
  auto meas = MeasureOp::create(builder, loc, q0);
  func::ReturnOp::create(builder, loc, ValueRange{meas.getQubitOut(), q1});
  EXPECT_TRUE(failed(computeTwoQubitUnitaryFromFunc(func)));
}

TEST_F(NativeGatesetMlirTest, SynthesisFailsWithoutEulerBasis) {
  const NativeGateset spec{.gates = {NativeGateKind::CX}};
  OpBuilder builder(mlir.ctx());
  const auto qubitTy = QubitType::get(mlir.ctx());
  const auto funcTy =
      builder.getFunctionType({qubitTy, qubitTy}, {qubitTy, qubitTy});
  auto func = func::FuncOp::create(builder, UnknownLoc::get(mlir.ctx()), "main",
                                   funcTy);
  auto* entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value out0;
  Value out1;
  EXPECT_TRUE(failed(synthesizeUnitary2QWeyl(
      builder, func.getLoc(), entry->getArgument(0), entry->getArgument(1),
      TWO_QUBIT_CONTROLLED_X01, spec, out0, out1)));
}

TEST_F(NativeGatesetMlirTest, SynthesisFailsWithoutEntangler) {
  const NativeGateset spec{.gates = {NativeGateKind::U}};
  OpBuilder builder(mlir.ctx());
  const auto qubitTy = QubitType::get(mlir.ctx());
  const auto funcTy =
      builder.getFunctionType({qubitTy, qubitTy}, {qubitTy, qubitTy});
  auto func = func::FuncOp::create(builder, UnknownLoc::get(mlir.ctx()), "main",
                                   funcTy);
  auto* entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value out0;
  Value out1;
  EXPECT_TRUE(failed(synthesizeUnitary2QWeyl(
      builder, func.getLoc(), entry->getArgument(0), entry->getArgument(1),
      TWO_QUBIT_CONTROLLED_X01, spec, out0, out1)));
}

TEST(WeylSynthesisTest, EntanglerCountFailsWithoutEntangler) {
  const NativeGateset spec{.gates = {NativeGateKind::U}};
  EXPECT_FALSE(spec.decomposeTarget(Matrix4x4::identity()).has_value());
}

TEST(NativeSpecTest, ParsesAndRejectsGatesets) {
  const auto ibm = NativeGateset::parse("x,sx,rz,cx");
  ASSERT_TRUE(ibm);
  EXPECT_TRUE(ibm->gates.contains(NativeGateKind::CX));
  EXPECT_TRUE(ibm->gates.contains(NativeGateKind::X));
  EXPECT_FALSE(NativeGateset::parse("x,sx,rz,not-a-gate").has_value());
  EXPECT_FALSE(NativeGateset::parse("u").has_value());

  const auto whitespaceToken = NativeGateset::parse("u, ,cx");
  ASSERT_TRUE(whitespaceToken);
  EXPECT_TRUE(whitespaceToken->gates.contains(NativeGateKind::U));
  EXPECT_TRUE(whitespaceToken->gates.contains(NativeGateKind::CX));

  EXPECT_FALSE(NativeGateset::parse("x,sx,p,cx").has_value());
  EXPECT_FALSE(NativeGateset::parse("ry,p,cz").has_value());

  const auto cxOnly = NativeGateset::parse("u,cx");
  ASSERT_TRUE(cxOnly);
  EXPECT_TRUE(cxOnly->gates.contains(NativeGateKind::U));
  EXPECT_TRUE(cxOnly->gates.contains(NativeGateKind::CX));
  EXPECT_FALSE(cxOnly->gates.contains(NativeGateKind::CZ));
  EXPECT_FALSE(cxOnly->gates.contains(NativeGateKind::X));

  const auto both = NativeGateset::parse("u,cx,cz");
  ASSERT_TRUE(both);
  EXPECT_TRUE(both->gates.contains(NativeGateKind::CX));
  EXPECT_TRUE(both->gates.contains(NativeGateKind::CZ));
  EXPECT_EQ(both->entangler, NativeGateKind::CZ);

  const auto ecrOnly = NativeGateset::parse("u,ecr");
  ASSERT_TRUE(ecrOnly);
  EXPECT_TRUE(ecrOnly->gates.contains(NativeGateKind::ECR));
  EXPECT_EQ(ecrOnly->entangler, NativeGateKind::ECR);

  // With CX/CZ also listed, CX/CZ win over ECR.
  const auto cxCzOverEcr = NativeGateset::parse("u,cz,cx,ecr");
  ASSERT_TRUE(cxCzOverEcr);
  EXPECT_EQ(cxCzOverEcr->entangler, NativeGateKind::CZ);

  const auto iswapOnly = NativeGateset::parse("u,iswap");
  ASSERT_TRUE(iswapOnly);
  EXPECT_TRUE(iswapOnly->gates.contains(NativeGateKind::ISWAP));
  EXPECT_EQ(iswapOnly->entangler, NativeGateKind::ISWAP);

  // iSWAP beats CZ/CX/ECR; two-qubit rotations still win when present.
  const auto iswapOverCtrlEcr = NativeGateset::parse("u,iswap,cz,cx,ecr");
  ASSERT_TRUE(iswapOverCtrlEcr);
  EXPECT_EQ(iswapOverCtrlEcr->entangler, NativeGateKind::ISWAP);

  // DCX is not a supported native-basis token.
  EXPECT_FALSE(NativeGateset::parse("u,dcx").has_value());
  EXPECT_FALSE(NativeGateset::parse("u,cx,dcx").has_value());

  const auto rzxOnly = NativeGateset::parse("u,rzx");
  ASSERT_TRUE(rzxOnly);
  EXPECT_EQ(rzxOnly->entangler, NativeGateKind::RZX);

  // Two-qubit rotations: RXX > RYY > RZX > RZZ (alphabetic).
  const auto rxxOverRest =
      NativeGateset::parse("u,rzx,rzz,ryy,rxx,iswap,cx,cz,ecr");
  ASSERT_TRUE(rxxOverRest);
  EXPECT_EQ(rxxOverRest->entangler, NativeGateKind::RXX);

  const auto ryyOverRzxRzz = NativeGateset::parse("u,rzx,ryy,rzz,iswap,cx");
  ASSERT_TRUE(ryyOverRzxRzz);
  EXPECT_EQ(ryyOverRzxRzz->entangler, NativeGateKind::RYY);

  const auto rzxOverRzz = NativeGateset::parse("u,rzx,rzz,iswap,cx,cz");
  ASSERT_TRUE(rzxOverRzz);
  EXPECT_EQ(rzxOverRzz->entangler, NativeGateKind::RZX);

  const auto rzzOverDiscrete = NativeGateset::parse("u,rzz,iswap,cx,cz,ecr");
  ASSERT_TRUE(rzzOverDiscrete);
  EXPECT_EQ(rzzOverDiscrete->entangler, NativeGateKind::RZZ);

  const auto rzzOnly = NativeGateset::parse("u,rzz");
  ASSERT_TRUE(rzzOnly);
  EXPECT_EQ(rzzOnly->entangler, NativeGateKind::RZZ);

  const auto ryyOnly = NativeGateset::parse("u,ryy");
  ASSERT_TRUE(ryyOnly);
  EXPECT_EQ(ryyOnly->entangler, NativeGateKind::RYY);

  const auto rxxOnly = NativeGateset::parse("u,rxx");
  ASSERT_TRUE(rxxOnly);
  EXPECT_EQ(rxxOnly->entangler, NativeGateKind::RXX);

  const auto rotationThenDiscrete =
      NativeGateset::parse("u,rzz,rxx,iswap,cz,cx,ecr");
  ASSERT_TRUE(rotationThenDiscrete);
  EXPECT_EQ(rotationThenDiscrete->entangler, NativeGateKind::RXX);

  const auto withoutRxx = NativeGateset::parse("u,ryy,iswap,cz,cx,ecr");
  ASSERT_TRUE(withoutRxx);
  EXPECT_EQ(withoutRxx->entangler, NativeGateKind::RYY);

  const auto withoutRotations = NativeGateset::parse("u,iswap,cz,cx,ecr");
  ASSERT_TRUE(withoutRotations);
  EXPECT_EQ(withoutRotations->entangler, NativeGateKind::ISWAP);
}

TEST(NativeSpecTest, RejectsGatesetWithoutSingleQubitStrategy) {
  EXPECT_FALSE(NativeGateset::parse("cx").has_value());
  EXPECT_FALSE(NativeGateset::parse("cz").has_value());
  EXPECT_FALSE(NativeGateset::parse("rx,cx").has_value());
}

TEST(NativeSpecTest, ResolvesEulerBasisFromGateset) {
  const auto uGateset = NativeGateset::parse("u,cx");
  ASSERT_TRUE(uGateset);
  EXPECT_EQ(*uGateset->eulerBasis, EulerBasis::U);

  const auto zsxx = NativeGateset::parse("x,sx,rz,cx");
  ASSERT_TRUE(zsxx);
  EXPECT_EQ(*zsxx->eulerBasis, EulerBasis::ZSXX);

  const auto rGateset = NativeGateset::parse("r,cz");
  ASSERT_TRUE(rGateset);
  EXPECT_EQ(*rGateset->eulerBasis, EulerBasis::R);

  const auto xzx = NativeGateset::parse("rx,rz,cz");
  ASSERT_TRUE(xzx);
  EXPECT_EQ(*xzx->eulerBasis, EulerBasis::XZX);

  const auto xyx = NativeGateset::parse("rx,ry,cz");
  ASSERT_TRUE(xyx);
  EXPECT_EQ(*xyx->eulerBasis, EulerBasis::XYX);

  const auto zyz = NativeGateset::parse("ry,rz,cz");
  ASSERT_TRUE(zyz);
  EXPECT_EQ(*zyz->eulerBasis, EulerBasis::ZYZ);
}

TEST_F(NativeGatesetMlirTest, AllowsOpMatchesGateset) {
  const auto spec = NativeGateset::parse("u,cx");
  ASSERT_TRUE(spec);

  OpBuilder builder(mlir.ctx());
  const Location loc = UnknownLoc::get(mlir.ctx());
  const auto qubitTy = QubitType::get(mlir.ctx());
  const auto funcTy =
      builder.getFunctionType({qubitTy, qubitTy}, {qubitTy, qubitTy});
  auto func = func::FuncOp::create(builder, loc, "allows_op", funcTy);
  auto* entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value q0 = entry->getArgument(0);
  Value q1 = entry->getArgument(1);

  EXPECT_TRUE(spec->allowsOp(
      BarrierOp::create(builder, loc, ValueRange{q0, q1}).getOperation()));
  EXPECT_TRUE(
      spec->allowsOp(GPhaseOp::create(builder, loc, 0.1).getOperation()));
  EXPECT_TRUE(spec->allowsOp(
      UOp::create(builder, loc, q0, 0.1, 0.2, 0.3).getOperation()));

  auto cx = CtrlOp::create(builder, loc, q0, q1, [&](Value target) {
    return XOp::create(builder, loc, target).getOutputQubit(0);
  });
  EXPECT_TRUE(spec->allowsOp(cx.getOperation()));

  auto cxWithInterleavedH =
      CtrlOp::create(builder, loc, q0, q1, [&](Value target) {
        auto wire = XOp::create(builder, loc, target).getOutputQubit(0);
        return HOp::create(builder, loc, wire).getOutputQubit(0);
      });
  EXPECT_FALSE(spec->allowsOp(cxWithInterleavedH.getOperation()));

  EXPECT_FALSE(spec->allowsOp(XOp::create(builder, loc, q0).getOperation()));
  EXPECT_FALSE(
      spec->allowsOp(RXXOp::create(builder, loc, q0, q1, 0.2).getOperation()));
  EXPECT_FALSE(
      spec->allowsOp(ECROp::create(builder, loc, q0, q1).getOperation()));

  const auto rzSpec = NativeGateset::parse("x,sx,rz,cx");
  ASSERT_TRUE(rzSpec);
  EXPECT_TRUE(
      rzSpec->allowsOp(RZOp::create(builder, loc, q0, 0.3).getOperation()));
  EXPECT_FALSE(
      rzSpec->allowsOp(POp::create(builder, loc, q0, 0.3).getOperation()));

  auto hCtrl = CtrlOp::create(builder, loc, q0, q1, [&](Value target) {
    return HOp::create(builder, loc, target).getOutputQubit(0);
  });
  EXPECT_FALSE(spec->allowsOp(hCtrl.getOperation()));

  const auto funcTy3 = builder.getFunctionType({qubitTy, qubitTy, qubitTy},
                                               {qubitTy, qubitTy, qubitTy});
  auto func3 = func::FuncOp::create(builder, loc, "allows_op_ccx", funcTy3);
  auto* entry3 = func3.addEntryBlock();
  builder.setInsertionPointToStart(entry3);
  Value c0 = entry3->getArgument(0);
  Value c1 = entry3->getArgument(1);
  Value target = entry3->getArgument(2);
  auto ccx =
      CtrlOp::create(builder, loc, ValueRange{c0, c1}, target, [&](Value t) {
        return XOp::create(builder, loc, t).getOutputQubit(0);
      });
  EXPECT_FALSE(spec->allowsOp(ccx.getOperation()));

  const auto czSpec = NativeGateset::parse("u,cz");
  ASSERT_TRUE(czSpec);
  auto cz = CtrlOp::create(builder, loc, q0, q1, [&](Value t) {
    return ZOp::create(builder, loc, t).getOutputQubit(0);
  });
  EXPECT_TRUE(czSpec->allowsOp(cz.getOperation()));
  EXPECT_FALSE(czSpec->allowsOp(cx.getOperation()));

  const auto rxxSpec = NativeGateset::parse("u,rxx");
  ASSERT_TRUE(rxxSpec);
  EXPECT_TRUE(rxxSpec->allowsOp(
      RXXOp::create(builder, loc, q0, q1, 0.2).getOperation()));
  EXPECT_TRUE(rxxSpec->allowsOp(
      RXXOp::create(builder, loc, q0, q1, std::numbers::pi / 2.0)
          .getOperation()));

  const auto ryySpec = NativeGateset::parse("u,ryy");
  ASSERT_TRUE(ryySpec);
  EXPECT_TRUE(ryySpec->allowsOp(
      RYYOp::create(builder, loc, q0, q1, 0.25).getOperation()));

  const auto rzxSpec = NativeGateset::parse("u,rzx");
  ASSERT_TRUE(rzxSpec);
  EXPECT_TRUE(rzxSpec->allowsOp(
      RZXOp::create(builder, loc, q0, q1, 0.25).getOperation()));

  const auto rzzSpec = NativeGateset::parse("u,rzz");
  ASSERT_TRUE(rzzSpec);
  EXPECT_TRUE(rzzSpec->allowsOp(
      RZZOp::create(builder, loc, q0, q1, 0.3).getOperation()));

  const auto iswapSpec = NativeGateset::parse("u,iswap");
  ASSERT_TRUE(iswapSpec);
  auto iswap = iSWAPOp::create(builder, loc, q0, q1);
  EXPECT_TRUE(iswapSpec->allowsOp(iswap.getOperation()));
  EXPECT_FALSE(iswapSpec->allowsOp(cx.getOperation()));

  const auto ecrSpec = NativeGateset::parse("u,ecr");
  ASSERT_TRUE(ecrSpec);
  auto ecr = ECROp::create(builder, loc, q0, q1);
  EXPECT_TRUE(ecrSpec->allowsOp(ecr.getOperation()));
  EXPECT_FALSE(ecrSpec->allowsOp(cx.getOperation()));

  const FloatType f64Float = builder.getF64Type();
  const Type f64Ty = Type::getFromOpaquePointer(f64Float.getAsOpaquePointer());
  const auto funcTyTheta =
      builder.getFunctionType({f64Ty, qubitTy, qubitTy}, {qubitTy, qubitTy});
  OpBuilder::InsertionGuard guard(builder);
  builder.clearInsertionPoint();
  auto funcTheta = func::FuncOp::create(
      builder, loc, "allows_op_runtime_two_qubit_rotations", funcTyTheta);
  auto* entryTheta = funcTheta.addEntryBlock();
  builder.setInsertionPointToStart(entryTheta);
  Value runtimeTheta = entryTheta->getArgument(0);
  Value runtimeQ0 = entryTheta->getArgument(1);
  Value runtimeQ1 = entryTheta->getArgument(2);
  auto runtimeRxx =
      RXXOp::create(builder, loc, runtimeQ0, runtimeQ1, runtimeTheta);
  EXPECT_TRUE(rxxSpec->allowsOp(runtimeRxx.getOperation()));
  EXPECT_FALSE(spec->allowsOp(runtimeRxx.getOperation()));
  auto runtimeRyy = RYYOp::create(builder, loc, runtimeRxx.getOutputQubit(0),
                                  runtimeRxx.getOutputQubit(1), runtimeTheta);
  EXPECT_TRUE(ryySpec->allowsOp(runtimeRyy.getOperation()));
  auto runtimeRzx = RZXOp::create(builder, loc, runtimeRyy.getOutputQubit(0),
                                  runtimeRyy.getOutputQubit(1), runtimeTheta);
  EXPECT_TRUE(rzxSpec->allowsOp(runtimeRzx.getOperation()));
  auto runtimeRzz = RZZOp::create(builder, loc, runtimeRzx.getOutputQubit(0),
                                  runtimeRzx.getOutputQubit(1), runtimeTheta);
  EXPECT_TRUE(rzzSpec->allowsOp(runtimeRzz.getOperation()));
  func::ReturnOp::create(
      builder, loc,
      ValueRange{runtimeRzz.getOutputQubit(0), runtimeRzz.getOutputQubit(1)});
  auto module = ModuleOp::create(loc);
  module.getBody()->push_back(funcTheta);
  EXPECT_TRUE(succeeded(verify(module)));
}
