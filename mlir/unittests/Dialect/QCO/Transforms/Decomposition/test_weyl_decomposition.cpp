/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeProfile.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/TypeSwitch.h>
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

using QubitId = std::size_t;

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

[[nodiscard]] static std::optional<QubitId>
lookupWireId(const llvm::DenseMap<Value, QubitId>& wireIds, Value wire) {
  if (const auto it = wireIds.find(wire); it != wireIds.end()) {
    return it->second;
  }
  return std::nullopt;
}

[[nodiscard]] static std::optional<Matrix4x4>
embeddedStepOnWires(UnitaryOpInterface op, QubitId q0,
                    std::optional<QubitId> q1) {
  if (op.isSingleQubit()) {
    Matrix2x2 matrix;
    if (!op.getUnitaryMatrix2x2(matrix)) {
      return std::nullopt;
    }
    return matrix.embedInTwoQubit(q0);
  }
  if (!q1.has_value()) {
    return std::nullopt;
  }
  Matrix4x4 matrix;
  if (!op.getUnitaryMatrix4x4(matrix)) {
    return std::nullopt;
  }
  return matrix.reorderForQubits(q0, *q1);
}

static std::optional<Matrix4x4>
computeTwoQubitUnitaryFromFunc(func::FuncOp funcOp) {
  Matrix4x4 unitary = Matrix4x4::identity();
  Complex global{1.0, 0.0};
  llvm::DenseMap<Value, QubitId> wireIds;
  wireIds[funcOp.getArgument(0)] = 0;
  wireIds[funcOp.getArgument(1)] = 1;

  for (Operation& op : funcOp.getBody().front()) {
    if (isa<arith::ConstantOp>(op)) {
      continue;
    }

    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      if (returnOp.getNumOperands() != 2) {
        return std::nullopt;
      }
      const auto out0 = lookupWireId(wireIds, returnOp.getOperand(0));
      const auto out1 = lookupWireId(wireIds, returnOp.getOperand(1));
      if (!out0.has_value() || !out1.has_value() || *out0 != 0 || *out1 != 1) {
        return std::nullopt;
      }
      continue;
    }

    if (auto gphase = dyn_cast<GPhaseOp>(op)) {
      if (const auto matrix = gphase.getUnitaryMatrix()) {
        global *= (*matrix)(0, 0);
      }
      continue;
    }

    auto unitaryOp = dyn_cast<UnitaryOpInterface>(op);
    if (!unitaryOp) {
      return std::nullopt;
    }

    const auto q0 = lookupWireId(wireIds, unitaryOp.getInputQubit(0));
    if (!q0.has_value()) {
      return std::nullopt;
    }
    std::optional<QubitId> q1;
    if (unitaryOp.isTwoQubit()) {
      q1 = lookupWireId(wireIds, unitaryOp.getInputQubit(1));
      if (!q1.has_value()) {
        return std::nullopt;
      }
    } else if (!unitaryOp.isSingleQubit()) {
      return std::nullopt;
    }

    const auto step = embeddedStepOnWires(unitaryOp, *q0, q1);
    if (!step.has_value()) {
      return std::nullopt;
    }
    unitary.premultiplyBy(*step);

    wireIds[unitaryOp.getOutputQubit(0)] = *q0;
    if (q1.has_value()) {
      wireIds[unitaryOp.getOutputQubit(1)] = *q1;
    }
  }

  unitary *= global;
  return unitary;
}

[[nodiscard]] static Synthesized2QCircuit
synthesize2QMatrix(MLIRContext* ctx, const Matrix4x4& target,
                   const NativeProfileSpec& spec) {
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
                                      const NativeProfileSpec& spec) {
  const auto circuit = synthesize2QMatrix(ctx, target, spec);
  ASSERT_TRUE(succeeded(verify(*circuit.mlirModule)));
  const auto actual = computeTwoQubitUnitaryFromFunc(circuit.func);
  ASSERT_TRUE(actual.has_value());
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

class NativeProfileMlirTest : public testing::Test {
protected:
  MlirTestContext mlir;

  void SetUp() override { mlir.setUp(); }
};

} // namespace

TEST_P(WeylSynthesisTest, PreservesTargetUnitary) {
  const auto spec = NativeProfileSpec::parse(GetParam().nativeGates);
  ASSERT_TRUE(spec);
  expectSynthesized2QMatrix(mlir.ctx(), GetParam().target(), *spec);
}

INSTANTIATE_TEST_SUITE_P(
    Profiles, WeylSynthesisTest,
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
        WeylSynthesisCase{"CzGeneric", "u,cz",
                          [] { return TWO_QUBIT_CONTROLLED_Z; }}),
    [](const testing::TestParamInfo<WeylSynthesisCase>& info) {
      return info.param.name;
    });

TEST(WeylSynthesisTest, IdentityRequiresNoEntanglers) {
  for (const char* gateset : {"u,cx", "u,cz"}) {
    const auto spec = NativeProfileSpec::parse(gateset);
    ASSERT_TRUE(spec) << gateset;
    const auto native = decomposition::detail::decomposeNativeTarget(
        Matrix4x4::identity(), *spec);
    ASSERT_TRUE(native.has_value()) << gateset;
    EXPECT_EQ(native->numBasisUses, 0U) << gateset;
  }
}

TEST(WeylSynthesisTest, RejectsGatesetWithoutEntangler) {
  EXPECT_FALSE(NativeProfileSpec::parse("u").has_value());
}

TEST_F(NativeProfileMlirTest, ReconstructionRejectsUnhandledOps) {
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
  BarrierOp::create(builder, loc, ValueRange{q0, q1});
  func::ReturnOp::create(builder, loc, ValueRange{q0, q1});
  EXPECT_FALSE(computeTwoQubitUnitaryFromFunc(func).has_value());
}

TEST_F(NativeProfileMlirTest, SynthesisFailsWithoutEulerBasis) {
  const NativeProfileSpec spec{.gates = {NativeGateKind::CX}};
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

TEST_F(NativeProfileMlirTest, SynthesisFailsWithoutEntangler) {
  const NativeProfileSpec spec{.gates = {NativeGateKind::U}};
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
  const NativeProfileSpec spec{.gates = {NativeGateKind::U}};
  EXPECT_FALSE(
      decomposition::detail::decomposeNativeTarget(Matrix4x4::identity(), spec)
          .has_value());
}

TEST(NativeSpecTest, ParsesAndRejectsGatesets) {
  const auto ibm = NativeProfileSpec::parse("x,sx,rz,cx");
  ASSERT_TRUE(ibm);
  EXPECT_TRUE(ibm->gates.contains(NativeGateKind::CX));
  EXPECT_TRUE(ibm->gates.contains(NativeGateKind::X));
  EXPECT_FALSE(NativeProfileSpec::parse("x,sx,rz,not-a-gate").has_value());
  EXPECT_FALSE(NativeProfileSpec::parse("u").has_value());

  const auto whitespaceToken = NativeProfileSpec::parse("u, ,cx");
  ASSERT_TRUE(whitespaceToken);
  EXPECT_TRUE(whitespaceToken->gates.contains(NativeGateKind::U));
  EXPECT_TRUE(whitespaceToken->gates.contains(NativeGateKind::CX));

  EXPECT_FALSE(NativeProfileSpec::parse("x,sx,p,cx").has_value());
  EXPECT_FALSE(NativeProfileSpec::parse("ry,p,cz").has_value());

  const auto cxOnly = NativeProfileSpec::parse("u,cx");
  ASSERT_TRUE(cxOnly);
  EXPECT_TRUE(cxOnly->gates.contains(NativeGateKind::U));
  EXPECT_TRUE(cxOnly->gates.contains(NativeGateKind::CX));
  EXPECT_FALSE(cxOnly->gates.contains(NativeGateKind::CZ));
  EXPECT_FALSE(cxOnly->gates.contains(NativeGateKind::X));

  const auto both = NativeProfileSpec::parse("u,cx,cz");
  ASSERT_TRUE(both);
  EXPECT_TRUE(both->gates.contains(NativeGateKind::CX));
  EXPECT_TRUE(both->gates.contains(NativeGateKind::CZ));
  EXPECT_EQ(both->entangler, NativeGateKind::CX);
}

TEST(NativeSpecTest, RejectsGatesetWithoutSingleQubitStrategy) {
  EXPECT_FALSE(NativeProfileSpec::parse("cx").has_value());
  EXPECT_FALSE(NativeProfileSpec::parse("cz").has_value());
  EXPECT_FALSE(NativeProfileSpec::parse("rx,cx").has_value());
}

TEST(NativeSpecTest, ResolvesEulerBasisFromGateset) {
  const auto uGateset = NativeProfileSpec::parse("u,cx");
  ASSERT_TRUE(uGateset);
  EXPECT_EQ(*uGateset->eulerBasis, EulerBasis::U);

  const auto zsxx = NativeProfileSpec::parse("x,sx,rz,cx");
  ASSERT_TRUE(zsxx);
  EXPECT_EQ(*zsxx->eulerBasis, EulerBasis::ZSXX);

  const auto rGateset = NativeProfileSpec::parse("r,cz");
  ASSERT_TRUE(rGateset);
  EXPECT_EQ(*rGateset->eulerBasis, EulerBasis::R);

  const auto xzx = NativeProfileSpec::parse("rx,rz,cz");
  ASSERT_TRUE(xzx);
  EXPECT_EQ(*xzx->eulerBasis, EulerBasis::XZX);

  const auto xyx = NativeProfileSpec::parse("rx,ry,cz");
  ASSERT_TRUE(xyx);
  EXPECT_EQ(*xyx->eulerBasis, EulerBasis::XYX);

  const auto zyz = NativeProfileSpec::parse("ry,rz,cz");
  ASSERT_TRUE(zyz);
  EXPECT_EQ(*zyz->eulerBasis, EulerBasis::ZYZ);
}

static std::optional<NativeGateKind> gateKindFor(UnitaryOpInterface op) {
  return llvm::TypeSwitch<Operation*, std::optional<NativeGateKind>>(
             op.getOperation())
      .Case<UOp>([](UOp) { return NativeGateKind::U; })
      .Case<XOp>([](XOp) { return NativeGateKind::X; })
      .Case<SXOp>([](SXOp) { return NativeGateKind::SX; })
      .Case<RZOp>([](RZOp) { return NativeGateKind::RZ; })
      .Case<RXOp>([](RXOp) { return NativeGateKind::RX; })
      .Case<RYOp>([](RYOp) { return NativeGateKind::RY; })
      .Case<ROp>([](ROp) { return NativeGateKind::R; })
      .Default([](Operation*) { return std::nullopt; });
}

static std::optional<NativeGateKind> entanglerKindFor(CtrlOp ctrl) {
  if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1 ||
      ctrl.getNumBodyUnitaries() != 1) {
    return std::nullopt;
  }
  return llvm::TypeSwitch<Operation*, std::optional<NativeGateKind>>(
             ctrl.getBodyUnitary(0).getOperation())
      .Case<XOp>([](XOp) { return NativeGateKind::CX; })
      .Case<ZOp>([](ZOp) { return NativeGateKind::CZ; })
      .Default([](Operation*) { return std::nullopt; });
}

static bool allowsOp(Operation* op, const NativeProfileSpec& spec) {
  return llvm::TypeSwitch<Operation*, bool>(op)
      .Case<BarrierOp, GPhaseOp>([](auto) { return true; })
      .Case<CtrlOp>([&](CtrlOp ctrl) {
        const auto kind = entanglerKindFor(ctrl);
        return kind && spec.gates.contains(*kind);
      })
      .Case<UnitaryOpInterface>([&](UnitaryOpInterface unitary) {
        if (!unitary.isSingleQubit()) {
          return false;
        }
        const auto gate = gateKindFor(unitary);
        return gate && spec.gates.contains(*gate);
      })
      .Default([](Operation*) { return false; });
}

TEST_F(NativeProfileMlirTest, AllowsOpMatchesGateset) {
  const auto spec = NativeProfileSpec::parse("u,cx");
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

  EXPECT_TRUE(allowsOp(
      BarrierOp::create(builder, loc, ValueRange{q0, q1}).getOperation(),
      *spec));
  EXPECT_TRUE(
      allowsOp(GPhaseOp::create(builder, loc, 0.1).getOperation(), *spec));
  EXPECT_TRUE(allowsOp(
      UOp::create(builder, loc, q0, 0.1, 0.2, 0.3).getOperation(), *spec));

  auto cx = CtrlOp::create(
      builder, loc, ValueRange{q0}, ValueRange{q1},
      [&builder, &loc](ValueRange targets) -> SmallVector<Value> {
        return {XOp::create(builder, loc, targets[0]).getOutputQubit(0)};
      });
  EXPECT_TRUE(allowsOp(cx.getOperation(), *spec));

  auto cxWithInterleavedH = CtrlOp::create(
      builder, loc, ValueRange{q0}, ValueRange{q1},
      [&builder, &loc](ValueRange targets) -> SmallVector<Value> {
        auto wire = XOp::create(builder, loc, targets[0]).getOutputQubit(0);
        return {HOp::create(builder, loc, wire).getOutputQubit(0)};
      });
  EXPECT_FALSE(allowsOp(cxWithInterleavedH.getOperation(), *spec));

  EXPECT_FALSE(allowsOp(XOp::create(builder, loc, q0).getOperation(), *spec));
  EXPECT_FALSE(
      allowsOp(RXXOp::create(builder, loc, q0, q1, 0.2).getOperation(), *spec));

  const auto rzSpec = NativeProfileSpec::parse("x,sx,rz,cx");
  ASSERT_TRUE(rzSpec);
  EXPECT_TRUE(
      allowsOp(RZOp::create(builder, loc, q0, 0.3).getOperation(), *rzSpec));
  EXPECT_FALSE(
      allowsOp(POp::create(builder, loc, q0, 0.3).getOperation(), *rzSpec));

  auto hCtrl = CtrlOp::create(
      builder, loc, ValueRange{q0}, ValueRange{q1},
      [&builder, &loc](ValueRange targets) -> SmallVector<Value> {
        return {HOp::create(builder, loc, targets[0]).getOutputQubit(0)};
      });
  EXPECT_FALSE(allowsOp(hCtrl.getOperation(), *spec));

  const auto funcTy3 = builder.getFunctionType({qubitTy, qubitTy, qubitTy},
                                               {qubitTy, qubitTy, qubitTy});
  auto func3 = func::FuncOp::create(builder, loc, "allows_op_ccx", funcTy3);
  auto* entry3 = func3.addEntryBlock();
  builder.setInsertionPointToStart(entry3);
  Value c0 = entry3->getArgument(0);
  Value c1 = entry3->getArgument(1);
  Value target = entry3->getArgument(2);
  auto ccx = CtrlOp::create(
      builder, loc, ValueRange{c0, c1}, ValueRange{target},
      [&builder, &loc](ValueRange targets) -> SmallVector<Value> {
        return {XOp::create(builder, loc, targets[0]).getOutputQubit(0)};
      });
  EXPECT_FALSE(allowsOp(ccx.getOperation(), *spec));

  const auto czSpec = NativeProfileSpec::parse("u,cz");
  ASSERT_TRUE(czSpec);
  auto cz = CtrlOp::create(
      builder, loc, ValueRange{q0}, ValueRange{q1},
      [&builder, &loc](ValueRange targets) -> SmallVector<Value> {
        return {ZOp::create(builder, loc, targets[0]).getOutputQubit(0)};
      });
  EXPECT_TRUE(allowsOp(cz.getOperation(), *czSpec));
  EXPECT_FALSE(allowsOp(cx.getOperation(), *czSpec));
}
