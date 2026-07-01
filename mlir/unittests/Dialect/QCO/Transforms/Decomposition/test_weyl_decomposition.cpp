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
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeProfile.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
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
#include <vector>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mqt::test;

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

static constexpr Matrix4x4 TWO_QUBIT_CONTROLLED_Z =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, //
                            0.0, 1.0, 0.0, 0.0, //
                            0.0, 0.0, 1.0, 0.0, //
                            0.0, 0.0, 0.0, -1.0);

template <typename MatrixT>
static bool isUnitaryMatrix(const MatrixT& matrix,
                            const double tolerance = MATRIX_TOLERANCE) {
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
  const Matrix4x4 unitary = Matrix4x4::fromElements(
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
  for (const auto [a, b, c] : {std::tuple{0.3, 0.2, 0.1},
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
  EXPECT_EQ(decomposed->singleQubitFactors.size(), 4U);
}

TEST(BasisDecomposerForcedCountTest, TwoBasisUsesProducesFactors) {
  const Matrix4x4 basis = TWO_QUBIT_CONTROLLED_X01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(TWO_QUBIT_CONTROLLED_X01, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{2});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 2);
  EXPECT_EQ(decomposed->singleQubitFactors.size(), 6U);
}

TEST(BasisDecomposerForcedCountTest, ThreeBasisUsesProducesFactors) {
  const Matrix4x4 basis = TWO_QUBIT_CONTROLLED_X01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(TWO_QUBIT_CONTROLLED_X01, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{3});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 3);
  EXPECT_EQ(decomposed->singleQubitFactors.size(), 8U);
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
      unitary = oneQ.embedInTwoQubit(*qid) * unitary;
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
      if (!op.getUnitaryMatrix4x4(twoQ)) {
        return std::nullopt;
      }
      unitary = twoQ.reorderForQubits(*q0id, *q1id) * unitary;
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
  const auto synthResult =
      synthesizeUnitary2QWeyl(builder, loc, entry->getArgument(0),
                              entry->getArgument(1), target, spec, out0, out1);
  if (failed(synthResult)) {
    ADD_FAILURE() << "synthesizeUnitary2QWeyl failed during test synthesis";
    return func;
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

class WeylSynthesisTest : public testing::TestWithParam<WeylSynthesisCase> {};

} // namespace

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
  const auto cxSpec = parseNativeSpec("u,cx");
  ASSERT_TRUE(cxSpec);
  const auto cxCount = twoQubitEntanglerCount(Matrix4x4::identity(), *cxSpec);
  ASSERT_TRUE(cxCount.has_value());
  EXPECT_EQ(*cxCount, 0U);

  const auto czSpec = parseNativeSpec("u,cz");
  ASSERT_TRUE(czSpec);
  const auto czCount = twoQubitEntanglerCount(Matrix4x4::identity(), *czSpec);
  ASSERT_TRUE(czCount.has_value());
  EXPECT_EQ(*czCount, 0U);
}

TEST(WeylSynthesisTest, RejectsMenuWithoutEntangler) {
  EXPECT_FALSE(parseNativeSpec("u").has_value());
}

TEST(WeylSynthesisTest, SynthesisFailsWithoutEntangler) {
  MlirTestContext fx;
  fx.setUp();
  const NativeProfileSpec spec{.gates = {NativeGateKind::U}};
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
      TWO_QUBIT_CONTROLLED_X01, spec, out0, out1)));
}

TEST(NativeSpecTest, ParsesAndRejectsMenus) {
  const auto ibm = parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(ibm);
  EXPECT_TRUE(ibm->gates.contains(NativeGateKind::CX));
  EXPECT_TRUE(ibm->gates.contains(NativeGateKind::X));
  EXPECT_FALSE(ibm->gates.contains(NativeGateKind::RZZ));
  EXPECT_FALSE(parseNativeSpec("x,sx,rz,not-a-gate").has_value());
  EXPECT_FALSE(parseNativeSpec("u").has_value());

  const auto pMenu = parseNativeSpec("x,sx,p,cx");
  const auto rzMenu = parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(pMenu);
  ASSERT_TRUE(rzMenu);
  EXPECT_EQ(pMenu->gates, rzMenu->gates);

  const auto cxOnly = parseNativeSpec("u,cx");
  ASSERT_TRUE(cxOnly);
  EXPECT_TRUE(cxOnly->gates.contains(NativeGateKind::CX));
  EXPECT_FALSE(cxOnly->gates.contains(NativeGateKind::CZ));

  const auto both = parseNativeSpec("u,cx,cz");
  ASSERT_TRUE(both);
  EXPECT_TRUE(both->gates.contains(NativeGateKind::CX));
  EXPECT_TRUE(both->gates.contains(NativeGateKind::CZ));

  const auto generic = parseNativeSpec("u,cx");
  ASSERT_TRUE(generic);
  EXPECT_TRUE(generic->gates.contains(NativeGateKind::U));
  EXPECT_FALSE(generic->gates.contains(NativeGateKind::X));
}

TEST(NativeSpecTest, ResolvesEulerBasisFromMenu) {
  const auto uMenu = parseNativeSpec("u,cx");
  ASSERT_TRUE(uMenu);
  EXPECT_EQ(uMenu->eulerBasis(), EulerBasis::U);

  const auto zsxx = parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(zsxx);
  EXPECT_EQ(zsxx->eulerBasis(), EulerBasis::ZSXX);

  const auto rMenu = parseNativeSpec("r,cz");
  ASSERT_TRUE(rMenu);
  EXPECT_EQ(rMenu->eulerBasis(), EulerBasis::R);

  const auto xzx = parseNativeSpec("rx,rz,cz");
  ASSERT_TRUE(xzx);
  EXPECT_EQ(xzx->eulerBasis(), EulerBasis::XZX);

  const auto xyx = parseNativeSpec("rx,ry,cz");
  ASSERT_TRUE(xyx);
  EXPECT_EQ(xyx->eulerBasis(), EulerBasis::XYX);

  const auto zyz = parseNativeSpec("ry,rz,cz");
  ASSERT_TRUE(zyz);
  EXPECT_EQ(zyz->eulerBasis(), EulerBasis::ZYZ);
}

TEST(NativeSpecTest, AllowsOpMatchesMenu) {
  MlirTestContext fx;
  fx.setUp();
  const auto spec = parseNativeSpec("u,cx,rzz");
  ASSERT_TRUE(spec);

  OpBuilder builder(fx.ctx());
  const Location loc = UnknownLoc::get(fx.ctx());
  const auto qubitTy = QubitType::get(fx.ctx());
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
  EXPECT_TRUE(
      allowsOp(RZZOp::create(builder, loc, q0, q1, 0.4).getOperation(), *spec));

  auto cx = CtrlOp::create(
      builder, loc, ValueRange{q0}, ValueRange{q1},
      [&](ValueRange targets) -> SmallVector<Value> {
        return {XOp::create(builder, loc, targets[0]).getOutputQubit(0)};
      });
  EXPECT_TRUE(allowsOp(cx.getOperation(), *spec));

  auto cxWithInterleavedH = CtrlOp::create(
      builder, loc, ValueRange{q0}, ValueRange{q1},
      [&](ValueRange targets) -> SmallVector<Value> {
        auto wire = XOp::create(builder, loc, targets[0]).getOutputQubit(0);
        return {HOp::create(builder, loc, wire).getOutputQubit(0)};
      });
  EXPECT_FALSE(allowsOp(cxWithInterleavedH.getOperation(), *spec));

  EXPECT_FALSE(allowsOp(XOp::create(builder, loc, q0).getOperation(), *spec));

  const auto czSpec = parseNativeSpec("u,cz");
  ASSERT_TRUE(czSpec);
  auto cz = CtrlOp::create(
      builder, loc, ValueRange{q0}, ValueRange{q1},
      [&](ValueRange targets) -> SmallVector<Value> {
        return {ZOp::create(builder, loc, targets[0]).getOutputQubit(0)};
      });
  EXPECT_TRUE(allowsOp(cz.getOperation(), *czSpec));
  EXPECT_FALSE(allowsOp(cx.getOperation(), *czSpec));
}
