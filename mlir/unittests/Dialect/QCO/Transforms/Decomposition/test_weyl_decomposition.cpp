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

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mqt::test;

// Weyl / basis / helpers.

TEST(DecompositionHelpersTest, RemEuclidNeverNegative) {
  EXPECT_DOUBLE_EQ(remEuclid(-1.0, 3.0), 2.0);
  EXPECT_DOUBLE_EQ(remEuclid(7.0, 3.0), 1.0);
  EXPECT_DOUBLE_EQ(remEuclid(0.0, 2.5), 0.0);
}

TEST(DecompositionHelpersTest, TraceToFidelityMatchesFormula) {
  const std::complex<double> x{3.0, 4.0};
  const double absx = 5.0;
  EXPECT_DOUBLE_EQ(traceToFidelity(x), (4.0 + (absx * absx)) / 20.0);
}

TEST(DecompositionHelpersTest, GlobalPhaseFactorUnitMagnitude) {
  const auto z = globalPhaseFactor(1.25);
  EXPECT_NEAR(std::abs(z), 1.0, 1e-14);
}

TEST(DecompositionHelpersTest, IsUnitaryMatrixRejectsNonUnitary) {
  const Matrix2x2 m = Matrix2x2::fromElements(2.0, 0.0, 0.0, 2.0);
  EXPECT_FALSE(isUnitaryMatrix(m));
}

TEST(DecompositionHelpersTest, IsUnitaryMatrixAcceptsUnitary) {
  const Matrix2x2 m = Matrix2x2::identity();
  EXPECT_TRUE(isUnitaryMatrix(m));
}

//===----------------------------------------------------------------------===//
// Weyl decomposition
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_P` at global scope
class WeylDecompositionTest : public testing::TestWithParam<Matrix4x4 (*)()> {
public:
  [[nodiscard]] static Matrix4x4
  restore(const TwoQubitWeylDecomposition& decomposition) {
    return k1(decomposition) * can(decomposition) * k2(decomposition) *
           globalPhaseFactor(decomposition);
  }

  [[nodiscard]] static std::complex<double>
  globalPhaseFactor(const TwoQubitWeylDecomposition& decomposition) {
    return mqt::test::globalPhaseFactor(decomposition.globalPhase());
  }
  [[nodiscard]] static Matrix4x4
  can(const TwoQubitWeylDecomposition& decomposition) {
    return decomposition.getCanonicalMatrix();
  }
  [[nodiscard]] static Matrix4x4
  k1(const TwoQubitWeylDecomposition& decomposition) {
    return kron(decomposition.k1l(), decomposition.k1r());
  }
  [[nodiscard]] static Matrix4x4
  k2(const TwoQubitWeylDecomposition& decomposition) {
    return kron(decomposition.k2l(), decomposition.k2r());
  }
};

TEST_P(WeylDecompositionTest, TestExact) {
  const auto& originalMatrix = GetParam()();
  auto decomposition = TwoQubitWeylDecomposition::create(
      originalMatrix, std::optional<double>{1.0});
  auto restoredMatrix = restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix));
}

TEST_P(WeylDecompositionTest, TestApproximation) {
  const auto& originalMatrix = GetParam()();
  auto decomposition = TwoQubitWeylDecomposition::create(
      originalMatrix, std::optional<double>{1.0 - 1e-12});
  auto restoredMatrix = restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix));
}

TEST(WeylDecompositionStandalone,
     CnotProducesValidWeylParametersAndUnitaryLocals) {
  const Matrix4x4 cnot = Matrix4x4::fromElements(1, 0, 0, 0, // row 0
                                                 0, 1, 0, 0, // row 1
                                                 0, 0, 0, 1, // row 2
                                                 0, 0, 1, 0);

  const auto decomp = TwoQubitWeylDecomposition::create(cnot, std::nullopt);
  EXPECT_GE(decomp.a(), -1e-10);
  EXPECT_GE(decomp.b(), -1e-10);
  EXPECT_GE(decomp.c(), -1e-10);
  constexpr double piOver4 = 0.7853981633974483;
  EXPECT_LE(decomp.a(), piOver4 + 1e-10);
  EXPECT_LE(decomp.b(), piOver4 + 1e-10);
  EXPECT_LE(decomp.c(), piOver4 + 1e-10);
  EXPECT_TRUE(isUnitaryMatrix(decomp.k1l()));
  EXPECT_TRUE(isUnitaryMatrix(decomp.k2l()));
  EXPECT_TRUE(isUnitaryMatrix(decomp.k1r()));
  EXPECT_TRUE(isUnitaryMatrix(decomp.k2r()));
}

TEST(WeylDecompositionStandalone, Random) {
  constexpr auto maxIterations = 5000;
  std::mt19937 rng{1234567UL};

  for (int i = 0; i < maxIterations; ++i) {
    auto originalMatrix = randomUnitary4x4(rng);
    auto decomposition = TwoQubitWeylDecomposition::create(
        originalMatrix, std::optional<double>{1.0 - 1e-12});
    auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

    // The reconstruction accuracy is bounded by the iterative diagonalization
    // residual rather than the (much tighter) default matrix tolerance.
    EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix, kSanityCheckPrecision));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ProductTwoQubitMatrices, WeylDecompositionTest,
    ::testing::Values([]() -> Matrix4x4 { return Matrix4x4::identity(); },
                      []() -> Matrix4x4 {
                        return kron(rzMatrix(1.0), ryMatrix(3.1));
                      },
                      []() -> Matrix4x4 {
                        return kron(Matrix2x2::identity(), rxMatrix(0.1));
                      }));

INSTANTIATE_TEST_SUITE_P(
    TwoQubitMatrices, WeylDecompositionTest,
    ::testing::Values(
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
          return kron(hGate(), ipz()) * cxGate01() * kron(ipx(), ipy());
        }));

//===----------------------------------------------------------------------===//
// Basis decomposer
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_P` at global scope
class BasisDecomposerTest : public testing::TestWithParam<
                                std::tuple<Matrix4x4 (*)(), Matrix4x4 (*)()>> {
public:
  [[nodiscard]] static Matrix4x4
  restore(const TwoQubitNativeDecomposition& decomposition,
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
    return matrix * ::globalPhaseFactor(decomposition.globalPhase);
  }

protected:
  void SetUp() override {
    basisMatrix = std::get<0>(GetParam())();
    target = std::get<1>(GetParam())();
    targetDecomposition = std::make_unique<TwoQubitWeylDecomposition>(
        TwoQubitWeylDecomposition::create(target, std::optional<double>{1.0}));
  }

  Matrix4x4 target;
  Matrix4x4 basisMatrix;
  std::unique_ptr<TwoQubitWeylDecomposition> targetDecomposition;
};

TEST_P(BasisDecomposerTest, TestExact) {
  const auto& originalMatrix = target;
  auto decomposer = TwoQubitBasisDecomposer::create(basisMatrix, 1.0);
  auto decomposed =
      decomposer.twoQubitDecompose(*targetDecomposition, std::nullopt);

  ASSERT_TRUE(decomposed.has_value());

  auto restoredMatrix = restore(*decomposed, basisMatrix);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix));
}

TEST_P(BasisDecomposerTest, TestApproximation) {
  const auto& originalMatrix = target;
  auto decomposer = TwoQubitBasisDecomposer::create(basisMatrix, 1.0 - 1e-12);
  auto decomposed =
      decomposer.twoQubitDecompose(*targetDecomposition, std::nullopt);

  ASSERT_TRUE(decomposed.has_value());

  auto restoredMatrix = restore(*decomposed, basisMatrix);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix));
}

TEST(BasisDecomposerTest, Random) {
  constexpr auto maxIterations = 2000;
  std::mt19937 rng{123456UL};

  const llvm::SmallVector<Matrix4x4, 2> basisMatrices{cxGate01(), cxGate10()};
  std::uniform_int_distribution<std::size_t> distBasisGate{
      0, basisMatrices.size() - 1};
  auto selectRandomBasisMatrix = [&]() {
    return basisMatrices[distBasisGate(rng)];
  };

  for (int i = 0; i < maxIterations; ++i) {
    auto originalMatrix = randomUnitary4x4(rng);

    auto targetDecomposition = TwoQubitWeylDecomposition::create(
        originalMatrix, std::optional<double>{1.0});
    const auto basisMatrix = selectRandomBasisMatrix();
    auto decomposer = TwoQubitBasisDecomposer::create(basisMatrix, 1.0);
    auto decomposed =
        decomposer.twoQubitDecompose(targetDecomposition, std::nullopt);

    ASSERT_TRUE(decomposed.has_value());

    auto restoredMatrix =
        BasisDecomposerTest::restore(*decomposed, basisMatrix);

    // Reconstruction accumulates the Weyl diagonalization residual through up
    // to three entangler layers, so allow a correspondingly relaxed tolerance.
    EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix, kSanityCheckPrecision));
  }
}

TEST(BasisDecomposerNumBasisTest, ForcesZeroBasisUsesForIdentityTarget) {
  const auto basis = cxGate01();
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const Matrix4x4 target = Matrix4x4::identity();
  const auto weyl =
      TwoQubitWeylDecomposition::create(target, std::optional<double>{1.0});
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{0});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 0);
  const Matrix4x4 restored = BasisDecomposerTest::restore(*decomposed, basis);
  EXPECT_TRUE(restored.isApprox(target));
}

INSTANTIATE_TEST_SUITE_P(
    ProductTwoQubitMatrices, BasisDecomposerTest,
    testing::Combine(
        // basis entanglers
        testing::Values([]() -> Matrix4x4 { return cxGate01(); },
                        []() -> Matrix4x4 { return cxGate10(); }),
        // targets to be decomposed
        testing::Values([]() -> Matrix4x4 { return Matrix4x4::identity(); },
                        []() -> Matrix4x4 {
                          return kron(rzMatrix(1.0), ryMatrix(3.1));
                        },
                        []() -> Matrix4x4 {
                          return kron(Matrix2x2::identity(), rxMatrix(0.1));
                        })));

INSTANTIATE_TEST_SUITE_P(
    TwoQubitMatrices, BasisDecomposerTest,
    testing::Combine(
        // basis entanglers
        testing::Values([]() -> Matrix4x4 { return cxGate01(); },
                        []() -> Matrix4x4 { return cxGate10(); }),
        // targets to be decomposed
        ::testing::Values(
            []() -> Matrix4x4 { return rzzMatrix(2.0); },
            []() -> Matrix4x4 {
              return ryyMatrix(1.0) * rzzMatrix(3.0) * rxxMatrix(2.0);
            },
            []() -> Matrix4x4 {
              return TwoQubitWeylDecomposition::getCanonicalMatrix(1.5, -0.2,
                                                                   0.0) *
                     kron(rxMatrix(1.0), Matrix2x2::identity());
            },
            []() -> Matrix4x4 {
              return kron(rxMatrix(1.0), ryMatrix(1.0)) *
                     TwoQubitWeylDecomposition::getCanonicalMatrix(1.1, 0.2,
                                                                   3.0) *
                     kron(rxMatrix(1.0), Matrix2x2::identity());
            },
            []() -> Matrix4x4 {
              return kron(hGate(), ipz()) * cxGate01() * kron(ipx(), ipy());
            })));

//===----------------------------------------------------------------------===//
// Two-qubit Weyl IR synthesis
//===----------------------------------------------------------------------===//

namespace {

struct WeylSynthesisFixture {
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

struct SynthesizedTwoQubitCircuit {
  OwningOpRef<ModuleOp> mlirModule;
  func::FuncOp func;
};

[[nodiscard]] bool extractTwoQubitMatrix(UnitaryOpInterface op,
                                         Matrix4x4& out) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op.getOperation())) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    Operation* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      out = cxGate01();
      return true;
    }
    if (llvm::isa<ZOp>(body)) {
      out = czGate();
      return true;
    }
    return false;
  }
  return op.getUnitaryMatrix4x4(out);
}

[[nodiscard]] std::optional<Matrix4x4>
compute2QUnitaryFromFunc(func::FuncOp funcOp) {
  Matrix4x4 acc = Matrix4x4::identity();
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
      if (!op.getUnitaryMatrix2x2(oneQ)) {
        DynamicMatrix dynamic;
        if (!op.getUnitaryMatrixDynamic(dynamic) || dynamic.rows() != 2 ||
            dynamic.cols() != 2) {
          return std::nullopt;
        }
        oneQ = Matrix2x2::fromElements(dynamic(0, 0), dynamic(0, 1),
                                       dynamic(1, 0), dynamic(1, 1));
      }
      acc = expandToTwoQubits(oneQ, *qid) * acc;
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
      acc = fixTwoQubitMatrixQubitOrder(twoQ, ids) * acc;
      wireIds[op->getResult(0)] = *q0id;
      wireIds[op->getResult(1)] = *q1id;
    }
  }

  return acc * global;
}

[[nodiscard]] SynthesizedTwoQubitCircuit
synthesize2QMatrix(MLIRContext* ctx, const Matrix4x4& target,
                   const NativeProfileSpec& spec) {
  OwningOpRef<ModuleOp> mlirModule = ModuleOp::create(UnknownLoc::get(ctx));
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(mlirModule->getBody());

  const auto qubitTy = QubitType::get(ctx);
  const auto funcTy =
      builder.getFunctionType({qubitTy, qubitTy}, {qubitTy, qubitTy});
  const Location loc = mlirModule->getLoc();
  auto func = func::FuncOp::create(builder, loc, "main", funcTy);
  auto* entry = func.addEntryBlock();

  builder.setInsertionPointToStart(entry);
  Value q0 = entry->getArgument(0);
  Value q1 = entry->getArgument(1);
  Value out0;
  Value out1;
  if (failed(synthesizeUnitary2QWeyl(builder, loc, q0, q1, target, spec, out0,
                                     out1))) {
    llvm::report_fatal_error(
        "synthesizeUnitary2QWeyl failed during test synthesis");
  }
  func::ReturnOp::create(builder, loc, ValueRange{out0, out1});
  return SynthesizedTwoQubitCircuit{.mlirModule = std::move(mlirModule),
                                    .func = func};
}

void expect2QMatrixPreserved(func::FuncOp funcOp, const Matrix4x4& original) {
  const auto actual = compute2QUnitaryFromFunc(funcOp);
  ASSERT_TRUE(actual.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*actual, original));
}

void expectSynthesized2QMatrix(MLIRContext* ctx, const Matrix4x4& target,
                               const NativeProfileSpec& spec) {
  const auto circuit = synthesize2QMatrix(ctx, target, spec);
  ASSERT_TRUE(succeeded(verify(*circuit.mlirModule)));
  expect2QMatrixPreserved(circuit.func, target);
}

} // namespace

TEST(WeylSynthesisTest, ReconstructsCxWithGenericProfile) {
  WeylSynthesisFixture fx;
  fx.setUp();
  const auto spec = parseNativeSpec("u,cx");
  ASSERT_TRUE(spec);
  expectSynthesized2QMatrix(fx.ctx(), cxGate01(), *spec);
}

TEST(WeylSynthesisTest, ReconstructsProductUnitaryWithGenericProfile) {
  WeylSynthesisFixture fx;
  fx.setUp();
  const auto spec = parseNativeSpec("u,cx");
  ASSERT_TRUE(spec);
  const Matrix4x4 target = kron(rzMatrix(1.0), ryMatrix(0.3));
  expectSynthesized2QMatrix(fx.ctx(), target, *spec);
}

TEST(WeylSynthesisTest, ReconstructsWithIbmBasicProfile) {
  WeylSynthesisFixture fx;
  fx.setUp();
  const auto spec = parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(spec);
  const Matrix4x4 target = kron(hGate(), Matrix2x2::identity()) * cxGate01() *
                           kron(rzMatrix(0.2), ryMatrix(0.1));
  expectSynthesized2QMatrix(fx.ctx(), target, *spec);
}

TEST(WeylSynthesisTest, IdentityRequiresNoEntanglers) {
  WeylSynthesisFixture fx;
  fx.setUp();
  const auto spec = parseNativeSpec("u,cx");
  ASSERT_TRUE(spec);
  const auto count = twoQubitEntanglerCount(Matrix4x4::identity(), *spec);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 0U);
}

TEST(WeylSynthesisTest, FailsWithoutEntanglerInSpec) {
  WeylSynthesisFixture fx;
  fx.setUp();
  const auto spec = parseNativeSpec("u");
  ASSERT_TRUE(spec);
  EXPECT_TRUE(spec->entanglerBases.empty());

  OwningOpRef<ModuleOp> mlirModule =
      ModuleOp::create(UnknownLoc::get(fx.ctx()));
  OpBuilder builder(fx.ctx());
  builder.setInsertionPointToStart(mlirModule->getBody());
  const auto qubitTy = QubitType::get(fx.ctx());
  const auto funcTy =
      builder.getFunctionType({qubitTy, qubitTy}, {qubitTy, qubitTy});
  const Location loc = mlirModule->getLoc();
  auto func = func::FuncOp::create(builder, loc, "main", funcTy);
  auto* entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value out0;
  Value out1;
  EXPECT_TRUE(failed(synthesizeUnitary2QWeyl(
      builder, loc, entry->getArgument(0), entry->getArgument(1), cxGate01(),
      *spec, out0, out1)));
}

//===----------------------------------------------------------------------===//
// Native gate menu parsing (Weyl library API)
//===----------------------------------------------------------------------===//

TEST(NativeSpecTest, ResolveIbmBasicCx) {
  const auto spec = parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(spec);
  EXPECT_TRUE(spec->allowedGates.contains(NativeGateKind::Cx));
  EXPECT_TRUE(spec->allowedGates.contains(NativeGateKind::X));
  EXPECT_FALSE(spec->allowRzz);
}

TEST(NativeSpecTest, ResolveRejectsUnknownToken) {
  EXPECT_FALSE(parseNativeSpec("x,sx,rz,not-a-gate").has_value());
}

TEST(NativeSpecTest, PhaseAliasPMatchesRzInIbmStyleMenu) {
  const auto pMenu = parseNativeSpec("x,sx,p,cx");
  const auto rzMenu = parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(pMenu);
  ASSERT_TRUE(rzMenu);
  EXPECT_EQ(pMenu->allowedGates, rzMenu->allowedGates);
}

TEST(NativePolicyTest, UsesCxAndCzFromResolvedSpec) {
  const auto cxOnly = parseNativeSpec("u,cx");
  ASSERT_TRUE(cxOnly);
  EXPECT_TRUE(llvm::is_contained(cxOnly->entanglerBases, EntanglerBasis::Cx));
  EXPECT_FALSE(llvm::is_contained(cxOnly->entanglerBases, EntanglerBasis::Cz));

  const auto both = parseNativeSpec("u,cx,cz");
  ASSERT_TRUE(both);
  EXPECT_TRUE(llvm::is_contained(both->entanglerBases, EntanglerBasis::Cx));
  EXPECT_TRUE(llvm::is_contained(both->entanglerBases, EntanglerBasis::Cz));
}

TEST(NativePolicyTest, ExcludesGateKindsNotInMenu) {
  const auto spec = parseNativeSpec("u,cx");
  ASSERT_TRUE(spec);
  EXPECT_TRUE(spec->allowedGates.contains(NativeGateKind::U));
  EXPECT_TRUE(spec->allowedGates.contains(NativeGateKind::Cx));
  EXPECT_FALSE(spec->allowedGates.contains(NativeGateKind::X));
  EXPECT_FALSE(spec->allowedGates.contains(NativeGateKind::Sx));
  EXPECT_FALSE(spec->allowedGates.contains(NativeGateKind::Rz));
}
