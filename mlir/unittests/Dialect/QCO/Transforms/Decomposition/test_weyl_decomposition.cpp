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
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/QC/IR/QCDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
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

static void fusionCxCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.cx(q0, q1);
}

static void fusionHCxInterleavedTCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.t(q1);
  b.s(q0);
  b.cx(q0, q1);
}

static void fusionThreeLineCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  b.cx(q0, q1);
  b.cx(q1, q2);
  b.cx(q0, q1);
}

static void fusionCxBarrierCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.barrier({q0, q1});
  b.cx(q0, q1);
}

static void fusionSwapCxPattern(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.cx(q1, q0);
  b.cx(q0, q1);
}

static void fusionHRzzSRzz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.rzz(-0.29, q0, q1);
  b.s(q1);
  b.rzz(0.17, q0, q1);
}

using mqt::test::cxGate01;
using mqt::test::czGate;
using mqt::test::expandToTwoQubits;
using mqt::test::fixTwoQubitMatrixQubitOrder;
using mqt::test::isEquivalentUpToGlobalPhase;
using mqt::test::QubitId;

[[nodiscard]] static std::optional<Value>
getUnitaryQubitOperand(qco::UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getOperand(index);
  if (!llvm::isa<qco::QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

[[nodiscard]] static std::optional<Value>
getUnitaryQubitResult(qco::UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getResult(index);
  if (!llvm::isa<qco::QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

[[nodiscard]] static bool extractSingleQubitMatrix(qco::UnitaryOpInterface op,
                                                   Matrix2x2& out) {
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

[[nodiscard]] static bool extractTwoQubitMatrix(qco::UnitaryOpInterface op,
                                                Matrix4x4& out) {
  if (auto ctrl = llvm::dyn_cast<qco::CtrlOp>(op.getOperation())) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<qco::XOp>(body)) {
      out = cxGate01();
      return true;
    }
    if (llvm::isa<qco::ZOp>(body)) {
      out = czGate();
      return true;
    }
    return false;
  }
  return op.getUnitaryMatrix4x4(out);
}

[[nodiscard]] static std::optional<Matrix4x4>
computeTwoQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp) {
  ModuleOp module = moduleOp.get();
  if (!module) {
    return std::nullopt;
  }
  Matrix4x4 unitary = Matrix4x4::identity();
  llvm::DenseMap<Value, std::size_t> qubitIds;
  std::size_t nextQubitId = 0;

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        if (auto alloc = llvm::dyn_cast<qco::AllocOp>(&rawOp)) {
          if (nextQubitId >= 2) {
            return std::nullopt;
          }
          qubitIds.try_emplace(alloc.getResult(), nextQubitId++);
        }
      }
    }
  }

  auto getQubitId = [&](Value qubit) -> std::optional<std::size_t> {
    auto it = qubitIds.find(qubit);
    if (it == qubitIds.end()) {
      return std::nullopt;
    }
    return it->second;
  };

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        auto op = llvm::dyn_cast<qco::UnitaryOpInterface>(&rawOp);
        if (!op) {
          continue;
        }
        if (llvm::isa<qco::BarrierOp, qco::GPhaseOp>(op.getOperation())) {
          continue;
        }

        if (op.isSingleQubit()) {
          const auto qIn = getUnitaryQubitOperand(op, 0);
          if (!qIn) {
            return std::nullopt;
          }
          auto qid = getQubitId(*qIn);
          if (!qid) {
            return std::nullopt;
          }
          Matrix2x2 oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary =
              expandToTwoQubits(oneQ, static_cast<QubitId>(*qid)) * unitary;
          const auto qOut = getUnitaryQubitResult(op, 0);
          if (!qOut) {
            return std::nullopt;
          }
          qubitIds[*qOut] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          const auto q0In = getUnitaryQubitOperand(op, 0);
          const auto q1In = getUnitaryQubitOperand(op, 1);
          if (!q0In || !q1In) {
            return std::nullopt;
          }
          auto q0id = getQubitId(*q0In);
          auto q1id = getQubitId(*q1In);
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Matrix4x4 twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          const llvm::SmallVector<QubitId, 2> ids{static_cast<QubitId>(*q0id),
                                                  static_cast<QubitId>(*q1id)};
          unitary = fixTwoQubitMatrixQubitOrder(twoQ, ids) * unitary;
          const auto q0Out = getUnitaryQubitResult(op, 0);
          const auto q1Out = getUnitaryQubitResult(op, 1);
          if (!q0Out || !q1Out) {
            return std::nullopt;
          }
          qubitIds[*q0Out] = *q0id;
          qubitIds[*q1Out] = *q1id;
          continue;
        }
      }
    }
  }

  if (nextQubitId != 2) {
    return std::nullopt;
  }
  return unitary;
}

static void
expectTwoQubitQcoModulesEquivalent(const OwningOpRef<ModuleOp>& lhs,
                                   const OwningOpRef<ModuleOp>& rhs) {
  const auto lhsUnitary = computeTwoQubitUnitaryFromModule(lhs);
  ASSERT_TRUE(lhsUnitary.has_value());
  const auto rhsUnitary = computeTwoQubitUnitaryFromModule(rhs);
  ASSERT_TRUE(rhsUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*lhsUnitary, *rhsUnitary));
}

struct TwoQFuseFixture {
  std::unique_ptr<MLIRContext> context;

  void setUp() {
    DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                    func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] MLIRContext* ctx() const { return context.get(); }
};

static std::size_t countCtrlOps(const OwningOpRef<ModuleOp>& moduleOp) {
  std::size_t count = 0;
  moduleOp.get()->walk([&](qco::CtrlOp) { ++count; });
  return count;
}

static LogicalResult runQcToQco(ModuleOp moduleOp) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(mlir::createQCToQCO());
  return pm.run(moduleOp);
}

static LogicalResult runTwoQFuse(ModuleOp moduleOp, StringRef nativeGates) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(mlir::qco::createFuseTwoQubitUnitaryRuns(
      mlir::qco::FuseTwoQubitUnitaryRunsOptions{
          .nativeGates = nativeGates.str(),
      }));
  return pm.run(moduleOp);
}

template <typename ProgramT>
static OwningOpRef<ModuleOp> buildProgram(MLIRContext* ctx, ProgramT program) {
  return mlir::qc::QCProgramBuilder::build(ctx, program);
}

template <typename ProgramT>
static void expectTwoQFusePreservesUnitary(MLIRContext* ctx, ProgramT program,
                                           StringRef nativeGates) {
  auto expected = buildProgram(ctx, program);
  ASSERT_TRUE(expected);
  ASSERT_TRUE(succeeded(runQcToQco(*expected)));

  auto fused = buildProgram(ctx, program);
  ASSERT_TRUE(fused);
  ASSERT_TRUE(succeeded(runQcToQco(*fused)));
  ASSERT_TRUE(succeeded(runTwoQFuse(*fused, nativeGates)));
  ASSERT_TRUE(succeeded(verify(*fused)));
  expectTwoQubitQcoModulesEquivalent(expected, fused);
}

//===----------------------------------------------------------------------===//
// FuseTwoQubitUnitaryRuns tests
//===----------------------------------------------------------------------===//

TEST(FuseTwoQubitUnitaryRunsTest, InvalidNativeGatesFailsPass) {
  TwoQFuseFixture fx;
  fx.setUp();
  auto module = buildProgram(fx.ctx(), fusionCxCx);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runQcToQco(*module)));
  EXPECT_TRUE(failed(runTwoQFuse(*module, "not-a-gate")));
}

TEST(FuseTwoQubitUnitaryRunsTest, AdjacentCxCancel) {
  TwoQFuseFixture fx;
  fx.setUp();
  expectTwoQFusePreservesUnitary(fx.ctx(), fusionCxCx, "u,cx");

  auto module = buildProgram(fx.ctx(), fusionCxCx);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runQcToQco(*module)));
  ASSERT_TRUE(succeeded(runTwoQFuse(*module, "u,cx")));
  EXPECT_EQ(countCtrlOps(module), 0U);
}

TEST(FuseTwoQubitUnitaryRunsTest, FusesCxThroughInterleavedOneQOps) {
  TwoQFuseFixture fx;
  fx.setUp();
  expectTwoQFusePreservesUnitary(fx.ctx(), fusionHCxInterleavedTCx, "u,cx");
}

TEST(FuseTwoQubitUnitaryRunsTest, StopsAtDifferentPairBoundary) {
  TwoQFuseFixture fx;
  fx.setUp();
  auto module = buildProgram(fx.ctx(), fusionThreeLineCx);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runQcToQco(*module)));
  ASSERT_TRUE(succeeded(runTwoQFuse(*module, "u,cx")));
  EXPECT_GE(countCtrlOps(module), 1U);
}

TEST(FuseTwoQubitUnitaryRunsTest, DoesNotFuseAcrossBarrier) {
  TwoQFuseFixture fx;
  fx.setUp();
  auto module = buildProgram(fx.ctx(), fusionCxBarrierCx);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runQcToQco(*module)));
  ASSERT_TRUE(succeeded(runTwoQFuse(*module, "u,cx")));
  EXPECT_EQ(countCtrlOps(module), 2U);
}

TEST(FuseTwoQubitUnitaryRunsTest, HandlesSwappedWireOrder) {
  TwoQFuseFixture fx;
  fx.setUp();
  expectTwoQFusePreservesUnitary(fx.ctx(), fusionSwapCxPattern, "u,cx");
}

TEST(FuseTwoQubitUnitaryRunsTest, HandlesRzzBlock) {
  TwoQFuseFixture fx;
  fx.setUp();
  expectTwoQFusePreservesUnitary(fx.ctx(), fusionHRzzSRzz, "x,sx,rz,rx,rzz,cz");
}
