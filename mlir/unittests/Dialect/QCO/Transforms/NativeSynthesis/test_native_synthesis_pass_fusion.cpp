/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// 1q run merging, 2q block consolidation, and RZX profile sweeps for the
// native-gate synthesis pass. Includes a few ``TEST_P`` matrices (U3 GPhase
// pair, generic-u3-cx two-qubit equivalence rows). Linked with sibling
// ``test_native_synthesis_*.cpp`` sources into
// ``mqt-core-mlir-unittest-native-synthesis``.

#include "native_synthesis_pass_test_fixture.h"

#include <array>
#include <numbers>
#include <optional>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::native_synth_test;

namespace {
// Count ops of a given MLIR op type across a module; used to assert the
// effects of the 1q-run-merging pre-synthesis step on concrete programs.
template <typename... OpTs>
std::size_t countOpsOfTypeInModule(const OwningOpRef<ModuleOp>& moduleOp) {
  std::size_t count = 0;
  moduleOp.get()->walk([&](mlir::Operation* op) {
    if (isa<OpTs...>(op)) {
      ++count;
    }
  });
  return count;
}

struct OneQU3FusionGPhaseRow {
  const char* name;
  void (*program)(mlir::qc::QCProgramBuilder&);
  unsigned expectGPhaseCount;
};

struct TwoQBlockEquivGenericU3CxRow {
  const char* name;
  void (*program)(mlir::qc::QCProgramBuilder&);
  std::optional<unsigned> expectExactCtrlOpCount;
};
} // namespace

class NativeSynthesisOneQFusionU3GPhaseTest
    : public NativeSynthesisPassTest,
      public testing::WithParamInterface<OneQU3FusionGPhaseRow> {
public:
  using NativeSynthesisPassTest::onlyGenericU3CxOps;
};

TEST_P(NativeSynthesisOneQFusionU3GPhaseTest, FusesAdjacentNativeUChain) {
  const OneQU3FusionGPhaseRow& param = GetParam();
  auto moduleOp =
      mlir::qc::QCProgramBuilder::build(context.get(), param.program);
  runNativeSynthesis(moduleOp, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(moduleOp));
  EXPECT_EQ(countOpsOfTypeInModule<qco::UOp>(moduleOp), 1U);
  EXPECT_EQ(countOpsOfTypeInModule<qco::GPhaseOp>(moduleOp),
            param.expectGPhaseCount);
}

INSTANTIATE_TEST_SUITE_P(
    OneQRunMergingU3GPhaseMatrix, NativeSynthesisOneQFusionU3GPhaseTest,
    testing::Values(OneQU3FusionGPhaseRow{"EmitsGlobalPhaseOnU3",
                                          mlir::qc::nativeSynthFusionTS,
                                          /*expectGPhaseCount=*/1U},
                    OneQU3FusionGPhaseRow{"OmitsGPhaseWhenResidualIsTrivial",
                                          mlir::qc::nativeSynthFusionUUTwoQDet1,
                                          /*expectGPhaseCount=*/0U}),
    [](const testing::TestParamInfo<OneQU3FusionGPhaseRow>& info) {
      return info.param.name;
    });

class NativeSynthesisTwoQBlockEquivGenericU3CxTest
    : public NativeSynthesisPassTest,
      public testing::WithParamInterface<TwoQBlockEquivGenericU3CxRow> {
public:
  using NativeSynthesisPassTest::onlyGenericU3CxOps;
};

TEST_P(NativeSynthesisTwoQBlockEquivGenericU3CxTest,
       EquivalentUnderConsolidation) {
  const TwoQBlockEquivGenericU3CxRow& param = GetParam();
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(context.get(), param.program);
  };

  auto expected = buildFn();
  runQcToQco(expected);
  const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
  ASSERT_TRUE(expectedUnitary.has_value());

  auto synth = buildFn();
  runNativeSynthesis(synth, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(synth));
  if (param.expectExactCtrlOpCount.has_value()) {
    EXPECT_EQ(countOpsOfTypeInModule<qco::CtrlOp>(synth),
              *param.expectExactCtrlOpCount);
  }
  const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
  ASSERT_TRUE(synthUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary));
}

INSTANTIATE_TEST_SUITE_P(
    TwoQBlockEquivGenericU3CxMatrix,
    NativeSynthesisTwoQBlockEquivGenericU3CxTest,
    testing::Values(
        TwoQBlockEquivGenericU3CxRow{"AdjacentCxCancel",
                                     mlir::qc::nativeSynthFusionCxCx,
                                     /*expectExactCtrlOpCount=*/0U},
        TwoQBlockEquivGenericU3CxRow{
            "FusesCxThroughInterleavedOneQOps",
            mlir::qc::nativeSynthFusionHCxInterleavedTCx, std::nullopt},
        TwoQBlockEquivGenericU3CxRow{"HandlesSwappedWireOrder",
                                     mlir::qc::nativeSynthFusionSwapCxPattern,
                                     std::nullopt},
        TwoQBlockEquivGenericU3CxRow{"EquivalentWhenBlockContainsDcx",
                                     mlir::qc::nativeSynthFusionHDcxSCx,
                                     std::nullopt},
        TwoQBlockEquivGenericU3CxRow{"EquivalentWhenBlockContainsRzx",
                                     mlir::qc::nativeSynthFusionXRzxTCx,
                                     std::nullopt}),
    [](const testing::TestParamInfo<TwoQBlockEquivGenericU3CxRow>& info) {
      return info.param.name;
    });

// --- 1q-run-merging pre-synthesis step ---
//
// The tests below exercise the in-pass run merging that fuses adjacent
// single-qubit `UnitaryOpInterface` ops on the same wire before per-op
// native-gate emission. They cover (a) the reductions unlocked by fusion,
// (b) that the fusion respects boundaries (CX, barrier, multi-use), and
// (c) unitary equivalence over longer mixed chains.

TEST_F(NativeSynthesisPassTest, OneQRunMergingCollapsesHadamardZHadamardToX) {
  // H * Z * H = X (up to global phase). With fusion enabled, the ibm-basic
  // emitter hits the ZSXX X-shortcut and emits a single X, whereas without
  // fusion we would expect at least 3 RZ gates from two H decompositions and
  // the Z.
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthFusionHadamardZHadamard);
  };

  auto moduleOp = buildFn();
  runNativeSynthesis(moduleOp, "x,sx,rz,cx");
  EXPECT_TRUE(onlyIbmBasicCxOps(moduleOp));
  EXPECT_EQ(countOpsOfTypeInModule<qco::XOp>(moduleOp), 1U);
  EXPECT_EQ(countOpsOfTypeInModule<qco::RZOp>(moduleOp), 0U);
  EXPECT_EQ(countOpsOfTypeInModule<qco::SXOp>(moduleOp), 0U);
}

TEST_F(NativeSynthesisPassTest, OneQRunMergingCancelsAdjacentSelfInverses) {
  // H * H = I. Fusion collapses the run to no 1q ops at all.
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthFusionHadamardHadamard);
  };

  auto moduleOp = buildFn();
  runNativeSynthesis(moduleOp, "x,sx,rz,cx");
  EXPECT_TRUE(onlyIbmBasicCxOps(moduleOp));
  EXPECT_EQ(countOpsOfTypeInModule<qco::XOp>(moduleOp), 0U);
  EXPECT_EQ(countOpsOfTypeInModule<qco::SXOp>(moduleOp), 0U);
  EXPECT_EQ(countOpsOfTypeInModule<qco::RZOp>(moduleOp), 0U);
}

TEST_F(NativeSynthesisPassTest, OneQRunMergingReducesMixedChainToSingleU) {
  // A long chain of distinct 1q ops on a single wire still collapses to a
  // single UOp on the generic-u3-cx profile via fusion, regardless of the
  // mix of non-native ops in the input.
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthFusionMixedChainHSTYSX);
  };

  auto moduleOp = buildFn();
  runNativeSynthesis(moduleOp, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(moduleOp));
  EXPECT_EQ(countOpsOfTypeInModule<qco::UOp>(moduleOp), 1U);
}

TEST_F(NativeSynthesisPassTest, OneQRunMergingDoesNotFuseAcrossCX) {
  // H(q0); CX(q0,q1); H(q0) must NOT be fused because CX breaks the run
  // on q0. Equivalence still holds; to witness that fusion did not happen
  // we assert we still see >=2 SX gates (one from each Hadamard expansion).
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthFusionHadamardCxHadamard);
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps,
      computeTwoQubitUnitaryFromModule);

  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::nativeSynthFusionHadamardCxHadamard);
  runNativeSynthesis(moduleOp, "x,sx,rz,cx");
  // Each H decomposes to rz(pi/2) sx rz(pi/2); without fusion we get two
  // separate decompositions => at least 2 SX gates total.
  EXPECT_GE(countOpsOfTypeInModule<qco::SXOp>(moduleOp), 2U);
}

TEST_F(NativeSynthesisPassTest, OneQRunMergingDoesNotFuseAcrossBarrier) {
  // A barrier between two 1q ops on the same wire interrupts the run:
  // `BarrierOp` is explicitly excluded from fusibility and its use of the
  // qubit breaks the single-use precondition on the intermediate value.
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::nativeSynthFusionHadamardBarrierHadamard);
  runNativeSynthesis(moduleOp, "x,sx,rz,cx");
  EXPECT_TRUE(onlyIbmBasicCxOps(moduleOp));
  // Two separate H decompositions survive => at least 2 SX gates.
  EXPECT_GE(countOpsOfTypeInModule<qco::SXOp>(moduleOp), 2U);
}

TEST_F(NativeSynthesisPassTest, OneQRunMergingSkipsFullyNativeRuns) {
  // A run consisting entirely of ops that are already native to the
  // ibm-basic-cx profile (rz; sx; rz) is pass-through: the cost gate only
  // fuses a fully-native run when fusion would produce strictly fewer ops
  // than the original run. For `rz; sx; rz` the ZSXX decomposition of the
  // fused matrix is itself three ops, so the run is left untouched.
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::nativeSynthFusionRzSxRz);
  runNativeSynthesis(moduleOp, "x,sx,rz,cx");
  EXPECT_TRUE(onlyIbmBasicCxOps(moduleOp));
  EXPECT_EQ(countOpsOfTypeInModule<qco::RZOp>(moduleOp), 2U);
  EXPECT_EQ(countOpsOfTypeInModule<qco::SXOp>(moduleOp), 1U);
}

TEST_F(NativeSynthesisPassTest,
       OneQRunMergingCostGateFusesFullyNativeUChainGenericU3) {
  // Phase-A cost-gate refinement (fully-native path): two adjacent native
  // `u` ops on the same wire fuse into a single `u` because U3 mode always
  // emits exactly one gate per fused 2x2 unitary. Without the cost gate,
  // the fully-native run would be skipped; without fusion, the run would
  // survive as two ops because there is no `MergeSubsequentU` canonicalizer.
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::nativeSynthFusionUUTwoQGenericU3);
  runNativeSynthesis(moduleOp, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(moduleOp));
  EXPECT_EQ(countOpsOfTypeInModule<qco::UOp>(moduleOp), 1U);
}

// GPhase expectations for adjacent native ``U`` fusion are covered by
// ``OneQRunMergingU3GPhaseMatrix`` (``EmitsGlobalPhaseOnU3`` /
// ``OmitsGPhaseWhenResidualIsTrivial``).

TEST_F(NativeSynthesisPassTest,
       OneQRunMergingLongMixedChainEquivalentAcrossProfiles) {
  // A ten-op mixed chain on a single wire must fuse to the correct unitary
  // on every CX-friendly reference profile (see
  // ``fiveCxEntanglerEquivalenceProfiles``), excluding IQM-default ``r,cz``,
  // which uses a different two-qubit path.
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthFusionLongMixedTenOpCx);
  };

  const auto profiles =
      NativeSynthesisPassTest::fiveCxEntanglerEquivalenceProfiles();
  for (const auto& pc : profiles) {
    // Expected and synthesized unitaries both come from the permissive
    // default helper, which understands the full alphabet the builder emits
    // and the R/RX/RY/RZ/U/P gates produced by synthesis.
    auto expected = buildFn();
    runQcToQco(expected);
    const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
    ASSERT_TRUE(expectedUnitary.has_value())
        << "native-gates=" << pc.nativeGates;

    auto synth = buildFn();
    runNativeSynthesis(synth, pc.nativeGates);
    EXPECT_TRUE(pc.isNative(synth)) << "native-gates=" << pc.nativeGates;
    const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
    ASSERT_TRUE(synthUnitary.has_value()) << "native-gates=" << pc.nativeGates;
    EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary))
        << "native-gates=" << pc.nativeGates;
  }
}

// --- 2q-block-consolidation pre-synthesis step (Phase B) ---
//
// These tests exercise the in-pass 2q block consolidation that collects
// adjacent two-qubit ops (plus interleaved single-qubit ops) acting on the
// same pair of wires, composes a 4x4 unitary, and re-synthesizes the block
// via `TwoQubitBasisDecomposer`. They cover (a) reductions unlocked by
// consolidation, (b) fully-native blocks that are only rewritten when
// strictly shorter, and (c) boundary conditions such as wire swaps and
// interleaved barriers.

// Generic-u3-cx two-qubit block equivalence rows (including adjacent-CX
// cancellation) live in ``TwoQBlockEquivGenericU3CxMatrix``.

TEST_F(NativeSynthesisPassTest,
       TwoQBlockConsolidationStopsAtDifferentPairBoundary) {
  // Consolidation must not cross a 2q op that touches a different pair of
  // wires. We arrange two back-to-back `cx(q0, q1)` separated by a
  // `cx(q1, q2)` so block consolidation cannot fuse the outer pair into a
  // single identity; equivalence still has to hold.
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthFusionThreeLineCx01Cx12Cx01);
  };

  auto synth = buildFn();
  runNativeSynthesis(synth, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(synth));
  // At least the middle CX(q1,q2) must survive because its pair differs
  // from the outer CX(q0,q1) block; consolidation cannot eliminate it.
  EXPECT_GE(countOpsOfTypeInModule<qco::CtrlOp>(synth), 1U);
}

TEST_F(NativeSynthesisPassTest,
       TwoQBlockConsolidationDoesNotFuseAcrossBarrier) {
  // A barrier between two CX(q0,q1) blocks must prevent them from being
  // fused into a single block. Each CX stays an individual entangler.
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthFusionCxBarrierCx);
  };

  auto synth = buildFn();
  runNativeSynthesis(synth, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(synth));
  // The barrier prevents block consolidation from cancelling the pair, so
  // both CX ops survive as separate entanglers.
  EXPECT_EQ(countOpsOfTypeInModule<qco::CtrlOp>(synth), 2U);
}

TEST_F(NativeSynthesisPassTest,
       TwoQBlockConsolidationHandlesRzzOnIbmFractional) {
  // Explicitly exercise a non-CX/CZ two-qubit gate inside a block on a
  // profile that supports it natively. Consolidation may keep/reshape the
  // block, but equivalence and profile validity must hold.
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthFusionHRzzSRzz);
  };

  auto expected = buildFn();
  runQcToQco(expected);
  const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
  ASSERT_TRUE(expectedUnitary.has_value());

  auto synth = buildFn();
  runNativeSynthesis(synth, "x,sx,rz,rx,rzz,cz");
  EXPECT_TRUE(onlyIbmFractionalOps(synth));
  const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
  ASSERT_TRUE(synthUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary));
}

TEST_F(NativeSynthesisPassTest,
       RzxStandaloneSynthesisEquivalentAcrossProfiles) {
  // Directed RZX tests (asymmetric 2q); both operand orders.
  const auto profiles = NativeSynthesisPassTest::allNineEquivalenceProfiles();

  // Four directed RZX fixtures: two angles × two operand orders.
  struct RzxStandaloneRow {
    double theta;
    bool swapOperands;
    void (*program)(mlir::qc::QCProgramBuilder&);
  };
  const std::array<RzxStandaloneRow, 4> rzxRows{{
      RzxStandaloneRow{.theta = 0.41,
                       .swapOperands = false,
                       .program = mlir::qc::nativeSynthFusionRzx041Q0First},
      RzxStandaloneRow{.theta = 0.41,
                       .swapOperands = true,
                       .program = mlir::qc::nativeSynthFusionRzx041Q1First},
      RzxStandaloneRow{.theta = std::numbers::pi / 2.0,
                       .swapOperands = false,
                       .program = mlir::qc::nativeSynthFusionRzxPiHalfQ0First},
      RzxStandaloneRow{.theta = std::numbers::pi / 2.0,
                       .swapOperands = true,
                       .program = mlir::qc::nativeSynthFusionRzxPiHalfQ1First},
  }};

  for (const auto& profileCase : profiles) {
    for (const RzxStandaloneRow& row : rzxRows) {
      auto buildFn = [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), row.program);
      };

      auto expected = buildFn();
      runQcToQco(expected);
      const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
      ASSERT_TRUE(expectedUnitary.has_value())
          << "native-gates=" << profileCase.nativeGates
          << " theta=" << row.theta << " swapped=" << row.swapOperands;

      auto synth = buildFn();
      runNativeSynthesis(synth, profileCase.nativeGates);
      EXPECT_TRUE(profileCase.isNative(synth))
          << "native-gates=" << profileCase.nativeGates
          << " theta=" << row.theta << " swapped=" << row.swapOperands;
      const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
      ASSERT_TRUE(synthUnitary.has_value())
          << "native-gates=" << profileCase.nativeGates
          << " theta=" << row.theta << " swapped=" << row.swapOperands;
      EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary))
          << "native-gates=" << profileCase.nativeGates
          << " theta=" << row.theta << " swapped=" << row.swapOperands;
    }
  }
}
