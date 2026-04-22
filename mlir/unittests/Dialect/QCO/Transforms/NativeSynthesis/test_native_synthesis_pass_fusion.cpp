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
// native-gate synthesis pass. Linked with sibling `test_native_synthesis_*.cpp`
// sources into `mqt-core-mlir-unittest-native-synthesis`.

#include "native_synthesis_pass_test_fixture.h"

#include <numbers>

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
} // namespace

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
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    builder.h(q0);
    builder.z(q0);
    builder.h(q0);
    builder.dealloc(q0);
    return builder.finalize();
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
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    builder.h(q0);
    builder.h(q0);
    builder.dealloc(q0);
    return builder.finalize();
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
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    builder.h(q0);
    builder.s(q0);
    builder.t(q0);
    builder.y(q0);
    builder.sx(q0);
    builder.dealloc(q0);
    return builder.finalize();
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
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.h(q0);
        builder.cx(q0, q1);
        builder.h(q0);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps,
      computeTwoQubitUnitaryFromModule);

  auto moduleOp = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.h(q0);
    builder.cx(q0, q1);
    builder.h(q0);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  }();
  runNativeSynthesis(moduleOp, "x,sx,rz,cx");
  // Each H decomposes to rz(pi/2) sx rz(pi/2); without fusion we get two
  // separate decompositions => at least 2 SX gates total.
  EXPECT_GE(countOpsOfTypeInModule<qco::SXOp>(moduleOp), 2U);
}

TEST_F(NativeSynthesisPassTest, OneQRunMergingDoesNotFuseAcrossBarrier) {
  // A barrier between two 1q ops on the same wire interrupts the run:
  // `BarrierOp` is explicitly excluded from fusibility and its use of the
  // qubit breaks the single-use precondition on the intermediate value.
  auto moduleOp = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    builder.h(q0);
    builder.barrier({q0});
    builder.h(q0);
    builder.dealloc(q0);
    return builder.finalize();
  }();
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
  auto moduleOp = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    builder.rz(0.4, q0);
    builder.sx(q0);
    builder.rz(-0.9, q0);
    builder.dealloc(q0);
    return builder.finalize();
  }();
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
  auto moduleOp = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    builder.u(0.3, 0.1, -0.2, q0);
    builder.u(-0.5, 0.7, 0.4, q0);
    builder.dealloc(q0);
    return builder.finalize();
  }();
  runNativeSynthesis(moduleOp, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(moduleOp));
  EXPECT_EQ(countOpsOfTypeInModule<qco::UOp>(moduleOp), 1U);
}

TEST_F(NativeSynthesisPassTest, OneQRunMergingEmitsGlobalPhaseOnU3) {
  // Phase-A GPhase refinement: fusing `T; S` on the generic-u3-cx profile
  // composes to a diagonal matrix whose SU(2) normalisation sheds a
  // non-trivial residual phase of `3*pi/8`. The fusion emitter preserves
  // the phase via a `qco.gphase` op so the synthesized IR reconstructs the
  // original unitary exactly (not merely up to global phase). `T; S` is
  // chosen over `T; T` because `MergeSubsequentT` would otherwise fold the
  // latter to `S` upstream: `T; S` is not matched by any existing
  // canonicalizer, so this test exercises the fusion path unambiguously.
  auto moduleOp = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    builder.t(q0);
    builder.s(q0);
    builder.dealloc(q0);
    return builder.finalize();
  }();
  runNativeSynthesis(moduleOp, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(moduleOp));
  EXPECT_EQ(countOpsOfTypeInModule<qco::UOp>(moduleOp), 1U);
  EXPECT_EQ(countOpsOfTypeInModule<qco::GPhaseOp>(moduleOp), 1U);
}

TEST_F(NativeSynthesisPassTest,
       OneQRunMergingOmitsGPhaseWhenResidualIsTrivial) {
  // Negative complement of OneQRunMergingEmitsGlobalPhaseOnU3: each U3 with
  // `lambda = -phi` has det = 1, so the composed unitary also has det = 1.
  // The fusion path computes an SU(2)-normalised decomposition whose
  // `globalPhase` is negligible, and `emitGPhaseIfNonTrivial` must skip
  // emitting any `qco.gphase` op.
  auto moduleOp = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    builder.u(0.3, 0.2, -0.2, q0);
    builder.u(0.5, 0.4, -0.4, q0);
    builder.dealloc(q0);
    return builder.finalize();
  }();
  runNativeSynthesis(moduleOp, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(moduleOp));
  EXPECT_EQ(countOpsOfTypeInModule<qco::UOp>(moduleOp), 1U);
  EXPECT_EQ(countOpsOfTypeInModule<qco::GPhaseOp>(moduleOp), 0U);
}

TEST_F(NativeSynthesisPassTest,
       OneQRunMergingLongMixedChainEquivalentAcrossProfiles) {
  // A ten-op mixed chain on a single wire must fuse to the correct unitary
  // on every CX-friendly reference profile (see
  // ``fiveCxEntanglerEquivalenceProfiles``), excluding IQM-default ``r,cz``,
  // which uses a different two-qubit path.
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.h(q0);
    builder.t(q0);
    builder.rx(0.37, q0);
    builder.s(q0);
    builder.ry(-0.21, q0);
    builder.h(q0);
    builder.z(q0);
    builder.rz(0.52, q0);
    builder.sx(q0);
    builder.y(q0);
    builder.cx(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
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

TEST_F(NativeSynthesisPassTest, TwoQBlockConsolidationCancelsAdjacentCx) {
  // Two CX(q0,q1) cancel to the identity. The consolidation step folds the
  // pair into a trivial 4x4, which the decomposer realises with zero basis
  // gate uses.
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.cx(q0, q1);
    builder.cx(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  };

  auto expected = buildFn();
  runQcToQco(expected);
  const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
  ASSERT_TRUE(expectedUnitary.has_value());

  auto synth = buildFn();
  runNativeSynthesis(synth, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(synth));
  EXPECT_EQ(countOpsOfTypeInModule<qco::CtrlOp>(synth), 0U);
  const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
  ASSERT_TRUE(synthUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary));
}

TEST_F(NativeSynthesisPassTest,
       TwoQBlockConsolidationFusesCxThroughInterleavedOneQOps) {
  // A non-native block containing interleaved single-qubit ops on the two
  // wires must consolidate into a single 4x4 unitary that the decomposer
  // synthesises with the target's entangler (CX). The resulting circuit
  // must be unitarily equivalent to the original.
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.h(q0);
    builder.cx(q0, q1);
    builder.t(q1);
    builder.s(q0);
    builder.cx(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  };

  auto expected = buildFn();
  runQcToQco(expected);
  const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
  ASSERT_TRUE(expectedUnitary.has_value());

  auto synth = buildFn();
  runNativeSynthesis(synth, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(synth));
  const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
  ASSERT_TRUE(synthUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary));
}

TEST_F(NativeSynthesisPassTest,
       TwoQBlockConsolidationStopsAtDifferentPairBoundary) {
  // Consolidation must not cross a 2q op that touches a different pair of
  // wires. We arrange two back-to-back `cx(q0, q1)` separated by a
  // `cx(q1, q2)` so block consolidation cannot fuse the outer pair into a
  // single identity; equivalence still has to hold.
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    const auto q2 = builder.allocQubit();
    builder.cx(q0, q1);
    builder.cx(q1, q2);
    builder.cx(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    builder.dealloc(q2);
    return builder.finalize();
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
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.cx(q0, q1);
    builder.barrier({q0, q1});
    builder.cx(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  };

  auto synth = buildFn();
  runNativeSynthesis(synth, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(synth));
  // The barrier prevents block consolidation from cancelling the pair, so
  // both CX ops survive as separate entanglers.
  EXPECT_EQ(countOpsOfTypeInModule<qco::CtrlOp>(synth), 2U);
}

TEST_F(NativeSynthesisPassTest, TwoQBlockConsolidationHandlesSwappedWireOrder) {
  // Three CXs in alternating direction form SWAP; consolidation must preserve
  // the unitary.
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.cx(q0, q1);
    builder.cx(q1, q0);
    builder.cx(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  };

  auto expected = buildFn();
  runQcToQco(expected);
  const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
  ASSERT_TRUE(expectedUnitary.has_value());

  auto synth = buildFn();
  runNativeSynthesis(synth, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(synth));
  const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
  ASSERT_TRUE(synthUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary));
}

TEST_F(NativeSynthesisPassTest,
       TwoQBlockConsolidationEquivalentWhenBlockContainsDcx) {
  // Convention audit: DCX is directional/asymmetric, so this checks that
  // Phase-B block accumulation preserves operand ordering when a DCX appears
  // inside an otherwise consolidatable block.
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.h(q0);
    builder.dcx(q0, q1);
    builder.s(q1);
    builder.cx(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  };

  auto expected = buildFn();
  runQcToQco(expected);
  const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
  ASSERT_TRUE(expectedUnitary.has_value());

  auto synth = buildFn();
  runNativeSynthesis(synth, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(synth));
  const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
  ASSERT_TRUE(synthUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary));
}

TEST_F(NativeSynthesisPassTest,
       TwoQBlockConsolidationEquivalentWhenBlockContainsRzx) {
  // Convention audit: RZX is directional/asymmetric. This test guards
  // against BE/LE mismatches in mixed blocks containing RZX.
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.x(q0);
    builder.rzx(0.41, q0, q1);
    builder.t(q1);
    builder.cx(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  };

  auto expected = buildFn();
  runQcToQco(expected);
  const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
  ASSERT_TRUE(expectedUnitary.has_value());

  auto synth = buildFn();
  runNativeSynthesis(synth, "u,cx");
  EXPECT_TRUE(onlyGenericU3CxOps(synth));
  const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
  ASSERT_TRUE(synthUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary));
}

TEST_F(NativeSynthesisPassTest,
       TwoQBlockConsolidationHandlesRzzOnIbmFractional) {
  // Explicitly exercise a non-CX/CZ two-qubit gate inside a block on a
  // profile that supports it natively. Consolidation may keep/reshape the
  // block, but equivalence and profile validity must hold.
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.h(q0);
    builder.rzz(-0.29, q0, q1);
    builder.s(q1);
    builder.rzz(0.17, q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
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

  // Representative generic and ``pi/2`` angles (operand order tested below).
  const std::array<double, 2> angles{{0.41, std::numbers::pi / 2.0}};

  for (const auto& profileCase : profiles) {
    for (const double theta : angles) {
      for (const bool swapOperands : {false, true}) {
        auto buildFn = [&] {
          mlir::qc::QCProgramBuilder builder(context.get());
          builder.initialize();
          const auto q0 = builder.allocQubit();
          const auto q1 = builder.allocQubit();
          if (swapOperands) {
            builder.rzx(theta, q1, q0);
          } else {
            builder.rzx(theta, q0, q1);
          }
          builder.dealloc(q0);
          builder.dealloc(q1);
          return builder.finalize();
        };

        auto expected = buildFn();
        runQcToQco(expected);
        const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
        ASSERT_TRUE(expectedUnitary.has_value())
            << "native-gates=" << profileCase.nativeGates << " theta=" << theta
            << " swapped=" << swapOperands;

        auto synth = buildFn();
        runNativeSynthesis(synth, profileCase.nativeGates);
        EXPECT_TRUE(profileCase.isNative(synth))
            << "native-gates=" << profileCase.nativeGates << " theta=" << theta
            << " swapped=" << swapOperands;
        const auto synthUnitary = computeTwoQubitUnitaryFromModule(synth);
        ASSERT_TRUE(synthUnitary.has_value())
            << "native-gates=" << profileCase.nativeGates << " theta=" << theta
            << " swapped=" << swapOperands;
        EXPECT_TRUE(
            isEquivalentUpToGlobalPhase(*expectedUnitary, *synthUnitary))
            << "native-gates=" << profileCase.nativeGates << " theta=" << theta
            << " swapped=" << swapOperands;
      }
    }
  }
}
