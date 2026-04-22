/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "native_synthesis_pass_test_fixture.h"

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::native_synth_test;

TEST_F(NativeSynthesisPassTest, DecomposesToIbmBasicCxProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.h(q0);
        builder.s(q0);
        builder.t(q0);
        builder.y(q0);
        builder.cx(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesSwapToIbmBasicCxProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.swap(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToGenericU3CxProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.h(q0);
        builder.s(q0);
        builder.t(q0);
        builder.y(q0);
        builder.cx(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "u,cx", &NativeSynthesisPassTest::onlyGenericU3CxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesSwapToGenericU3CxProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.swap(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "u,cx", &NativeSynthesisPassTest::onlyGenericU3CxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesCxToCzForIbmBasicCzProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.h(q1);
        builder.cx(q0, q1);
        builder.t(q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "x,sx,rz,cz", &NativeSynthesisPassTest::onlyIbmBasicCzOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesSwapToIbmBasicCzProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.swap(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "x,sx,rz,cz", &NativeSynthesisPassTest::onlyIbmBasicCzOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesSwapToGenericU3CzProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.swap(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "u,cz", &NativeSynthesisPassTest::onlyGenericU3CzOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToIqmDefaultProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.x(q0);
        builder.y(q0);
        builder.sx(q0);
        builder.cz(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "r,cz", &NativeSynthesisPassTest::onlyIqmDefaultOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToIbmFractionalProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.h(q0);
        builder.ry(0.37, q0);
        builder.sxdg(q0);
        builder.cx(q0, q1);
        builder.rzz(0.23, q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "x,sx,rz,rx,rzz,cz", &NativeSynthesisPassTest::onlyIbmFractionalOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesSwapToIbmFractionalProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.swap(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "x,sx,rz,rx,rzz,cz", &NativeSynthesisPassTest::onlyIbmFractionalOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToAxisPairRxRzCxProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.h(q0);
        builder.y(q0);
        builder.cx(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "rx,rz,cx", &NativeSynthesisPassTest::onlyAxisPairRxRzCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesSwapViaBasisDecomposerAxisPairCx) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.swap(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "rx,rz,cx", &NativeSynthesisPassTest::onlyAxisPairRxRzCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesRzToAxisPairRxRyCxProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.z(q0);
        builder.cx(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "rx,ry,cx", &NativeSynthesisPassTest::onlyAxisPairRxRyCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToAxisPairRyRzCzProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.x(q0);
        builder.h(q0);
        builder.cz(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "ry,rz,cz", &NativeSynthesisPassTest::onlyAxisPairRyRzCzOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesSwapViaBasisDecomposerAxisPairCz) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.swap(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "ry,rz,cz", &NativeSynthesisPassTest::onlyAxisPairRyRzCzOps);
}

TEST_F(NativeSynthesisPassTest, ConvertsCxToCzForAxisPairRyRzCzProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.cx(q0, q1);
        builder.y(q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "ry,rz,cz", &NativeSynthesisPassTest::onlyAxisPairRyRzCzOps);
}

TEST_F(NativeSynthesisPassTest, ConvertsCxToCzForIqmDefaultProfile) {
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.cx(q0, q1);
        builder.y(q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "r,cz", &NativeSynthesisPassTest::onlyIqmDefaultOps);
}

TEST_F(NativeSynthesisPassTest, BroadOneQCanonicalizationIqmDefaultNoLeakage) {
  auto moduleOp = buildBroadOneQCanonicalizationCircuit();
  runNativeSynthesis(moduleOp, "r,cz");
  EXPECT_TRUE(onlyIqmDefaultOps(moduleOp));
}

TEST_F(NativeSynthesisPassTest,
       BroadOneQCanonicalizationAxisPairRyRzCzNoLeakage) {
  auto moduleOp = buildBroadOneQCanonicalizationCircuit();
  runNativeSynthesis(moduleOp, "ry,rz,cz");
  EXPECT_TRUE(onlyAxisPairRyRzCzOps(moduleOp));
}

TEST_F(NativeSynthesisPassTest, BroadOneQCanonicalizationGenericU3CzNoLeakage) {
  auto moduleOp = buildBroadOneQCanonicalizationCircuit();
  runNativeSynthesis(moduleOp, "u,cz");
  EXPECT_TRUE(onlyGenericU3CzOps(moduleOp));
}

TEST_F(NativeSynthesisPassTest, GenericProfileMatchesGenericU3CxBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.h(q0);
        builder.y(q1);
        builder.cx(q0, q1);
        builder.s(q0);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "u,cx", &NativeSynthesisPassTest::onlyGenericU3CxOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, GenericProfileMatchesAxisPairRyRzCzBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.x(q0);
        builder.h(q0);
        builder.cz(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "ry,rz,cz", &NativeSynthesisPassTest::onlyAxisPairRyRzCzOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, ZeroAngleCanonicalizationIqmDefaultNoLeakage) {
  auto moduleOp = buildZeroAngleCanonicalizationCircuit();
  runNativeSynthesis(moduleOp, "r,cz");
  EXPECT_TRUE(onlyIqmDefaultOps(moduleOp));
}

TEST_F(NativeSynthesisPassTest,
       ZeroAngleCanonicalizationAxisPairRyRzNoLeakage) {
  auto moduleOp = buildZeroAngleCanonicalizationCircuit();
  runNativeSynthesis(moduleOp, "ry,rz,cz");
  EXPECT_TRUE(onlyAxisPairRyRzCzOps(moduleOp));
}

TEST_F(NativeSynthesisPassTest, FailsForUnsupportedNativeGateMenu) {
  expectSynthesisFailure(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        builder.h(q0);
        builder.dealloc(q0);
        return builder.finalize();
      },
      "not-a-gate");
}

TEST_F(NativeSynthesisPassTest,
       CustomProfileAcceptsOverlappingOneQSupersetMenu) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.h(q0);
        builder.y(q0);
        builder.cx(q0, q1);
        builder.s(q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "u,rx,rz,cx", &NativeSynthesisPassTest::onlyUOrAxisPairRxRzCxOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, CustomProfileMatchesIbmFractionalBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] { return buildIbmFractionalAllGateFamiliesCircuit(); },
      "x,sx,rz,rx,cz,rzz", &NativeSynthesisPassTest::onlyIbmFractionalOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, CustomProfileAcceptsMultipleEntanglersMenu) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.h(q0);
        builder.cx(q0, q1);
        builder.s(q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "u,cx,cz", &NativeSynthesisPassTest::onlyGenericU3CxOrCzOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest,
       FailsForUnsupportedNativeGateMenuWithoutEmitter) {
  expectSynthesisFailure(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        builder.h(q0);
        builder.dealloc(q0);
        return builder.finalize();
      },
      "rz,cx");
}

TEST_F(NativeSynthesisPassTest, MinimalIbmBasicCustomMenuAcceptsPhaseAlias) {
  // `x,sx,rz,cx` is the minimal IBM-basic style menu. The synthesis pass may
  // represent Z-axis phases using `p`, which should be accepted as an alias of
  // `rz` for custom menus.
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.p(0.13, q0);
        builder.h(q0);
        builder.cx(q0, q1);
        builder.p(-0.27, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps);
}

TEST_F(NativeSynthesisPassTest, LargeMultiQubitCircuitStaysWithinMinimalMenu) {
  // Stress-test: larger circuit (>2 qubits) with many 1Q/2Q ops that should
  // still synthesize into the minimal IBM-basic custom menu.
  expectNativeAfterSynthesis(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();

        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        const auto q2 = builder.allocQubit();
        const auto q3 = builder.allocQubit();
        const auto q4 = builder.allocQubit();

        // A mix of non-native 1Q ops (h/s/t/y) and entanglers (cx/cz/swap)
        // across different pairs.
        builder.h(q0);
        builder.s(q1);
        builder.t(q2);
        builder.y(q3);
        builder.h(q4);

        builder.cx(q0, q1);
        builder.cz(q1, q2);
        builder.swap(q2, q3);
        builder.cx(q3, q4);

        // Add depth with repeated layers.
        for (int layer = 0; layer < 8; ++layer) {
          builder.h(q0);
          builder.s(q0);
          builder.t(q0);

          builder.y(q1);
          builder.h(q2);
          builder.s(q3);
          builder.t(q4);

          builder.cx(q0, q2);
          builder.cz(q1, q3);
          builder.cx(q2, q4);

          if ((layer % 2) == 0) {
            builder.swap(q0, q1);
            builder.swap(q3, q4);
          } else {
            builder.cx(q4, q0);
            builder.cz(q2, q1);
          }
        }

        // Include explicit phases too (these should end up as `rz`/`p`).
        builder.p(0.25, q0);
        builder.p(-0.5, q2);
        builder.p(0.75, q4);

        builder.dealloc(q0);
        builder.dealloc(q1);
        builder.dealloc(q2);
        builder.dealloc(q3);
        builder.dealloc(q4);
        return builder.finalize();
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps);
}

TEST_F(NativeSynthesisPassTest, FailsForNativeGateMenuWithoutSingleQEmitter) {
  expectSynthesisFailure(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        builder.cx(q0, q1);
        builder.dealloc(q0);
        builder.dealloc(q1);
        return builder.finalize();
      },
      "cx,cz");
}

TEST_F(NativeSynthesisPassTest, FailsForNegativeScoreWeight) {
  expectSynthesisFailure(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        builder.h(q0);
        builder.dealloc(q0);
        return builder.finalize();
      },
      "u,cx", -1.0, 0.1, 0.01);
}

TEST_F(NativeSynthesisPassTest, CandidateSelectionIsDeterministicAcrossRuns) {
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.swap(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  };

  auto firstModule = buildFn();
  runNativeSynthesis(firstModule, "u,cx");
  auto secondModule = buildFn();
  runNativeSynthesis(secondModule, "u,cx");

  EXPECT_EQ(moduleToString(firstModule), moduleToString(secondModule));
}

TEST_F(NativeSynthesisPassTest,
       RichCustomMenuSelectionRemainsDeterministicAcrossWeightsAndRuns) {
  auto buildFn = [&] {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    builder.swap(q0, q1);
    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  };

  auto firstModule = buildFn();
  runNativeSynthesis(firstModule, "u,rx,rz,cx,cz", 1.0, 0.1, 0.01);
  auto secondModule = buildFn();
  runNativeSynthesis(secondModule, "u,rx,rz,cx,cz", 1.0, 0.1, 0.01);
  EXPECT_EQ(moduleToString(firstModule), moduleToString(secondModule));

  auto alternateWeightsModule = buildFn();
  runNativeSynthesis(alternateWeightsModule, "u,rx,rz,cx,cz", 3.0, 0.5, 0.0);
  EXPECT_TRUE(onlyUOrAxisPairRxRzCxOps(alternateWeightsModule) ||
              onlyGenericU3CxOrCzOps(alternateWeightsModule));
}

TEST_F(NativeSynthesisPassTest, FailsForMultiControlledGateStructure) {
  expectSynthesisFailure(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        const auto q2 = builder.allocQubit();
        builder.mcx({q0, q1}, q2);
        builder.dealloc(q0);
        builder.dealloc(q1);
        builder.dealloc(q2);
        return builder.finalize();
      },
      "x,sx,rz,cx");
}

TEST_F(NativeSynthesisPassTest, FailsForControlledTwoTargetGateStructure) {
  expectSynthesisFailure(
      [&] {
        mlir::qc::QCProgramBuilder builder(context.get());
        builder.initialize();
        const auto q0 = builder.allocQubit();
        const auto q1 = builder.allocQubit();
        const auto q2 = builder.allocQubit();
        builder.cswap(q0, q1, q2);
        builder.dealloc(q0);
        builder.dealloc(q1);
        builder.dealloc(q2);
        return builder.finalize();
      },
      "x,sx,rz,cx");
}

TEST_F(NativeSynthesisPassTest,
       RandomizedEquivalentAcrossProfilesWithFixedSeed) {
  auto buildStressCircuit = [&](MLIRContext* ctx, const char* nativeGates) {
    mlir::qc::QCProgramBuilder builder(ctx);
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();
    const std::string menu(nativeGates);
    if (menu == "r,cz") {
      builder.r(0.37, -0.42, q0);
      builder.cz(q0, q1);
      builder.r(-0.11, 0.21, q1);
    } else if (menu == "ry,rz,cz") {
      builder.ry(0.37, q0);
      builder.rz(-0.42, q1);
      builder.cz(q0, q1);
      builder.rz(0.21, q0);
    } else if (menu == "rx,ry,cx") {
      builder.rx(0.37, q0);
      builder.ry(-0.42, q1);
      builder.cx(q0, q1);
      builder.ry(0.21, q0);
    } else if (menu == "rx,rz,cx") {
      builder.rx(0.37, q0);
      builder.rz(-0.42, q1);
      builder.cx(q0, q1);
      builder.rz(0.21, q0);
    } else {
      builder.h(q0);
      builder.y(q1);
      builder.cx(q0, q1);
      builder.s(q0);
      builder.cx(q1, q0);
    }

    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  };

  const auto profiles = NativeSynthesisPassTest::allNineEquivalenceProfiles();

  for (const auto& profileCase : profiles) {
    auto synthesizedModule =
        buildStressCircuit(context.get(), profileCase.nativeGates);
    PassManager prePm(synthesizedModule->getContext());
    prePm.addPass(createQCToQCO());
    ASSERT_TRUE(succeeded(prePm.run(*synthesizedModule)));
    const auto expectedUnitary =
        computeTwoQubitUnitaryFromModule(synthesizedModule);
    ASSERT_TRUE(expectedUnitary.has_value());

    PassManager synthPm(synthesizedModule->getContext());
    synthPm.addPass(
        qco::createNativeGateSynthesisPass(qco::NativeGateSynthesisOptions{
            .nativeGates = profileCase.nativeGates,
        }));
    ASSERT_TRUE(succeeded(synthPm.run(*synthesizedModule)))
        << "native-gates=" << profileCase.nativeGates;
    EXPECT_TRUE(profileCase.isNative(synthesizedModule))
        << "native-gates=" << profileCase.nativeGates;

    const auto synthesizedUnitary =
        computeTwoQubitUnitaryFromModule(synthesizedModule);
    ASSERT_TRUE(synthesizedUnitary.has_value());
    EXPECT_TRUE(
        isEquivalentUpToGlobalPhase(*expectedUnitary, *synthesizedUnitary))
        << "native-gates=" << profileCase.nativeGates;
  }
}
