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

namespace {

/// Row for ``native-gates`` menu + IR predicate used by several profile
/// matrices.
struct NativeSynthMenuRow {
  const char* name;
  const char* nativeGates;
  bool (*isNative)(OwningOpRef<ModuleOp>&);
};

} // namespace

class NativeSynthesisSwapProfileTest
    : public NativeSynthesisPassTest,
      public testing::WithParamInterface<NativeSynthMenuRow> {
public:
  using NativeSynthesisPassTest::onlyAxisPairRxRzCxOps;
  using NativeSynthesisPassTest::onlyAxisPairRyRzCzOps;
  using NativeSynthesisPassTest::onlyGenericU3CxOps;
  using NativeSynthesisPassTest::onlyGenericU3CzOps;
  using NativeSynthesisPassTest::onlyIbmBasicCxOps;
  using NativeSynthesisPassTest::onlyIbmBasicCzOps;
  using NativeSynthesisPassTest::onlyIbmFractionalOps;
};

TEST_P(NativeSynthesisSwapProfileTest, DecomposesSwapToProfile) {
  const NativeSynthMenuRow& param = GetParam();
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::swap);
      },
      param.nativeGates, param.isNative);
}

INSTANTIATE_TEST_SUITE_P(
    SwapMenuMatrix, NativeSynthesisSwapProfileTest,
    testing::Values(
        NativeSynthMenuRow{"IbmBasicCx", "x,sx,rz,cx",
                           &NativeSynthesisSwapProfileTest::onlyIbmBasicCxOps},
        NativeSynthMenuRow{"GenericU3Cx", "u,cx",
                           &NativeSynthesisSwapProfileTest::onlyGenericU3CxOps},
        NativeSynthMenuRow{"IbmBasicCz", "x,sx,rz,cz",
                           &NativeSynthesisSwapProfileTest::onlyIbmBasicCzOps},
        NativeSynthMenuRow{"GenericU3Cz", "u,cz",
                           &NativeSynthesisSwapProfileTest::onlyGenericU3CzOps},
        NativeSynthMenuRow{
            "IbmFractional", "x,sx,rz,rx,rzz,cz",
            &NativeSynthesisSwapProfileTest::onlyIbmFractionalOps},
        NativeSynthMenuRow{
            "AxisPairRxRzCx", "rx,rz,cx",
            &NativeSynthesisSwapProfileTest::onlyAxisPairRxRzCxOps},
        NativeSynthMenuRow{
            "AxisPairRyRzCz", "ry,rz,cz",
            &NativeSynthesisSwapProfileTest::onlyAxisPairRyRzCzOps}),
    [](const testing::TestParamInfo<NativeSynthMenuRow>& info) {
      return info.param.name;
    });

class NativeSynthesisHstycxMenuTest
    : public NativeSynthesisPassTest,
      public testing::WithParamInterface<NativeSynthMenuRow> {
public:
  using NativeSynthesisPassTest::onlyGenericU3CxOps;
  using NativeSynthesisPassTest::onlyIbmBasicCxOps;
};

TEST_P(NativeSynthesisHstycxMenuTest, DecomposesHstycxTwoQToProfile) {
  const NativeSynthMenuRow& param = GetParam();
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesHstycxTwoQ);
      },
      param.nativeGates, param.isNative);
}

INSTANTIATE_TEST_SUITE_P(
    HstycxTwoQMenuMatrix, NativeSynthesisHstycxMenuTest,
    testing::Values(
        NativeSynthMenuRow{"IbmBasicCx", "x,sx,rz,cx",
                           &NativeSynthesisHstycxMenuTest::onlyIbmBasicCxOps},
        NativeSynthMenuRow{"GenericU3Cx", "u,cx",
                           &NativeSynthesisHstycxMenuTest::onlyGenericU3CxOps}),
    [](const testing::TestParamInfo<NativeSynthMenuRow>& info) {
      return info.param.name;
    });

class NativeSynthesisCxYOnQ1MenuTest
    : public NativeSynthesisPassTest,
      public testing::WithParamInterface<NativeSynthMenuRow> {
public:
  using NativeSynthesisPassTest::onlyAxisPairRyRzCzOps;
  using NativeSynthesisPassTest::onlyIqmDefaultOps;
};

TEST_P(NativeSynthesisCxYOnQ1MenuTest, ConvertsCxToCzForProfile) {
  const NativeSynthMenuRow& param = GetParam();
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesCxYOnQ1);
      },
      param.nativeGates, param.isNative);
}

INSTANTIATE_TEST_SUITE_P(
    CxYOnQ1MenuMatrix, NativeSynthesisCxYOnQ1MenuTest,
    testing::Values(
        NativeSynthMenuRow{
            "AxisPairRyRzCz", "ry,rz,cz",
            &NativeSynthesisCxYOnQ1MenuTest::onlyAxisPairRyRzCzOps},
        NativeSynthMenuRow{"IqmDefault", "r,cz",
                           &NativeSynthesisCxYOnQ1MenuTest::onlyIqmDefaultOps}),
    [](const testing::TestParamInfo<NativeSynthMenuRow>& info) {
      return info.param.name;
    });

class NativeSynthesisBroadOneQMenuTest
    : public NativeSynthesisPassTest,
      public testing::WithParamInterface<NativeSynthMenuRow> {
public:
  using NativeSynthesisPassTest::onlyAxisPairRyRzCzOps;
  using NativeSynthesisPassTest::onlyGenericU3CzOps;
  using NativeSynthesisPassTest::onlyIqmDefaultOps;
};

TEST_P(NativeSynthesisBroadOneQMenuTest, CanonicalizationNoLeakage) {
  const NativeSynthMenuRow& param = GetParam();
  auto moduleOp = buildBroadOneQCanonicalizationCircuit();
  runNativeSynthesis(moduleOp, param.nativeGates);
  EXPECT_TRUE(param.isNative(moduleOp));
}

INSTANTIATE_TEST_SUITE_P(
    BroadOneQMenuMatrix, NativeSynthesisBroadOneQMenuTest,
    testing::Values(
        NativeSynthMenuRow{
            "IqmDefault", "r,cz",
            &NativeSynthesisBroadOneQMenuTest::onlyIqmDefaultOps},
        NativeSynthMenuRow{
            "AxisPairRyRzCz", "ry,rz,cz",
            &NativeSynthesisBroadOneQMenuTest::onlyAxisPairRyRzCzOps},
        NativeSynthMenuRow{
            "GenericU3Cz", "u,cz",
            &NativeSynthesisBroadOneQMenuTest::onlyGenericU3CzOps}),
    [](const testing::TestParamInfo<NativeSynthMenuRow>& info) {
      return info.param.name;
    });

class NativeSynthesisZeroAngleMenuTest
    : public NativeSynthesisPassTest,
      public testing::WithParamInterface<NativeSynthMenuRow> {
public:
  using NativeSynthesisPassTest::onlyAxisPairRyRzCzOps;
  using NativeSynthesisPassTest::onlyIqmDefaultOps;
};

TEST_P(NativeSynthesisZeroAngleMenuTest, CanonicalizationNoLeakage) {
  const NativeSynthMenuRow& param = GetParam();
  auto moduleOp = buildZeroAngleCanonicalizationCircuit();
  runNativeSynthesis(moduleOp, param.nativeGates);
  EXPECT_TRUE(param.isNative(moduleOp));
}

INSTANTIATE_TEST_SUITE_P(
    ZeroAngleMenuMatrix, NativeSynthesisZeroAngleMenuTest,
    testing::Values(
        NativeSynthMenuRow{
            "IqmDefault", "r,cz",
            &NativeSynthesisZeroAngleMenuTest::onlyIqmDefaultOps},
        NativeSynthMenuRow{
            "AxisPairRyRzCz", "ry,rz,cz",
            &NativeSynthesisZeroAngleMenuTest::onlyAxisPairRyRzCzOps}),
    [](const testing::TestParamInfo<NativeSynthMenuRow>& info) {
      return info.param.name;
    });

TEST_F(NativeSynthesisPassTest, DecomposesCxToCzForIbmBasicCzProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesHCxTOnQ1);
      },
      "x,sx,rz,cz", &NativeSynthesisPassTest::onlyIbmBasicCzOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToIqmDefaultProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesXYSXCz);
      },
      "r,cz", &NativeSynthesisPassTest::onlyIqmDefaultOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToIbmFractionalProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesFractionalChain);
      },
      "x,sx,rz,rx,rzz,cz", &NativeSynthesisPassTest::onlyIbmFractionalOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToAxisPairRxRzCxProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesHYcx);
      },
      "rx,rz,cx", &NativeSynthesisPassTest::onlyAxisPairRxRzCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesRzToAxisPairRxRyCxProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesZCx);
      },
      "rx,ry,cx", &NativeSynthesisPassTest::onlyAxisPairRxRyCxOps);
}

/// Single-control / single-target QC→QCO ``ctrl`` shells from
/// ``allSingleControlledGateFamiliesOneCtrlOneTarget`` must reach the generic
/// ``u,cx`` menu.
TEST_F(NativeSynthesisPassTest,
       AllSingleControlledOneCtrlOneTargetFamiliesReachesU3Cx) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(),
            mlir::qc::
                nativeSynthAllSingleControlledGateFamiliesOneCtrlOneTarget);
      },
      "u,cx", &NativeSynthesisPassTest::onlyGenericU3CxOps);
}

TEST_F(NativeSynthesisPassTest, GenericProfileMatchesGenericU3CxBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesHq0Yq1CxSq0);
      },
      "u,cx", &NativeSynthesisPassTest::onlyGenericU3CxOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, GenericProfileMatchesAxisPairRyRzCzBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesXHCz);
      },
      "ry,rz,cz", &NativeSynthesisPassTest::onlyAxisPairRyRzCzOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, FailsForUnsupportedNativeGateMenu) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::h);
      },
      "not-a-gate");
}

TEST_F(NativeSynthesisPassTest,
       CustomProfileAcceptsOverlappingOneQSupersetMenu) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesHYSameWireCxSq1);
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
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesHCxSq1);
      },
      "u,cx,cz", &NativeSynthesisPassTest::onlyGenericU3CxOrCzOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest,
       FailsForUnsupportedNativeGateMenuWithoutEmitter) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::h);
      },
      "rz,cx");
}

TEST_F(NativeSynthesisPassTest, MinimalIbmBasicCustomMenuAcceptsPhaseAlias) {
  // `x,sx,rz,cx` is the minimal IBM-basic style menu. The synthesis pass may
  // represent Z-axis phases using `p`, which should be accepted as an alias of
  // `rz` for custom menus.
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthProfilesPhaseHCxPhase);
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps);
}

TEST_F(NativeSynthesisPassTest, LargeMultiQubitCircuitStaysWithinMinimalMenu) {
  // Stress-test: larger circuit (>2 qubits) with many 1Q/2Q ops that should
  // still synthesize into the minimal IBM-basic custom menu.
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(),
            mlir::qc::nativeSynthProfilesLargeFiveQStressEightLayers);
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps);
}

TEST_F(NativeSynthesisPassTest, FailsForNativeGateMenuWithoutSingleQEmitter) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 mlir::qc::singleControlledX);
      },
      "cx,cz");
}

TEST_F(NativeSynthesisPassTest, FailsForNegativeScoreWeight) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::h);
      },
      "u,cx", -1.0, 0.1, 0.01);
}

TEST_F(NativeSynthesisPassTest, CandidateSelectionIsDeterministicAcrossRuns) {
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthDeterminismTwoQubitSwap);
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
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthDeterminismTwoQubitSwap);
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
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 mlir::qc::multipleControlledX);
      },
      "x,sx,rz,cx");
}

TEST_F(NativeSynthesisPassTest, FailsForControlledTwoTargetGateStructure) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::singleControlledSwap);
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
