/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// Scoring helpers for native-gate synthesis plus ``XXPlusYY`` / ``XXMinusYY``
// rewrite metric checks.

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Scoring.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/TwoQubit.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"
#include "native_synthesis_pass_test_fixture.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Conversion/QCToQCO/QCToQCO.h>
#include <mlir/Dialect/QC/Builder/QCProgramBuilder.h>
#include <mlir/Dialect/QCO/IR/QCOInterfaces.h>
#include <mlir/Dialect/QCO/IR/QCOOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::native_synth_test;

namespace {

/// Dummy payload: scoring helpers do not inspect the type.
struct ScoringTag {};

} // namespace

static std::pair<unsigned, unsigned>
countSingleAndTwoQubitUnitariesForXxRzzMetrics(ModuleOp module) {
  unsigned numOneQ = 0;
  unsigned numTwoQ = 0;
  module.walk([&](Operation* op) {
    if (llvm::isa<qco::BarrierOp, qco::GPhaseOp>(op)) {
      return;
    }
    if (llvm::isa_and_present<qco::CtrlOp>(op->getParentOp())) {
      return;
    }
    auto unitary = llvm::dyn_cast<qco::UnitaryOpInterface>(op);
    if (!unitary) {
      return;
    }
    if (unitary.isSingleQubit()) {
      ++numOneQ;
      return;
    }
    if (unitary.isTwoQubit()) {
      ++numTwoQ;
    }
  });
  return {numOneQ, numTwoQ};
}

TEST(NativeSynthesisScoringTest, ValidScoreWeights) {
  using namespace mlir::qco::native_synth;
  EXPECT_TRUE(areValidScoreWeights(ScoreWeights{}));
  EXPECT_TRUE(areValidScoreWeights(
      ScoreWeights{.twoQ = 0.0, .oneQ = 0.0, .depth = 0.0}));
  EXPECT_TRUE(areValidScoreWeights(
      ScoreWeights{.twoQ = 5.0, .oneQ = 2.5, .depth = 0.1}));

  EXPECT_FALSE(areValidScoreWeights(ScoreWeights{.twoQ = -1.0}));
  EXPECT_FALSE(areValidScoreWeights(ScoreWeights{.oneQ = -0.1}));
  EXPECT_FALSE(areValidScoreWeights(ScoreWeights{.depth = -0.01}));

  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(areValidScoreWeights(ScoreWeights{.twoQ = inf}));
  EXPECT_FALSE(areValidScoreWeights(ScoreWeights{.oneQ = inf}));
  EXPECT_FALSE(areValidScoreWeights(ScoreWeights{.depth = inf}));
  EXPECT_FALSE(areValidScoreWeights(ScoreWeights{.twoQ = nan}));
  EXPECT_FALSE(areValidScoreWeights(ScoreWeights{.oneQ = nan}));
  EXPECT_FALSE(areValidScoreWeights(ScoreWeights{.depth = nan}));
}

TEST(NativeSynthesisScoringTest, ScoreCandidateAppliesWeightsLinearly) {
  using namespace mlir::qco::native_synth;
  SynthesisCandidate<ScoringTag> candidate;
  candidate.metrics.numTwoQ = 3;
  candidate.metrics.numOneQ = 5;
  candidate.metrics.depth = 7;
  candidate.candidateClass = CandidateClass::DirectSingleQ;
  candidate.enumerationIndex = 11;

  const ScoreWeights weights{.twoQ = 2.0, .oneQ = 0.5, .depth = 0.1};
  const auto score = scoreCandidate(candidate, weights);

  EXPECT_DOUBLE_EQ(score.weighted, (2.0 * 3.0) + (0.5 * 5.0) + (0.1 * 7.0));
  EXPECT_EQ(score.numTwoQ, 3U);
  EXPECT_EQ(score.numOneQ, 5U);
  EXPECT_EQ(score.depth, 7U);
  EXPECT_EQ(score.tieBreakClass,
            static_cast<unsigned>(CandidateClass::DirectSingleQ));
  EXPECT_EQ(score.enumerationIndex, 11U);
}

TEST(NativeSynthesisScoringTest, IsBetterScoreComparesWeightedFirst) {
  using namespace mlir::qco::native_synth;
  const CandidateScore lower{
      .weighted = 1.0, .numTwoQ = 10, .depth = 100, .numOneQ = 1000};
  const CandidateScore higher{.weighted = 2.0};
  EXPECT_TRUE(isBetterScore(lower, higher));
  EXPECT_FALSE(isBetterScore(higher, lower));
  EXPECT_FALSE(isBetterScore(lower, lower));
}

TEST(NativeSynthesisScoringTest, IsBetterScoreTieBreaksInDeclaredOrder) {
  using namespace mlir::qco::native_synth;
  const CandidateScore anchor{.weighted = 1.0,
                              .numTwoQ = 5,
                              .depth = 5,
                              .numOneQ = 5,
                              .tieBreakClass = 5,
                              .enumerationIndex = 5};

  const CandidateScore fewerTwoQ{.weighted = 1.0,
                                 .numTwoQ = 4,
                                 .depth = 99,
                                 .numOneQ = 99,
                                 .tieBreakClass = 99,
                                 .enumerationIndex = 99};
  EXPECT_TRUE(isBetterScore(fewerTwoQ, anchor));

  const CandidateScore lowerDepth{.weighted = 1.0,
                                  .numTwoQ = 5,
                                  .depth = 4,
                                  .numOneQ = 99,
                                  .tieBreakClass = 99,
                                  .enumerationIndex = 99};
  EXPECT_TRUE(isBetterScore(lowerDepth, anchor));

  const CandidateScore fewerOneQ{.weighted = 1.0,
                                 .numTwoQ = 5,
                                 .depth = 5,
                                 .numOneQ = 4,
                                 .tieBreakClass = 99,
                                 .enumerationIndex = 99};
  EXPECT_TRUE(isBetterScore(fewerOneQ, anchor));

  const CandidateScore lowerClass{.weighted = 1.0,
                                  .numTwoQ = 5,
                                  .depth = 5,
                                  .numOneQ = 5,
                                  .tieBreakClass = 0,
                                  .enumerationIndex = 99};
  EXPECT_TRUE(isBetterScore(lowerClass, anchor));

  const CandidateScore lowerEnum{.weighted = 1.0,
                                 .numTwoQ = 5,
                                 .depth = 5,
                                 .numOneQ = 5,
                                 .tieBreakClass = 5,
                                 .enumerationIndex = 0};
  EXPECT_TRUE(isBetterScore(lowerEnum, anchor));
}

TEST(NativeSynthesisScoringTest, IsBetterScoreTreatsCloseWeightedAsTie) {
  using namespace mlir::qco::native_synth;
  const CandidateScore a{.weighted = 1.0, .numTwoQ = 1};
  const CandidateScore b{.weighted = 1.0 + 1e-13, .numTwoQ = 0};
  EXPECT_TRUE(isBetterScore(b, a));
}

TEST(NativeSynthesisScoringTest, IsBetterScoreFallsBackToTupleTieBreak) {
  using namespace mlir::qco::native_synth;
  // Within tolerance: force the lexicographic tuple comparison path.
  const CandidateScore lhs{.weighted = 2.0 + 1e-13,
                           .numTwoQ = 3,
                           .depth = 4,
                           .numOneQ = 5,
                           .tieBreakClass = 6,
                           .enumerationIndex = 7};
  const CandidateScore rhs{.weighted = 2.0,
                           .numTwoQ = 3,
                           .depth = 4,
                           .numOneQ = 5,
                           .tieBreakClass = 6,
                           .enumerationIndex = 8};
  EXPECT_TRUE(isBetterScore(lhs, rhs));
}

TEST(NativeSynthesisScoringTest, SelectBestCandidateReturnsNullForEmptyInput) {
  using namespace mlir::qco::native_synth;
  const llvm::SmallVector<SynthesisCandidate<ScoringTag>, 0> empty;
  EXPECT_EQ(selectBestCandidate(llvm::ArrayRef(empty), ScoreWeights{}),
            nullptr);
}

TEST(NativeSynthesisScoringTest, SelectBestCandidatePicksLowestWeighted) {
  using namespace mlir::qco::native_synth;
  llvm::SmallVector<SynthesisCandidate<ScoringTag>, 3> candidates(3);
  candidates[0].metrics.numTwoQ = 4U;
  candidates[1].metrics.numTwoQ = 1U;
  candidates[2].metrics.numTwoQ = 2U;

  const auto* best =
      selectBestCandidate(llvm::ArrayRef(candidates), ScoreWeights{});
  ASSERT_NE(best, nullptr);
  EXPECT_EQ(best, &candidates[1]);
}

TEST(NativeSynthesisScoringTest, SelectBestCandidateHonoursWeightPreferences) {
  using namespace mlir::qco::native_synth;
  llvm::SmallVector<SynthesisCandidate<ScoringTag>, 2> candidates(2);
  candidates[0].metrics.numTwoQ = 2U;
  candidates[0].metrics.numOneQ = 0U;
  candidates[1].metrics.numTwoQ = 1U;
  candidates[1].metrics.numOneQ = 20U;

  EXPECT_EQ(selectBestCandidate(llvm::ArrayRef(candidates), ScoreWeights{}),
            candidates.data());

  const ScoreWeights heavyTwoQ{.twoQ = 10.0, .oneQ = 0.01, .depth = 0.0};
  EXPECT_EQ(selectBestCandidate(llvm::ArrayRef(candidates), heavyTwoQ),
            &candidates[1]);
}

TEST(NativeSynthesisScoringTest,
     SelectBestCandidateTieBreaksByEnumerationOrder) {
  using namespace mlir::qco::native_synth;
  llvm::SmallVector<SynthesisCandidate<ScoringTag>, 3> candidates(3);
  candidates[0].enumerationIndex = 2U;
  candidates[1].enumerationIndex = 0U;
  candidates[2].enumerationIndex = 1U;

  const auto* best =
      selectBestCandidate(llvm::ArrayRef(candidates), ScoreWeights{});
  ASSERT_NE(best, nullptr);
  EXPECT_EQ(best, &candidates[1]);
}

TEST_F(NativeSynthesisPassTest, XxPlusMinusYyEmittedCountsMatchScoringMetrics) {
  using namespace mlir::qco::native_synth;

  const auto runRewriteCase =
      [&](void (*emitProgram)(mlir::qc::QCProgramBuilder&)) {
        OwningOpRef<ModuleOp> module =
            mlir::qc::QCProgramBuilder::build(context.get(), emitProgram);

        PassManager pm(context.get());
        pm.addPass(createQCToQCO());
        ASSERT_TRUE(succeeded(pm.run(*module)));

        Operation* twoQOp = nullptr;
        module->walk([&](Operation* op) {
          if (llvm::isa<qco::XXPlusYYOp, qco::XXMinusYYOp>(op)) {
            twoQOp = op;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        ASSERT_NE(twoQOp, nullptr);

        IRRewriter rewriter(context.get());
        ASSERT_TRUE(succeeded(rewriteXXPlusMinusYYViaRzz(rewriter, twoQOp)));

        const auto expected = xxPlusMinusYyRzzRewriteScoringMetrics();
        const auto [numOneQ, numTwoQ] =
            countSingleAndTwoQubitUnitariesForXxRzzMetrics(*module);
        EXPECT_EQ(numOneQ, expected.numOneQ);
        EXPECT_EQ(numTwoQ, expected.numTwoQ);
      };

  runRewriteCase(mlir::qc::nativeSynthScoringXxPlusYyOnly);
  runRewriteCase(mlir::qc::nativeSynthScoringXxMinusYyOnly);
}
