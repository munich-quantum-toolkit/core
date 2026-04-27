/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>

#include <cmath>
#include <tuple>

/// Deterministic candidate scoring and selection. All comparisons are total
/// orders, so the same input always picks the same candidate.

namespace mlir::qco::native_synth {

/// Primary cost `weighted`; when two weighted scores agree within FP tolerance,
/// `isBetterScore` breaks ties in order: `numTwoQ`, `depth`, `numOneQ`,
/// `tieBreakClass`, `enumerationIndex`.
struct CandidateScore {
  double weighted = 0.0;
  unsigned numTwoQ = 0;
  unsigned depth = 0;
  unsigned numOneQ = 0;
  unsigned tieBreakClass = 0;
  unsigned enumerationIndex = 0;
};

/// Project a candidate onto its `CandidateScore`.
template <typename Payload>
CandidateScore scoreCandidate(const SynthesisCandidate<Payload>& candidate,
                              const ScoreWeights& weights) {
  return {
      .weighted =
          (weights.twoQ * static_cast<double>(candidate.metrics.numTwoQ)) +
          (weights.oneQ * static_cast<double>(candidate.metrics.numOneQ)) +
          (weights.depth * static_cast<double>(candidate.metrics.depth)),
      .numTwoQ = candidate.metrics.numTwoQ,
      .depth = candidate.metrics.depth,
      .numOneQ = candidate.metrics.numOneQ,
      .tieBreakClass = static_cast<unsigned>(candidate.candidateClass),
      .enumerationIndex = candidate.enumerationIndex,
  };
}

/// Strict less-than: `true` iff `lhs` is a strictly better candidate than
/// `rhs`.
inline bool isBetterScore(const CandidateScore& lhs,
                          const CandidateScore& rhs) {
  constexpr double scoreTolerance = 1e-12;
  if (std::abs(lhs.weighted - rhs.weighted) > scoreTolerance) {
    return lhs.weighted < rhs.weighted;
  }
  const auto lhsTie = std::tie(lhs.numTwoQ, lhs.depth, lhs.numOneQ,
                               lhs.tieBreakClass, lhs.enumerationIndex);
  const auto rhsTie = std::tie(rhs.numTwoQ, rhs.depth, rhs.numOneQ,
                               rhs.tieBreakClass, rhs.enumerationIndex);
  return lhsTie < rhsTie;
}

/// Return the best candidate by `isBetterScore`, or `nullptr` on empty input.
template <typename Candidate>
const Candidate* selectBestCandidate(llvm::ArrayRef<Candidate> candidates,
                                     const ScoreWeights& weights) {
  if (candidates.empty()) {
    return nullptr;
  }
  const auto* best = &candidates.front();
  auto bestScore = scoreCandidate(*best, weights);
  for (const auto& candidate : llvm::drop_begin(candidates)) {
    const auto candidateScore = scoreCandidate(candidate, weights);
    if (isBetterScore(candidateScore, bestScore)) {
      best = &candidate;
      bestScore = candidateScore;
    }
  }
  return best;
}

} // namespace mlir::qco::native_synth
