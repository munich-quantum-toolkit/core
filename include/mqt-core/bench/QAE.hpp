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

#include "bench/Evaluation.hpp"
#include "bench/mqt_core_bench_export.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mqt::bench {

/// State preparation and oracle for quantum amplitude estimation.
struct QAEProblem {
  /// Distinct big-endian good states of equal width, leaving at least one bad
  /// state.
  std::vector<std::string> goodStates;
  /// Total probability of the good states before amplification.
  ///
  /// Preparation distributes each class's probability uniformly within that
  /// class.
  double amplitude;
};

/// Parameters for one QAE circuit.
///
/// Supply exactly one of precision and iterations.
struct QAEOptions {
  static constexpr size_t MAX_QUBITS = 63;
  static constexpr size_t MAX_PRECISION = 31;
  static constexpr size_t MAX_ITERATIONS = std::numeric_limits<int32_t>::max();

  QAEProblem problem;
  /// Phase-register width for canonical QAE.
  std::optional<size_t> precision = std::nullopt;
  /// Grover iterations before reading the data register in an amplification
  /// experiment.
  std::optional<size_t> iterations = std::nullopt;
};

/// A validated QAE phase-estimation or amplitude-amplification circuit.
class MQT_CORE_BENCH_EXPORT QAE final {
public:
  [[nodiscard]] static llvm::Expected<QAE> create(QAEOptions options);

  [[nodiscard]] const QAEOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  ///
  /// Requires a binary outcome of output().width bits.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] llvm::Expected<Evaluation> evaluate(const Counts& counts) const;

private:
  explicit QAE(QAEOptions options);

  QAEOptions options_;
  Output output_;
};

/// Execute one circuit and return fresh counts.
///
/// The callback selects the backend and shots and must be nonempty. Report
/// failures through Expected. The workflow borrows the callback for the call.
using QAERun = llvm::function_ref<llvm::Expected<Counts>(const QAE&)>;

/// Estimate the good-state probability from the most frequent folded phase bin.
[[nodiscard]] MQT_CORE_BENCH_EXPORT llvm::Expected<double>
phaseEstimation(const QAEProblem& problem, size_t precision, QAERun run);

/// Fit the joint binomial likelihood at powers 1, 2, ..., 2^(rounds-1).
[[nodiscard]] MQT_CORE_BENCH_EXPORT llvm::Expected<double>
maximumLikelihood(const QAEProblem& problem, size_t rounds, QAERun run);

/// Adapt the Grover power to a confidence interval inferred from measured
/// counts.
///
/// Return probability bounds at most twice epsilon apart. For the stated
/// confidence, each callback must supply an independent fixed-size shot batch.
/// Return an error if the observations are inconsistent or the batch budget is
/// exhausted.
[[nodiscard]] MQT_CORE_BENCH_EXPORT llvm::Expected<std::pair<double, double>>
iterativeAmplitudeEstimation(const QAEProblem& problem, double epsilon,
                             double confidence, size_t maxBatches, QAERun run);

} // namespace mqt::bench
