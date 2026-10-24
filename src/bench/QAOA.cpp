/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QAOA.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <system_error>

namespace mqt::bench {

QAOA::QAOA(const QAOAOptions options)
    : options_(options), output_{.name = "result", .width = options_.qubits} {}

llvm::Expected<QAOA> QAOA::create(QAOAOptions options) {
  if (options.qubits < 3 ||
      options.qubits >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "QAOA qubits must be at least 3 and fit circuit dimensions");
  }
  if (options.depth == 0 || options.depth > QAOAOptions::MAX_DEPTH) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "QAOA depth must be positive and fit circuit input dimensions");
  }
  return QAOA(options);
}

const QAOAOptions& QAOA::options() const noexcept { return options_; }

const Output& QAOA::output() const noexcept { return output_; }

llvm::Expected<size_t> QAOA::cutSize(const std::string_view outcome) const {
  if (auto error = detail::validateOutcome(outcome, output_.width)) {
    return error;
  }
  auto cut = size_t{outcome.front() != outcome.back()};
  for (size_t index = 1; index < outcome.size(); ++index) {
    if (outcome[index - 1] != outcome[index]) {
      ++cut;
    }
  }
  return cut;
}

llvm::Expected<QAOAEvaluation> QAOA::evaluate(const Counts& counts) const {
  auto shots = detail::countShots(counts);
  if (!shots) {
    return shots.takeError();
  }
  // Extended precision preserves weighted scores without integer overflow.
  // NOLINTBEGIN(google-runtime-float)
  long double weightedCut = 0.L;
  for (const auto& [outcome, count] : counts) {
    auto cut = cutSize(outcome);
    if (!cut) {
      return cut.takeError();
    }
    weightedCut +=
        static_cast<long double>(*cut) * static_cast<long double>(count);
  }
  const auto optimum = options_.qubits - (options_.qubits % 2);
  const auto mean = std::min(weightedCut / static_cast<long double>(*shots),
                             static_cast<long double>(optimum));
  return QAOAEvaluation{
      .meanCut = static_cast<double>(mean),
      .approximationRatio =
          static_cast<double>(mean / static_cast<long double>(optimum)),
  };
  // NOLINTEND(google-runtime-float)
}

} // namespace mqt::bench
