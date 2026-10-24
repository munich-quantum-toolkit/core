/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFT.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <system_error>

namespace mqt::bench {

QFT::QFT(QFTOptions options)
    : options_(options), output_{.name = "result", .width = options_.qubits} {}

llvm::Expected<QFT> QFT::create(QFTOptions options) {
  if (options.qubits == 0 || options.qubits > QFTOptions::MAX_QUBITS) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "QFT qubits must be between 1 and 1000000");
  }
  if (options.periodExponent > options.qubits ||
      options.periodExponent > QFTOptions::MAX_PERIOD_EXPONENT) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "QFT period exponent must be at most the qubit count and 1074");
  }
  if (options.method != QFTMethod::Standard &&
      options.method != QFTMethod::Semiclassical) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "unknown QFT method");
  }
  return QFT(options);
}

const QFTOptions& QFT::options() const noexcept { return options_; }

const Output& QFT::output() const noexcept { return output_; }

llvm::Expected<double> QFT::probability(const std::string_view outcome) const {
  if (auto error = detail::validateOutcome(outcome, output_.width)) {
    return error;
  }
  if (!std::ranges::all_of(outcome.substr(options_.periodExponent),
                           [](const char bit) { return bit == '0'; })) {
    return 0.;
  }
  return std::ldexp(1., -static_cast<int>(options_.periodExponent));
}

llvm::Expected<Evaluation> QFT::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
