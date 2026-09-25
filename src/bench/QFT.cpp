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
#include "support/Diagnostics.hpp"

#include "mlir/Support/LogicalResult.h"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace mqt::bench {

mlir::FailureOr<QFT> QFT::create(QFTOptions options) {
  if (options.qubits == 0 || options.qubits > QFTOptions::MAX_QUBITS) {
    return ::mqt::emitError("QFT qubits must be between 1 and 1000000",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  if (options.periodExponent > options.qubits ||
      options.periodExponent > QFTOptions::MAX_PERIOD_EXPONENT) {
    return ::mqt::emitError(
        "QFT period exponent must be at most the qubit count and 1074",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  if (options.method != QFTMethod::Standard &&
      options.method != QFTMethod::Semiclassical) {
    return ::mqt::emitError("unknown QFT method",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  return QFT(options);
}

QFT::QFT(QFTOptions options)
    : options_(options), output_{.name = "result", .width = options_.qubits} {}

const QFTOptions& QFT::options() const noexcept { return options_; }

const Output& QFT::output() const noexcept { return output_; }

mlir::FailureOr<double> QFT::probability(const std::string_view outcome) const {
  if (mlir::failed(detail::validateOutcome(outcome, output_.width))) {
    return mlir::failure();
  }
  if (!std::ranges::all_of(outcome.substr(options_.periodExponent),
                           [](const char bit) { return bit == '0'; })) {
    return 0.;
  }
  return std::ldexp(1., -static_cast<int>(options_.periodExponent));
}

mlir::FailureOr<Evaluation> QFT::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
