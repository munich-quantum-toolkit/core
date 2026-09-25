/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/GHZ.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"
#include "support/Diagnostics.hpp"

#include "mlir/Support/LogicalResult.h"

#include <cmath>
#include <cstddef>
#include <string_view>

namespace mqt::bench {

mlir::FailureOr<GHZ> GHZ::create(GHZOptions options) {
  if (options.qubits == 0 || options.qubits > GHZOptions::MAX_QUBITS) {
    return ::mqt::emitError("GHZ qubits must be between 1 and 1000000",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  if (options.topology != GHZTopology::Linear &&
      options.topology != GHZTopology::Star) {
    return ::mqt::emitError("unknown GHZ topology",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  if (options.basis != GHZBasis::Z && options.basis != GHZBasis::X) {
    return ::mqt::emitError("unknown GHZ measurement basis",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  if (options.basis == GHZBasis::X &&
      options.qubits > GHZOptions::MAX_X_BASIS_QUBITS) {
    return ::mqt::emitError("GHZ X-basis qubits must be between 1 and 1075",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  return GHZ(options);
}

GHZ::GHZ(GHZOptions options)
    : options_(options), output_{.name = "result", .width = options_.qubits} {}

const GHZOptions& GHZ::options() const noexcept { return options_; }

const Output& GHZ::output() const noexcept { return output_; }

mlir::FailureOr<double> GHZ::probability(const std::string_view outcome) const {
  if (mlir::failed(detail::validateOutcome(outcome, output_.width))) {
    return mlir::failure();
  }

  if (options_.basis == GHZBasis::Z) {
    const auto allZero = outcome.find('1') == std::string_view::npos;
    const auto allOne = outcome.find('0') == std::string_view::npos;
    return allZero || allOne ? 0.5 : 0.;
  }

  size_t ones = 0;
  for (const auto bit : outcome) {
    ones += bit == '1' ? 1U : 0U;
  }
  if (ones % 2U != 0U) {
    return 0.;
  }
  return std::ldexp(1., 1 - static_cast<int>(options_.qubits));
}

mlir::FailureOr<Evaluation> GHZ::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
