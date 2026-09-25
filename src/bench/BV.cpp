/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"
#include "support/Diagnostics.hpp"

#include "mlir/Support/LogicalResult.h"

#include <string_view>
#include <utility>

namespace mqt::bench {

mlir::FailureOr<BV> BV::create(BVOptions options) {
  const auto width = options.hiddenBitstring.size();
  if (width == 0 || width > BVOptions::MAX_BITS) {
    return ::mqt::emitError(
        "Bernstein--Vazirani requires a hidden bitstring of width 1 "
        "through "
        "1000000",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  if (mlir::failed(detail::validateOutcome(options.hiddenBitstring, width))) {
    return mlir::failure();
  }
  if (options.method != BVMethod::Static &&
      options.method != BVMethod::Dynamic) {
    return ::mqt::emitError("unknown Bernstein--Vazirani method",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  return BV(std::move(options));
}

BV::BV(BVOptions options)
    : options_(std::move(options)),
      output_{.name = "result", .width = options_.hiddenBitstring.size()} {}

const BVOptions& BV::options() const noexcept { return options_; }

const Output& BV::output() const noexcept { return output_; }

mlir::FailureOr<double> BV::probability(const std::string_view outcome) const {
  if (mlir::failed(detail::validateOutcome(outcome, output_.width))) {
    return mlir::failure();
  }
  return outcome == options_.hiddenBitstring ? 1. : 0.;
}

mlir::FailureOr<Evaluation> BV::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, options_.hiddenBitstring);
}

} // namespace mqt::bench
