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

#include "llvm/Support/LogicalResult.h"

#include <string_view>
#include <utility>

namespace mqt::bench {

llvm::FailureOr<BV> BV::create(BVOptions options) {
  const auto width = options.hiddenBitstring.size();
  if (width == 0 || width > BVOptions::MAX_BITS) {
    return ::mqt::emitError(
        "Bernstein--Vazirani requires a hidden bitstring of width 1 "
        "through "
        "1000000",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  if (llvm::failed(detail::validateOutcome(options.hiddenBitstring, width))) {
    return llvm::failure();
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

llvm::FailureOr<double> BV::probability(const std::string_view outcome) const {
  if (llvm::failed(detail::validateOutcome(outcome, output_.width))) {
    return llvm::failure();
  }
  return outcome == options_.hiddenBitstring ? 1. : 0.;
}

llvm::FailureOr<Evaluation> BV::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, options_.hiddenBitstring);
}

} // namespace mqt::bench
