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

#include "llvm/Support/Error.h"

#include <string_view>
#include <system_error>
#include <utility>

namespace mqt::bench {

BV::BV(BVOptions options)
    : options_(std::move(options)),
      output_{.name = "result", .width = options_.hiddenBitstring.size()} {}

llvm::Expected<BV> BV::create(BVOptions options) {
  const auto width = options.hiddenBitstring.size();
  if (width == 0 || width > BVOptions::MAX_BITS) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Bernstein--Vazirani requires a hidden bitstring of width 1 through "
        "1000000");
  }
  if (auto error = detail::validateOutcome(options.hiddenBitstring, width)) {
    return error;
  }
  if (options.method != BVMethod::Static &&
      options.method != BVMethod::Dynamic) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "unknown Bernstein--Vazirani method");
  }
  return BV(std::move(options));
}

const BVOptions& BV::options() const noexcept { return options_; }

const Output& BV::output() const noexcept { return output_; }

llvm::Expected<double> BV::probability(const std::string_view outcome) const {
  if (auto error = detail::validateOutcome(outcome, output_.width)) {
    return error;
  }
  return outcome == options_.hiddenBitstring ? 1. : 0.;
}

llvm::Expected<Evaluation> BV::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, options_.hiddenBitstring);
}

} // namespace mqt::bench
