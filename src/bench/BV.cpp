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

#include "EvaluationUtils.hpp"
#include "bench/Evaluation.hpp"

#include <stdexcept>
#include <string_view>
#include <utility>

namespace mqt::bench {

BV::BV(BVOptions options)
    : options_(std::move(options)),
      output_{.name = "result", .width = options_.hiddenBitstring.size()} {
  const auto width = options_.hiddenBitstring.size();
  if (width == 0 || width > BVOptions::MAX_BITS) {
    throw std::invalid_argument(
        "Bernstein--Vazirani requires a hidden bitstring of width 1 through "
        "1000000");
  }
  detail::validateOutcome(options_.hiddenBitstring, width);
  if (options_.method != BVMethod::Static &&
      options_.method != BVMethod::Dynamic) {
    throw std::invalid_argument("unknown Bernstein--Vazirani method");
  }
}

const BVOptions& BV::options() const noexcept { return options_; }

const Output& BV::output() const noexcept { return output_; }

double BV::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  return outcome == options_.hiddenBitstring ? 1. : 0.;
}

Evaluation BV::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); },
      options_.hiddenBitstring);
}

} // namespace mqt::bench
