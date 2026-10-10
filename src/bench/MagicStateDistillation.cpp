/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/MagicStateDistillation.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"
#include "support/Diagnostics.hpp"

#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace mqt::bench {

llvm::FailureOr<MagicStateDistillation>
MagicStateDistillation::create(MagicStateDistillationOptions options) {
  if (options.levels == 0 ||
      options.levels >
          static_cast<size_t>(std::numeric_limits<int64_t>::max()) / 5) {
    return ::mqt::emitError("magic-state-distillation levels must be positive "
                            "and fit circuit dimensions",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  return MagicStateDistillation(options);
}

MagicStateDistillation::MagicStateDistillation(
    MagicStateDistillationOptions options)
    : options_(options), output_{.name = "result", .width = 2} {}

const MagicStateDistillationOptions&
MagicStateDistillation::options() const noexcept {
  return options_;
}

const Output& MagicStateDistillation::output() const noexcept {
  return output_;
}

llvm::FailureOr<double>
MagicStateDistillation::probability(const std::string_view outcome) const {
  if (llvm::failed(detail::validateOutcome(outcome, output_.width))) {
    return llvm::failure();
  }
  return outcome == "00" ? 1. : 0.;
}

llvm::FailureOr<Evaluation>
MagicStateDistillation::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, "00");
}

} // namespace mqt::bench
