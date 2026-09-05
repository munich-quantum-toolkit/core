/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFTAdderQuantum.hpp"

#include "EvaluationUtils.hpp"
#include "bench/Evaluation.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace mqt::bench {
namespace {

[[nodiscard]] bool isIncrement(const std::string_view addend,
                               const std::string_view sum) {
  auto carry = true;
  for (size_t index = addend.size(); index > 0; --index) {
    const auto addendBit = addend[index - 1] == '1';
    const auto expectedSumBit = addendBit != carry;
    if ((sum[index - 1] == '1') != expectedSumBit) {
      return false;
    }
    carry = addendBit && carry;
  }
  return true;
}

} // namespace

QFTAdderQuantum::QFTAdderQuantum(QFTAdderQuantumOptions options)
    : options_(options),
      output_{.name = "result", .width = 2 * options_.qubits} {
  if (options_.qubits == 0 ||
      options_.qubits > QFTAdderQuantumOptions::MAX_QUBITS) {
    throw std::invalid_argument(
        "quantum QFT adder qubits must be between 1 and 1024");
  }
}

const QFTAdderQuantumOptions& QFTAdderQuantum::options() const noexcept {
  return options_;
}

const Output& QFTAdderQuantum::output() const noexcept { return output_; }

double QFTAdderQuantum::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  const auto addend = outcome.substr(0, options_.qubits);
  const auto sum = outcome.substr(options_.qubits);
  if (!isIncrement(addend, sum)) {
    return 0.;
  }
  return std::ldexp(1., -static_cast<int>(options_.qubits));
}

Evaluation QFTAdderQuantum::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); });
}

} // namespace mqt::bench
