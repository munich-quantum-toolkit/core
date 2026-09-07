/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "bench/Evaluation.hpp"
#include "bench/mqt_core_bench_export.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace mqt::bench {

/// Store the addend in a quantum register or compile it into constant phases.
enum class QFTAdderMethod : uint8_t { Register, Constant };
/// Wrap the sum at the operand width or retain an extra carry bit.
enum class QFTAdderOverflow : uint8_t { Wrap, Carry };

/// Parameters for a QFT adder benchmark.
struct QFTAdderOptions {
  static constexpr size_t MAX_QUBITS = 1'024;

  /// Big-endian addend; register inputs also allow '+' for a |+> qubit.
  std::string addend;
  /// Big-endian binary accumulator, with the same width as the addend.
  std::string accumulator;
  QFTAdderMethod method = QFTAdderMethod::Register;
  QFTAdderOverflow overflow = QFTAdderOverflow::Wrap;
};

/// Add two configured operands using an exact no-swap QFT circuit.
/// Register results concatenate the addend and sum; constant results contain
/// only the sum. Carry mode extends the sum by one bit. All strings are
/// big-endian; leading zeros determine the operand width.
class MQT_CORE_BENCH_EXPORT QFTAdder final {
public:
  explicit QFTAdder(QFTAdderOptions options);

  [[nodiscard]] const QFTAdderOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the unique logical outcome, or no value for a superposed addend.
  [[nodiscard]] const std::optional<std::string>&
  expectedResult() const noexcept;
  /// Return the ideal probability of a logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  QFTAdderOptions options_;
  Output output_;
  std::optional<std::string> expectedResult_;
};

} // namespace mqt::bench
