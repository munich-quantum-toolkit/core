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
enum class QFTAdderMethod : uint8_t {
  /// Store the addend in a quantum register.
  Register,
  /// Compile the addend into constant phases.
  Constant,
};
/// Wrap the sum at the operand width or retain an extra carry bit.
enum class QFTAdderOverflow : uint8_t {
  /// Compute the sum modulo \f$2^n\f$.
  Wrap,
  /// Retain the carry in an extra sum qubit.
  Carry,
};

/// Parameters for a QFT adder benchmark.
struct QFTAdderOptions {
  /// Maximum sum-register width, including an optional carry bit.
  static constexpr size_t MAX_QUBITS = 1'024;

  /// Big-endian addend; register inputs also allow '+' for a
  /// \f$|+\rangle\f$ qubit.
  std::string addend;
  /// Big-endian binary accumulator, with the same width as the addend.
  std::string accumulator;
  /// How the addend enters the circuit.
  QFTAdderMethod method = QFTAdderMethod::Register;
  /// Wrap or carry behavior for the sum.
  QFTAdderOverflow overflow = QFTAdderOverflow::Wrap;
};

/// A validated QFT adder benchmark.
///
/// Register results concatenate the addend and sum; constant results contain
/// only the sum. Wrap mode computes \f$(a+b) \bmod 2^n\f$; carry mode extends
/// the sum by one bit. All strings are big-endian; leading zeros determine the
/// operand width.
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
