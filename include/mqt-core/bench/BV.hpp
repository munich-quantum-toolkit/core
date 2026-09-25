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

#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace mqt::bench {

/// Circuit method used for Bernstein--Vazirani.
enum class BVMethod : uint8_t {
  /// Allocate one query qubit for each hidden bit.
  Static,
  /// Measure, reset, and reuse one query qubit for every hidden bit.
  Dynamic,
};

/// Parameters for one Bernstein--Vazirani benchmark instance.
struct BVOptions {
  static constexpr size_t MAX_BITS = 1'000'000;

  /// Big-endian hidden bitstring.
  std::string hiddenBitstring;
  /// Qubit-allocation and measurement method.
  BVMethod method = BVMethod::Static;
};

/// A validated Bernstein--Vazirani benchmark.
class MQT_CORE_BENCH_EXPORT BV final {
public:
  [[nodiscard]] static mlir::FailureOr<BV> create(BVOptions options);

  [[nodiscard]] const BVOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] mlir::FailureOr<double>
  probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] mlir::FailureOr<Evaluation>
  evaluate(const Counts& counts) const;

private:
  explicit BV(BVOptions options);

  BVOptions options_;
  Output output_;
};

} // namespace mqt::bench
