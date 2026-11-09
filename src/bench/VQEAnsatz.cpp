/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/VQEAnsatz.hpp"

#include "bench/Evaluation.hpp"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>

namespace mqt::bench {

VQEAnsatz::VQEAnsatz(const VQEAnsatzOptions options)
    : options_(options), output_{.name = "result", .width = options_.qubits} {}

llvm::Expected<VQEAnsatz> VQEAnsatz::create(VQEAnsatzOptions options) {
  if (options.qubits == 0) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "VQE ansatz qubits must be positive");
  }
  if (options.depth == 0) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "VQE ansatz depth must be positive");
  }
  const auto maximum = static_cast<size_t>(std::numeric_limits<int64_t>::max());
  if (options.depth >= maximum / options.qubits) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "VQE ansatz parameter count must fit circuit input dimensions");
  }
  return VQEAnsatz(options);
}

const VQEAnsatzOptions& VQEAnsatz::options() const noexcept { return options_; }

const Output& VQEAnsatz::output() const noexcept { return output_; }

size_t VQEAnsatz::numParameters() const noexcept {
  return options_.qubits * (options_.depth + 1);
}

} // namespace mqt::bench
