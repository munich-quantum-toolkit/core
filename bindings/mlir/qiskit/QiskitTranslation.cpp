/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QiskitTranslation.h"

#include <llvm/ADT/StringSet.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mqt::bindings::qiskit {

uint32_t validateRegisterLayout(const std::vector<Register>& registers,
                                const uint32_t total,
                                const std::string_view kind) {
  std::vector<bool> inRegister(total, false);
  llvm::StringSet<> names;
  for (const auto& reg : registers) {
    if (reg.name.empty() || !names.insert(reg.name).second) {
      throw std::runtime_error("Qiskit requires unique, non-empty " +
                               std::string(kind) + " register names");
    }
    if (reg.bits.empty()) {
      throw std::runtime_error("Qiskit does not support empty " +
                               std::string(kind) + " registers");
    }
    for (const auto bit : reg.bits) {
      if (bit >= total || inRegister[bit]) {
        throw std::runtime_error(
            "Qiskit circuit translation requires disjoint " +
            std::string(kind) + " register membership");
      }
      inRegister[bit] = true;
    }
  }
  const auto firstRegistered = std::ranges::find(inRegister, true);
  const auto loose =
      static_cast<uint32_t>(firstRegistered - inRegister.begin());
  uint32_t expected = loose;
  for (const auto& reg : registers) {
    for (const auto bit : reg.bits) {
      if (bit != expected) {
        throw std::runtime_error("Qiskit circuit translation requires loose " +
                                 std::string(kind) +
                                 " bits before contiguous registers");
      }
      ++expected;
    }
  }
  if (expected != total) {
    throw std::runtime_error("Qiskit circuit translation requires loose " +
                             std::string(kind) +
                             " bits before contiguous registers");
  }
  return loose;
}

std::vector<std::complex<double>>
reverseQubitOrder(const std::span<const std::complex<double>> matrix,
                  const size_t numQubits) {
  if (numQubits >= std::numeric_limits<size_t>::digits / 2U) {
    throw std::runtime_error("unitary matrix is too large to represent safely");
  }
  const auto dimension = size_t{1} << numQubits;
  if (matrix.size() != dimension * dimension) {
    throw std::runtime_error(
        "unitary matrix size does not match its qubit count");
  }
  const auto reverseIndex = [numQubits](const size_t index) {
    size_t reversed = 0U;
    for (size_t bit = 0U; bit < numQubits; ++bit) {
      reversed = (reversed << 1U) | ((index >> bit) & 1U);
    }
    return reversed;
  };
  std::vector<std::complex<double>> result(matrix.size());
  for (size_t row = 0U; row < dimension; ++row) {
    for (size_t column = 0U; column < dimension; ++column) {
      result[(row * dimension) + column] =
          matrix[(reverseIndex(row) * dimension) + reverseIndex(column)];
    }
  }
  return result;
}

} // namespace mqt::bindings::qiskit
