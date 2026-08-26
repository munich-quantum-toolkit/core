/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file OpenQASMSerializer.hpp
 * @brief OpenQASM serialization for the classic intermediate representation.
 */

#pragma once

#include "ir/Definitions.hpp"
#include "ir/Register.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>

namespace qc {

class Operation;
class QuantumComputation;

using QubitIndexToRegisterMap =
    std::unordered_map<Qubit, std::pair<QuantumRegister, std::string>>;
using BitIndexToRegisterMap =
    std::unordered_map<Bit, std::pair<ClassicalRegister, std::string>>;

/**
 * @brief Serializes classic IR circuits and operations to OpenQASM.
 */
class OpenQASMSerializer final {
public:
  explicit OpenQASMSerializer(std::ostream& output,
                              Format format = Format::OpenQASM3)
      : output(output), format(format) {}

  /**
   * @brief Serialize a complete quantum computation.
   * @param computation The computation to serialize.
   */
  void serialize(const QuantumComputation& computation) const;

  /**
   * @brief Serialize a single operation without a circuit header.
   * @param operation The operation to serialize.
   * @param qubitMap Map from qubit indices to their register and operand name.
   * @param bitMap Map from bit indices to their register and operand name.
   * @param indent Nesting level, using two spaces per level.
   */
  void serialize(const Operation& operation,
                 const QubitIndexToRegisterMap& qubitMap,
                 const BitIndexToRegisterMap& bitMap,
                 std::size_t indent = 0U) const;

private:
  std::ostream& output;
  Format format;
};

} // namespace qc
